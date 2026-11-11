# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1 · Notebook 3 — Your first Impulse report + an AI/BI dashboard
# MAGIC
# MAGIC **⏱ ~45 minutes · 🎯 Goal: compute a fleet report for all 100 trips, store it in *your* schema as a
# MAGIC gold star schema, and put it on an AI/BI dashboard.**
# MAGIC
# MAGIC ```
# MAGIC   shared silver (dbx_workshop_dev.shared)         your gold (USER_SCHEMA)                  AI/BI dashboard
# MAGIC   impulse_silver_*  ──►  Report: events + aggregations  ──►  impulse_w1_*_fact / _dimension  ──►  charts & filters
# MAGIC ```
# MAGIC
# MAGIC | # | Section |
# MAGIC |---|---|
# MAGIC | 1 | Configure the report (source = shared silver, sink = your schema) |
# MAGIC | 2 | Signals: physical channels and an **alias** |
# MAGIC | 3 | Events: *when* to look |
# MAGIC | 4 | Aggregations: *what* to compute |
# MAGIC | 5 | Run it |
# MAGIC | 6 | Explore the gold star schema with SQL |
# MAGIC | 7 | Build the AI/BI dashboard (by hand, and automatically) |

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

require_user_schema()
setup_impulse()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient

user_catalog, user_schema = USER_SCHEMA.split(".")
GOLD = f"{USER_SCHEMA}.{W1_GOLD_PREFIX}"     # every gold table is <your schema>.impulse_w1_<entity>
print("gold tables will be written as", GOLD + "_*")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Configure the report
# MAGIC
# MAGIC The config is a plain dict (validated by Impulse with Pydantic):
# MAGIC
# MAGIC * **`source`** — the shared silver tables (same as notebook 1).
# MAGIC * **`unity_sink`** — where to write: **your** catalog + schema, and a `table_prefix` so every output
# MAGIC   table is called `impulse_w1_<something>`.
# MAGIC * **`query_engine`** — `DefaultSolver` over RLE data.
# MAGIC * **`measurement_dimensions`** — which `container_metrics` columns to copy into the gold
# MAGIC   `measurement_dimension` table. These become your dashboard's filters and groupings.

# COMMAND ----------

config = {
    "source": impulse_source(SILVER),
    "unity_sink": {"catalog": user_catalog, "schema": user_schema, "table_prefix": W1_GOLD_PREFIX,
                   "cleanup_temp_tables": True},
    "query_engine": {"solver": "DefaultSolver", "data_type": "RLE"},
    "measurement_dimensions": ["container_id", "start_ts", "stop_ts", "trip_start_dt", "duration_s", "vehicle_key",
                               "manufacturer", "powertrain_type", "scenario", "driver", "starting_temperature_c"],
}

from impulse_reporting.core.report import Report
report = Report(name=W1_REPORT_NAME, spark=spark, workspace_client=WorkspaceClient(), config=config)
db = report.get_db()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Signals
# MAGIC
# MAGIC The instructor's silver layer defines **channel aliases**: stable, logical names that resolve to the
# MAGIC physical signal and **convert units** on the way. `channel_with_alias(channel_alias="vehicle_speed")`
# MAGIC returns `Vhcl.v` already in **km/h**. (Workshop 2 shows how aliases work under the hood.)
# MAGIC
# MAGIC The data quirks from `DATA_NOTES.md` are handled right here, in the signal definitions:
# MAGIC
# MAGIC | Signal | Handling |
# MAGIC |---|---|
# MAGIC | speed | drop spikes above 100 m/s (= 360 km/h) |
# MAGIC | SOC (*state of charge*: % the HV battery is full) | ignore SOC ≤ 0 (battery management system start-up) |
# MAGIC | `PT.BattHV.Temp` | logged in **Kelvin** although labelled °C → subtract 273.15 |

# COMMAND ----------

v_kmh = db.query.channel_with_alias(channel_alias="vehicle_speed")       # Vhcl.v converted to km/h
speed_kmh = v_kmh.where(v_kmh < VHCL_V_MAX_MPS * 3.6)                    # without the > 100 m/s spikes
accel = db.query.channel(channel_name="Car.ax")                          # longitudinal acceleration [m/s²]
soc_raw = db.query.channel(channel_name="PT.BCU.BattHV.SOC")
soc = soc_raw.where(soc_raw > 0)                                         # [%], BMS start-up zeros ignored
current = db.query.channel(channel_name="PT.BattHV.Current")             # [A], negative = into the battery
batt_temp = db.query.channel(channel_name="PT.BattHV.Temp") - KELVIN_OFFSET   # logged in K -> [°C]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Events — *when* to look
# MAGIC
# MAGIC | Event | Type | Definition |
# MAGIC |---|---|---|
# MAGIC | `trip` | `ContainerEvent` | the whole recording — for per-trip statistics |
# MAGIC | `driving` | `BasicEvent` | vehicle faster than 3.6 km/h |
# MAGIC | `charging` | `BasicEvent` | battery current < −10 A at standstill, for more than 60 s |
# MAGIC
# MAGIC Each contiguous window where the condition holds becomes one **event instance** in the gold
# MAGIC `event_instance_fact` table. Every event must be registered with `report.add_event(...)`.

# COMMAND ----------

from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.container_event import ContainerEvent

trip = ContainerEvent(name="trip", desc="The whole trip")
driving = BasicEvent(name="driving", expr=speed_kmh > 3.6, desc="Vehicle moving faster than 3.6 km/h",
                     required_channels=["vehicle_speed"])
charging = BasicEvent(name="charging", expr=((current < -10) & (speed_kmh < 0.36)).filter(60e6),
                      desc="HV current below -10 A at standstill for more than 60 s",
                      required_channels=["PT.BattHV.Current", "vehicle_speed"])
for event in (trip, driving, charging):
    report.add_event(event)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Aggregations — *what* to compute
# MAGIC
# MAGIC Aggregations live on **pages** (just a way to group them). Every aggregation is computed **per trip**
# MAGIC and, if given an event, only inside that event's windows.
# MAGIC
# MAGIC * **`HistogramDuration`** — time spent per value band (duration-weighted; values in µs).
# MAGIC * **`Histogram2DDuration`** — time spent per *combination* of two signals (a heatmap).
# MAGIC * **`StatsAggregator`** — `min / max / mean / start / end` per **event instance**.

# COMMAND ----------

from impulse_reporting.core.page import Page
from impulse_reporting.aggregations.histogram import HistogramDuration
from impulse_reporting.aggregations.histogram2d import Histogram2DDuration
from impulse_reporting.aggregations.stats_aggregator import StatsAggregator

driving_page = Page(page_number=1)
driving_page.add_aggregation(HistogramDuration(
    name="speed_distribution", base_expr=speed_kmh, event=driving,
    bins=[float(b) for b in range(0, 230, 10)], channel_name="vehicle_speed",
    bins_unit="km/h", values_unit="us", desc="Time per 10 km/h speed band while driving"))
driving_page.add_aggregation(Histogram2DDuration(
    name="speed_vs_acceleration", x_expr=speed_kmh, y_expr=accel, event=driving,
    x_bins=[float(b) for b in range(0, 220, 20)], y_bins=[float(b) for b in range(-6, 7)],
    x_channel_name="vehicle_speed", y_channel_name="Car.ax", x_bins_unit="km/h", y_bins_unit="m/s²",
    values_unit="us", desc="How hard each vehicle accelerates / brakes at which speed"))
driving_page.add_aggregation(StatsAggregator(
    name="trip_stats", event=trip,
    input_expressions=[speed_kmh, soc, current, batt_temp],
    channel_names=["vehicle_speed_kmh", "soc_pct", "battery_current_a", "battery_temp_c"],
    statistics=["min", "max", "mean", "start", "end"], desc="Per-trip summary statistics"))
report.add_page(driving_page)

battery_page = Page(page_number=2)
battery_page.add_aggregation(HistogramDuration(
    name="battery_current_distribution", base_expr=current, event=driving,
    bins=[float(b) for b in range(-300, 325, 25)], channel_name="PT.BattHV.Current",
    bins_unit="A", values_unit="us", desc="HV battery current while driving (negative = regeneration)"))
battery_page.add_aggregation(StatsAggregator(
    name="charging_stats", event=charging,
    input_expressions=[soc, current, batt_temp],
    channel_names=["soc_pct", "battery_current_a", "battery_temp_c"],
    statistics=["min", "max", "mean", "start", "end"], desc="Statistics per charging session"))
report.add_page(battery_page)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Run it
# MAGIC
# MAGIC `report.run()` resolves every channel for every trip, solves events and aggregations in parallel
# MAGIC across the cluster and writes the **gold star schema** to your schema. ⏳ ~3–6 minutes for 100 trips.

# COMMAND ----------

import time
t = time.time()
report.run()
print(f"✅ report computed and written in {(time.time() - t) / 60:.1f} min")
display(spark.sql(f"SHOW TABLES IN {USER_SCHEMA} LIKE '{W1_GOLD_PREFIX}_*'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · The gold star schema
# MAGIC
# MAGIC ```
# MAGIC                         impulse_w1_measurement_dimension  (one row per trip: vehicle, scenario, …)
# MAGIC                                     │ container_id
# MAGIC   impulse_w1_event_dimension ── event_id ──┬── impulse_w1_event_instance_fact   (one row per event window)
# MAGIC                                             ├── impulse_w1_histogram_fact        ── visual_id ── impulse_w1_histogram_dimension
# MAGIC                                             ├── impulse_w1_histogram2d_fact      ── visual_id ── impulse_w1_histogram2d_dimension
# MAGIC                                             └── impulse_w1_stats_aggregator_fact ── visual_id ── impulse_w1_stats_aggregator_dimension
# MAGIC ```
# MAGIC
# MAGIC *Facts* hold numbers, *dimensions* describe them. Everything is plain Delta, so any SQL tool,
# MAGIC dashboard or Genie space can use it. A few example queries:

# COMMAND ----------

# MAGIC %md
# MAGIC **Q1 — Time spent per speed band, by scenario** (histogram fact ⨝ dimension ⨝ trips)

# COMMAND ----------

display(spark.sql(f"""
  SELECT m.scenario, f.lower_bound AS speed_from_kmh, round(sum(f.hist_value) / 60e6, 1) AS minutes
  FROM {GOLD}_histogram_fact f
  JOIN {GOLD}_histogram_dimension d USING (visual_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE d.name = 'speed_distribution'
  GROUP BY ALL ORDER BY scenario, speed_from_kmh
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Q2 — Per-trip statistics as one wide row** (pivot the long `stats_aggregator_fact`)

# COMMAND ----------

display(spark.sql(f"""
  SELECT m.trip_start_dt, m.container_id, m.vehicle_key, m.scenario,
         max(CASE WHEN f.channel_name = 'vehicle_speed_kmh' AND f.aggregation_label = 'max'   THEN f.statistic_value END) AS max_speed_kmh,
         max(CASE WHEN f.channel_name = 'soc_pct'           AND f.aggregation_label = 'start' THEN f.statistic_value END) AS soc_start_pct,
         max(CASE WHEN f.channel_name = 'soc_pct'           AND f.aggregation_label = 'end'   THEN f.statistic_value END) AS soc_end_pct,
         max(CASE WHEN f.channel_name = 'battery_temp_c'    AND f.aggregation_label = 'max'   THEN f.statistic_value END) AS max_batt_temp_c
  FROM {GOLD}_stats_aggregator_fact f
  JOIN {GOLD}_stats_aggregator_dimension d USING (visual_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE d.name = 'trip_stats'
  GROUP BY ALL ORDER BY max_speed_kmh DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **Q3 — Charging sessions: how long, how much SOC gained** (event instances ⨝ their statistics)

# COMMAND ----------

display(spark.sql(f"""
  SELECT m.vehicle_key, e.container_id, e.event_instance_id,
         round((e.end_ts - e.start_ts) / 60e6, 1) AS charging_min,
         round(max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'end'   THEN s.statistic_value END)
             - max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'start' THEN s.statistic_value END), 1) AS soc_gained_pct
  FROM {GOLD}_event_instance_fact e
  JOIN {GOLD}_event_dimension ed USING (event_id)
  JOIN {GOLD}_stats_aggregator_fact s USING (container_id, event_id, event_instance_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE ed.event_name = 'charging'
  GROUP BY ALL ORDER BY soc_gained_pct DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Put it on an AI/BI dashboard
# MAGIC
# MAGIC ### Option A — build it yourself (recommended to try once, ~10 min)
# MAGIC 1. In the left sidebar click **Dashboards → Create dashboard**.
# MAGIC 2. Open the **Data** tab → **Create from SQL**, paste query **Q1** above (replace `{GOLD}` with the
# MAGIC    value printed at the top of this notebook, e.g. `dbx_workshop_dev.jane_doe.impulse_w1`) and run it.
# MAGIC    Rename the dataset to *speed histogram*.
# MAGIC 3. Back on the **canvas**, click **Add a visualization**, pick the dataset, choose **Bar**:
# MAGIC    X = `speed_from_kmh`, Y = `SUM(minutes)`, Color = `scenario`.
# MAGIC 4. Add a **filter** widget on `scenario`; it applies to every dataset with that column.
# MAGIC 5. Repeat with Q2 (a table, or a bar of max speed by vehicle) and Q3.
# MAGIC 6. **Publish** (top right) — viewers then see it with your credentials embedded.
# MAGIC
# MAGIC 💡 Or open the dashboard's **AI assistant** and ask: *"bar chart of minutes by speed_from_kmh coloured by scenario"*.
# MAGIC
# MAGIC ### Option B — let the notebook build it for you
# MAGIC The next cells describe a complete dashboard (datasets = SQL over your gold tables, plus widgets) and
# MAGIC create + publish it in your home folder via the Lakeview API. Open the link, then keep editing in the UI.

# COMMAND ----------

from common.dashboard_builder import Dashboard

trips_sql = f"""
  WITH stats AS (
    SELECT f.container_id,
           max(CASE WHEN f.channel_name = 'vehicle_speed_kmh' AND f.aggregation_label = 'max'  THEN f.statistic_value END) AS max_speed_kmh,
           max(CASE WHEN f.channel_name = 'vehicle_speed_kmh' AND f.aggregation_label = 'mean' THEN f.statistic_value END) AS mean_speed_kmh,
           max(CASE WHEN f.channel_name = 'soc_pct' AND f.aggregation_label = 'start' THEN f.statistic_value END) AS soc_start_pct,
           max(CASE WHEN f.channel_name = 'soc_pct' AND f.aggregation_label = 'end'   THEN f.statistic_value END) AS soc_end_pct,
           max(CASE WHEN f.channel_name = 'battery_temp_c' AND f.aggregation_label = 'max' THEN f.statistic_value END) AS max_batt_temp_c
    FROM {GOLD}_stats_aggregator_fact f JOIN {GOLD}_stats_aggregator_dimension d USING (visual_id)
    WHERE d.name = 'trip_stats' GROUP BY f.container_id),
  ev AS (
    SELECT e.container_id,
           sum(CASE WHEN d.event_name = 'driving'  THEN (e.end_ts - e.start_ts) / 3600e6 ELSE 0 END) AS driving_h,
           sum(CASE WHEN d.event_name = 'charging' THEN (e.end_ts - e.start_ts) / 3600e6 ELSE 0 END) AS charging_h
    FROM {GOLD}_event_instance_fact e JOIN {GOLD}_event_dimension d USING (event_id) GROUP BY e.container_id)
  SELECT m.container_id, m.trip_start_dt, m.vehicle_key, m.powertrain_type, m.scenario, m.driver, m.duration_s / 60 AS duration_min,
         s.max_speed_kmh, s.mean_speed_kmh, s.soc_start_pct, s.soc_end_pct, s.max_batt_temp_c,
         coalesce(ev.driving_h, 0) AS driving_h, coalesce(ev.charging_h, 0) AS charging_h, 1 AS trips
  FROM {GOLD}_measurement_dimension m LEFT JOIN stats s USING (container_id) LEFT JOIN ev USING (container_id)
"""
speed_hist_sql = f"""
  SELECT m.vehicle_key, m.scenario, f.lower_bound AS speed_kmh, f.hist_value / 60e6 AS minutes
  FROM {GOLD}_histogram_fact f JOIN {GOLD}_histogram_dimension d USING (visual_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id) WHERE d.name = 'speed_distribution'
"""
current_hist_sql = f"""
  SELECT m.vehicle_key, m.scenario, f.lower_bound AS current_a, f.hist_value / 60e6 AS minutes
  FROM {GOLD}_histogram_fact f JOIN {GOLD}_histogram_dimension d USING (visual_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE d.name = 'battery_current_distribution' AND m.powertrain_type <> 'generic'
"""
heatmap_sql = f"""
  SELECT m.vehicle_key, m.scenario, cast(f.x_lower_bound AS INT) AS speed_kmh, cast(f.y_lower_bound AS INT) AS accel_mps2,
         f.hist_value / 60e6 AS minutes
  FROM {GOLD}_histogram2d_fact f JOIN {GOLD}_histogram2d_dimension d USING (visual_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id) WHERE d.name = 'speed_vs_acceleration'
"""
charging_sql = f"""
  SELECT m.vehicle_key, m.scenario, e.container_id, e.event_instance_id,
         (e.end_ts - e.start_ts) / 60e6 AS charging_min,
         max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'start' THEN s.statistic_value END) AS soc_start_pct,
         max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'end'   THEN s.statistic_value END) AS soc_end_pct,
         max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'end'   THEN s.statistic_value END)
       - max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'start' THEN s.statistic_value END) AS soc_gained_pct,
         max(CASE WHEN s.channel_name = 'battery_current_a' AND s.aggregation_label = 'mean' THEN -s.statistic_value END) AS mean_charge_current_a
  FROM {GOLD}_event_instance_fact e JOIN {GOLD}_event_dimension ed USING (event_id)
  JOIN {GOLD}_stats_aggregator_fact s USING (container_id, event_id, event_instance_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE ed.event_name = 'charging' GROUP BY ALL
"""

# COMMAND ----------

dash = Dashboard(f"Impulse W1 - Fleet overview ({user_schema})")
for name, sql in [("trips", trips_sql), ("speed_hist", speed_hist_sql), ("current_hist", current_hist_sql),
                  ("heatmap", heatmap_sql), ("charging", charging_sql)]:
    dash.dataset(name, sql)
ALL = ["trips", "speed_hist", "current_hist", "heatmap", "charging"]

dash.page("Fleet overview")
dash.text("## 🚗 Impulse fleet report\nComputed by **Impulse** from 100 simulated trips "
          "(Renumics synthetic vehicle fleet telemetry). Every number is duration-weighted.", width=6, height=2)
dash.filter(ALL, "vehicle_key", "Vehicle", width=3, height=2)
dash.filter(ALL, "scenario", "Scenario", width=3, height=2, x=3, new_row=False)
dash.counter("trips", "trips", "Trips", width=2)
dash.counter("trips", "driving_h", "Driving hours", width=2, x=2, new_row=False)
dash.counter("trips", "charging_h", "Charging hours", width=2, x=4, new_row=False)
dash.chart("bar", "speed_hist", "speed_kmh", "minutes", "Time per speed band [min]", color="scenario",
           x_scale="quantitative", width=6, height=6, x_title="speed [km/h]", y_title="minutes", stacked=True)
dash.chart("bar", "trips", "vehicle_key", "max_speed_kmh", "Top speed by vehicle [km/h]", agg="MAX", width=3, height=6)
dash.heatmap("heatmap", "speed_kmh", "accel_mps2", "minutes", "Speed [km/h] × acceleration [m/s²] (minutes)",
             width=3, height=6, x=3, new_row=False)
dash.table("trips", ["trip_start_dt", "container_id", "vehicle_key", "scenario", "driver", "duration_min", "max_speed_kmh",
                     "soc_start_pct", "soc_end_pct", "max_batt_temp_c"], "Trips", width=6, height=7)

dash.page("Battery & charging")
dash.filter(["current_hist", "charging"], "vehicle_key", "Vehicle", width=3, height=2)
dash.chart("bar", "current_hist", "current_a", "minutes", "HV current while driving [A] — negative = regeneration",
           color="vehicle_key", x_scale="quantitative", width=6, height=6, x_title="current [A]", stacked=True)
dash.chart("bar", "charging", "vehicle_key", "soc_gained_pct", "Average SOC gained per charging session [%-points]",
           agg="AVG", width=3, height=6)
dash.chart("scatter", "charging", "charging_min", "soc_gained_pct", "Charging time vs SOC gained", agg="AVG",
           color="vehicle_key", x_scale="quantitative", width=3, height=6, x=3, new_row=False)
dash.table("charging", ["vehicle_key", "container_id", "charging_min", "soc_start_pct", "soc_end_pct",
                        "soc_gained_pct", "mean_charge_current_a"], "Charging sessions", width=6, height=6)

url = dash.publish(WorkspaceClient(), parent_path=f"/Users/{current_user()}", warehouse_id=pick_warehouse_id())
displayHTML(f'<h3>📊 <a href="{url}" target="_blank">Open your dashboard</a></h3><p>{url}</p>')

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Recap — and what you just did
# MAGIC
# MAGIC * One `Report` turned a fleet of **~370 million samples** into a compact star schema of a few thousand rows.
# MAGIC * Events (*when*) × aggregations (*what*) × trips (*where*), with **duration weighting**, **signal
# MAGIC   alignment** and **aliases / unit conversion** handled for you — and data quirks handled in the
# MAGIC   signal definitions, not by editing the data.
# MAGIC * The gold layer is plain Delta: SQL, dashboards and Genie spaces all work on it directly.
# MAGIC * Re-run `report.run()` after new trips land and the dashboard updates (with `incremental` enabled,
# MAGIC   only new trips are processed).
# MAGIC
# MAGIC 🎓 **Workshop 1 complete.** Workshop 2 goes under the hood: building the silver layer yourself,
# MAGIC every TSAL result type, custom UDFs, unit conversion and a deeper dashboard.
