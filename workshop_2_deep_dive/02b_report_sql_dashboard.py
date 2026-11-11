# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2 · Notebook 2b — A deep fleet report, SQL on gold, and a multi-page dashboard
# MAGIC
# MAGIC **⏱ ~60 minutes · 🎯 Goal: use (almost) every reporting building block on the full fleet, query the
# MAGIC gold star schema with SQL, and publish a dashboard far beyond workshop 1.**
# MAGIC
# MAGIC | Building block | Used for |
# MAGIC |---|---|
# MAGIC | `ContainerEvent` | whole-trip statistics |
# MAGIC | `BasicEvent` | driving · regeneration · charging · hard braking |
# MAGIC | `SequenceOfEvents` | **launches**: standstill *followed by* pulling away (to 50 km/h) |
# MAGIC | `TimeWindowEvent` | fixed **5-minute** segments → a time series per trip |
# MAGIC | `PointsInTimeEvent` | each **HARSH_BRAKE** alert from the POI event stream |
# MAGIC | `HistogramDuration` / `HistogramDistance` | time **and distance** per speed band |
# MAGIC | `Histogram2DDuration` | the **e-motor operating map** (rpm × torque) |
# MAGIC | `StatsAggregator` + **custom statistics** | recovered energy (kWh), charge rate (%/min), p95 |
# MAGIC | `PointValueAggregator` | speed / deceleration sampled at each alert |
# MAGIC | `CalculatedChannel` | battery power `U·I` materialized as a new channel |
# MAGIC | `incremental` | re-run only what changed |

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

require_user_schema()
setup_impulse()

import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient

user_catalog, user_schema = USER_SCHEMA.split(".")
GOLD = f"{USER_SCHEMA}.{W2_GOLD_PREFIX}"
print("gold:", GOLD + "_*")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Configuration
# MAGIC
# MAGIC New compared with workshop 1:
# MAGIC * **`incremental`** — on re-runs, unchanged definitions only process **new or updated trips** (detected
# MAGIC   via `container_metrics.last_modified`, which the silver build stamps). Changed definitions are
# MAGIC   recomputed everywhere. The first run is always full.
# MAGIC * **`calculated_channels.emit_channel_metrics`** — also write a `channel_metrics`-shaped table for the
# MAGIC   calculated channel, so it could serve as a silver source itself.
# MAGIC * **`solver_config.channel_time_unit = "us"`** — tells `TimeWindowEvent` our time base.

# COMMAND ----------

config = {
    "source": impulse_source(SILVER),
    "unity_sink": {"catalog": user_catalog, "schema": user_schema, "table_prefix": W2_GOLD_PREFIX,
                   "cleanup_temp_tables": True},
    "query_engine": {"solver": "DefaultSolver", "data_type": "RLE",
                     "solver_config": {"channel_time_unit": "us"}},
    "incremental": {"enabled": True, "silver_last_modified_column": "last_modified"},
    "measurement_dimensions": ["container_id", "start_ts", "stop_ts", "trip_start_dt", "duration_s", "vehicle_key", "manufacturer",
                               "powertrain_type", "scenario", "driver", "tires", "number_of_passengers",
                               "traffic", "starting_temperature_c", "empty_weight_kg"],
    "calculated_channels": {"emit_channel_metrics": True, "attribute_columns": ["unit"]},
}

from impulse_reporting.core.report import Report
from impulse_reporting.core.page import Page

report = Report(name=W2_REPORT_NAME, spark=spark, workspace_client=WorkspaceClient(), config=config)
q = report.get_db().query

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Signals — aliases do the unit work, the definitions handle the data quirks
# MAGIC
# MAGIC Same rules as before (`DATA_NOTES.md`): speed spikes > 100 m/s dropped, SOC (state of charge, % battery full) ≤ 0 ignored, battery
# MAGIC temperature converted from Kelvin. The silver data itself is untouched.

# COMMAND ----------

v_kmh = q.channel_with_alias(channel_alias="vehicle_speed")          # Vhcl.v in km/h
speed_kmh = v_kmh.where(v_kmh < VHCL_V_MAX_MPS * 3.6)                # without the spikes
odometer_km = q.channel_with_alias(channel_alias="odometer")         # km
motor_rpm = q.channel_with_alias(channel_alias="motor_speed")        # rpm (e-motor only)
motor_power_kw = q.channel_with_alias(channel_alias="motor_power")   # kW, negative = regeneration
motor_torque = q.channel(channel_name="PT.Motor.Trq")                # Nm
accel = q.channel(channel_name="Car.ax")                             # m/s²
lat_accel = q.channel(channel_name="Car.ay")                         # m/s²
soc_raw = q.channel(channel_name="PT.BCU.BattHV.SOC")
soc = soc_raw.where(soc_raw > 0)                                     # %, BMS start-up zeros ignored
current = q.channel(channel_name="PT.BattHV.Current")                # A
voltage = q.channel(channel_name="PT.Motor.VoltBatt")                # V
batt_temp = q.channel(channel_name="PT.BattHV.Temp") - KELVIN_OFFSET  # logged in K -> °C
alerts = q.poi_channel(channel_name="Event.DriverAlert", dtype="string")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Events — every type

# COMMAND ----------

from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.container_event import ContainerEvent
from impulse_reporting.events.sequence_of_events import SequenceOfEvents
from impulse_reporting.events.time_window_event import TimeWindowEvent
from impulse_reporting.events.points_in_time_event import PointsInTimeEvent

trip = ContainerEvent(name="trip", desc="The whole trip")
driving = BasicEvent(name="driving", expr=speed_kmh > 3.6, desc="Moving faster than 3.6 km/h")
regen = BasicEvent(name="regeneration",
                   expr=((motor_power_kw < 0) & (speed_kmh > 3.6)).merge_intervals(2e6).filter(5e6),
                   desc="E-motor generating while moving; gaps < 2 s bridged, windows > 5 s")
charging = BasicEvent(name="charging", expr=((current < -10) & (speed_kmh < 0.36)).filter(60e6),
                      desc="HV current < -10 A at standstill for > 60 s")
hard_braking = BasicEvent(name="hard_braking", expr=(accel < -3).filter(0.3e6),
                          desc="Deceleration stronger than 3 m/s² for > 0.3 s")
# SequenceOfEvents joins windows that OVERLAP. "Standing still" and "moving" never overlap, so the
# standstill window is extended 1 s to the right with expand_right() to reach into the pull-away.
launch = SequenceOfEvents(
    name="launch",
    expressions=[(speed_kmh < 0.36).filter(2e6).expand_right(1e6),      # stood still > 2 s (+1 s overlap)
                 (speed_kmh > 0.36) & (speed_kmh < 50)],                  # then pulled away, until 50 km/h
    desc="Standing still for > 2 s, then pulling away up to 50 km/h")
five_min = TimeWindowEvent(name="five_minute_windows", window_length=300e6, desc="Consecutive 5-minute segments")
harsh_alert = PointsInTimeEvent(name="harsh_brake_alert", expr=alerts == "HARSH_BRAKE",
                                desc="Each HARSH_BRAKE alert of the telematics event stream")

for e in (trip, driving, regen, charging, hard_braking, launch, five_min, harsh_alert):
    report.add_event(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Custom statistics
# MAGIC
# MAGIC A custom statistic is a plain function `func(series, t_start, t_end) -> [values]` (per channel) or
# MAGIC `func([series…], t_start, t_end) -> [values]` (cross channel), called **once per event instance** with
# MAGIC the series clipped to it. Each returned value becomes an ordinary row in `stats_aggregator_fact`.
# MAGIC Define them **in the notebook** (they are pickled to the workers) and follow the UDF rules: handle
# MAGIC empty input, return the same number of values, respect gaps.
# MAGIC
# MAGIC ℹ️ A per-channel statistic runs on **every** input of its aggregator (energy of `speed_kmh` is computed
# MAGIC too); the SQL below simply reads the rows that make sense, e.g. `channel_name = 'regen_power_kw'`.

# COMMAND ----------

US = 1e6


def energy_kwh(series, t_start, t_end):
    """Energy of a power signal in **kW**, integrated per continuous segment -> [kWh]."""
    if len(series) == 0:
        return [float("nan")]
    total = 0.0
    for a, b in series.continuous_interval_indices:
        t = np.append(series.tstarts[a:b + 1], series.tends[b]) / US / 3600        # hours
        p = np.append(series.values[a:b + 1], series.values[b])
        total += np.trapezoid(p, t)
    return [float(total)]


def rate_per_minute(series, t_start, t_end):
    """(last - first) / duration in minutes -> e.g. SOC %-points per minute."""
    if len(series) < 2 or t_end <= t_start:
        return [float("nan")]
    return [float((series.values[-1] - series.values[0]) / ((t_end - t_start) / 60e6))]


def p95(series, t_start, t_end):
    return [float(np.nanpercentile(series.values, 95))] if len(series) else [float("nan")]


def peak_power_kw(series_list, t_start, t_end):
    """Cross-channel: peak battery power from voltage [V] and current [A], aligned -> [kW]."""
    v, i = series_list
    if len(v) == 0 or len(i) == 0:
        return [float("nan")]
    vs, cs = v.synchronized(i)                         # rule 4: align before combining raw arrays
    return [float(np.nanmax(vs.values * cs.values) / 1000)] if len(vs) else [float("nan")]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Aggregations, pages and a calculated channel

# COMMAND ----------

from impulse_reporting.aggregations.histogram import HistogramDuration, HistogramDistance
from impulse_reporting.aggregations.histogram2d import Histogram2DDuration
from impulse_reporting.aggregations.stats_aggregator import StatsAggregator
from impulse_reporting.aggregations.point_value_aggregator import PointValueAggregator
from impulse_reporting.channels.calculated_channel import CalculatedChannel
from impulse_query_engine.analyze.query.aggregations.custom_statistic import PerChannelStatistic, CrossChannelStatistic

speed_bins = [float(b) for b in range(0, 230, 10)]

p1 = Page(page_number=1)   # --- driving dynamics -------------------------------------------------
p1.add_aggregation(HistogramDuration(name="speed_time", base_expr=speed_kmh, bins=speed_bins, event=driving,
                                     channel_name="vehicle_speed", bins_unit="km/h", values_unit="us"))
p1.add_aggregation(HistogramDistance(name="speed_distance", base_expr=speed_kmh, weights_expr=odometer_km,
                                     bins=speed_bins, event=driving, channel_name="vehicle_speed",
                                     weights_channel_name="odometer", bins_unit="km/h", values_unit="km"))
p1.add_aggregation(Histogram2DDuration(name="lateral_vs_longitudinal", x_expr=lat_accel, y_expr=accel,
                                       x_bins=[float(b) for b in range(-10, 11)], y_bins=[float(b) for b in range(-8, 7)],
                                       event=driving, x_channel_name="Car.ay", y_channel_name="Car.ax",
                                       x_bins_unit="m/s²", y_bins_unit="m/s²", values_unit="us"))
p1.add_aggregation(StatsAggregator(name="trip_stats", event=trip,
                                   input_expressions=[speed_kmh, lat_accel, soc, batt_temp],
                                   channel_names=["speed_kmh", "lat_accel_mps2", "soc_pct", "battery_temp_c"],
                                   statistics=["min", "max", "mean", "start", "end"],
                                   per_channel_custom_statistics=[PerChannelStatistic(func=p95, aggregation_labels=["p95"])]))
p1.add_aggregation(StatsAggregator(name="window_stats", event=five_min,
                                   input_expressions=[speed_kmh, motor_power_kw, batt_temp],
                                   channel_names=["speed_kmh", "motor_power_kw", "battery_temp_c"],
                                   statistics=["mean", "max"],
                                   per_channel_custom_statistics=[PerChannelStatistic(func=energy_kwh, aggregation_labels=["energy_kwh"])]))
p1.add_aggregation(StatsAggregator(name="launch_stats", event=launch, input_expressions=[accel, speed_kmh],
                                   channel_names=["accel_mps2", "speed_kmh"], statistics=["max", "end"]))
report.add_page(p1)

p2 = Page(page_number=2)   # --- e-motor & energy ------------------------------------------------
p2.add_aggregation(Histogram2DDuration(name="motor_operating_map", x_expr=motor_rpm, y_expr=motor_torque,
                                       x_bins=[float(b) for b in range(0, 20001, 1000)],
                                       y_bins=[float(b) for b in range(-400, 701, 50)], event=driving,
                                       x_channel_name="motor_speed", y_channel_name="PT.Motor.Trq",
                                       x_bins_unit="rpm", y_bins_unit="Nm", values_unit="us",
                                       desc="E-motor operating points while driving"))
p2.add_aggregation(StatsAggregator(name="regen_stats", event=regen,
                                   input_expressions=[motor_power_kw * -1, speed_kmh],
                                   channel_names=["regen_power_kw", "speed_kmh"], statistics=["max", "start", "end"],
                                   per_channel_custom_statistics=[PerChannelStatistic(func=energy_kwh, aggregation_labels=["energy_kwh"])]))
p2.add_aggregation(StatsAggregator(name="drive_energy", event=driving, input_expressions=[motor_power_kw, voltage, current],
                                   channel_names=["motor_power_kw", "battery_voltage_v", "battery_current_a"],
                                   statistics=["mean"],
                                   per_channel_custom_statistics=[PerChannelStatistic(func=energy_kwh, aggregation_labels=["energy_kwh"])],
                                   cross_channel_custom_statistics=[CrossChannelStatistic(
                                       func=peak_power_kw, aggregation_labels=["peak_battery_power_kw"],
                                       inputs=["battery_voltage_v", "battery_current_a"], channel_name="battery_power")]))
report.add_page(p2)

p3 = Page(page_number=3)   # --- battery & charging ----------------------------------------------
p3.add_aggregation(StatsAggregator(name="charging_stats", event=charging, input_expressions=[soc, current, batt_temp],
                                   channel_names=["soc_pct", "battery_current_a", "battery_temp_c"],
                                   statistics=["start", "end", "mean", "max", "min"],
                                   per_channel_custom_statistics=[PerChannelStatistic(func=rate_per_minute, aggregation_labels=["rate_per_min"])]))
p3.add_aggregation(HistogramDuration(name="battery_temp_charging", base_expr=batt_temp, event=charging,
                                     bins=[float(b) for b in range(-10, 61, 2)], channel_name="PT.BattHV.Temp",
                                     bins_unit="°C", values_unit="us"))
report.add_page(p3)

p4 = Page(page_number=4)   # --- driving style / safety -----------------------------------------
p4.add_aggregation(PointValueAggregator(name="at_harsh_brake_alert", event=harsh_alert,
                                        input_expressions=[speed_kmh, accel], channel_names=["speed_kmh", "accel_mps2"]))
p4.add_aggregation(StatsAggregator(name="hard_braking_stats", event=hard_braking, input_expressions=[accel, speed_kmh],
                                   channel_names=["accel_mps2", "speed_kmh"], statistics=["min", "start", "end"]))
report.add_page(p4)

report.add_calculated_channel(CalculatedChannel(
    name="battery_power_kw", expr=voltage * current / 1000,
    identity={"channel_name": "Calc.BattHV.Power", "data_key": "IMPULSE"},
    desc="HV battery power U·I in kW (positive = discharging)", attributes={"unit": "kW"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Run (full), then run again (incremental)
# MAGIC
# MAGIC ⏳ The first run processes 100 trips × ~15 signals and writes everything: expect ~10–15 minutes.
# MAGIC The second run finds every definition unchanged and no new trips, so it has almost nothing to do.

# COMMAND ----------

t = time.time()
report.run()
print(f"✅ first run: {(time.time() - t) / 60:.1f} min")

# COMMAND ----------

t = time.time()
report.run()          # incremental: definitions unchanged, no new trips -> fast no-op
print(f"✅ incremental re-run: {(time.time() - t) / 60:.1f} min")
display(spark.sql(f"SHOW TABLES IN {USER_SCHEMA} LIKE '{W2_GOLD_PREFIX}_*'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · SQL on the gold layer
# MAGIC
# MAGIC Facts join to `measurement_dimension` on `container_id`, to `event_dimension` on `event_id` and to their
# MAGIC own dimension on `visual_id`; `stats_aggregator_fact` also joins `event_instance_fact` on
# MAGIC `event_instance_id`. Some questions the fleet team would ask:

# COMMAND ----------

# MAGIC %md **Q1 · Which vehicles recover the most energy by regenerative braking?**

# COMMAND ----------

display(spark.sql(f"""
  SELECT m.vehicle_key, count(*) AS regen_events,
         round(sum(s.statistic_value), 2) AS recovered_kwh,
         round(sum(s.statistic_value) / count(DISTINCT s.container_id), 3) AS kwh_per_trip,
         round(max(s.statistic_value), 3) AS best_single_event_kwh
  FROM {GOLD}_stats_aggregator_fact s
  JOIN {GOLD}_stats_aggregator_dimension d USING (visual_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE d.name = 'regen_stats' AND s.channel_name = 'regen_power_kw' AND s.aggregation_label = 'energy_kwh'
  GROUP BY ALL ORDER BY recovered_kwh DESC
"""))

# COMMAND ----------

# MAGIC %md **Q2 · Energy intensity: kWh per 100 km while driving, by vehicle and driving style**
# MAGIC (energy from a custom statistic, distance from the *distance-weighted* histogram)

# COMMAND ----------

display(spark.sql(f"""
  WITH energy AS (
    SELECT s.container_id, sum(s.statistic_value) AS kwh
    FROM {GOLD}_stats_aggregator_fact s JOIN {GOLD}_stats_aggregator_dimension d USING (visual_id)
    WHERE d.name = 'drive_energy' AND s.channel_name = 'motor_power_kw' AND s.aggregation_label = 'energy_kwh'
    GROUP BY 1),
  dist AS (
    SELECT f.container_id, sum(f.hist_value) AS km
    FROM {GOLD}_histogram_fact f JOIN {GOLD}_histogram_dimension d USING (visual_id)
    WHERE d.name = 'speed_distance' GROUP BY 1)
  SELECT m.vehicle_key, m.driver, round(sum(e.kwh) / sum(d.km) * 100, 1) AS kwh_per_100km, round(sum(d.km), 1) AS km
  FROM energy e JOIN dist d USING (container_id) JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE m.powertrain_type = 'ev' AND d.km > 1
  GROUP BY ALL ORDER BY vehicle_key, kwh_per_100km
"""))

# COMMAND ----------

# MAGIC %md **Q3 · Charging performance: SOC rate (%/min) per session** (custom statistic `rate_per_min`)

# COMMAND ----------

display(spark.sql(f"""
  SELECT m.vehicle_key, s.container_id, s.event_instance_id,
         round((e.end_ts - e.start_ts) / 60e6, 1) AS minutes,
         round(max(CASE WHEN s.aggregation_label = 'start' THEN s.statistic_value END), 1) AS soc_start,
         round(max(CASE WHEN s.aggregation_label = 'end' THEN s.statistic_value END), 1) AS soc_end,
         round(max(CASE WHEN s.aggregation_label = 'rate_per_min' THEN s.statistic_value END), 2) AS soc_pct_per_min
  FROM {GOLD}_stats_aggregator_fact s
  JOIN {GOLD}_stats_aggregator_dimension d USING (visual_id)
  JOIN {GOLD}_event_instance_fact e USING (container_id, event_id, event_instance_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE d.name = 'charging_stats' AND s.channel_name = 'soc_pct'
  GROUP BY ALL ORDER BY soc_pct_per_min DESC
"""))

# COMMAND ----------

# MAGIC %md **Q4 · How fast were cars going when a HARSH_BRAKE alert fired?** (PointValueAggregator rows)

# COMMAND ----------

display(spark.sql(f"""
  SELECT m.driver, count(*) AS alerts, round(avg(s.statistic_value), 1) AS avg_speed_kmh,
         round(percentile(s.statistic_value, 0.9), 1) AS p90_speed_kmh
  FROM {GOLD}_stats_aggregator_fact s
  JOIN {GOLD}_stats_aggregator_dimension d USING (visual_id)
  JOIN {GOLD}_measurement_dimension m USING (container_id)
  WHERE d.name = 'at_harsh_brake_alert' AND s.channel_name = 'speed_kmh'
  GROUP BY ALL ORDER BY alerts DESC
"""))

# COMMAND ----------

# MAGIC %md **Q5 · Event inventory — every event type in one table**

# COMMAND ----------

display(spark.sql(f"""
  SELECT d.event_name, d.event_type, count(*) AS instances, count(DISTINCT e.container_id) AS trips,
         round(avg((e.end_ts - e.start_ts) / 1e6), 1) AS avg_duration_s
  FROM {GOLD}_event_instance_fact e JOIN {GOLD}_event_dimension d USING (event_id)
  GROUP BY ALL ORDER BY instances DESC
"""))

# COMMAND ----------

# MAGIC %md **Q6 · The calculated channel is a real (silver-shaped) time series** — peak battery power per trip

# COMMAND ----------

display(spark.sql(f"""
  SELECT m.vehicle_key, c.container_id, round(max(c.value), 1) AS peak_battery_kw, round(min(c.value), 1) AS peak_charge_kw,
         count(*) AS intervals
  FROM {GOLD}_calculated_channel_fact c JOIN {GOLD}_measurement_dimension m USING (container_id)
  GROUP BY ALL ORDER BY peak_battery_kw DESC LIMIT 15
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · The deep-dive dashboard
# MAGIC
# MAGIC Five pages built from SQL datasets over your gold tables — plus one straight on the **silver** layer
# MAGIC (data quality). Same builder as workshop 1; open the result and keep editing in the UI.

# COMMAND ----------

from common.dashboard_builder import Dashboard

M = f"{GOLD}_measurement_dimension"
stat = lambda name: f"""{GOLD}_stats_aggregator_fact s JOIN {GOLD}_stats_aggregator_dimension d ON s.visual_id = d.visual_id AND d.name = '{name}'
  JOIN {M} m ON m.container_id = s.container_id"""

datasets = {
    "kpi": f"""
      WITH km AS (
        SELECT h.container_id, sum(h.hist_value) AS km
        FROM {GOLD}_histogram_fact h JOIN {GOLD}_histogram_dimension hd USING (visual_id)
        WHERE hd.name = 'speed_distance' GROUP BY 1),
      rg AS (
        SELECT s.container_id, sum(s.statistic_value) AS regen_kwh
        FROM {GOLD}_stats_aggregator_fact s JOIN {GOLD}_stats_aggregator_dimension d USING (visual_id)
        WHERE d.name = 'regen_stats' AND s.channel_name = 'regen_power_kw' AND s.aggregation_label = 'energy_kwh' GROUP BY 1),
      al AS (
        SELECT e.container_id, count(*) AS harsh_alerts
        FROM {GOLD}_event_instance_fact e JOIN {GOLD}_event_dimension ed USING (event_id)
        WHERE ed.event_name = 'harsh_brake_alert' GROUP BY 1)
      SELECT m.container_id, m.vehicle_key, m.powertrain_type, m.scenario, m.driver, 1 AS trips,
             coalesce(km.km, 0) AS km, coalesce(rg.regen_kwh, 0) AS regen_kwh, coalesce(al.harsh_alerts, 0) AS harsh_alerts
      FROM {M} m LEFT JOIN km USING (container_id) LEFT JOIN rg USING (container_id) LEFT JOIN al USING (container_id)""",
    "speed_bands": f"""
      SELECT m.vehicle_key, m.powertrain_type, m.scenario, m.driver, f.lower_bound AS speed_kmh,
             CASE WHEN hd.name = 'speed_distance' THEN f.hist_value ELSE f.hist_value / 60e6 END AS amount, hd.name AS measure
      FROM {GOLD}_histogram_fact f JOIN {GOLD}_histogram_dimension hd USING (visual_id) JOIN {M} m USING (container_id)
      WHERE hd.name IN ('speed_time', 'speed_distance')""",
    "motor_map": f"""
      SELECT m.vehicle_key, m.powertrain_type, m.scenario, m.driver, cast(f.x_lower_bound AS INT) AS motor_rpm,
             cast(f.y_lower_bound AS INT) AS torque_nm, f.hist_value / 60e6 AS minutes
      FROM {GOLD}_histogram2d_fact f JOIN {GOLD}_histogram2d_dimension d USING (visual_id) JOIN {M} m USING (container_id)
      WHERE d.name = 'motor_operating_map' AND f.hist_value > 0""",
    "gg": f"""
      SELECT m.vehicle_key, m.powertrain_type, m.scenario, m.driver, cast(f.x_lower_bound AS INT) AS lateral_mps2,
             cast(f.y_lower_bound AS INT) AS longitudinal_mps2, f.hist_value / 60e6 AS minutes
      FROM {GOLD}_histogram2d_fact f JOIN {GOLD}_histogram2d_dimension d USING (visual_id) JOIN {M} m USING (container_id)
      WHERE d.name = 'lateral_vs_longitudinal' AND f.hist_value > 0""",
    "regen": f"""
      SELECT m.vehicle_key, m.powertrain_type, m.scenario, m.driver, s.container_id, s.event_instance_id,
             max(CASE WHEN s.channel_name = 'regen_power_kw' AND s.aggregation_label = 'energy_kwh' THEN s.statistic_value END) AS energy_kwh,
             max(CASE WHEN s.channel_name = 'regen_power_kw' AND s.aggregation_label = 'max' THEN s.statistic_value END) AS peak_kw,
             max(CASE WHEN s.channel_name = 'speed_kmh' AND s.aggregation_label = 'start' THEN s.statistic_value END)
           - max(CASE WHEN s.channel_name = 'speed_kmh' AND s.aggregation_label = 'end' THEN s.statistic_value END) AS speed_drop_kmh
      FROM {stat('regen_stats')} GROUP BY ALL""",
    "windows": f"""
      SELECT m.vehicle_key, m.powertrain_type, m.scenario, m.driver, s.container_id,
             concat(m.vehicle_key, ' #', s.container_id) AS trip,
             dense_rank() OVER (PARTITION BY s.container_id ORDER BY e.start_ts) * 5 AS minute,
             max(CASE WHEN s.channel_name = 'speed_kmh' AND s.aggregation_label = 'mean' THEN s.statistic_value END) AS mean_speed_kmh,
             max(CASE WHEN s.channel_name = 'battery_temp_c' AND s.aggregation_label = 'max' THEN s.statistic_value END) AS max_batt_temp_c,
             max(CASE WHEN s.channel_name = 'motor_power_kw' AND s.aggregation_label = 'energy_kwh' THEN s.statistic_value END) AS energy_kwh
      FROM {stat('window_stats')} JOIN {GOLD}_event_instance_fact e
        ON e.container_id = s.container_id AND e.event_id = s.event_id AND e.event_instance_id = s.event_instance_id
      GROUP BY m.vehicle_key, m.powertrain_type, m.scenario, m.driver, s.container_id, e.start_ts""",
    "charging": f"""
      SELECT m.vehicle_key, m.powertrain_type, m.scenario, m.driver, s.container_id, s.event_instance_id,
             max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'rate_per_min' THEN s.statistic_value END) AS soc_pct_per_min,
             max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'start' THEN s.statistic_value END) AS soc_start,
             max(CASE WHEN s.channel_name = 'soc_pct' AND s.aggregation_label = 'end' THEN s.statistic_value END) AS soc_end,
             max(CASE WHEN s.channel_name = 'battery_current_a' AND s.aggregation_label = 'mean' THEN -s.statistic_value END) AS charge_current_a,
             max(CASE WHEN s.channel_name = 'battery_temp_c' AND s.aggregation_label = 'max' THEN s.statistic_value END) AS max_temp_c
      FROM {stat('charging_stats')} GROUP BY ALL""",
    "alerts": f"""
      SELECT m.vehicle_key, m.powertrain_type, m.scenario, m.driver, s.statistic_value AS speed_kmh, 1 AS alerts
      FROM {stat('at_harsh_brake_alert')} WHERE s.channel_name = 'speed_kmh'""",
    "dq": f"""
      SELECT q.container_id, t.vehicle_key, t.powertrain_type, t.scenario, t.driver, q.channel_name, q.finding,
             round(q.observed_max, 1) AS observed_max, q.implausible_pct, q.handling
      FROM {SHARED_SCHEMA}.{TABLE_PREFIX}_silver_dq_findings q JOIN {SILVER['container_metrics']} t USING (container_id)""",
    "coverage": f"""
      SELECT t.vehicle_key, t.powertrain_type, t.scenario, t.driver, c.channel_name,
             sum(c.raw_sample_count) AS samples, sum(c.sample_count) AS rle_intervals, avg(c.sample_rate_hz) AS rate_hz
      FROM {SILVER['channel_metrics']} c JOIN {SILVER['container_metrics']} t USING (container_id)
      WHERE c.value_type = 'DOUBLE' GROUP BY ALL""",
}

dash = Dashboard(f"Impulse W2 - Fleet deep dive ({user_schema})")
for name, sql in datasets.items():
    dash.dataset(name, sql)
ALL = list(datasets)

def filters():
    dash.filter(ALL, "vehicle_key", "Vehicle", width=2, height=2)
    dash.filter(ALL, "powertrain_type", "Powertrain", width=2, height=2, x=2, new_row=False)
    dash.filter(ALL, "driver", "Driving style", width=2, height=2, x=4, new_row=False)

dash.page("1 · Fleet")
dash.text("## 🚗 Fleet deep dive\nAll numbers computed by **Impulse** from ~370 M samples · duration- and "
          "distance-weighted · units resolved through channel aliases.", width=6, height=2)
filters()
dash.counter("kpi", "trips", "Trips", width=2)
dash.counter("kpi", "km", "km driven", width=2, x=2, new_row=False)
dash.counter("kpi", "regen_kwh", "kWh recovered", width=1, x=4, new_row=False)
dash.counter("kpi", "harsh_alerts", "Harsh-brake alerts", width=1, x=5, new_row=False)
dash.chart("bar", "speed_bands", "speed_kmh", "amount", "Distance per speed band [km] vs time [min]", color="measure",
           x_scale="quantitative", width=6, height=6, x_title="speed [km/h]", y_title="km  |  minutes")
dash.chart("bar", "kpi", "vehicle_key", "km", "Distance by vehicle [km]", color="scenario", width=3, height=6, stacked=True)
dash.chart("bar", "kpi", "vehicle_key", "harsh_alerts", "Harsh-brake alerts by vehicle", color="driver",
           width=3, height=6, x=3, new_row=False, stacked=True)

dash.page("2 · E-motor & energy")
filters()
dash.heatmap("motor_map", "motor_rpm", "torque_nm", "minutes", "E-motor operating map: rpm × torque [Nm] (minutes)", width=6, height=8)
dash.chart("bar", "regen", "vehicle_key", "energy_kwh", "Recovered energy by vehicle [kWh]", color="driver",
           width=3, height=6, stacked=True)
dash.chart("scatter", "regen", "speed_drop_kmh", "energy_kwh", "Regen events: speed drop vs energy", agg="AVG",
           color="vehicle_key", x_scale="quantitative", width=3, height=6, x=3, new_row=False)

dash.page("3 · Battery & charging")
filters()
dash.chart("bar", "charging", "vehicle_key", "soc_pct_per_min", "Average charge rate [SOC %-points / min]", agg="AVG", width=3, height=6)
dash.chart("scatter", "charging", "charge_current_a", "soc_pct_per_min", "Charge current vs rate", agg="AVG",
           color="vehicle_key", x_scale="quantitative", width=3, height=6, x=3, new_row=False)
dash.table("charging", ["vehicle_key", "container_id", "soc_start", "soc_end", "soc_pct_per_min", "charge_current_a",
                        "max_temp_c"], "Charging sessions", width=6, height=6)

dash.page("4 · Driving style")
filters()
dash.heatmap("gg", "lateral_mps2", "longitudinal_mps2", "minutes", "g-g diagram: lateral × longitudinal acceleration [m/s²]",
             width=3, height=7)
dash.chart("bar", "alerts", "driver", "speed_kmh", "Avg speed at HARSH_BRAKE alert [km/h]", agg="AVG",
           width=3, height=7, x=3, new_row=False)
dash.chart("line", "windows", "minute", "mean_speed_kmh", "Mean speed per 5-minute window (pick a vehicle)", agg="AVG",
           color="trip", x_scale="quantitative", width=6, height=6, x_title="minute of trip")

dash.page("5 · Data quality (silver)")
filters()
dash.text("Silver keeps every measured value. Findings are **flagged** here and handled in the TSAL "
          "signal definitions — see `DATA_NOTES.md` (Kelvin battery temperature, lat/long in radians, "
          "speed spikes, SOC = 0 at BMS start-up, time-normalized timestamps).", width=6, height=2)
dash.chart("bar", "dq", "channel_name", "implausible_pct", "Share of implausible samples per finding [%]", agg="AVG",
           color="vehicle_key", width=6, height=5)
dash.table("dq", ["container_id", "vehicle_key", "scenario", "channel_name", "finding", "observed_max",
                  "implausible_pct", "handling"], "Data-quality findings", width=6, height=5)
dash.chart("bar", "coverage", "channel_name", "samples", "Samples per signal (silver)", width=6, height=7,
           x_title="signal", y_title="samples")

url = dash.publish(WorkspaceClient(), parent_path=f"/Users/{current_user()}", warehouse_id=pick_warehouse_id())
displayHTML(f'<h3>📊 <a href="{url}" target="_blank">Open the deep-dive dashboard</a></h3><p>{url}</p>')

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Recap
# MAGIC * Every event type: whole trip, conditions, **sequences**, **fixed windows**, **points in time**.
# MAGIC * Aggregations beyond histograms: **distance weighting**, **2-D operating maps**, **custom statistics**
# MAGIC   (energy, rates, percentiles, cross-channel physics), **point values** at events.
# MAGIC * A **calculated channel** materializes a derived signal in silver shape.
# MAGIC * **Incremental** runs only reprocess what changed.
# MAGIC * The gold star schema answers fleet questions in plain SQL and feeds a multi-page AI/BI dashboard.
# MAGIC
# MAGIC 🎓 **Workshop 2 complete!** Ideas to continue: add a Genie space on the gold tables, schedule the report
# MAGIC as a job after the silver pipeline (see `declarative_pipeline/`), or extract ML features (`impulse-ml` skill).
