# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1 · Notebook 1 — Introduction to Impulse and TSAL
# MAGIC
# MAGIC **⏱ ~45 minutes · 🎯 Goal: understand what Impulse is, why it exists, and write your first TSAL queries.**
# MAGIC
# MAGIC [Impulse](https://github.com/databrickslabs/impulse) is a Databricks Labs library for analyzing
# MAGIC **large-scale time-series measurement data** — think vehicle test drives, battery bench runs,
# MAGIC or industrial sensor recordings — on Spark and Delta, *without* needing to be a Spark expert.
# MAGIC
# MAGIC In this notebook you will:
# MAGIC
# MAGIC | # | Section | You will learn |
# MAGIC |---|---|---|
# MAGIC | 1 | Setup | how the workshop finds the shared data and the shared Impulse library |
# MAGIC | 2 | The data | the **silver data model**: containers (trips) and channels (signals) |
# MAGIC | 3 | Connect | create a `MeasurementDB` — Impulse's window onto the silver tables |
# MAGIC | 4 | First TSAL query | select a channel, reduce it, get a **DataFrame** back (great for testing) |
# MAGIC | 5 | Signals as objects | pull a whole signal into Python and plot it |
# MAGIC | 6 | Derived signals & conditions | arithmetic → new signals, comparisons → **time windows** |
# MAGIC | 7 | Why this matters | duration weighting & automatic alignment — the things SQL makes hard |
# MAGIC | 8 | A first report | events + aggregations computed for every trip in one call |
# MAGIC
# MAGIC > **Before you start:** open the **`CONFIG`** notebook at the root of the repo and set
# MAGIC > `USER_SCHEMA` to your own schema (e.g. `dbx_workshop_dev.jane_doe`). That is the only thing you
# MAGIC > ever need to change. Run this notebook on **Serverless** compute.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Setup
# MAGIC
# MAGIC Three things happen here:
# MAGIC
# MAGIC 1. **`%run ../CONFIG`** — loads every shared name (tables, paths) and *your* schema from one place.
# MAGIC 2. **`%pip install $IMPULSE_PIP_TARGET`** — installs Impulse the way your instructor set it up
# MAGIC    (`IMPULSE_INSTALL_MODE` in CONFIG). There is exactly **one** copy for everyone, in one of two forms:
# MAGIC    * **`wheel`** (default): a `.whl` file the instructor built once and stored in the Unity Catalog volume
# MAGIC      `dbx_workshop_dev.shared.impulse_libs` — a normal, versioned Python package install.
# MAGIC    * **`repo`**: the instructor's shared Git folder `/Workspace/Shared/impulse`; then this cell only
# MAGIC      installs `lz4` and the code is imported straight from the Git folder.
# MAGIC
# MAGIC    (Impulse ships signals between Spark workers as `lz4`-compressed objects; everything else it needs —
# MAGIC    pydantic, scipy, numpy … — is already on Databricks Serverless.)
# MAGIC 3. **`setup_impulse()`** — checks Impulse is importable in the chosen mode and tells you where from.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

require_user_schema()   # checks you set USER_SCHEMA in CONFIG (and creates the schema if needed)
setup_impulse()         # makes `import impulse_query_engine` / `impulse_reporting` work

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # keep the output readable

import pandas as pd
import pyspark.sql.functions as F
from common.workshop_helpers import plot_signals, plot_histogram, intervals_to_pandas


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · The data: a simulated vehicle fleet
# MAGIC
# MAGIC We use the open [Synthetic Vehicle Fleet Telemetry](https://huggingface.co/datasets/renumics/synthetic-vehicle-fleet-telemetry)
# MAGIC dataset (Renumics / VW Group Innovation / IPG Automotive, CC BY-SA 4.0): **100 simulated trips**
# MAGIC of **13 vehicle models** (EVs, hybrids and combustion cars) across three scenarios — mountain
# MAGIC drives, race-track laps and EV charging — sampled at roughly **100 Hz**.
# MAGIC
# MAGIC Your instructor has already converted it into the **Impulse silver data model**. You don't need to
# MAGIC know how (that's workshop 2) — only two words of vocabulary:
# MAGIC
# MAGIC | Impulse term | Meaning | In this dataset |
# MAGIC |---|---|---|
# MAGIC | **container** | one recording | one **trip** (`container_id` = trip / session id) |
# MAGIC | **channel** | one signal inside a recording | one **signal** of that trip, e.g. `Vhcl.v` (vehicle speed) |
# MAGIC
# MAGIC A few **vehicle terms** used throughout the workshops:
# MAGIC
# MAGIC | Term | Meaning |
# MAGIC |---|---|
# MAGIC | **SOC** — *state of charge* | how full the high-voltage traction battery is, in **% of its usable capacity** (0 % = empty, 100 % = full). The EV equivalent of a fuel gauge. Signal `PT.BCU.BattHV.SOC`. |
# MAGIC | **HV battery** | the high-voltage battery (hundreds of volts) that drives the e-motor — not the 12 V battery |
# MAGIC | **BMS** | *battery management system* — the controller that measures and reports SOC, current, temperature |
# MAGIC | **Regeneration (regen)** | the e-motor works as a generator while braking and charges the battery: motor power and battery current turn **negative** |
# MAGIC | **Powertrain types** | `ev` battery-electric · `hev_*` hybrid (engine + e-motor) · `generic` combustion engine only |
# MAGIC
# MAGIC Three tables are all Impulse strictly needs. Let's peek at each.

# COMMAND ----------

# MAGIC %md
# MAGIC ### `container_metrics` — one row per trip
# MAGIC The "what was recorded" table: start/stop time (epoch **microseconds**), vehicle, scenario, driver style…
# MAGIC
# MAGIC ⏰ The signal timestamps are **time-normalized**: every trip's samples start at 2026-03-20 12:00:00, so
# MAGIC `start_ts` is the same for all trips. The real start of each trip is `trip_start_dt` — use it to order trips.

# COMMAND ----------

trips = spark.table(SILVER["container_metrics"])
display(trips.select("container_id", "trip_start_dt", "vehicle_key", "powertrain_type", "scenario", "driver",
                     "duration_s", "num_channels", "start_ts", "stop_ts").orderBy("trip_start_dt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### `channel_metrics` — one row per (trip, signal)
# MAGIC Which signals exist in which trip, with their name, unit and summary statistics. Impulse uses it to
# MAGIC **find** channels *before* it reads any sample data — that's why queries over thousands of
# MAGIC recordings stay fast.

# COMMAND ----------

display(spark.table(SILVER["channel_metrics"])
        .where("container_id = 0")
        .select("channel_id", "channel_name", "unit", "sample_count", "min", "max", "mean", "sample_rate_hz")
        .orderBy("channel_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### `channels` — the samples themselves
# MAGIC Each row says *"from `tstart` to `tend`, this channel had this `value`"*. This **interval**
# MAGIC (run-length encoded) form means a signal that doesn't change — a gear, a pedal at rest — is a single
# MAGIC row instead of thousands. It's by far the biggest table: hundreds of millions of rows.

# COMMAND ----------

display(spark.table(SILVER["channels"]).where("container_id = 0 AND channel_id = 680").orderBy("tstart").limit(10))
print(f"channels table: {spark.table(SILVER['channels']).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Connect Impulse to the silver layer
# MAGIC
# MAGIC A **`MeasurementDB`** is Impulse's handle on the silver tables. You give it the table names
# MAGIC (CONFIG's `impulse_source()` builds that dict for you) and it gives you **`db.query`**, the entry
# MAGIC point of **TSAL** — the *Time Series Analytics Language*.
# MAGIC
# MAGIC A **solver** turns TSAL into Spark jobs. We always use the `DefaultSolver`, which reads the silver
# MAGIC tables and processes every trip **in parallel** on the cluster.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver

source = impulse_source(SILVER)          # {"container_metrics_table": ..., "channels_uri": ..., ...}
for k, v in source.items():
    print(f"{k:26s} -> {v}")

db = MeasurementDB(MeasurementDBConfig(**source, table_locations="unity_catalog"), WorkspaceClient())
solver = DefaultSolver(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Your first TSAL query — signals in, DataFrame out
# MAGIC
# MAGIC ### 4a · Select a channel
# MAGIC Channels are selected by **metadata**, not by table columns: `channel(channel_name="Vhcl.v")` means
# MAGIC *"in every trip, the signal whose `channel_name` tag is `Vhcl.v`"*. You could add more tags
# MAGIC (`unit="m/s"`, `category="Vhcl"`) and they must all match.
# MAGIC
# MAGIC 👉 Nothing is read yet. A selection is a **lazy expression** — a recipe Impulse executes later.
# MAGIC
# MAGIC ### Real data has quirks — handle them in the signal definition
# MAGIC Like any real measurement data, this dataset has known quirks (full list: **`DATA_NOTES.md`** at the repo
# MAGIC root). The silver layer stores what was measured, **unchanged**; each analysis handles the quirk where it
# MAGIC defines the signal. Two matter right away:
# MAGIC
# MAGIC | Signal | Quirk | Handling |
# MAGIC |---|---|---|
# MAGIC | `Vhcl.v` | spikes above 2,000 m/s in three trips | keep only values < 100 m/s |
# MAGIC | `PT.BCU.BattHV.SOC` (state of charge, % battery full) | 0 for a moment while the battery management system (BMS) starts | keep only SOC > 0 |
# MAGIC
# MAGIC `signal.where(condition)` keeps the signal only where the condition holds — more on that in section 6.

# COMMAND ----------

vhcl_v  = db.query.channel(channel_name="Vhcl.v")              # vehicle speed [m/s], as logged
soc_raw = db.query.channel(channel_name="PT.BCU.BattHV.SOC")   # HV battery state of charge [%], as logged

speed = vhcl_v.where(vhcl_v < VHCL_V_MAX_MPS)                   # drop implausible spikes (> 100 m/s)
soc   = soc_raw.where(soc_raw > 0)                              # ignore SOC = 0 while the BMS initialises

print(type(speed).__name__, "->", speed)                        # just an expression, no data yet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b · Reduce it and solve
# MAGIC `.max()`, `.min()`, `.mean()` reduce a signal to **one number per trip**. `.alias()` names the result
# MAGIC column. `select(...)` lists what you want, and **`.solve(spark, solver)`** runs it across all trips and
# MAGIC returns a regular **Spark DataFrame** with **one row per trip** — `container_id` first, then one column
# MAGIC per expression.
# MAGIC
# MAGIC This is the quickest way to **test** an idea: write the expression, get a table back, eyeball it.
# MAGIC
# MAGIC Like any Spark DataFrame the result is **lazy**: every `display()` would recompute it. Because we
# MAGIC reuse it below, we save it once as a small table in **your** schema — Impulse output is just data.
# MAGIC
# MAGIC ⏳ This first query touches **all 100 trips × 2 signals (~40 million samples)**, so give it a minute
# MAGIC or two. The next ones filter trips first and come back in seconds.

# COMMAND ----------

kpis_df = (
    db.query
    .select(
        vhcl_v.max().alias("raw_max_speed_mps"),     # as logged, for comparison
        speed.max().alias("max_speed_mps"),          # cleaned
        speed.mean().alias("mean_speed_mps"),        # duration-weighted mean -- see section 7
        soc.min().alias("min_soc_pct"),
        soc.max().alias("max_soc_pct"),
    )
    .solve(spark, solver=solver)                     # -> Spark DataFrame, one row per trip
)

KPI_TABLE = f"{USER_SCHEMA}.{W1_GOLD_PREFIX}_trip_kpis"
save_table(kpis_df, KPI_TABLE)          # compute once (= df.write.mode("overwrite").saveAsTable, with retry)
kpis = spark.table(KPI_TABLE)
print(f"{kpis.count()} trips -> {KPI_TABLE}")
display(kpis.orderBy("container_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c · Where Impulse stops and plain Spark takes over
# MAGIC
# MAGIC This is an important idea: **Impulse does the time-series work, and hands you an ordinary table.**
# MAGIC
# MAGIC | Step | Done by | What happens |
# MAGIC |---|---|---|
# MAGIC | 1. resolve channels, read samples, align, weight by duration, reduce per trip | **Impulse** (the `kpis_df` cell, 4b) | ~40 M samples → **one row per trip** |
# MAGIC | 2. join with trip metadata, convert units, group, sort | **generic Spark** (this cell) | 100 rows → one row per vehicle |
# MAGIC
# MAGIC Impulse results always start with **`container_id`**, the same key as `container_metrics`. So
# MAGIC enriching them is a normal Spark join — nothing Impulse-specific. Everything you know from Spark /
# MAGIC SQL applies: joins, `groupBy`, window functions, `saveAsTable`, or `spark.sql` on the saved table.
# MAGIC (`.toPandas(spark, solver)` returns a **pandas** DataFrame instead — we only need that when we want the
# MAGIC signals themselves as Python objects, as in section 5.)
# MAGIC
# MAGIC **Question answered:** *what is the top speed of each vehicle model across all its trips?*

# COMMAND ----------

# --- Generic Spark from here on: Impulse already did the hard part ---------------------------------

by_vehicle = (
    kpis                                                                     # Impulse result: 1 row per trip
    .join(trips.select("container_id", "vehicle_key"), "container_id")       # + vehicle from container_metrics
    .withColumn("max_speed_kmh", F.col("max_speed_mps") * 3.6)               # plain unit conversion
    .groupBy("vehicle_key")                                                  # one row per vehicle model ...
    .agg(F.round(F.max("max_speed_kmh"), 1).alias("max_speed_kmh"),          # ... with its fastest trip
         F.count("*").alias("trips"))
    .orderBy(F.desc("max_speed_kmh"))                                        # fastest first
)
display(by_vehicle)

# COMMAND ----------

# MAGIC %md
# MAGIC The same question in **SQL** on the saved table — again, nothing Impulse-specific:

# COMMAND ----------

display(spark.sql(f"""
  SELECT t.vehicle_key, round(max(k.max_speed_mps) * 3.6, 1) AS max_speed_kmh, count(*) AS trips
  FROM {KPI_TABLE} k JOIN {SILVER['container_metrics']} t USING (container_id)
  GROUP BY t.vehicle_key ORDER BY max_speed_kmh DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC > 🔎 **Why the `.where(v < 100)` matters:** without it, three trips report top speeds of up to
# MAGIC > ~2,100 m/s (7,700 km/h!). The instructor's silver build **flags** such findings in a data-quality
# MAGIC > table — it never deletes measured data — so every analyst can see them and decide how to handle them.

# COMMAND ----------

display(kpis.where(F.col("raw_max_speed_mps") > VHCL_V_MAX_MPS)
        .join(trips.select("container_id", "vehicle_key", "scenario"), "container_id")
        .select("container_id", "vehicle_key", "scenario",
                F.round("raw_max_speed_mps", 1).alias("raw_max_speed_mps"),
                F.round("max_speed_mps", 1).alias("max_speed_mps")))
display(spark.table(f"{SHARED_SCHEMA}.{TABLE_PREFIX}_silver_dq_findings")
        .select("container_id", "channel_name", "finding", "implausible_pct", "observed_max", "handling")
        .orderBy("channel_name", "container_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4d · Filter which trips are processed
# MAGIC `havingTag(...)` restricts the query to trips whose **container tags** match — here only the
# MAGIC Mercedes EQS trips. Fewer trips → less data read → faster answers.

# COMMAND ----------

eqs = (
    db.query.havingTag(vehicle_key="MB_EQS_350")
    .select(speed.max().alias("max_speed_mps"), soc.min().alias("min_soc_pct"), soc.max().alias("max_soc_pct"))
    .solve(spark, solver=solver)
)
display(eqs.join(trips.select("container_id", "vehicle_key", "scenario"), "container_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Signals as Python objects
# MAGIC
# MAGIC Select a channel **without** reducing it and each row holds the whole signal as a **`SampleSeries`**:
# MAGIC three numpy arrays `tstarts`, `tends`, `values`. This is how you test expressions on real data —
# MAGIC look at the numbers, plot them.
# MAGIC
# MAGIC 👉 This is the one case where we use **`.toPandas(spark, solver)`** instead of `.solve(...)`: in a Spark
# MAGIC DataFrame a signal is a compressed binary column, and `.toPandas` turns it back into a Python
# MAGIC `SampleSeries` object you can inspect and plot.
# MAGIC
# MAGIC We filter to one trip with `where(query.metric("container_id") == 0)` — a filter on a
# MAGIC `container_metrics` column. Trip 0 is a Tesla Model S mountain drive.

# COMMAND ----------

TRIP = 0
q = db.query    # a query builder; .where() adds a filter to it, .select() sets what to compute
one_trip = (
    q.where(q.metric("container_id") == TRIP)
    .select(speed.alias("speed"), soc.alias("soc"))
    .toPandas(spark, solver=solver)          # pandas: we want the signals as Python objects to plot
)
s = one_trip.loc[0, "speed"]
print(type(s).__name__, "with", len(s), "intervals")
print("first tstarts:", s.tstarts[:3], "\nfirst values :", s.values[:3])

# COMMAND ----------

fig = plot_signals({"speed [m/s]": one_trip.loc[0, "speed"], "SOC [%]": one_trip.loc[0, "soc"]},
                   title=f"Trip {TRIP} — raw signals from Impulse")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Derived signals and conditions
# MAGIC
# MAGIC TSAL uses plain Python operators — and the **type of the result tells you what you can do next**:
# MAGIC
# MAGIC | You write | You get | Think of it as |
# MAGIC |---|---|---|
# MAGIC | `speed * 3.6`, `a + b`, `a / b` | **`SampleSeries`** | a new (virtual) signal |
# MAGIC | `speed_kmh > 80`, `a < b` | **`Intervals`** | the time windows where the condition is true |
# MAGIC | `cond_a & cond_b`, `cond_a \| cond_b` | **`Intervals`** | intersection / union of windows |
# MAGIC | `signal.where(windows)` | **`SampleSeries`** | the signal *only inside* those windows |
# MAGIC
# MAGIC Below: convert speed to km/h, find when the driver is braking **while** going faster than 50 km/h,
# MAGIC and keep the speed only inside those windows.

# COMMAND ----------

brake = db.query.channel(channel_name="DM.Brake")      # brake pedal demand, 0..1 (sampled at a different rate!)

speed_kmh = speed * 3.6                                 # SampleSeries: a virtual signal
braking_fast = (brake > 0.1) & (speed_kmh > 50)         # Intervals: time windows
speed_while_braking = speed_kmh.where(braking_fast)     # SampleSeries restricted to the windows

q = db.query    # a fresh builder: .where() adds filters to the builder it's called on
res = (
    q.where(q.metric("container_id") == TRIP)
    .select(speed_kmh.alias("speed_kmh"), brake.alias("brake"),
            braking_fast.alias("braking_fast"), speed_while_braking.alias("speed_while_braking"))
    .toPandas(spark, solver=solver)          # pandas again: we plot the signals
)
windows = intervals_to_pandas(res.loc[0, "braking_fast"], t0=res.loc[0, "speed_kmh"].tstarts[0])
print(f"{len(windows)} braking windows above 50 km/h, {windows.duration_s.sum():.0f} s in total")
windows.sort_values("duration_s", ascending=False).head()

# COMMAND ----------

fig = plot_signals({"speed [km/h]": res.loc[0, "speed_kmh"], "brake demand": res.loc[0, "brake"],
                    "speed while\nbraking": res.loc[0, "speed_while_braking"]},
                   intervals=res.loc[0, "braking_fast"], title="Braking above 50 km/h (shaded)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Why this matters — what Impulse does for you
# MAGIC
# MAGIC Two problems make measurement data painful in plain SQL. Impulse handles both, on every trip, in
# MAGIC parallel:
# MAGIC
# MAGIC **1. Time weighting.** A sample is a *value that holds for a duration*. A gear that stays in 6th for
# MAGIC 60 s must count 60× more than one that flicks through 3rd for 1 s. `AVG(value)` in SQL weights every
# MAGIC row equally — and with change-only or run-length-encoded data, that is simply wrong.
# MAGIC
# MAGIC **2. Alignment.** Brake (≈30 Hz) and speed (≈100 Hz) are sampled at *different* instants. Above,
# MAGIC `(brake > 0.1) & (speed_kmh > 50)` just worked — Impulse synchronized the two signals on their
# MAGIC overlapping validity intervals. In SQL that is an as-of join per signal pair, per trip.
# MAGIC
# MAGIC Let's prove point 1 with the gear signal of trip 7 — a Volvo XC90 hybrid with an 8-speed gearbox
# MAGIC (trip 0 is a single-speed Tesla, so it wouldn't show anything).

# COMMAND ----------

GEAR_TRIP = 7
gear = db.query.channel(channel_name="Vhcl.GearNo")
q = db.query
impulse_gear = (q.where(q.metric("container_id") == GEAR_TRIP)
                .select(gear.mean().alias("impulse_duration_weighted_mean"))
                .solve(spark, solver=solver)
                .first().impulse_duration_weighted_mean)

naive = spark.sql(f"""
    SELECT avg(c.value) AS naive_sql_avg
    FROM {SILVER['channels']} c JOIN {SILVER['channel_metrics']} m USING (container_id, channel_id)
    WHERE c.container_id = {GEAR_TRIP} AND m.channel_name = 'Vhcl.GearNo'
""").first().naive_sql_avg

print(f"Naive SQL AVG(gear)           : {naive:.2f}")
print(f"Impulse duration-weighted mean: {impulse_gear:.2f}")
print("The SQL average over-counts the many short gear changes and under-counts long cruising in high gear.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Point 2 in practice: the same question in Impulse and in plain SQL
# MAGIC
# MAGIC **Question:** for each lap-time trip, *how long was the driver braking above 50 km/h, in how many
# MAGIC separate windows, and what was the (duration-weighted) speed during them?*
# MAGIC
# MAGIC Brake (`DM.Brake`, ≈30 Hz) and speed (`Vhcl.v`, ≈100 Hz) have **different timestamps**, so the
# MAGIC answer requires aligning two signals. Impulse: the three lines from section 6. Plain SQL: see the
# MAGIC cell after next. Both run on the same 20 trips; compare the code — and the run times.

# COMMAND ----------

# DBTITLE 1,Impulse
import time

t = time.time()
lap = db.query.havingTag(scenario="lap_time")
braking_impulse = (
    lap.select(
        braking_fast.alias("windows"),                                        # (brake > 0.1) & (speed_kmh > 50)
        speed_kmh.where(braking_fast).mean().alias("mean_speed_kmh"),         # duration-weighted, inside windows
    )
    .solve(spark, solver=solver)
    # the windows come back as an array of [start, end] pairs -> count them and add up their lengths.
    # (A condition still true at the very end of a recording also yields a zero-length [t, t] window at
    #  the last sample; we count only windows with a real duration.)
    .select("container_id",
            F.expr("size(filter(coalesce(windows, array()), w -> w[1] > w[0]))").alias("n_windows"),
            F.expr("aggregate(coalesce(windows, array()), 0D, (acc, w) -> acc + (w[1] - w[0])) / 1e6").alias("braking_s"),
            "mean_speed_kmh")
)
save_table(braking_impulse, f"{USER_SCHEMA}.{W1_GOLD_PREFIX}_braking_impulse")
display(spark.table(f"{USER_SCHEMA}.{W1_GOLD_PREFIX}_braking_impulse").orderBy("container_id"))
print(f"Impulse: {time.time() - t:.0f} s")

# COMMAND ----------

# DBTITLE 1,The same in plain SQL
t = time.time()
braking_sql = spark.sql(f"""
  WITH lap_channels AS (            -- 1. the channel ids of both signals in every lap-time trip
    SELECT m.container_id, m.channel_id, m.channel_name
    FROM {SILVER['channel_metrics']} m
    JOIN {SILVER['container_metrics']} t USING (container_id)
    WHERE t.scenario = 'lap_time' AND m.channel_name IN ('DM.Brake', 'Vhcl.v')
  ),
  intervals AS (                    -- 2. their RLE intervals [tstart, tend), each flagged with its condition
    SELECT c.container_id, l.channel_name, c.tstart, c.tend, c.value,
           CASE WHEN l.channel_name = 'DM.Brake' THEN c.value > 0.1
                ELSE c.value < {VHCL_V_MAX_MPS} AND c.value * 3.6 > 50 END AS cond
    FROM {SILVER['channels']} c JOIN lap_channels l USING (container_id, channel_id)
  ),
  island_starts AS (                -- 3. gaps-and-islands, part 1: does a "true" interval start a new window?
    SELECT *,                        --    (not if the previous interval was also true AND touches this one)
           CASE WHEN lag(cond) OVER w AND lag(tend) OVER w = tstart THEN 0 ELSE 1 END AS new_window
    FROM intervals
    WINDOW w AS (PARTITION BY container_id, channel_name ORDER BY tstart)
  ),
  islands AS (                      --    part 2: number the windows with a running sum ...
    SELECT container_id, channel_name, tstart, tend,
           sum(new_window) OVER (PARTITION BY container_id, channel_name ORDER BY tstart
                                 ROWS UNBOUNDED PRECEDING) AS window_no
    FROM island_starts WHERE cond
  ),
  windows AS (                      --    ... and collapse each one into [start, end)
    SELECT container_id, channel_name, min(tstart) AS s, max(tend) AS e
    FROM islands GROUP BY container_id, channel_name, window_no
  ),
  both_true AS (                    -- 4. ALIGNMENT: intersect every brake window with every speed window
    SELECT b.container_id, greatest(b.s, v.s) AS s, least(b.e, v.e) AS e
    FROM windows b JOIN windows v
      ON b.container_id = v.container_id AND b.channel_name = 'DM.Brake' AND v.channel_name = 'Vhcl.v'
     AND b.s < v.e AND v.s < b.e                  -- overlap condition
  ),
  speed_in_windows AS (             -- 5. clip every speed interval to the windows it overlaps (for weighting)
    SELECT w.container_id, i.value * 3.6 AS kmh,
           least(i.tend, w.e) - greatest(i.tstart, w.s) AS overlap_us
    FROM both_true w JOIN intervals i
      ON i.container_id = w.container_id AND i.channel_name = 'Vhcl.v' AND i.value < {VHCL_V_MAX_MPS}
     AND i.tstart < w.e AND w.s < i.tend
  )
  SELECT w.container_id, w.n_windows, w.braking_s, s.mean_speed_kmh
  FROM (SELECT container_id, count(*) AS n_windows, sum(e - s) / 1e6 AS braking_s
        FROM both_true GROUP BY container_id) w
  LEFT JOIN (SELECT container_id, sum(kmh * overlap_us) / sum(overlap_us) AS mean_speed_kmh   -- 6. weighted mean
             FROM speed_in_windows GROUP BY container_id) s USING (container_id)
""")
save_table(braking_sql, f"{USER_SCHEMA}.{W1_GOLD_PREFIX}_braking_sql")
display(spark.table(f"{USER_SCHEMA}.{W1_GOLD_PREFIX}_braking_sql").orderBy("container_id"))
print(f"plain SQL: {time.time() - t:.0f} s")

# COMMAND ----------

# MAGIC %md
# MAGIC Do the two agree? Largest difference per column over all lap-time trips (trip 95 only appears on the
# MAGIC Impulse side, with 0 windows: its speed signal is corrupted throughout, so the SQL finds no rows):

# COMMAND ----------

a, b = (spark.table(f"{USER_SCHEMA}.{W1_GOLD_PREFIX}_braking_{x}") for x in ("impulse", "sql"))
display(a.join(b, "container_id", "full").select(
    F.count("*").alias("trips"),
    *[F.max(F.abs(a[c] - b[c])).alias(f"max_diff_{c}") for c in ("n_windows", "braking_s", "mean_speed_kmh")]))

# COMMAND ----------

# MAGIC %md
# MAGIC 👉 **Same answer — very different effort.** The SQL needs gaps-and-islands window functions *per
# MAGIC signal*, an interval-overlap join to align them and a second overlap join with clipping for the
# MAGIC weighted mean; one wrong `<` vs `<=` silently changes the result. Every extra signal or condition
# MAGIC (`& (gear > 3)`, `.filter(2e6)`, an `expand` around events) adds another round of that. In Impulse it
# MAGIC stays one readable expression — and the very same `braking_fast` can become an **event** in a report
# MAGIC (notebook 3), a histogram scope, or a calculated channel.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · A first report — all trips, one call
# MAGIC
# MAGIC Ad-hoc queries are perfect for exploring. When you want the **same analysis on every recording** —
# MAGIC and results you can put on a dashboard — you build a **report**:
# MAGIC
# MAGIC * an **event** defines *when* to look (time windows, e.g. "speed above 100 km/h"),
# MAGIC * an **aggregation** defines *what* to compute in those windows (histograms, statistics),
# MAGIC * a **page** groups aggregations, and `report.run()` computes everything for every trip.
# MAGIC
# MAGIC Here we keep it in memory (`persist_results=False`) and only process the **lap-time** trips via a
# MAGIC `container_filters` tag filter. In notebook 3 you'll write a full report to **your** schema.

# COMMAND ----------

from impulse_reporting.core.report import Report
from impulse_reporting.core.page import Page
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.aggregations.histogram import HistogramDuration

preview_config = {
    "source": impulse_source(SILVER),
    "query_engine": {"solver": "DefaultSolver", "data_type": "RLE"},
    "container_filters": {"tag_filters": [[{"tag_name": "scenario", "comparator": "==", "value": "lap_time"}]]},
}
report = Report(name="w1_preview", spark=spark, workspace_client=WorkspaceClient(), config=preview_config)

r_v = report.get_db().query.channel(channel_name="Vhcl.v")
r_speed_kmh = r_v.where(r_v < VHCL_V_MAX_MPS) * 3.6        # same clean definition as above

fast = BasicEvent(name="above_100_kmh", expr=r_speed_kmh > 100, desc="Vehicle faster than 100 km/h")
report.add_event(fast)                     # every event must be registered with the report

page = Page(page_number=1)
page.add_aggregation(HistogramDuration(
    name="speed_distribution_fast", base_expr=r_speed_kmh, event=fast,
    bins=[float(b) for b in range(100, 210, 10)], bins_unit="km/h", values_unit="us",
    channel_name="Vhcl.v [km/h]", desc="Time spent per 10 km/h speed band while above 100 km/h"))
report.add_page(page)

report.run(persist_results=False)          # compute, don't write
print("aggregations computed:", list(report.aggregation_dfs))

# COMMAND ----------

# MAGIC %md
# MAGIC The histogram is **duration-weighted**: each bin holds the *time* (µs) spent in that speed band — not
# MAGIC a sample count — so trips recorded at different rates are directly comparable.

# COMMAND ----------

# Results are keyed by aggregation type; "changed" = freshly computed in this run.
hist = report.aggregation_dfs["HISTOGRAM"]["changed"]
summary = (hist.groupBy("lower_bound", "upper_bound").agg((F.sum("hist_value") / 1e6).alias("seconds"))
           .orderBy("lower_bound"))
display(summary)

# Plot it: one bar per 10 km/h band, bar height = time spent in that band across all lap-time trips.
pdf = summary.toPandas()          # 10 rows -- tiny, so pandas/matplotlib is fine here
fig = plot_histogram(pdf.lower_bound, pdf.upper_bound, pdf.seconds,
                     title="Lap-time trips: time spent per speed band while above 100 km/h",
                     xlabel="speed [km/h]", ylabel="time [s]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Recap
# MAGIC
# MAGIC * The **silver model** = containers (trips) + channels (signals), with samples stored as intervals.
# MAGIC * Real data has quirks: silver keeps it unchanged, your **signal definition** handles it (`DATA_NOTES.md`).
# MAGIC * **TSAL** expressions are lazy: `channel()` selects by metadata, operators derive signals (`SampleSeries`)
# MAGIC   and time windows (`Intervals`), `.where()` combines them, reducers produce one number per trip.
# MAGIC * `select(...).solve(spark, solver)` turns any expression into a **Spark DataFrame** — ideal for testing,
# MAGIC   and from there it's plain Spark/SQL. Use `.toPandas(...)` only when you need signals as Python objects.
# MAGIC * Impulse does the hard parts for you: **duration weighting**, **signal alignment**, **parallelism**.
# MAGIC * A **report** applies events + aggregations to every trip in one call.
# MAGIC
# MAGIC **Next → `02_genie_code_with_impulse_skills`**: let Genie Code write TSAL for you.
