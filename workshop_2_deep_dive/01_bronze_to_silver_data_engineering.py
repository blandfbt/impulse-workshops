# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2 · Notebook 1 — Data engineering: bronze → the Impulse silver model
# MAGIC
# MAGIC **⏱ ~60 minutes · 🎯 Goal: build an Impulse silver layer yourself and understand every table in it.**
# MAGIC
# MAGIC In workshop 1 the silver layer "just existed". Real projects start from whatever the loggers,
# MAGIC simulators or vendors produce. Impulse **does not ship an ingestion component** — shaping data into
# MAGIC its silver model is your job, and the decisions you make here (time base, encoding, metadata, data
# MAGIC quality) decide how easy and how fast every downstream analysis will be.
# MAGIC
# MAGIC You will turn the **bronze** copy of the Renumics fleet dataset (in `dbx_workshop_dev.shared`) into a
# MAGIC complete Impulse silver layer **in your own schema**, for six hand-picked trips:
# MAGIC
# MAGIC ```
# MAGIC  BRONZE (source star schema)                    SILVER (Impulse model)                      required?
# MAGIC  ─────────────────────────────                  ──────────────────────────────────────────  ─────────
# MAGIC  impulse_bronze_signal_data ──┬─ samples ──DQ──► channels           [tstart,tend) → value (RLE)  ✅
# MAGIC   (1 row per observation)     │                 channel_metrics    1 row / (trip, signal)       ✅
# MAGIC  impulse_bronze_signal_info ──┤                 container_metrics  1 row / trip                 ✅
# MAGIC  impulse_bronze_trip_info   ──┤                 channel_tags       EAV signal metadata          optional
# MAGIC  impulse_bronze_vehicle_info──┤                 container_tags     EAV trip metadata            optional
# MAGIC  impulse_bronze_vehicle_state─┘                 poi_channels       point events                 optional
# MAGIC                                                 channel_mapping    logical aliases              optional
# MAGIC                                                 unit_conversion    unit families + factors      optional
# MAGIC ```
# MAGIC
# MAGIC | Step | Builds | Key idea |
# MAGIC |---|---|---|
# MAGIC | 0 | — | explore bronze and spot its traps |
# MAGIC | 1 | samples | one time base (epoch µs, UTC), one value column |
# MAGIC | 2 | `dq_findings` | flag data-quality issues — never modify measured data |
# MAGIC | 3 | `channels` | **run-length encoding (RLE)** |
# MAGIC | 4 | `channel_metrics` | how Impulse finds channels without reading samples |
# MAGIC | 5 | `channel_tags` | EAV metadata = select by *anything* |
# MAGIC | 6 | `container_metrics`, `container_tags` | the recording-level view |
# MAGIC | 7 | `poi_channels` | events that only exist at an instant |
# MAGIC | 8 | `channel_mapping`, `unit_conversion` | logical names, fallbacks, unit conversion |
# MAGIC | 9 | — | write Delta tables, then **validate with Impulse** |
# MAGIC
# MAGIC The same logic as a production **Lakeflow Declarative Pipeline** lives in `declarative_pipeline/`.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

require_user_schema()
setup_impulse()

import json
import warnings
import pyspark.sql.functions as F
from pyspark.sql import Window

warnings.filterwarnings("ignore", category=UserWarning)

# One rule for every timestamp conversion below: interpret the source's zone-less timestamps as UTC.
spark.conf.set("spark.sql.session.timeZone", "UTC")

TRIPS = W2_ENGINEERING_TRIPS                      # the six trips we engineer by hand (see CONFIG)
MY_SILVER = silver_tables(USER_SCHEMA, W2_SILVER_PREFIX)
print("trips:", TRIPS)
print("writing silver to:", USER_SCHEMA, "with prefix", W2_SILVER_PREFIX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0 · Explore bronze — know your source before you model it
# MAGIC
# MAGIC Bronze is a faithful copy of the source: one **fact** table of observations and four **dimension**
# MAGIC tables. Grain of the fact table: *one value of one signal at one instant in one trip*.

# COMMAND ----------

bronze = spark.table(BRONZE["signal_data"]).where(F.col("session_id").isin(TRIPS))
signal_info = spark.table(BRONZE["signal_info"])
bronze.printSchema()
display(bronze.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trap 1 — signals are *not* sampled together
# MAGIC Every signal has its own timestamps and rate. Some stream at ~100 Hz, others are only written **when
# MAGIC they change** (a gear, a status flag — sometimes just 2 rows per trip). Any model that assumes a
# MAGIC shared time grid ("one wide row per timestamp") would either explode in size or invent data.

# COMMAND ----------

per_signal = (
    bronze.join(signal_info.select("signal_id", "signal_name", "unit"), "signal_id")
    .groupBy("signal_name", "unit")
    .agg(F.count("*").alias("observations"),
         F.round(F.count("*") / F.countDistinct("session_id"), 0).alias("obs_per_trip"),
         F.sum(F.col("int_value").isNotNull().cast("int")).alias("int_values"),
         F.min(F.coalesce("float_value", "int_value")).alias("min"),
         F.max(F.coalesce("float_value", "int_value")).alias("max"))
    .orderBy("observations")
)
display(per_signal)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trap 2 — labels you can't blindly trust
# MAGIC Check every signal's values against its *declared* unit. The query below does it for a few signals:
# MAGIC
# MAGIC * `PT.BattHV.Temp` is labelled **degC** but sits around 293–335 → it's logged in **Kelvin**.
# MAGIC * `Vhcl.PoI.GCS.Lat` / `.Long` are labelled **deg** but are tiny numbers (≈ 0.8 / 0.17) → **radians**
# MAGIC   (× 180/π gives the Swiss Engadin / Bernina pass for mountain drives, Hockenheimring for laps).
# MAGIC * `Env.Temperature` is labelled degC, but some trips log Kelvin (−10 … 299): unreliable — we won't use it.
# MAGIC * `PT.BCU.BattHV.SOC` — the **state of charge** (how full the HV battery is) — is labelled `none` but holds **percent**, and reads **0** for a moment while the
# MAGIC   battery management system starts.
# MAGIC * Values live in two columns: `float_value` *or* `int_value`.

# COMMAND ----------

display(spark.sql(f"""
  SELECT s.signal_name, s.unit AS declared_unit, b.session_id,
         round(min(coalesce(b.float_value, b.int_value)), 3) AS min, round(max(coalesce(b.float_value, b.int_value)), 3) AS max
  FROM {BRONZE['signal_data']} b JOIN {BRONZE['signal_info']} s USING (signal_id)
  WHERE b.session_id IN ({','.join(map(str, TRIPS))})
    AND s.signal_name IN ('PT.BattHV.Temp', 'Vhcl.PoI.GCS.Lat', 'Vhcl.PoI.GCS.Long', 'Env.Temperature', 'PT.BCU.BattHV.SOC')
  GROUP BY ALL ORDER BY signal_name, session_id
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trap 3 — a broken sensor
# MAGIC Look at the maximum of `Vhcl.v` per trip. Trip 95 reports thousands of m/s.

# COMMAND ----------

display(spark.sql(f"""
  SELECT b.session_id, s.signal_name, round(max(b.float_value), 1) AS max_mps, round(avg(b.float_value), 1) AS avg_mps
  FROM {BRONZE['signal_data']} b JOIN {BRONZE['signal_info']} s USING (signal_id)
  WHERE b.session_id IN ({','.join(map(str, TRIPS))}) AND s.signal_name IN ('Vhcl.v', 'Car.Gen.vx_1')
  GROUP BY ALL ORDER BY session_id, signal_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC `Car.Gen.vx_1` (the body's longitudinal velocity — an independent measurement) agrees with `Vhcl.v`
# MAGIC everywhere **except trip 95**, where `Vhcl.v` is garbage (benchmark question DQ-01 of the dataset).
# MAGIC
# MAGIC ### What do we do with all this?
# MAGIC **We do not modify the measured data.** Silver must be a faithful record — "fixing" it would hide the
# MAGIC problem and bake one team's assumption into everybody's data. Instead:
# MAGIC 1. keep the **declared** unit in silver (it's what the source says),
# MAGIC 2. **flag** checkable findings in a data-quality table (step 2),
# MAGIC 3. **document** every quirk (`DATA_NOTES.md`) and handle it in the analysis's **TSAL signal
# MAGIC    definition** — `temp - 273.15`, `lat * 180/π`, `v.where(v < 100)`, `soc.where(soc > 0)`.
# MAGIC
# MAGIC ⏰ One more: timestamps are **time-normalized** — every trip starts at 2026-03-20 12:00:00. The real trip
# MAGIC start lives in `trip_info.trip_start_timestamp`; we carry it into silver as `trip_start_dt`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Samples — one key, one time base, one value
# MAGIC
# MAGIC Impulse identifies everything by **`container_id`** (the recording) and **`channel_id`** (a signal
# MAGIC *within* that recording):
# MAGIC
# MAGIC | Silver column | From bronze | Why |
# MAGIC |---|---|---|
# MAGIC | `container_id` (bigint) | `session_id` | one trip = one recording |
# MAGIC | `channel_id` (int) | `signal_id` | stable per signal; only has to be unique *within* a container |
# MAGIC | `timestamp` (bigint) | `timestamp` → **epoch microseconds** | integer time: exact, compact, sortable; µs keeps the full source precision. All Impulse time parameters (windows, durations) are then in µs. |
# MAGIC | `value` (double) | `coalesce(float_value, int_value)` | one numeric value column |
# MAGIC
# MAGIC The source timestamps carry no time zone. Converting with the session time zone pinned to UTC
# MAGIC means "12:00:00" becomes exactly 12:00:00 UTC — no silent shift.

# COMMAND ----------

samples = bronze.select(
    F.col("session_id").cast("bigint").alias("container_id"),
    F.col("signal_id").cast("int").alias("channel_id"),
    F.unix_micros(F.col("timestamp").cast("timestamp")).alias("timestamp"),       # epoch µs
    F.coalesce("float_value", F.col("int_value").cast("double")).alias("value"),
)
display(samples.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Data-quality findings — flag, don't drop
# MAGIC
# MAGIC We declare **plausible ranges** for signals with known failure modes, measure every (trip, signal) and
# MAGIC record each one with implausible samples — with the evidence and the recommended handling — in a
# MAGIC `dq_findings` table. Nothing is removed: analysts see the findings and apply the handling in TSAL. A
# MAGIC dashboard page on this table makes data quality visible to everyone (notebook 2b).

# COMMAND ----------

dq_rules = spark.createDataFrame([
    ("Vhcl.v", 0.0, 100.0, "implausible speed spikes (up to ~2,100 m/s)", "drop values > 100 m/s: v.where(v < 100)"),
    ("PT.BCU.BattHV.SOC", 1e-9, 100.0, "SOC = 0 while the BMS initialises", "ignore SOC <= 0: soc.where(soc > 0)"),
], "signal_name string, plausible_min double, plausible_max double, finding string, handling string")
rules = dq_rules.join(signal_info, "signal_name").select(
    F.col("signal_id").cast("int").alias("channel_id"), F.col("signal_name").alias("channel_name"),
    "unit", "plausible_min", "plausible_max", "finding", "handling")

out_of_range = (F.col("value") < F.col("plausible_min")) | (F.col("value") > F.col("plausible_max"))
dq_findings = (
    samples.join(F.broadcast(rules), "channel_id")
    .groupBy("container_id", "channel_id", "channel_name", "unit", "plausible_min", "plausible_max", "finding", "handling")
    .agg(F.count("*").alias("sample_count"), F.sum(out_of_range.cast("int")).alias("implausible_count"),
         F.min("value").alias("observed_min"), F.max("value").alias("observed_max"))
    .where(F.col("implausible_count") > 0)
    .withColumn("implausible_pct", F.round(F.col("implausible_count") / F.col("sample_count") * 100, 3))
    .withColumn("checked_at", F.current_timestamp())
)
display(dq_findings)

clean = samples   # flagged, not dropped: every measured value goes on to silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · `channels` — run-length encoding (RLE)
# MAGIC
# MAGIC Impulse treats a sample as **a value that holds for a duration** (zero-order hold): it is valid from its
# MAGIC timestamp until the next sample arrives. So the natural storage format is an interval
# MAGIC `[tstart, tend) → value`, and **consecutive samples with the same value can be merged into one
# MAGIC interval** without losing anything:
# MAGIC
# MAGIC ```
# MAGIC  RAW samples (timestamp, value)          RLE intervals [tstart, tend) → value
# MAGIC  t=0   gear 3                             [0,  30) → 3
# MAGIC  t=10  gear 3        ── collapse ──►      [30, 40) → 4
# MAGIC  t=20  gear 3                             [40, 50) → 5
# MAGIC  t=30  gear 4                             (last interval ends at the last sample)
# MAGIC  t=40  gear 5
# MAGIC  t=50  gear 5
# MAGIC ```
# MAGIC
# MAGIC Impulse can also read RAW `(timestamp, value)` and encode on the fly (`data_type="RAW"`), but encoding
# MAGIC **once at ingestion** saves that work on every single query. The algorithm, per (container, channel),
# MAGIC ordered by time:
# MAGIC
# MAGIC 1. `new_run` = 1 when the value differs from the previous sample (or it's the first sample)
# MAGIC 2. `run_id` = running sum of `new_run` → every run of equal values shares a `run_id`
# MAGIC 3. per run: `tstart` = first timestamp, `tend` = timestamp of the **next** run's first sample, `value`
# MAGIC
# MAGIC Let's watch it work on the gear signal of one trip, intermediate columns included:

# COMMAND ----------

w = Window.partitionBy("container_id", "channel_id").orderBy("timestamp")
gear_id = signal_info.where("signal_name = 'Vhcl.GearNo'").first().signal_id

steps = (
    clean.where((F.col("container_id") == TRIPS[0]) & (F.col("channel_id") == gear_id))
    .withColumn("prev_value", F.lag("value").over(w))
    .withColumn("next_ts", F.lead("timestamp").over(w))
    .withColumn("new_run", F.when(F.col("prev_value").isNull() | (F.col("value") != F.col("prev_value")), 1).otherwise(0))
    .withColumn("run_id", F.sum("new_run").over(w.rowsBetween(Window.unboundedPreceding, 0)))
)
display(steps.orderBy("timestamp").limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC Now the real thing for all six trips and all signals. (`eqNullSafe` also treats two NULLs as "equal".)

# COMMAND ----------

new_run = F.lag("timestamp").over(w).isNull() | ~F.col("value").eqNullSafe(F.lag("value").over(w))
channels = (
    clean
    .withColumn("_next_ts", F.lead("timestamp").over(w))
    .withColumn("_new_run", new_run.cast("int"))
    .withColumn("_run_id", F.sum("_new_run").over(w.rowsBetween(Window.unboundedPreceding, 0)))
    .groupBy("container_id", "channel_id", "_run_id")
    .agg(F.min("timestamp").alias("tstart"),
         F.max(F.coalesce("_next_ts", "timestamp")).alias("tend"),   # interval ends where the next run starts
         F.first("value").alias("value"))
    .select("container_id", "channel_id", "tstart", "tend", "value")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### How much did RLE save?
# MAGIC Honest answer: **it depends entirely on the signal.** Step-like signals (gear, flags, pedals at rest,
# MAGIC change-only counters) collapse dramatically. This simulator emits continuous signals at full float64
# MAGIC precision, so nearly every sample differs from the last and they barely compress.

# COMMAND ----------

raw_n = clean.groupBy("channel_id").agg(F.count("*").alias("raw_samples"))
rle_n = channels.groupBy("channel_id").agg(F.count("*").alias("rle_intervals"))
compression = (raw_n.join(rle_n, "channel_id")
               .join(signal_info.select(F.col("signal_id").alias("channel_id"), "signal_name"), "channel_id")
               .withColumn("compression_x", F.round(F.col("raw_samples") / F.col("rle_intervals"), 2))
               .orderBy(F.desc("compression_x")))
display(compression)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🧪 Experiment: what if values were quantized like real ECU signals?
# MAGIC A real battery-management system doesn't send SOC as `40.149707923363046`; it sends it in fixed steps
# MAGIC (e.g. 0.1 %) over CAN. Quantize SOC and speed to realistic resolutions and re-measure.
# MAGIC *(Experiment only — we keep full precision in silver: quantization is lossy and a business decision.)*

# COMMAND ----------

resolution = {"PT.BCU.BattHV.SOC": 0.1, "Vhcl.v": 0.01, "PT.BattHV.Temp": 0.5}
res_df = spark.createDataFrame(list(resolution.items()), "signal_name string, resolution double").join(
    signal_info.select("signal_id", "signal_name"), "signal_name")
quantized = (clean.join(F.broadcast(res_df.select(F.col("signal_id").alias("channel_id"), "resolution", "signal_name")), "channel_id")
             .withColumn("value", F.round(F.col("value") / F.col("resolution")) * F.col("resolution")))
q_runs = (quantized
          .withColumn("_new_run", (F.lag("value").over(w).isNull() | (F.col("value") != F.lag("value").over(w))).cast("int"))
          .groupBy("signal_name", "resolution")
          .agg(F.count("*").alias("raw_samples"), F.sum("_new_run").alias("rle_intervals"))
          .withColumn("compression_x", F.round(F.col("raw_samples") / F.col("rle_intervals"), 1)))
display(q_runs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · `channel_metrics` — the channel catalogue
# MAGIC
# MAGIC One row per (trip, signal). It plays two roles:
# MAGIC
# MAGIC 1. **Selection.** When you write `channel(channel_name="Vhcl.v")`, Impulse resolves *which*
# MAGIC    `(container_id, channel_id)` pairs match by reading this small table (and/or `channel_tags`)
# MAGIC    **before** touching the huge `channels` table — then reads only those channels. `channel_name`,
# MAGIC    `unit` and `data_key` live here for that reason (`data_key` + `channel_name` are also the join keys
# MAGIC    for aliases in step 8).
# MAGIC 2. **Statistics.** Cheap per-channel stats for data-quality dashboards and for filtering. We compute
# MAGIC    the mean **duration-weighted** — each interval counts by its length — matching Impulse's semantics.
# MAGIC
# MAGIC Impulse doesn't fix the statistic columns: add what your analyses need.

# COMMAND ----------

weight = (F.col("tend") - F.col("tstart")).cast("double")
names = signal_info.select(F.col("signal_id").cast("int").alias("channel_id"),
                           F.col("signal_name").alias("channel_name"), "unit")
raw_counts = clean.groupBy("container_id", "channel_id").agg(F.count("*").alias("raw_sample_count"))
channel_metrics = (
    channels.groupBy("container_id", "channel_id")
    .agg(F.count("*").alias("sample_count"),
         F.avg(F.when(F.isnan("value") | F.col("value").isNull(), 1.0).otherwise(0.0)).alias("nan_ratio"),
         F.min("value").alias("min"), F.max("value").alias("max"),
         F.coalesce(F.try_divide(F.sum(F.col("value") * weight), F.when(F.sum(weight) > 0, F.sum(weight))),
                    F.avg("value")).alias("mean"),                                  # duration-weighted
         F.stddev("value").alias("std"),
         F.percentile_approx("value", [0.01, 0.10, 0.90, 0.99], 1000).alias("_pz"),
         F.min("tstart").alias("begin_ts"), F.max("tend").alias("end_ts"))
    .join(raw_counts, ["container_id", "channel_id"], "left")
    .join(F.broadcast(names), "channel_id", "left")
    .select("container_id", "channel_id", "channel_name",
            F.lit("CarMaker").alias("data_key"),                # the "data source" key used by aliases
            "unit", F.lit("DOUBLE").alias("value_type"),
            "sample_count", "raw_sample_count",
            F.round(F.col("raw_sample_count") / F.col("sample_count"), 2).alias("rle_compression_ratio"),
            "nan_ratio", "min", "max", "mean", "std",
            F.col("_pz")[0].alias("pz1"), F.col("_pz")[1].alias("pz10"),
            F.col("_pz")[2].alias("pz90"), F.col("_pz")[3].alias("pz99"),
            "begin_ts", "end_ts", ((F.col("end_ts") - F.col("begin_ts")) / 1e6).alias("duration_s"),
            F.try_divide(F.col("raw_sample_count"), (F.col("end_ts") - F.col("begin_ts")) / 1e6).alias("sample_rate_hz"))
)
display(channel_metrics.where(F.col("container_id") == TRIPS[0]).orderBy("channel_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · `channel_tags` — EAV metadata
# MAGIC
# MAGIC `channel_tags` stores signal metadata as **entity–attribute–value** rows: `(container_id, channel_id,
# MAGIC key, value)`. When it is configured, *every* keyword of `channel(...)` becomes a lookup here — so you can
# MAGIC select by any metadata (`category="PT"`, `physical_quantity="velocity"`, `subsystem=...`) and add new
# MAGIC metadata later **without changing any schema**. We unpivot the signal dimension (plus a `series_type`
# MAGIC tag) into this shape.

# COMMAND ----------

TAG_COLUMNS = {"channel_name": "signal_name", "signal_description": "signal_description", "unit": "unit",
               "physical_quantity": "physical_quantity", "category": "category", "subsystem": "subsystem",
               "component": "component", "group": "group", "is_categorical": "is_categorical",
               "dtype": "dtype", "signal_mapping": "signal_mapping"}

def unpivot(df, id_cols, value_cols):
    """Wide columns -> (id_cols..., key, value) rows; drop empty values."""
    s = df.select(*id_cols, *[F.col(f"`{c}`").cast("string").alias(c) for c in value_cols])
    return s.unpivot(id_cols, value_cols, "key", "value").where(F.col("value").isNotNull() & (F.trim("value") != ""))

signal_tags = unpivot(
    signal_info.select(F.col("signal_id").cast("int").alias("channel_id"),
                       *[F.col(f"`{src}`").alias(k) for k, src in TAG_COLUMNS.items()],
                       F.lit("CarMaker").alias("data_key"), F.lit("SAMPLE").alias("series_type")),
    ["channel_id"], [*TAG_COLUMNS, "data_key", "series_type"])
display(signal_tags.where("channel_id = 680"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · `container_metrics` + `container_tags` — the recording view
# MAGIC
# MAGIC One row per trip. Impulse hard-references only a few names: **`container_id`**, and **`start_ts` /
# MAGIC `stop_ts`** (used by `ContainerEvent` and `TimeWindowEvent` — so keep them in the *same unit as the
# MAGIC channels*, epoch µs). Everything else is pass-through: list columns in a report's
# MAGIC `measurement_dimensions` and they land in the gold layer for dashboards.
# MAGIC
# MAGIC The source hides useful parameters inside the trip name, e.g.
# MAGIC `mountain_drive_scenario_Tesla_S_nopax_traf2_soc30_80kw_len700_13C_aggressive_RT_265_35R19` — we parse
# MAGIC `scenario`, `traffic`, `soc_target_pct`, `charge_power_kw`, `route_length` into real columns.
# MAGIC (The dataset docs warn the precise meaning of some tokens is undocumented — we keep their raw numbers.)

# COMMAND ----------

keys = bronze.groupBy("session_id").agg(F.first("vehicle_id").alias("vehicle_id"),
                                        F.first("vehicle_state_id").alias("vehicle_state_id"),
                                        F.first("trip_hash").alias("trip_hash"))
name = F.col("trip_name")
token = lambda pattern: F.nullif(F.regexp_extract(name, pattern, 1), F.lit("")).cast("int")
trips = spark.table(BRONZE["trip_info"]).select(
    F.col("session_id").cast("bigint").alias("container_id"), "trip_name",
    F.regexp_extract(name, r"^(.+?)_scenario_", 1).alias("scenario"),
    F.col("starting_temperature").cast("int").alias("starting_temperature_c"),
    F.col("trip_start_timestamp").cast("timestamp").alias("trip_start_dt"),
    token(r"_traf(\d+)_").alias("traffic"), token(r"_soc(\d+)_").alias("soc_target_pct"),
    token(r"_(\d+)kw_").alias("charge_power_kw"), token(r"_len(\d+)_").alias("route_length"))
vehicles = spark.table(BRONZE["vehicle_info"]).select(
    F.col("vehicle_id").cast("int"), F.col("vehicle_name").alias("vehicle_key"), "manufacturer", "vehicle_type",
    "powertrain_type", F.col("battery_voltage_oc").cast("double"), F.col("empty_weight").cast("double").alias("empty_weight_kg"))
states = spark.table(BRONZE["vehicle_state"]).select(
    F.col("vehicle_state_id").cast("int"), F.col("number_of_passengers").cast("int"), "tires", "driver")
bounds = channel_metrics.groupBy("container_id").agg(
    F.countDistinct("channel_id").cast("int").alias("num_channels"),
    F.min("begin_ts").alias("start_ts"), F.max("end_ts").alias("stop_ts"))

container_metrics = (
    bounds.join(keys.select(F.col("session_id").cast("bigint").alias("container_id"), "vehicle_id",
                            "vehicle_state_id", "trip_hash"), "container_id")
    .join(trips, "container_id", "left").join(vehicles, "vehicle_id", "left").join(states, "vehicle_state_id", "left")
    .select("container_id", "start_ts", "stop_ts",
            F.timestamp_micros("start_ts").alias("start_dt"), F.timestamp_micros("stop_ts").alias("stop_dt"),
            ((F.col("stop_ts") - F.col("start_ts")) / 1e6).alias("duration_s"), "num_channels",
            "vehicle_key", "manufacturer", "vehicle_type", "powertrain_type", "battery_voltage_oc", "empty_weight_kg",
            "scenario", "trip_name", "driver", "tires", "number_of_passengers", "traffic", "soc_target_pct",
            "charge_power_kw", "route_length", "starting_temperature_c", "trip_start_dt",
            F.col("container_id").cast("int").alias("session_id"), "vehicle_id", "vehicle_state_id", "trip_hash",
            F.current_timestamp().alias("last_modified"))       # lets incremental reports detect updates
)
CONTAINER_TAGS = ["vehicle_key", "manufacturer", "vehicle_type", "powertrain_type", "scenario", "driver", "tires",
                  "number_of_passengers", "traffic", "soc_target_pct", "charge_power_kw", "route_length",
                  "starting_temperature_c", "trip_name"]
container_tags = unpivot(container_metrics, ["container_id"], CONTAINER_TAGS)
display(container_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 · `poi_channels` — Points-in-Time events
# MAGIC
# MAGIC Some data is not a signal but a stream of **discrete events** — fault codes, log messages, alerts. A
# MAGIC value exists *only at its instant*; there is no "value in between". Storing those as intervals would
# MAGIC be wrong (an alert would appear to last until the next one). Impulse has a dedicated table:
# MAGIC `(container_id, channel_id, timestamp, value_double, value_string)` and selects them with
# MAGIC `poi_channel(...)`.
# MAGIC
# MAGIC The source has no event log, so we derive two from the signals (and say so in the metadata):
# MAGIC
# MAGIC * **`Event.GearShift`** (numeric): at every gear change, the new gear.
# MAGIC * **`Event.DriverAlert`** (string): what a telematics box would emit — `HARSH_BRAKE` (ax < −4 m/s²),
# MAGIC   `HARSH_ACCEL` (ax > 3 m/s²), `SHARP_TURN` (|ay| > 5 m/s²) at the *rising edge* of each condition,
# MAGIC   at most one per code every 3 s (debounce).
# MAGIC
# MAGIC POI channels get IDs above the source range (10001, 10002) and rows in `channel_metrics` /
# MAGIC `channel_tags`, so they are found exactly like any other channel.

# COMMAND ----------

sid = {r.signal_name: r.signal_id for r in signal_info.where(F.col("signal_name").isin("Vhcl.GearNo", "Car.ax", "Car.ay")).collect()}

gear_shifts = (clean.where(F.col("channel_id") == sid["Vhcl.GearNo"])
               .withColumn("_prev", F.lag("value").over(w))
               .where(F.col("_prev").isNotNull() & (F.col("value") != F.col("_prev")))
               .select("container_id", F.lit(10001).alias("channel_id"), "timestamp",
                       F.col("value").alias("value_double"), F.lit(None).cast("string").alias("value_string")))

rules_alert = [("HARSH_BRAKE", "Car.ax", F.col("value") < -4.0), ("HARSH_ACCEL", "Car.ax", F.col("value") > 3.0),
               ("SHARP_TURN", "Car.ay", F.abs("value") > 5.0)]
edges = None
for code, signal, cond in rules_alert:
    e = (clean.where(F.col("channel_id") == sid[signal]).withColumn("_on", cond)
         .withColumn("_prev_on", F.lag("_on").over(w))
         .where(F.col("_on") & ~F.coalesce("_prev_on", F.lit(False)))          # rising edge only
         .select("container_id", "timestamp", F.lit(code).alias("code")))
    edges = e if edges is None else edges.unionByName(e)
wc = Window.partitionBy("container_id", "code").orderBy("timestamp")
driver_alerts = (edges.withColumn("_prev_ts", F.lag("timestamp").over(wc))
                 .where(F.col("_prev_ts").isNull() | (F.col("timestamp") - F.col("_prev_ts") > 3_000_000))  # 3 s debounce
                 .select("container_id", F.lit(10002).alias("channel_id"), "timestamp",
                         F.lit(None).cast("double").alias("value_double"), F.col("code").alias("value_string")))

poi_channels = gear_shifts.unionByName(driver_alerts).select(
    F.col("container_id").cast("bigint"), F.col("channel_id").cast("int"), "timestamp", "value_double", "value_string")
display(poi_channels.groupBy("container_id", "channel_id", "value_string").count().orderBy("container_id", "channel_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC Register the POI channels in the catalogue tables so `poi_channel(channel_name=...)` can find them:

# COMMAND ----------

POI_META = [(10001, "Event.GearShift", "DOUBLE", "Gear-shift events: the new gear number at each change of Vhcl.GearNo"),
            (10002, "Event.DriverAlert", "STRING", "Simulated telematics event log: HARSH_BRAKE / HARSH_ACCEL / SHARP_TURN alerts")]
poi_meta = spark.createDataFrame(POI_META, "channel_id int, channel_name string, value_type string, description string")

poi_metrics = (poi_channels.groupBy("container_id", "channel_id")
               .agg(F.count("*").alias("sample_count"), F.min("value_double").alias("min"), F.max("value_double").alias("max"),
                    F.avg("value_double").alias("mean"), F.min("timestamp").alias("begin_ts"), F.max("timestamp").alias("end_ts"))
               .join(poi_meta, "channel_id")
               .select("container_id", "channel_id", "channel_name", F.lit("CarMaker").alias("data_key"),
                       F.lit("none").alias("unit"), "value_type", "sample_count",
                       F.col("sample_count").alias("raw_sample_count"), F.lit(1.0).alias("rle_compression_ratio"),
                       F.lit(0.0).alias("nan_ratio"), "min", "max", "mean", F.lit(None).cast("double").alias("std"),
                       *[F.lit(None).cast("double").alias(c) for c in ("pz1", "pz10", "pz90", "pz99")],
                       "begin_ts", "end_ts", ((F.col("end_ts") - F.col("begin_ts")) / 1e6).alias("duration_s"),
                       F.lit(None).cast("double").alias("sample_rate_hz")))
channel_metrics_all = channel_metrics.unionByName(poi_metrics)

poi_tags = unpivot(poi_meta.select("channel_id", "channel_name", F.lit("none").alias("unit"),
                                   F.col("description").alias("signal_description"), F.lit("Event").alias("category"),
                                   F.col("value_type").alias("dtype"), F.lit("CarMaker").alias("data_key"),
                                   F.lit("POI").alias("series_type")),
                   ["channel_id"], ["channel_name", "unit", "signal_description", "category", "dtype", "data_key", "series_type"])
channel_tags = (channel_metrics_all.select("container_id", "channel_id").distinct()
                .join(F.broadcast(signal_tags.unionByName(poi_tags)), "channel_id")
                .select("container_id", "channel_id", "key", "value"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 · `channel_mapping` + `unit_conversion` — logical names and units
# MAGIC
# MAGIC Physical signal names differ between vehicles, loggers and software versions. A **channel alias**
# MAGIC gives analysts one stable name and lets the data team decide what it resolves to:
# MAGIC
# MAGIC * `channel_mapping`: `(channel_alias, source_channel, data_key, priority, source_unit, target_unit)`.
# MAGIC   Per trip, Impulse joins it to `channel_metrics` on `source_channel = channel_name` and `data_key`,
# MAGIC   and keeps the **lowest priority that exists**. `drive_unit_speed` → `PT.Motor.rotv` (1) or
# MAGIC   `Vhcl.Engine.rotv` (2): EVs and hybrids resolve to the e-motor, combustion cars (which have no
# MAGIC   e-motor channel) fall back to the engine **automatically** — one name for "main drive unit rpm".
# MAGIC * `unit_conversion`: `(group_id, unit, conversion_factor, is_base)`. Each factor converts a unit into its
# MAGIC   family's base unit; Impulse converts *source → target* with `factor(source) / factor(target)`
# MAGIC   (m/s → km/h = 1 / 0.2778 = 3.6). Conversion happens only through aliases, using the channel's
# MAGIC   **`channel_metrics.unit`** as the source unit when present.
# MAGIC
# MAGIC ⚠️ Conversions are **multiplicative only**, and they trust the **declared** unit. Kelvin → °C needs an
# MAGIC offset, and lat/long are *declared* `deg` while holding radians — both are handled in TSAL instead.

# COMMAND ----------

channel_mapping = spark.createDataFrame([
    ("vehicle_speed", "Vhcl.v", "CarMaker", 1, "m/s", "km/h", "Vehicle speed in km/h"),
    ("drive_unit_speed", "PT.Motor.rotv", "CarMaker", 1, "rad/s", "rpm", "Speed of the main drive unit in rpm: e-motor if present"),
    ("drive_unit_speed", "Vhcl.Engine.rotv", "CarMaker", 2, "rad/s", "rpm", "Speed of the main drive unit in rpm: engine otherwise"),
    ("motor_speed", "PT.Motor.rotv", "CarMaker", 1, "rad/s", "rpm", "E-motor speed in rpm"),
    ("engine_speed", "Vhcl.Engine.rotv", "CarMaker", 1, "rad/s", "rpm", "Engine speed in rpm"),
    ("motor_power", "PT.Motor.PwrElec", "CarMaker", 1, "W", "kW", "E-motor electrical power in kW"),
    ("battery_energy", "PT.BattHV.Energy", "CarMaker", 1, "J", "kWh", "HV battery energy content in kWh"),
    ("battery_max_power", "PT.BattHV.Pwr_max", "CarMaker", 1, "W", "kW", "HV battery max power in kW"),
    ("odometer", "Vhcl.Distance", "CarMaker", 1, "m", "km", "Distance driven in km"),
    ("steering_angle", "Vhcl.Steer.Ang", "CarMaker", 1, "rad", "deg", "Steering angle in degrees"),
    ("ambient_pressure", "Env.AirPressure", "CarMaker", 1, "Pa", "kPa", "Ambient air pressure in kPa"),
], "channel_alias string, source_channel string, data_key string, priority int, source_unit string, target_unit string, description string")

PI = 3.141592653589793
unit_conversion = spark.createDataFrame([
    ("speed", "m/s", 1.0, True), ("speed", "km/h", 1 / 3.6, False), ("speed", "mph", 0.44704, False),
    ("angular_velocity", "rad/s", 1.0, True), ("angular_velocity", "rpm", 2 * PI / 60, False),
    ("energy", "J", 1.0, True), ("energy", "Wh", 3600.0, False), ("energy", "kWh", 3.6e6, False),
    ("power", "W", 1.0, True), ("power", "kW", 1000.0, False),
    ("length", "m", 1.0, True), ("length", "km", 1000.0, False), ("length", "mi", 1609.344, False),
    ("pressure", "Pa", 1.0, True), ("pressure", "kPa", 1000.0, False), ("pressure", "bar", 1e5, False),
    ("angle", "rad", 1.0, True), ("angle", "deg", PI / 180, False),
], "group_id string, unit string, conversion_factor double, is_base boolean")
display(channel_mapping)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 · Write the silver tables
# MAGIC
# MAGIC Plain managed Delta tables. **Liquid clustering** on `(container_id, channel_id)` matters most for
# MAGIC `channels` — by far the biggest table — because Impulse always reads it for a set of
# MAGIC (container, channel) pairs, and clustering lets Delta skip every other file. Table comments help
# MAGIC humans *and* Genie understand the model.

# COMMAND ----------

def save(df, table, cluster_by=None, comment=""):
    writer = df.write.mode("overwrite").option("overwriteSchema", "true")
    if cluster_by:
        writer = writer.clusterBy(*cluster_by)
    writer.saveAsTable(MY_SILVER[table])
    spark.sql(f"COMMENT ON TABLE {MY_SILVER[table]} IS '{comment}'")
    print(f"✅ {MY_SILVER[table]:65s} {spark.table(MY_SILVER[table]).count():>12,} rows")

save(channels, "channels", ["container_id", "channel_id"], "Impulse silver: RLE samples [tstart, tend) epoch us")
save(poi_channels, "poi_channels", ["container_id", "channel_id"], "Impulse silver: Points-in-Time events")
save(channel_metrics_all, "channel_metrics", ["container_id", "channel_id"], "Impulse silver: one row per (trip, channel)")
save(channel_tags, "channel_tags", ["container_id", "channel_id"], "Impulse silver: EAV channel tags")
save(container_metrics, "container_metrics", ["container_id"], "Impulse silver: one row per trip")
save(container_tags, "container_tags", ["container_id"], "Impulse silver: EAV container tags")
save(channel_mapping, "channel_mapping", comment="Impulse silver: logical channel aliases")
save(unit_conversion, "unit_conversion", comment="Impulse silver: unit families and factors")
dq_findings.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{USER_SCHEMA}.{W2_SILVER_PREFIX}_dq_findings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 · Validate — does Impulse understand *your* silver layer?
# MAGIC
# MAGIC The real test of a silver layer is whether Impulse can query it. Point a `MeasurementDB` at your
# MAGIC tables and solve one expression of each kind — including the **alias with fallback and unit
# MAGIC conversion** and a **POI** channel.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver

my_db = MeasurementDB(MeasurementDBConfig(**impulse_source(MY_SILVER), table_locations="unity_catalog"), WorkspaceClient())
q = my_db.query
raw_speed = q.channel(channel_name="Vhcl.v")                         # direct: m/s as logged
alias_speed = q.channel_with_alias(channel_alias="vehicle_speed")    # alias: converted to km/h
clean_speed = alias_speed.where(alias_speed < 360)                   # quirk handled in TSAL (100 m/s)
drive_rpm = q.channel_with_alias(channel_alias="drive_unit_speed")   # e-motor, or engine as fallback
alerts = q.poi_channel(channel_name="Event.DriverAlert", dtype="string")

check = (q.select(raw_speed.max().alias("vhcl_v_max_mps"),
                  alias_speed.max().alias("vehicle_speed_max_kmh"),
                  clean_speed.max().alias("clean_speed_max_kmh"),
                  drive_rpm.max().alias("drive_unit_max_rpm"),
                  alerts.alias("driver_alerts"))
         .solve(spark, solver=DefaultSolver(spark)))                  # a Spark DataFrame, one row per trip
display(check
        .withColumn("n_alerts", F.coalesce(F.size("driver_alerts"), F.lit(0)))   # POI result = array of events
        .drop("driver_alerts")
        .join(container_metrics.select("container_id", "vehicle_key", "powertrain_type"), "container_id")
        .orderBy("container_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC 👀 **Read the result:**
# MAGIC * `vhcl_v_max_mps × 3.6 = vehicle_speed_max_kmh` — the alias converted units.
# MAGIC * Trip 95's raw maximum is absurd; `clean_speed_max_kmh` shows the TSAL handling at work.
# MAGIC * `drive_unit_max_rpm` exists for **every** trip: the e-motor for EVs, the engine (priority 2 fallback)
# MAGIC   for the combustion cars 94 and 95. Analysts just ask for `drive_unit_speed`.
# MAGIC
# MAGIC Finally, compare with the instructor's shared silver for the same trips — the row counts must match.

# COMMAND ----------

compare = []
for t in ["channels", "channel_metrics", "channel_tags", "container_metrics", "container_tags", "poi_channels"]:
    mine = spark.table(MY_SILVER[t]).count()
    shared = spark.table(SILVER[t]).where(F.col("container_id").isin(TRIPS)).count()
    compare.append((t, mine, shared, "✅" if mine == shared else "❌"))
display(spark.createDataFrame(compare, "table string, yours long, shared long, match string"))
assert all(c[3] == "✅" for c in compare), "your silver differs from the shared silver"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Recap
# MAGIC
# MAGIC | Decision | What we chose | Why |
# MAGIC |---|---|---|
# MAGIC | Keys | trip → `container_id`, signal → `channel_id` | Impulse's two identities |
# MAGIC | Time | epoch **µs**, bigint, UTC | exact, compact; all TSAL durations in µs |
# MAGIC | Encoding | **RLE** `[tstart, tend)` at ingestion | merge equal runs once instead of per query |
# MAGIC | Data quality | flag findings, document quirks, handle in TSAL | silver stays a faithful record |
# MAGIC | Metadata | stats in `*_metrics`, everything descriptive in EAV `*_tags` | fast channel lookup; select by anything |
# MAGIC | Events | separate `poi_channels` | instants are not intervals |
# MAGIC | Semantics | aliases + unit families | stable names, fallbacks, units handled once |
# MAGIC
# MAGIC **Next:** `declarative_pipeline/` (the same as a production pipeline + job) or straight to
# MAGIC **`02a_tsal_deep_dive`**.
