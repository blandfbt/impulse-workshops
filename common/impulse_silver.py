"""Bronze (Renumics star schema) -> Impulse silver layer transformations.

Pure ``DataFrame -> DataFrame`` functions shared by
  * ``instructor_setup/03_create_silver_tables``   (builds the shared silver layer), and
  * ``workshop_2_deep_dive/declarative_pipeline``  (the same logic as a declarative pipeline).

Workshop 2 / notebook 1 walks through exactly this logic cell by cell, so read that notebook for
the *why*; this module is the compact *how*.

Mapping from the source dataset to the Impulse silver model
-----------------------------------------------------------
* container  = one trip            -> ``container_id = session_id``
* channel    = one signal in a trip -> ``channel_id = signal_id``
* time       = epoch **microseconds** of the (time-normalised, timezone-less) source timestamp
* value      = ``coalesce(float_value, int_value)`` as double

All functions expect ``spark.sql.session.timeZone = UTC`` (see :func:`use_utc`), so TIMESTAMP_NTZ
values convert to epoch microseconds without a timezone shift.
"""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
DATA_KEY = "CarMaker"  # the "data source" key; joins channel_mapping <-> channel_metrics

# Data-quality checks. Silver stores the measured data unchanged -- these rules only *flag*
# (trip, signal) pairs in the dq_findings table; analyses handle the quirk in the TSAL signal
# definition (see DATA_NOTES.md), e.g. `vhcl_v.where(vhcl_v < 100)` or `soc.where(soc > 0)`.
DQ_RULES = [
    # signal, plausible_min, plausible_max, finding, handling in TSAL
    ("Vhcl.v", 0.0, 100.0, "implausible speed spikes (up to ~2,100 m/s)",
     "drop values > 100 m/s: v.where(v < 100)"),
    ("PT.BCU.BattHV.SOC", 1e-9, 100.0, "SOC = 0 while the BMS initialises",
     "ignore SOC <= 0: soc.where(soc > 0)"),
]

# Points-in-Time (POI) channels derived during ingestion. They get channel ids above the source
# signal id range (0..680) so they can never collide.
POI_GEAR_SHIFT_ID = 10001        # numeric POI: the new gear at every gear change
POI_DRIVER_ALERT_ID = 10002      # string POI: a simulated telematics event log
DRIVER_ALERT_RULES = [            # (code, signal, comparator, threshold)
    ("HARSH_BRAKE", "Car.ax", "<", -4.0),   # m/s^2
    ("HARSH_ACCEL", "Car.ax", ">", 3.0),    # m/s^2
    ("SHARP_TURN", "Car.ay", "abs>", 5.0),  # m/s^2
]
DRIVER_ALERT_DEBOUNCE_US = 3_000_000  # one alert of the same code per 3 s at most

POI_CHANNELS = [
    # channel_id, channel_name, value_type, unit, description
    (POI_GEAR_SHIFT_ID, "Event.GearShift", "DOUBLE", "none",
     "Gear-shift events: the new gear number at each change of Vhcl.GearNo"),
    (POI_DRIVER_ALERT_ID, "Event.DriverAlert", "STRING", "none",
     "Simulated telematics event log: HARSH_BRAKE / HARSH_ACCEL / SHARP_TURN alerts"),
]

# Unit families for the optional unit_conversion table. Each factor converts a value in `unit`
# into the family's base unit; Impulse converts S -> T with factor(S) / factor(T).
# Only multiplicative conversions are supported (K -> degC needs an offset: do it in TSAL).
UNIT_CONVERSIONS = [
    # group_id, unit, conversion_factor, is_base
    ("speed", "m/s", 1.0, True),
    ("speed", "km/h", 1 / 3.6, False),
    ("speed", "mph", 0.44704, False),
    ("angular_velocity", "rad/s", 1.0, True),
    ("angular_velocity", "rpm", 2 * 3.141592653589793 / 60, False),
    ("energy", "J", 1.0, True),
    ("energy", "Wh", 3600.0, False),
    ("energy", "kWh", 3.6e6, False),
    ("power", "W", 1.0, True),
    ("power", "kW", 1000.0, False),
    ("length", "m", 1.0, True),
    ("length", "km", 1000.0, False),
    ("length", "mi", 1609.344, False),
    ("pressure", "Pa", 1.0, True),
    ("pressure", "kPa", 1000.0, False),
    ("pressure", "bar", 1e5, False),
    ("angle", "rad", 1.0, True),
    ("angle", "deg", 3.141592653589793 / 180, False),
]

# Optional channel_mapping table: stable, *logical* channel names (aliases) resolved to physical
# signals. Lower priority wins per trip; if the preferred signal is missing (e.g. a combustion car
# has no e-motor) Impulse falls back to the next one. source_unit/target_unit drive unit conversion.
CHANNEL_MAPPING = [
    # channel_alias, source_channel, priority, source_unit, target_unit, description
    ("vehicle_speed", "Vhcl.v", 1, "m/s", "km/h", "Vehicle speed in km/h"),
    ("drive_unit_speed", "PT.Motor.rotv", 1, "rad/s", "rpm", "Speed of the main drive unit in rpm: e-motor if present"),
    ("drive_unit_speed", "Vhcl.Engine.rotv", 2, "rad/s", "rpm", "Speed of the main drive unit in rpm: engine otherwise"),
    ("motor_speed", "PT.Motor.rotv", 1, "rad/s", "rpm", "E-motor speed in rpm"),
    ("engine_speed", "Vhcl.Engine.rotv", 1, "rad/s", "rpm", "Engine speed in rpm"),
    ("motor_power", "PT.Motor.PwrElec", 1, "W", "kW", "E-motor electrical power in kW"),
    ("battery_energy", "PT.BattHV.Energy", 1, "J", "kWh", "HV battery energy content in kWh"),
    ("battery_max_power", "PT.BattHV.Pwr_max", 1, "W", "kW", "HV battery max power in kW"),
    ("odometer", "Vhcl.Distance", 1, "m", "km", "Distance driven in km"),
    ("steering_angle", "Vhcl.Steer.Ang", 1, "rad", "deg", "Steering angle in degrees"),
    ("ambient_pressure", "Env.AirPressure", 1, "Pa", "kPa", "Ambient air pressure in kPa"),
]

CHANNEL_TAG_COLUMNS = {  # channel_tags key -> bronze signal_info column
    "channel_name": "signal_name",
    "signal_description": "signal_description",
    "unit": "unit",
    "physical_quantity": "physical_quantity",
    "category": "category",
    "subsystem": "subsystem",
    "component": "component",
    "group": "group",
    "is_categorical": "is_categorical",
    "dtype": "dtype",
    "signal_mapping": "signal_mapping",
}

CONTAINER_TAG_COLUMNS = (
    "vehicle_key", "manufacturer", "vehicle_type", "powertrain_type", "scenario", "driver",
    "tires", "number_of_passengers", "traffic", "soc_target_pct", "charge_power_kw",
    "route_length", "starting_temperature_c", "trip_name",
)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def use_utc(spark) -> None:
    """Pin the session time zone so TIMESTAMP_NTZ -> epoch conversions are shift-free."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")


def ntz_to_epoch_us(col):
    """Epoch microseconds of a TIMESTAMP_NTZ, read as UTC wall-clock time."""
    return F.unix_micros(col.cast("timestamp")).cast("bigint")


def _is_finite(col):
    return col.isNotNull() & ~F.isnan(col) & (F.abs(col) != F.lit(float("inf")))


def _unpivot_tags(df: DataFrame, id_cols: list[str], value_cols: list[str]) -> DataFrame:
    """Wide columns -> strict EAV ``(id_cols..., key, value)`` rows, dropping empty values."""
    as_strings = df.select(*id_cols, *[F.col(f"`{c}`").cast("string").alias(c) for c in value_cols])
    return as_strings.unpivot(id_cols, value_cols, "key", "value").where(
        F.col("value").isNotNull() & (F.trim("value") != "")
    )


# ---------------------------------------------------------------------------------------------
# 1. Samples and data-quality findings
# ---------------------------------------------------------------------------------------------
def samples(signal_data: DataFrame) -> DataFrame:
    """Bronze fact rows -> RAW samples ``(container_id, channel_id, timestamp[us], value)``."""
    return signal_data.select(
        F.col("session_id").cast("bigint").alias("container_id"),
        F.col("signal_id").cast("int").alias("channel_id"),
        ntz_to_epoch_us(F.col("timestamp")).alias("timestamp"),
        F.coalesce("float_value", F.col("int_value").cast("double")).cast("double").alias("value"),
    )


def dq_findings(samples_df: DataFrame, signal_info: DataFrame) -> DataFrame:
    """(trip, signal) pairs with samples outside their plausible range -- flagged, never dropped."""
    ranges = signal_info.sparkSession.createDataFrame(
        [(n, float(lo), float(hi), f, h) for n, lo, hi, f, h in DQ_RULES],
        "signal_name string, plausible_min double, plausible_max double, finding string, handling string")
    rules = signal_info.join(ranges, "signal_name").select(
        F.col("signal_id").cast("int").alias("channel_id"), F.col("signal_name").alias("channel_name"),
        "unit", "plausible_min", "plausible_max", "finding", "handling")
    out_of_range = (F.col("value") < F.col("plausible_min")) | (F.col("value") > F.col("plausible_max"))
    return (
        samples_df.join(F.broadcast(rules), "channel_id")
        .groupBy("container_id", "channel_id", "channel_name", "unit", "plausible_min", "plausible_max",
                 "finding", "handling")
        .agg(F.count("*").alias("sample_count"), F.sum(out_of_range.cast("int")).alias("implausible_count"),
             F.min("value").alias("observed_min"), F.max("value").alias("observed_max"))
        .where(F.col("implausible_count") > 0)
        .withColumn("implausible_pct", F.round(F.col("implausible_count") / F.col("sample_count") * 100, 3))
        .withColumn("checked_at", F.current_timestamp())
    )


# ---------------------------------------------------------------------------------------------
# 2. channels (RLE)
# ---------------------------------------------------------------------------------------------
def rle_encode(samples_df: DataFrame) -> DataFrame:
    """Run-length encode samples into ``[tstart, tend)`` intervals per (container, channel).

    Consecutive samples with the same value collapse into one interval. Each interval ends at
    the first sample of the next run; the last interval of a channel ends at its last sample.
    """
    w = Window.partitionBy("container_id", "channel_id").orderBy("timestamp")
    new_run = F.col("value").eqNullSafe(F.lag("value").over(w)) == F.lit(False)
    is_first = F.lag("timestamp").over(w).isNull()
    return (
        samples_df
        .withColumn("_next_ts", F.lead("timestamp").over(w))
        .withColumn("_new_run", F.when(is_first | new_run, 1).otherwise(0))
        .withColumn("_run_id", F.sum("_new_run").over(w.rowsBetween(Window.unboundedPreceding, 0)))
        .groupBy("container_id", "channel_id", "_run_id")
        .agg(
            F.min("timestamp").alias("tstart"),
            F.max(F.coalesce("_next_ts", "timestamp")).alias("tend"),
            F.first("value").alias("value"),
        )
        .select("container_id", "channel_id", "tstart", "tend", "value")
    )


# ---------------------------------------------------------------------------------------------
# 3. channel_metrics + channel_tags
# ---------------------------------------------------------------------------------------------
def channel_metrics(channels_df: DataFrame, samples_df: DataFrame, signal_info: DataFrame) -> DataFrame:
    """One row per (container, channel): duration-weighted stats + selection columns."""
    weight = (F.col("tend") - F.col("tstart")).cast("double")
    valid = _is_finite(F.col("value"))
    w_sum = F.sum(F.when(valid, weight))
    stats = (
        channels_df.groupBy("container_id", "channel_id")
        .agg(
            F.count("*").alias("sample_count"),  # RLE intervals
            F.avg(F.when(valid, 0.0).otherwise(1.0)).alias("nan_ratio"),
            F.min(F.when(valid, F.col("value"))).alias("min"),
            F.max(F.when(valid, F.col("value"))).alias("max"),
            F.coalesce(F.try_divide(F.sum(F.when(valid, F.col("value") * weight)), F.when(w_sum > 0, w_sum)),
                       F.avg(F.when(valid, F.col("value")))).alias("mean"),
            F.stddev(F.when(valid, F.col("value"))).alias("std"),
            F.percentile_approx(F.when(valid, F.col("value")), [0.01, 0.10, 0.90, 0.99], 1000).alias("_pz"),
            F.min("tstart").alias("begin_ts"),
            F.max("tend").alias("end_ts"),
        )
    )
    raw_counts = samples_df.groupBy("container_id", "channel_id").agg(F.count("*").alias("raw_sample_count"))
    names = signal_info.select(F.col("signal_id").cast("int").alias("channel_id"),
                               F.col("signal_name").alias("channel_name"), "unit")
    return (
        stats.join(raw_counts, ["container_id", "channel_id"], "left")
        .join(F.broadcast(names), "channel_id", "left")
        .select(
            F.col("container_id").cast("bigint"), F.col("channel_id").cast("int"),
            "channel_name", F.lit(DATA_KEY).alias("data_key"), "unit", F.lit("DOUBLE").alias("value_type"),
            F.col("sample_count").cast("bigint"), F.col("raw_sample_count").cast("bigint"),
            F.round(F.col("raw_sample_count") / F.col("sample_count"), 2).alias("rle_compression_ratio"),
            "nan_ratio", "min", "max", "mean", "std",
            F.col("_pz")[0].alias("pz1"), F.col("_pz")[1].alias("pz10"),
            F.col("_pz")[2].alias("pz90"), F.col("_pz")[3].alias("pz99"),
            F.col("begin_ts").cast("bigint"), F.col("end_ts").cast("bigint"),
            ((F.col("end_ts") - F.col("begin_ts")) / 1e6).alias("duration_s"),
            F.try_divide(F.col("raw_sample_count"), (F.col("end_ts") - F.col("begin_ts")) / 1e6).alias("sample_rate_hz"),
        )
    )


def poi_channel_metrics(poi_df: DataFrame) -> DataFrame:
    """channel_metrics rows for the derived POI channels (so they are selectable like any channel)."""
    meta = poi_df.sparkSession.createDataFrame(
        [(cid, name, vt, unit) for cid, name, vt, unit, _ in POI_CHANNELS],
        "channel_id int, channel_name string, value_type string, unit string")
    return (
        poi_df.groupBy("container_id", "channel_id")
        .agg(F.count("*").alias("sample_count"), F.min("value_double").alias("min"),
             F.max("value_double").alias("max"), F.avg("value_double").alias("mean"),
             F.min("timestamp").alias("begin_ts"), F.max("timestamp").alias("end_ts"))
        .join(F.broadcast(meta), "channel_id")
        .select(
            F.col("container_id").cast("bigint"), F.col("channel_id").cast("int"), "channel_name",
            F.lit(DATA_KEY).alias("data_key"), "unit", "value_type",
            F.col("sample_count").cast("bigint"), F.col("sample_count").cast("bigint").alias("raw_sample_count"),
            F.lit(1.0).alias("rle_compression_ratio"), F.lit(0.0).alias("nan_ratio"),
            "min", "max", "mean", F.lit(None).cast("double").alias("std"),
            *[F.lit(None).cast("double").alias(c) for c in ("pz1", "pz10", "pz90", "pz99")],
            F.col("begin_ts").cast("bigint"), F.col("end_ts").cast("bigint"),
            ((F.col("end_ts") - F.col("begin_ts")) / 1e6).alias("duration_s"),
            F.lit(None).cast("double").alias("sample_rate_hz"),
        )
    )


def channel_tags(channel_metrics_df: DataFrame, signal_info: DataFrame) -> DataFrame:
    """Strict EAV ``(container_id, channel_id, key, value)`` tags for every silver channel."""
    signal_tags = _unpivot_tags(
        signal_info.select(F.col("signal_id").cast("int").alias("channel_id"),
                           *[F.col(f"`{src}`").alias(key) for key, src in CHANNEL_TAG_COLUMNS.items()]),
        ["channel_id"], list(CHANNEL_TAG_COLUMNS),
    ).withColumn("_series_type", F.lit("SAMPLE"))
    poi_tags = channel_metrics_df.sparkSession.createDataFrame(
        [(cid, k, v) for cid, name, vt, unit, desc in POI_CHANNELS
         for k, v in [("channel_name", name), ("unit", unit), ("signal_description", desc),
                      ("category", "Event"), ("dtype", vt)]],
        "channel_id int, key string, value string").withColumn("_series_type", F.lit("POI"))
    tags = signal_tags.unionByName(poi_tags)
    extra = tags.select("channel_id", "_series_type").distinct().select(
        "channel_id", F.lit("data_key").alias("key"), F.lit(DATA_KEY).alias("value"), "_series_type")
    series_type = tags.select("channel_id", "_series_type").distinct().select(
        "channel_id", F.lit("series_type").alias("key"), F.col("_series_type").alias("value"), "_series_type")
    all_tags = tags.unionByName(extra).unionByName(series_type).drop("_series_type")
    return (
        channel_metrics_df.select("container_id", "channel_id").distinct()
        .join(F.broadcast(all_tags), "channel_id")
        .select(F.col("container_id").cast("bigint"), F.col("channel_id").cast("int"), "key", "value")
    )


# ---------------------------------------------------------------------------------------------
# 4. container_metrics + container_tags
# ---------------------------------------------------------------------------------------------
def trip_keys(signal_data: DataFrame) -> DataFrame:
    """session -> vehicle / vehicle-state / trip-hash keys (one row per trip)."""
    return signal_data.groupBy("session_id").agg(
        F.first("vehicle_id").alias("vehicle_id"),
        F.first("vehicle_state_id").alias("vehicle_state_id"),
        F.first("trip_hash").alias("trip_hash"),
    )


def container_metrics(channel_metrics_df: DataFrame, keys: DataFrame, trip_info: DataFrame,
                      vehicle_info: DataFrame, vehicle_state: DataFrame) -> DataFrame:
    """One row per trip: time bounds + vehicle / configuration / scenario metadata."""
    bounds = channel_metrics_df.where(F.col("value_type") == "DOUBLE").groupBy("container_id").agg(
        F.countDistinct("channel_id").cast("int").alias("num_channels"),
        F.min("begin_ts").alias("start_ts"), F.max("end_ts").alias("stop_ts"))
    name = F.col("trip_name")
    trips = trip_info.select(
        F.col("session_id").cast("bigint").alias("container_id"),
        "trip_name",
        F.regexp_extract(name, r"^(.+?)_scenario_", 1).alias("scenario"),
        F.col("starting_temperature").cast("int").alias("starting_temperature_c"),
        F.col("trip_start_timestamp").cast("timestamp").alias("trip_start_dt"),
        F.when(name.rlike(r"_pax_"), 1).when(name.rlike(r"_nopax_"), 0).alias("pax_flag"),
        F.nullif(F.regexp_extract(name, r"_traf(\d+)_", 1), F.lit("")).cast("int").alias("traffic"),
        F.nullif(F.regexp_extract(name, r"_soc(\d+)_", 1), F.lit("")).cast("int").alias("soc_target_pct"),
        F.nullif(F.regexp_extract(name, r"_(\d+)kw_", 1), F.lit("")).cast("int").alias("charge_power_kw"),
        F.nullif(F.regexp_extract(name, r"_len(\d+)_", 1), F.lit("")).cast("int").alias("route_length"),
    )
    vehicles = vehicle_info.select(
        F.col("vehicle_id").cast("int"), F.col("vehicle_name").alias("vehicle_key"), "manufacturer",
        "vehicle_type", "powertrain_type", F.col("battery_voltage_oc").cast("double"),
        F.col("empty_weight").cast("double").alias("empty_weight_kg"))
    states = vehicle_state.select(F.col("vehicle_state_id").cast("int"),
                                  F.col("number_of_passengers").cast("int"), "tires", "driver")
    return (
        bounds
        .join(keys.select(F.col("session_id").cast("bigint").alias("container_id"), "vehicle_id",
                          "vehicle_state_id", "trip_hash"), "container_id")
        .join(F.broadcast(trips), "container_id", "left")
        .join(F.broadcast(vehicles), "vehicle_id", "left")
        .join(F.broadcast(states), "vehicle_state_id", "left")
        .select(
            F.col("container_id").cast("bigint"),
            F.col("start_ts").cast("bigint"), F.col("stop_ts").cast("bigint"),
            F.timestamp_micros("start_ts").alias("start_dt"), F.timestamp_micros("stop_ts").alias("stop_dt"),
            ((F.col("stop_ts") - F.col("start_ts")) / 1e6).alias("duration_s"),
            "num_channels", "vehicle_key", "manufacturer", "vehicle_type", "powertrain_type",
            "battery_voltage_oc", "empty_weight_kg", "scenario", "trip_name", "driver", "tires",
            "number_of_passengers", "traffic", "soc_target_pct", "charge_power_kw", "route_length",
            "starting_temperature_c", "trip_start_dt",
            F.col("container_id").cast("int").alias("session_id"), "vehicle_id", "vehicle_state_id",
            "trip_hash", F.current_timestamp().alias("last_modified"),
        )
    )


def container_tags(container_metrics_df: DataFrame) -> DataFrame:
    """Strict EAV container tags from the descriptive container_metrics columns."""
    return _unpivot_tags(container_metrics_df, ["container_id"], list(CONTAINER_TAG_COLUMNS))


# ---------------------------------------------------------------------------------------------
# 5. poi_channels (Points-in-Time events)
# ---------------------------------------------------------------------------------------------
def poi_channels(samples_df: DataFrame, signal_info: DataFrame) -> DataFrame:
    """Derive discrete events: gear shifts (numeric) and driver alerts (string)."""
    names = F.broadcast(signal_info.where(F.col("signal_name").isin("Vhcl.GearNo", "Car.ax", "Car.ay"))
                        .select(F.col("signal_id").cast("int").alias("channel_id"), "signal_name"))
    named = samples_df.join(names, "channel_id")  # only the three source signals, tagged by name
    w = Window.partitionBy("container_id", "channel_id").orderBy("timestamp")

    # Gear shifts: every change of Vhcl.GearNo, valued with the new gear.
    gear = (
        named.where(F.col("signal_name") == "Vhcl.GearNo")
        .withColumn("_prev", F.lag("value").over(w))
        .where(F.col("_prev").isNotNull() & (F.col("value") != F.col("_prev")))
        .select("container_id", F.lit(POI_GEAR_SHIFT_ID).alias("channel_id"), "timestamp",
                F.col("value").alias("value_double"), F.lit(None).cast("string").alias("value_string"))
    )

    # Driver alerts: the rising edge of each rule, debounced per code.
    alerts = []
    for code, signal, op, thr in DRIVER_ALERT_RULES:
        v = F.col("value")
        cond = {"<": v < thr, ">": v > thr, "abs>": F.abs(v) > thr}[op]
        alerts.append(
            named.where(F.col("signal_name") == signal)
            .withColumn("_on", cond)
            .withColumn("_prev_on", F.lag("_on").over(w))
            .where(F.col("_on") & ~F.coalesce(F.col("_prev_on"), F.lit(False)))
            .select("container_id", "timestamp", F.lit(code).alias("code"))
        )
    edges = alerts[0]
    for a in alerts[1:]:
        edges = edges.unionByName(a)
    wc = Window.partitionBy("container_id", "code").orderBy("timestamp")
    debounced = (
        edges.withColumn("_prev_ts", F.lag("timestamp").over(wc))
        .where(F.col("_prev_ts").isNull() | (F.col("timestamp") - F.col("_prev_ts") > DRIVER_ALERT_DEBOUNCE_US))
        .select("container_id", F.lit(POI_DRIVER_ALERT_ID).alias("channel_id"), "timestamp",
                F.lit(None).cast("double").alias("value_double"), F.col("code").alias("value_string"))
    )
    return gear.unionByName(debounced).select(
        F.col("container_id").cast("bigint"), F.col("channel_id").cast("int"),
        F.col("timestamp").cast("bigint"), "value_double", "value_string")


# ---------------------------------------------------------------------------------------------
# 6. Reference tables: channel_mapping + unit_conversion
# ---------------------------------------------------------------------------------------------
def channel_mapping(spark) -> DataFrame:
    return spark.createDataFrame(
        [(a, s, DATA_KEY, p, su, tu, d) for a, s, p, su, tu, d in CHANNEL_MAPPING],
        "channel_alias string, source_channel string, data_key string, priority int, "
        "source_unit string, target_unit string, description string")


def unit_conversion(spark) -> DataFrame:
    return spark.createDataFrame(
        UNIT_CONVERSIONS, "group_id string, unit string, conversion_factor double, is_base boolean")
