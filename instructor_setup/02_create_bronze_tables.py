# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Bronze tables
# MAGIC
# MAGIC **Instructor only — run after `01_download_dataset`.**
# MAGIC
# MAGIC Bronze is a **faithful copy of the source** — same star schema, same column names — plus
# MAGIC ingestion metadata (`_source_file`, `_ingested_at`). No business logic happens here; that is
# MAGIC silver's job. Workshop 2 / notebook 1 starts from exactly these tables.
# MAGIC
# MAGIC | Bronze table (in `dbx_workshop_dev.shared`) | Source | Grain |
# MAGIC |---|---|---|
# MAGIC | `impulse_bronze_signal_data`   | `fact_signal_data`  | one row per signal observation |
# MAGIC | `impulse_bronze_signal_info`   | `dim_signal_info`   | one row per signal definition (681) |
# MAGIC | `impulse_bronze_trip_info`     | `dim_trip_info`     | one row per trip / session (100) |
# MAGIC | `impulse_bronze_vehicle_info`  | `dim_vehicle_info`  | one row per vehicle model (13) |
# MAGIC | `impulse_bronze_vehicle_state` | `dim_vehicle_state` | one row per passenger/tire/driver config (49) |
# MAGIC
# MAGIC ℹ️ The fact table keeps the curated `BRONZE_SIGNALS` from CONFIG (40 of 681 signals — everything
# MAGIC the labs use). That keeps bronze at a few hundred million rows instead of 5.1 billion. Set
# MAGIC `BRONZE_SIGNALS = ["ALL"]` in CONFIG to load everything.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T

spark.conf.set("spark.sql.session.timeZone", "UTC")

# The 11 columns present in every fact file. Some files additionally carry two internal DuckLake
# columns; an explicit schema drops them and keeps every read consistent.
FACT_SCHEMA = T.StructType([
    T.StructField("timestamp", T.TimestampNTZType()),
    T.StructField("signal_id", T.IntegerType()),
    T.StructField("session_id", T.IntegerType()),
    T.StructField("vehicle_id", T.IntegerType()),
    T.StructField("vehicle_state_id", T.IntegerType()),
    T.StructField("float_value", T.DoubleType()),
    T.StructField("int_value", T.LongType()),
    T.StructField("string_value", T.StringType()),
    T.StructField("trip_hash", T.StringType()),
    T.StructField("partition", T.IntegerType()),
    T.StructField("vehicle_name", T.StringType()),
])


def with_ingest_metadata(df):
    return (df.withColumn("_source_file", F.col("_metadata.file_path"))
              .withColumn("_ingested_at", F.current_timestamp()))

# COMMAND ----------

# DBTITLE 1,Dimension tables (tiny — straight copies)
DIMS = {
    "signal_info": ("dim_signal_info", "Signal definitions: name, unit, category, value type (681 rows)"),
    "trip_info": ("dim_trip_info", "Simulated trips / sessions and scenario parameters (100 rows)"),
    "vehicle_info": ("dim_vehicle_info", "Simulated vehicle models and powertrain parameters (13 rows)"),
    "vehicle_state": ("dim_vehicle_state", "Passenger, tire and driving-style configurations (49 rows)"),
}
for key, (folder, comment) in DIMS.items():
    df = with_ingest_metadata(spark.read.parquet(f"{RAW_DATASET_PATH}/{folder}"))
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BRONZE[key])
    spark.sql(f"COMMENT ON TABLE {BRONZE[key]} IS 'Bronze copy of {folder} (renumics/synthetic-vehicle-fleet-telemetry). {comment}'")
    print(f"{BRONZE[key]}: {spark.table(BRONZE[key]).count()} rows")

# COMMAND ----------

# DBTITLE 1,Signal data (the big one)
signal_info = spark.table(BRONZE["signal_info"])
fact_root = f"{RAW_DATASET_PATH}/fact_signal_data"
if BRONZE_SIGNALS == ["ALL"]:
    keep_ids, paths = None, [fact_root]
else:
    found = signal_info.where(F.col("signal_name").isin(BRONZE_SIGNALS))
    keep_ids = [r.signal_id for r in found.select("signal_id").collect()]
    missing = set(BRONZE_SIGNALS) - {r.signal_name for r in found.select("signal_name").collect()}
    assert not missing, f"signals not in dim_signal_info: {missing}"
    # Only read the signal partitions that hold the wanted signals (partition pruning on folders).
    partitions = sorted({r.partition for r in found.select("partition").collect()})
    paths = [f"{fact_root}/trip_hash=*/partition={p}" for p in partitions]
    print(f"keeping {len(keep_ids)} signals; reading storage partitions {partitions} of 0-9")

fact = (
    spark.read.schema(FACT_SCHEMA)
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.parquet")
    .parquet(*paths)
)
if keep_ids is not None:
    fact = fact.where(F.col("signal_id").isin(keep_ids))

(
    with_ingest_metadata(fact)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .clusterBy("session_id", "signal_id")
    .saveAsTable(BRONZE["signal_data"])
)
spark.sql(f"COMMENT ON TABLE {BRONZE['signal_data']} IS 'Bronze copy of fact_signal_data "
          "(renumics/synthetic-vehicle-fleet-telemetry): one row per signal observation, curated workshop signals.'")

# COMMAND ----------

# DBTITLE 1,Check what landed
summary = spark.sql(f"""
  SELECT count(*)                    AS observations,
         count(DISTINCT session_id)  AS trips,
         count(DISTINCT signal_id)   AS signals,
         min(timestamp)              AS first_ts,
         max(timestamp)              AS last_ts
  FROM {BRONZE['signal_data']}
""")
display(summary)
row = summary.first()

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"observations": row.observations, "trips": row.trips, "signals": row.signals}))
