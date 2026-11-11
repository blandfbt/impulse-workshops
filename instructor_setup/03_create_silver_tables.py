# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Silver tables — the Impulse data model
# MAGIC
# MAGIC **Instructor only — run after `02_create_bronze_tables`.**
# MAGIC
# MAGIC Converts the bronze star schema into the **Impulse silver model** in `dbx_workshop_dev.shared`.
# MAGIC Workshop 1 reads these tables directly, so participants never have to think about this step.
# MAGIC (Workshop 2 / notebook 1 rebuilds the same thing by hand for a few trips, cell by cell.)
# MAGIC
# MAGIC | Silver table | Required by Impulse? | What it holds |
# MAGIC |---|---|---|
# MAGIC | `impulse_silver_container_metrics` | **required** | one row per trip (= *container*): time bounds, vehicle, scenario |
# MAGIC | `impulse_silver_channel_metrics`   | **required** | one row per (trip, signal) (= *channel*): stats + name / unit / data_key |
# MAGIC | `impulse_silver_channels`          | **required** | the samples, **run-length encoded** `[tstart, tend) → value` (epoch µs) |
# MAGIC | `impulse_silver_container_tags`    | optional | EAV trip tags → container filtering (`havingTag`, `tag_filters`) |
# MAGIC | `impulse_silver_channel_tags`      | optional | EAV signal tags → channel selection by any metadata |
# MAGIC | `impulse_silver_poi_channels`      | optional | Points-in-Time events: gear shifts, driver alerts |
# MAGIC | `impulse_silver_channel_mapping`   | optional | logical aliases (`vehicle_speed` → `Vhcl.v` / `Car.Gen.vx_1`) |
# MAGIC | `impulse_silver_unit_conversion`   | optional | unit families + factors (m/s → km/h, rad/s → rpm, J → kWh, …) |
# MAGIC | `impulse_silver_dq_findings`       | (not Impulse) | data-quality findings per (trip, signal) — flagged, never dropped |
# MAGIC
# MAGIC **Stored data is never modified.** Known quirks (Kelvin temperatures, speed spikes, SOC = 0 at BMS
# MAGIC start-up, lat/long in radians, …) are *flagged* in `impulse_silver_dq_findings` and documented in
# MAGIC `DATA_NOTES.md`; the labs handle them in their TSAL signal definitions.
# MAGIC
# MAGIC The transformation code lives in `common/impulse_silver.py` (shared with the declarative pipeline).

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

import json
import pyspark.sql.functions as F
from common import impulse_silver as silver

silver.use_utc(spark)

bronze = spark.table(BRONZE["signal_data"])
signal_info = spark.table(BRONZE["signal_info"])
TABLE_COMMENTS = {
    "container_metrics": "Impulse silver (required): one row per trip (container) with time bounds [epoch us], vehicle, configuration and scenario metadata.",
    "channel_metrics": "Impulse silver (required): one row per (trip, signal) channel with duration-weighted statistics, channel_name, unit and data_key.",
    "channels": "Impulse silver (required): run-length-encoded samples. One row per interval [tstart, tend) in epoch microseconds with a constant value.",
    "container_tags": "Impulse silver (optional): EAV tags per trip (vehicle_key, scenario, driver, ...) used for container filtering.",
    "channel_tags": "Impulse silver (optional): EAV tags per channel (channel_name, unit, category, subsystem, ...) used for channel selection.",
    "poi_channels": "Impulse silver (optional): Points-in-Time events (gear shifts, driver alerts), value only defined at the timestamp [epoch us].",
    "channel_mapping": "Impulse silver (optional): logical channel aliases resolved to physical signals by priority, with source/target units.",
    "unit_conversion": "Impulse silver (optional): unit families with multiplicative factors to the family base unit.",
}


def save(df, name, cluster_by=None, comment=None):
    writer = df.write.mode("overwrite").option("overwriteSchema", "true")
    if cluster_by:
        writer = writer.clusterBy(*cluster_by)
    writer.saveAsTable(name)
    if comment:
        spark.sql(f"COMMENT ON TABLE {name} IS '{comment}'")
    print(f"✅ {name}: {spark.table(name).count():,} rows")

# COMMAND ----------

# DBTITLE 1,1 · Samples + data-quality findings (flagged, not dropped)
samples = silver.samples(bronze)
DQ_TABLE = f"{SHARED_SCHEMA}.{TABLE_PREFIX}_silver_dq_findings"
save(silver.dq_findings(samples, signal_info), DQ_TABLE,
     comment="Data-quality findings per (trip, signal). Flagged only: silver keeps the measured values; handle in TSAL.")
display(spark.table(DQ_TABLE).orderBy("channel_name", "container_id"))
clean = samples  # nothing is removed: silver keeps every measured value

# COMMAND ----------

# DBTITLE 1,2 · channels (RLE)
save(silver.rle_encode(clean), SILVER["channels"], ["container_id", "channel_id"], TABLE_COMMENTS["channels"])

# COMMAND ----------

# DBTITLE 1,3 · poi_channels (derived events)
save(silver.poi_channels(clean, signal_info), SILVER["poi_channels"], ["container_id", "channel_id"],
     TABLE_COMMENTS["poi_channels"])

# COMMAND ----------

# DBTITLE 1,4 · channel_metrics + channel_tags
cm = silver.channel_metrics(spark.table(SILVER["channels"]), clean, signal_info).unionByName(
    silver.poi_channel_metrics(spark.table(SILVER["poi_channels"])))
save(cm, SILVER["channel_metrics"], ["container_id", "channel_id"], TABLE_COMMENTS["channel_metrics"])
save(silver.channel_tags(spark.table(SILVER["channel_metrics"]), signal_info), SILVER["channel_tags"],
     ["container_id", "channel_id"], TABLE_COMMENTS["channel_tags"])

# COMMAND ----------

# DBTITLE 1,5 · container_metrics + container_tags
ctm = silver.container_metrics(
    spark.table(SILVER["channel_metrics"]), silver.trip_keys(bronze),
    spark.table(BRONZE["trip_info"]), spark.table(BRONZE["vehicle_info"]), spark.table(BRONZE["vehicle_state"]))
save(ctm, SILVER["container_metrics"], ["container_id"], TABLE_COMMENTS["container_metrics"])
save(silver.container_tags(spark.table(SILVER["container_metrics"])), SILVER["container_tags"], ["container_id"],
     TABLE_COMMENTS["container_tags"])

# COMMAND ----------

# DBTITLE 1,6 · Reference tables: channel_mapping + unit_conversion
save(silver.channel_mapping(spark), SILVER["channel_mapping"], comment=TABLE_COMMENTS["channel_mapping"])
save(silver.unit_conversion(spark), SILVER["unit_conversion"], comment=TABLE_COMMENTS["unit_conversion"])

# COMMAND ----------

# DBTITLE 1,Summary: how much did RLE save?
rle = spark.sql(f"""
  SELECT sum(raw_sample_count) AS raw_samples, sum(sample_count) AS rle_intervals,
         round(sum(raw_sample_count) / sum(sample_count), 2) AS compression_ratio
  FROM {SILVER['channel_metrics']} WHERE value_type = 'DOUBLE'
""").first()
print(f"RLE: {rle.raw_samples:,} samples -> {rle.rle_intervals:,} intervals ({rle.compression_ratio}x)")
display(spark.table(SILVER["container_metrics"]).orderBy("container_id"))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"raw_samples": rle.raw_samples, "rle_intervals": rle.rle_intervals,
                                  "containers": spark.table(SILVER["container_metrics"]).count()}))
