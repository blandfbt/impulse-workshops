# Databricks notebook source
# MAGIC %md
# MAGIC # Smoke test: can Impulse query the pipeline's silver tables?
# MAGIC
# MAGIC Second task of the `impulse_w2_silver_pipeline_job`. Runs after the declarative pipeline refreshed
# MAGIC and fails the job if the silver layer it produced is not usable by Impulse — so a broken silver layer
# MAGIC never reaches downstream reports.

# COMMAND ----------

# MAGIC %run ../../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

require_user_schema()
setup_impulse()
import json
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver

PIPE_SILVER = silver_tables(USER_SCHEMA, W2_PIPELINE_SILVER_PREFIX)
TRIPS = W2_ENGINEERING_TRIPS

# COMMAND ----------

# DBTITLE 1,1 · Same row counts as the shared silver layer for these trips
compare = []
for t in ["channels", "channel_metrics", "channel_tags", "container_metrics", "container_tags", "poi_channels"]:
    mine = spark.table(PIPE_SILVER[t]).count()
    shared = spark.table(SILVER[t]).where(F.col("container_id").isin(TRIPS)).count()
    compare.append((t, mine, shared))
display(spark.createDataFrame(compare, "table string, pipeline long, shared long"))
assert all(m == s for _, m, s in compare), f"pipeline silver differs from shared silver: {compare}"

# COMMAND ----------

# DBTITLE 1,2 · Impulse query: direct channel, alias with fallback + unit conversion, POI channel
db = MeasurementDB(MeasurementDBConfig(**impulse_source(PIPE_SILVER), table_locations="unity_catalog"), WorkspaceClient())
q = db.query
res = (q.select(q.channel(channel_name="Vhcl.v").max().alias("vhcl_v_max_mps"),
                q.channel_with_alias(channel_alias="vehicle_speed").max().alias("vehicle_speed_max_kmh"),
                q.poi_channel(channel_name="Event.GearShift").alias("gear_shifts"))
       .solve(spark, solver=DefaultSolver(spark))
       .withColumn("n_gear_shifts", F.coalesce(F.size("gear_shifts"), F.lit(0))).drop("gear_shifts"))
display(res)
stats = res.agg(F.count("*").alias("n"), F.count("vehicle_speed_max_kmh").alias("with_speed")).first()
assert stats.n == len(TRIPS) and stats.with_speed == stats.n, f"alias did not resolve for every trip: {stats}"
print("✅ the pipeline's silver layer is Impulse-ready")
dbutils.notebook.exit(json.dumps({"trips": stats.n}))
