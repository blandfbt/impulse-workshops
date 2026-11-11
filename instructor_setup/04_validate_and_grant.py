# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Validate the shared layer and grant participants access
# MAGIC
# MAGIC **Instructor only — run after `03_create_silver_tables`.**
# MAGIC
# MAGIC 1. Checks every bronze and silver table exists and is non-empty.
# MAGIC 2. Runs a real **Impulse** query against the shared silver layer, installing Impulse exactly the way
# MAGIC    participants will (`IMPULSE_INSTALL_MODE`: the wheel from `impulse_libs`, or the shared Git folder).
# MAGIC 3. Prints (and, with `apply_grants = true`, executes) the Unity Catalog grants participants need:
# MAGIC    read the `impulse_*` tables in the shared schema, and create their own schema.
# MAGIC
# MAGIC Grants are scoped to the `impulse_*` objects only, so other content in the shared schema is untouched.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

dbutils.widgets.dropdown("apply_grants", "false", ["true", "false"], "Execute the GRANT statements")
dbutils.widgets.text("grant_to", "account users", "Principal (group) to grant to")

# COMMAND ----------

# DBTITLE 1,1 · Every table exists and has rows
import json

checks = {f"bronze.{k}": v for k, v in BRONZE.items()} | {f"silver.{k}": v for k, v in SILVER.items()}
rows = {name: spark.table(table).count() for name, table in checks.items()}
for name, n in rows.items():
    print(f"{'✅' if n else '❌'} {checks[name]:70s} {n:>15,}")
assert all(rows.values()), "some tables are empty"

# COMMAND ----------

# DBTITLE 1,2 · Impulse smoke test (Impulse installed as participants will get it)
setup_impulse()
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver

db = MeasurementDB(MeasurementDBConfig(**impulse_source(SILVER), table_locations="unity_catalog"), WorkspaceClient())
vhcl_v = db.query.channel(channel_name="Vhcl.v")
speed = vhcl_v.where(vhcl_v < VHCL_V_MAX_MPS)          # handle the documented spikes (DATA_NOTES.md)
smoke = (
    db.query.havingTag(vehicle_key="MB_EQS_350")
    .select(speed.max().alias("max_speed_mps"), speed.mean().alias("mean_speed_mps"))
    .solve(spark, solver=DefaultSolver(spark))
)
display(smoke)
summary = smoke.agg(F.count("*").alias("n"), F.max("max_speed_mps").alias("top")).first()
assert summary.n > 0 and summary.top < VHCL_V_MAX_MPS, f"unexpected smoke-test result: {summary}"
print("✅ Impulse can query the shared silver layer")

# COMMAND ----------

# DBTITLE 1,3 · Grants for participants
apply = dbutils.widgets.get("apply_grants") == "true"
principal = dbutils.widgets.get("grant_to")
impulse_tables = [r.tableName for r in spark.sql(f"SHOW TABLES IN {SHARED_SCHEMA} LIKE '{TABLE_PREFIX}_*'").collect()]

grants = [
    f"GRANT USE CATALOG ON CATALOG {SHARED_CATALOG} TO `{principal}`",
    f"GRANT CREATE SCHEMA ON CATALOG {SHARED_CATALOG} TO `{principal}`",  # participants create USER_SCHEMA
    f"GRANT USE SCHEMA ON SCHEMA {SHARED_SCHEMA} TO `{principal}`",
    f"GRANT READ VOLUME ON VOLUME {SHARED_SCHEMA}.{RAW_VOLUME_NAME} TO `{principal}`",
    f"GRANT READ VOLUME ON VOLUME {SHARED_SCHEMA}.{LIBS_VOLUME_NAME} TO `{principal}`",   # the Impulse wheel
] + [f"GRANT SELECT ON TABLE {SHARED_SCHEMA}.{t} TO `{principal}`" for t in impulse_tables]

for g in grants:
    print(g)
    if apply:
        spark.sql(g)
print("\n✅ grants applied" if apply else "\nℹ️ dry run — set apply_grants = true to execute these statements")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"tables": rows, "grants_applied": apply}))
