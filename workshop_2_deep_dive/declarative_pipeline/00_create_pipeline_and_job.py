# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2 · The silver build as a Lakeflow Declarative Pipeline + job
# MAGIC
# MAGIC Notebook `01_bronze_to_silver_data_engineering` built the silver layer step by step. In production you
# MAGIC want the same logic **declarative, testable and scheduled**. This folder holds exactly that:
# MAGIC
# MAGIC | File | Role |
# MAGIC |---|---|
# MAGIC | `transformations/impulse_silver_pipeline.py` | the pipeline: one `@dp.materialized_view` per silver table, data-quality **expectations** |
# MAGIC | `validate_pipeline_silver.py` | job task 2: fails the job unless Impulse can query the result |
# MAGIC | this notebook | creates the pipeline and the job **without** the bundle (for Git-folder users) |
# MAGIC
# MAGIC ```
# MAGIC  job: impulse_w2_silver_pipeline_job
# MAGIC   ├─ task 1  refresh_silver_pipeline  (pipeline task)  bronze ─► samples ─► channels ─► metrics ─► … (+ dq_findings)
# MAGIC   └─ task 2  impulse_smoke_test       (notebook task)  Impulse query on the new silver tables
# MAGIC ```
# MAGIC
# MAGIC **Declarative vs. the notebook:** you declare *what* each table is; the pipeline works out the order,
# MAGIC runs independent tables in parallel, records lineage, and enforces the expectations
# MAGIC (`expect_or_drop` / `expect_or_fail`) with pass/fail metrics in the pipeline UI.
# MAGIC
# MAGIC > Deployed with the Asset Bundle? Then both already exist (`resources/w2_silver.pipeline.yml`) — run the
# MAGIC > job from **Jobs & Pipelines** instead. Otherwise run this notebook.

# COMMAND ----------

# MAGIC %run ../../CONFIG

# COMMAND ----------

require_user_schema()
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs, pipelines

ws = WorkspaceClient()
me = current_user()
catalog, schema = USER_SCHEMA.split(".")
here = os.path.join(REPO_ROOT, "workshop_2_deep_dive", "declarative_pipeline")
pipeline_name = f"impulse_w2_silver_pipeline_{schema}"
job_name = f"impulse_w2_silver_pipeline_job_{schema}"

settings = dict(
    name=pipeline_name, catalog=catalog, schema=schema, serverless=True, channel="CURRENT", photon=True,
    root_path=here,
    libraries=[pipelines.PipelineLibrary(file=pipelines.FileLibrary(path=f"{here}/transformations/impulse_silver_pipeline.py"))],
    configuration={"impulse_workshop.repo_root": REPO_ROOT,
                   "impulse_workshop.bronze_schema": SHARED_SCHEMA,
                   "impulse_workshop.table_prefix": W2_PIPELINE_SILVER_PREFIX,
                   "impulse_workshop.trips": ",".join(map(str, W2_ENGINEERING_TRIPS))},
)
existing = [p for p in ws.pipelines.list_pipelines(filter=f"name LIKE '{pipeline_name}'")]
if existing:
    pipeline_id = existing[0].pipeline_id
    ws.pipelines.update(pipeline_id, **settings)
else:
    pipeline_id = ws.pipelines.create(**settings).pipeline_id
print("pipeline:", pipeline_id)

# COMMAND ----------

job_settings = dict(
    name=job_name,
    parameters=[jobs.JobParameterDefinition(name="user_schema", default=USER_SCHEMA),
                jobs.JobParameterDefinition(name="impulse_repo_path", default=IMPULSE_REPO_PATH)],
    environments=[jobs.JobEnvironment(environment_key="serverless",
                                      spec=compute.Environment(environment_version="4"))],
    tasks=[
        jobs.Task(task_key="refresh_silver_pipeline",
                  pipeline_task=jobs.PipelineTask(pipeline_id=pipeline_id, full_refresh=True)),
        jobs.Task(task_key="impulse_smoke_test", depends_on=[jobs.TaskDependency(task_key="refresh_silver_pipeline")],
                  environment_key="serverless",
                  notebook_task=jobs.NotebookTask(notebook_path=f"{here}/validate_pipeline_silver")),
    ],
)
found = [j for j in ws.jobs.list(name=job_name)]
if found:
    job_id = found[0].job_id
    ws.jobs.reset(job_id, new_settings=jobs.JobSettings(**job_settings))
else:
    job_id = ws.jobs.create(**job_settings).job_id
host = ws.config.host.rstrip("/")
displayHTML(f'<p>Pipeline: <a href="{host}/pipelines/{pipeline_id}" target="_blank">{pipeline_name}</a><br>'
            f'Job: <a href="{host}/jobs/{job_id}" target="_blank">{job_name}</a></p>')

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job (≈5–8 minutes). Open the pipeline link while it runs to watch the graph and the
# MAGIC expectation metrics; the job link shows both tasks.

# COMMAND ----------

run = ws.jobs.run_now(job_id).result()
print("job finished:", run.state.result_state)
display(spark.sql(f"SHOW TABLES IN {USER_SCHEMA} LIKE '{W2_PIPELINE_SILVER_PREFIX}_*'"))
