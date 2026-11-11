# Silver build as a Lakeflow Declarative Pipeline

Notebook `../01_bronze_to_silver_data_engineering` builds the silver layer step by step. Here the same
logic runs as a **Lakeflow Spark Declarative Pipeline**, orchestrated by a **job**:

```
job impulse_w2_silver_pipeline_job
 ├─ refresh_silver_pipeline  (pipeline task)   transformations/impulse_silver_pipeline.py
 │     bronze ─► samples (view, expectations) ─► channels (RLE) ─► channel_metrics ─► container_metrics …
 │                                           └► dq_findings, poi_channels, *_tags, channel_mapping, unit_conversion
 └─ impulse_smoke_test       (notebook task)   validate_pipeline_silver.py — fails unless Impulse can query it
```

Output: `<USER_SCHEMA>.impulse_w2_sdp_*` for the six trips in `W2_ENGINEERING_TRIPS`.

* **With the bundle:** `databricks bundle deploy …` creates both (see `resources/w2_silver.pipeline.yml`);
  run `databricks bundle run impulse_w2_silver_pipeline_job`.
* **Without the bundle (Git folder):** run [`00_create_pipeline_and_job`](00_create_pipeline_and_job.py),
  which creates the pipeline + job for your schema via the SDK and runs it.

The transformation functions are shared with the instructor setup in `common/impulse_silver.py`; the
pipeline file adds the declarative table definitions and data-quality **expectations**.
