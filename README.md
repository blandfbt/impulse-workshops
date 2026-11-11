# Databricks Impulse Workshops

Two hands-on workshops for **[Impulse](https://github.com/databrickslabs/impulse)** — the Databricks Labs
library for analyzing large-scale time-series measurement data (vehicle test drives, battery runs,
industrial sensors) on Spark and Delta, using the **TSAL** (Time Series Analytics Language) DSL.

The data is the open [Synthetic Vehicle Fleet Telemetry](https://huggingface.co/datasets/renumics/synthetic-vehicle-fleet-telemetry)
dataset: 100 simulated trips of 13 vehicle models (EVs, hybrids, combustion), ~100 Hz, 5.1 B observations.

| Workshop | For | You will |
|---|---|---|
| **[1 · Introduction](workshop_1_intro/)** (~2 h) | anyone new to Impulse | query signals with TSAL, let **Genie Code** + the Impulse skills write TSAL, build a report into your schema and an **AI/BI dashboard** |
| **[2 · Deep dive](workshop_2_deep_dive/)** (~3.5 h) | data engineers & analysts who did workshop 1 | engineer bronze → the Impulse **silver model** (RLE, optional tables) as a notebook *and* a **declarative pipeline + job**, master all TSAL result types, aliases, unit conversion, UDFs, and a multi-page dashboard |

## Quick start (participants)

1. **Clone this repo** into your workspace: *Workspace → your folder → Create → Git folder* →
   `https://github.com/blandfbt/impulse-workshops`.
2. Open **[`CONFIG`](CONFIG.py)** and set **one** value — your own schema:
   ```python
   USER_SCHEMA = "dbx_workshop_dev.<your_name>"
   ```
3. Attach **Serverless** compute and start with
   [`workshop_1_intro/01_intro_to_impulse_tsal`](workshop_1_intro/01_intro_to_impulse_tsal.py).

That's it: the shared data and the shared Impulse library are prepared by your instructor.

## How it fits together

```
 Hugging Face ──► /Volumes/<catalog>/shared/impulse_raw          (raw parquet, instructor)
                    │
                    ▼
            shared.impulse_bronze_*   faithful copy of the source star schema      (instructor)
                    │
                    ▼
            shared.impulse_silver_*   the Impulse silver model                      (instructor; rebuilt by
                    │                                                                participants in workshop 2)
                    ▼   Impulse (one shared clone: /Workspace/Shared/impulse)
            <USER_SCHEMA>.impulse_w1_* / impulse_w2_*   gold star schemas           (participants)
                    │
                    ▼
            AI/BI dashboards in your home folder
```

* **One copy of Impulse.** The instructor clones `databrickslabs/impulse` once into a shared Git folder
  (`IMPULSE_REPO_PATH`, default `/Workspace/Shared/impulse`) and packages it as a **wheel** in the volume
  `dbx_workshop_dev.shared.impulse_libs`. Notebooks `%pip install` that wheel (`IMPULSE_INSTALL_MODE =
  "wheel"`, default) or import straight from the Git folder (`"repo"`). The Genie Code **skills** are
  published next to the wheel (`impulse_libs/skills/<version>/`) and copied into each user's Genie Code
  skills folder in workshop 1 / notebook 2.
* **One config file.** Every notebook starts with `%run ../CONFIG`. Participants only set `USER_SCHEMA`.
* **Data quirks are documented, not hidden.** Silver keeps the measured values unchanged; see
  **[`DATA_NOTES.md`](DATA_NOTES.md)** for how each quirk is handled in the TSAL signal definitions.

## Repository layout

```
CONFIG.py                      the single config notebook (participants set USER_SCHEMA)
DATA_NOTES.md                  known data quirks and how the labs handle them
databricks.yml, resources/     Databricks Asset Bundle: setup job, workshop test jobs, pipeline
common/                        shared Python: bronze->silver transforms, plotting, dashboard builder
instructor_setup/              00 guide · 01 download · 01b Impulse wheel · 02 bronze · 03 silver · 04 validate & grant
workshop_1_intro/              01 TSAL intro · 02 Genie Code skills · 03 report + dashboard
workshop_2_deep_dive/          01 bronze->silver · 02a TSAL deep dive · 02b report, SQL, dashboard
  declarative_pipeline/        the silver build as a Lakeflow Declarative Pipeline + job
```

## Instructors

Start with [`instructor_setup/00_instructor_guide`](instructor_setup/00_instructor_guide.py) — it covers
cloning Impulse, Genie Code skills, building the shared data (~20 min) and granting access.

### Deploying with the Databricks Asset Bundle

The bundle is optional for participants but handy for instructors and automated end-to-end tests:

```bash
databricks bundle validate -p <profile>
databricks bundle deploy   -p <profile> --var user_schema_name=<your_schema>
databricks bundle run impulse_instructor_setup        -p <profile>   # download, bronze, silver, validate
databricks bundle run impulse_workshop_1              -p <profile>   # runs every workshop-1 notebook
databricks bundle run impulse_workshop_2              -p <profile>   # runs every workshop-2 notebook
databricks bundle run impulse_w2_silver_pipeline_job  -p <profile>   # declarative pipeline + Impulse smoke test
```

Targets: `dev` (default, personal copy, jobs prefixed `[dev <you>]`) and `shared` (production mode, deployed
under `/Workspace/Shared/.bundle/`). Variables: `catalog`, `shared_schema`, `user_catalog`,
`user_schema_name`, `impulse_repo_path`, `impulse_install_mode`, `warehouse_id`, `max_trips`.

## Requirements

* Databricks workspace with Unity Catalog and Serverless (notebooks, jobs, SQL warehouse)
* Impulse **v0.11.0** (Python 3.12 → Serverless environment version 2+), installed from the shared wheel
  (or imported from the shared Git folder)
* For instructors: outbound access to `huggingface.co`, ~55 GB volume storage

## License & attribution

Workshop code and notebooks: [MIT](LICENSE). The MIT license covers this repository only — not the
dataset or Impulse. Dataset: *Synthetic Vehicle Fleet Telemetry* by
[Renumics](https://renumics.com/), CC BY-SA 4.0, based on joint work of Volkswagen AG Group Innovation,
IPG Automotive and Renumics — see [NOTICE.md](NOTICE.md). Impulse is a Databricks Labs project, provided
as-is without SLAs. This repository is not an official Databricks product.
