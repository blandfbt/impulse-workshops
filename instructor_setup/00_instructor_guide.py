# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Instructor guide — set up the Impulse workshops
# MAGIC
# MAGIC **Audience: the workshop instructor / workspace administrator. Do this once per workspace, ideally the
# MAGIC day before.** Participants never run anything in `instructor_setup/`.
# MAGIC
# MAGIC | Step | What | Where | Time |
# MAGIC |---|---|---|---|
# MAGIC | 1 | *(repo mode only)* Clone **Impulse** once into a shared Git folder | Workspace UI or CLI | 2 min |
# MAGIC | 1b | Package Impulse as a **wheel** in the volume `shared.impulse_libs` | `01b_build_impulse_wheel` | 1 min |
# MAGIC | 2 | (Optional) turn the Impulse **Genie Code skills** on workspace-wide | notebook snippet | 2 min |
# MAGIC | 3 | Check `CONFIG` (catalog, shared schema, Impulse path) | `../CONFIG` | 2 min |
# MAGIC | 4 | Run `01` → `04` (or the `impulse_instructor_setup` job) | this folder | ~20 min |
# MAGIC | 5 | Grant participants access | `04_validate_and_grant` | 1 min |
# MAGIC | 6 | Smoke-test a participant run | workshop notebooks | 15 min |
# MAGIC
# MAGIC ## Prerequisites
# MAGIC * A workspace with **Unity Catalog** and **Serverless** compute (notebooks + jobs + SQL warehouse).
# MAGIC * Outbound internet from serverless to `huggingface.co` (for the one-time download).
# MAGIC * You can `CREATE SCHEMA` / `CREATE VOLUME` in the catalog (default `dbx_workshop_dev`), and grant on it.
# MAGIC * ~55 GB of storage for the raw download, ~10 GB for bronze + silver Delta tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Clone Impulse **once** into a shared Git folder *(only for `IMPULSE_INSTALL_MODE = "repo"`)*
# MAGIC
# MAGIC > **Default wheel mode: skip this step.** `01b_build_impulse_wheel` downloads the `IMPULSE_VERSION` tag
# MAGIC > from GitHub into a temp directory and needs nothing in `/Workspace/Shared`.
# MAGIC
# MAGIC Every workshop notebook imports Impulse from **one** shared location instead of installing it per
# MAGIC user: same version for everyone, one place to upgrade, no per-user installs. The location is
# MAGIC `IMPULSE_REPO_PATH` in `CONFIG` — **default `/Workspace/Shared/impulse`**.
# MAGIC
# MAGIC **Option A — Workspace UI**
# MAGIC 1. **Workspace** → **Shared** → **Create** → **Git folder**.
# MAGIC 2. Git repository URL: `https://github.com/databrickslabs/impulse` · provider **GitHub** ·
# MAGIC    folder name **`impulse`** → **Create Git folder**.
# MAGIC 3. Pin the tested release: open the Git folder → branch selector → **Tags** → `v0.11.0`
# MAGIC    (`IMPULSE_VERSION` in CONFIG).
# MAGIC
# MAGIC **Option B — Databricks CLI**
# MAGIC ```bash
# MAGIC databricks repos create https://github.com/databrickslabs/impulse gitHub --path /Workspace/Shared/impulse
# MAGIC databricks repos update /Workspace/Shared/impulse --tag v0.11.0
# MAGIC ```
# MAGIC
# MAGIC **Permissions:** participants need **Can Read** on `/Workspace/Shared/impulse` (`Shared` grants this to
# MAGIC all users by default). Only instructors should have *Can Manage* — so nobody switches the branch
# MAGIC mid-workshop.
# MAGIC
# MAGIC > Using a different path (e.g. a team folder)? Change `IMPULSE_REPO_PATH` in `CONFIG` — it's the only
# MAGIC > place it's referenced. To upgrade Impulse later, pull a newer tag in the Git folder, bump
# MAGIC > `IMPULSE_VERSION` in CONFIG, re-run `01b_build_impulse_wheel` and re-test.
# MAGIC
# MAGIC ## Step 1b · Package Impulse as a wheel (default install mode)
# MAGIC
# MAGIC Run **`01b_build_impulse_wheel`**. It downloads the `IMPULSE_VERSION` tag from GitHub (or uses the Git
# MAGIC folder from step 1, if present), builds `databricks_impulse-<version>-py3-none-any.whl` and stores it (plus a small JSON manifest with version + commit) in the Unity Catalog volume
# MAGIC **`dbx_workshop_dev.shared.impulse_libs`**. Notebooks then run `%pip install $IMPULSE_PIP_TARGET`.
# MAGIC
# MAGIC CONFIG supports **both** options via `IMPULSE_INSTALL_MODE`:
# MAGIC
# MAGIC | Mode | Notebooks get Impulse from | Pros |
# MAGIC |---|---|---|
# MAGIC | `"wheel"` (default) | `/Volumes/<catalog>/shared/impulse_libs/databricks_impulse-<v>-py3-none-any.whl` | immutable, versioned, UC-governed (`READ VOLUME`); also usable as a job/cluster library |
# MAGIC | `"repo"` | `sys.path` → `/Workspace/Shared/impulse/src` | no build step; always the Git folder's checked-out tag |
# MAGIC
# MAGIC Either way there is **one** copy for the whole workspace. The Genie Code skills (Markdown, not part of the
# MAGIC wheel) are published by `01b` to `impulse_libs/skills/<version>/` too, so wheel mode never needs the Git folder.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Genie Code skills
# MAGIC
# MAGIC Impulse ships agent skills (Markdown `SKILL.md` files) in its `skills/` folder. They are **not** part of
# MAGIC the wheel, so `01b_build_impulse_wheel` publishes them next to it, same version:
# MAGIC `/Volumes/dbx_workshop_dev/shared/impulse_libs/skills/v0.11.0/`. Genie Code reads skills from
# MAGIC **workspace** folders, so workshop 1 / notebook 2 copies them from there (or, in `repo` mode, from
# MAGIC `/Workspace/Shared/impulse/skills`) into each participant's `/Users/<you>/.assistant/skills/`.
# MAGIC
# MAGIC Optional — turn them on for **everybody** instead (workspace admins), from a notebook after `01b`:
# MAGIC
# MAGIC ```python
# MAGIC import shutil
# MAGIC shutil.copytree("/Volumes/dbx_workshop_dev/shared/impulse_libs/skills/v0.11.0",
# MAGIC                 "/Workspace/.assistant/skills", dirs_exist_ok=True)
# MAGIC ```
# MAGIC
# MAGIC Users can see (and toggle) the skills Genie Code loads under **Genie Code pane → ⋮ → Customizations**.
# MAGIC Copies are snapshots: repeat after an Impulse upgrade.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Check CONFIG
# MAGIC
# MAGIC Open `../CONFIG`. As instructor you only care about section 2 (shared locations) and 3 (dataset):
# MAGIC
# MAGIC | Setting | Default | Meaning |
# MAGIC |---|---|---|
# MAGIC | `IMPULSE_REPO_PATH` | `/Workspace/Shared/impulse` | the Git folder from step 1 (repo mode only) |
# MAGIC | `IMPULSE_INSTALL_MODE` | `wheel` | `wheel` (from `impulse_libs`, step 1b) or `repo` (the Git folder) |
# MAGIC | `LIBS_VOLUME_NAME` | `impulse_libs` | volume holding the Impulse wheel |
# MAGIC | `SHARED_CATALOG` / `SHARED_SCHEMA_NAME` | `dbx_workshop_dev` / `shared` | where bronze + silver live |
# MAGIC | `RAW_VOLUME_NAME` | `impulse_raw` | volume for the Hugging Face download |
# MAGIC | `HF_REVISION` | pinned commit | reproducible data |
# MAGIC | `BRONZE_SIGNALS` | 40 curated signals | `["ALL"]` loads all 681 (≈5.1 B rows) |
# MAGIC | `W2_ENGINEERING_TRIPS` | 6 trips | trips participants transform by hand in workshop 2 |
# MAGIC
# MAGIC Everything this setup creates is prefixed **`impulse_`**, so it coexists with other content in the
# MAGIC shared schema.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Build the shared data
# MAGIC
# MAGIC Run the notebooks in order on **Serverless** (or run the job `impulse_instructor_setup` after
# MAGIC `databricks bundle deploy`, which chains them):
# MAGIC
# MAGIC | Notebook | Creates (in `dbx_workshop_dev.shared`) | Typical time |
# MAGIC |---|---|---|
# MAGIC | `01_download_dataset` | volume `impulse_raw` with the 52 GB dataset (resumable) | 10–20 min |
# MAGIC | `02_create_bronze_tables` | `impulse_bronze_signal_data` (≈370 M rows) + 4 dimension tables | ~5 min |
# MAGIC | `03_create_silver_tables` | the 8 `impulse_silver_*` tables + `impulse_silver_dq_findings` | ~3 min |
# MAGIC | `04_validate_and_grant` | — (checks + an Impulse smoke test + grants) | ~2 min |
# MAGIC
# MAGIC Tip: for a dry run, start `01` with the widget `max_trips = 10`; re-run later with 100 (it resumes).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · Grant participants access
# MAGIC
# MAGIC Run `04_validate_and_grant` with `apply_grants = true` and `grant_to = <participant group>` (default
# MAGIC `account users`). It grants exactly:
# MAGIC
# MAGIC * `USE CATALOG` + `CREATE SCHEMA` on the catalog — participants create their own `USER_SCHEMA`
# MAGIC   (skip `CREATE SCHEMA` if you pre-create schemas for them),
# MAGIC * `USE SCHEMA` on the shared schema and `SELECT` on each `impulse_*` table,
# MAGIC * `READ VOLUME` on `impulse_raw` and on `impulse_libs` (the Impulse wheel).
# MAGIC
# MAGIC Participants also need to use a **SQL warehouse** (`CAN USE`) for the dashboards; set `WAREHOUSE_ID`
# MAGIC in CONFIG to pin one, otherwise the notebooks pick a serverless warehouse automatically.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · Smoke-test as a participant
# MAGIC
# MAGIC Either clone this repo into your own Git folder, set `USER_SCHEMA` in `CONFIG` and run the notebooks,
# MAGIC or use the bundle:
# MAGIC
# MAGIC ```bash
# MAGIC databricks bundle deploy -p <profile> --var user_schema_name=<your_schema>
# MAGIC databricks bundle run impulse_workshop_1 -p <profile>
# MAGIC databricks bundle run impulse_workshop_2 -p <profile>
# MAGIC databricks bundle run impulse_w2_silver_pipeline_job -p <profile>
# MAGIC ```
# MAGIC
# MAGIC Expected runtimes on serverless: workshop 1 ≈ 15 min, workshop 2 ≈ 30 min, pipeline job ≈ 8 min.
# MAGIC
# MAGIC ## Running the day
# MAGIC * Ask participants to open `README.md`, then `CONFIG`, set `USER_SCHEMA`, and start with
# MAGIC   `workshop_1_intro/01_intro_to_impulse_tsal`.
# MAGIC * Most common issue: **`USER_SCHEMA` not set** → the first cell tells them exactly what to type.
# MAGIC * `ImportError: cannot import name 'Self'` → they're on an old compute (Python < 3.12). Use Serverless
# MAGIC   (environment version 2+).
# MAGIC * Clean-up afterwards: participants' objects are all prefixed `impulse_w1_*` / `impulse_w2_*` in their
# MAGIC   own schemas; dashboards are named `Impulse W1 …` / `Impulse W2 …` in their home folders.
