# Instructor setup

Run once per workspace, before the workshops. Full instructions: **[`00_instructor_guide`](00_instructor_guide.py)**.

| Notebook | Does | Time |
|---|---|---|
| `00_instructor_guide` | step-by-step guide: clone Impulse once, Genie Code skills, CONFIG, grants | read |
| `01b_build_impulse_wheel` | downloads the Impulse release tag from GitHub and packages it as a wheel — plus its Genie Code skills — into volume `dbx_workshop_dev.shared.impulse_libs` | 1 min |
| `01_download_dataset` | Hugging Face → volume `dbx_workshop_dev.shared.impulse_raw` (52 GB, resumable, pinned revision) | 10–20 min |
| `02_create_bronze_tables` | `impulse_bronze_*` — faithful copy of the source star schema (40 curated signals, ≈370 M rows) | ~5 min |
| `03_create_silver_tables` | `impulse_silver_*` — the Impulse silver model incl. all optional tables + `impulse_silver_dq_findings` | ~4 min |
| `04_validate_and_grant` | row-count checks, a real Impulse query, participant grants (dry run unless `apply_grants=true`) | ~2 min |

Or deploy the bundle and run the chained job: `databricks bundle run impulse_instructor_setup -p <profile>`.

**Where to clone Impulse (repo mode only):** wheel mode needs no clone, because `01b` downloads tag `v0.11.0`
from GitHub. For `"repo"` mode, create one Databricks Git folder at **`/Workspace/Shared/impulse`**, readable by all participants. If you choose another path, change `IMPULSE_REPO_PATH` in `../CONFIG`.

**How notebooks install it** (`IMPULSE_INSTALL_MODE` in `../CONFIG`): `"wheel"` (default) installs
`/Volumes/dbx_workshop_dev/shared/impulse_libs/databricks_impulse-0.11.0-py3-none-any.whl` built by `01b`;
`"repo"` imports straight from the Git folder.
