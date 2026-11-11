# Databricks notebook source
# MAGIC %md
# MAGIC # 01b · Package Impulse as a wheel in a Unity Catalog volume
# MAGIC
# MAGIC **Instructor only — run once per Impulse version, after cloning Impulse (guide step 1).**
# MAGIC
# MAGIC Builds a standard Python wheel from the shared Git folder (`IMPULSE_REPO_PATH`, pinned to
# MAGIC `IMPULSE_VERSION`) and stores it — together with the **Genie Code skills** (Markdown, same version) —
# MAGIC in the volume **`dbx_workshop_dev.shared.impulse_libs`**:
# MAGIC
# MAGIC ```
# MAGIC /Workspace/Shared/impulse  (Git folder, tag v0.11.0)
# MAGIC        │  pip wheel
# MAGIC        ▼
# MAGIC /Volumes/dbx_workshop_dev/shared/impulse_libs/databricks_impulse-0.11.0-py3-none-any.whl
# MAGIC /Volumes/dbx_workshop_dev/shared/impulse_libs/skills/v0.11.0/impulse*/SKILL.md   (workshop 1 / notebook 2)
# MAGIC        │  %pip install $IMPULSE_PIP_TARGET   (every workshop notebook, IMPULSE_INSTALL_MODE = "wheel")
# MAGIC        ▼
# MAGIC participants' notebooks / jobs
# MAGIC ```
# MAGIC
# MAGIC **Why a wheel?** It's an immutable, versioned artifact governed by Unity Catalog (`READ VOLUME`), it
# MAGIC installs like any package (`%pip`, job/cluster libraries, serverless environments), and nobody's
# MAGIC notebooks change if someone switches the Git folder's branch. The Git folder stays the single source:
# MAGIC re-run this notebook after pulling a new Impulse tag (and bump `IMPULSE_VERSION` in CONFIG).
# MAGIC
# MAGIC Prefer the Git-folder approach? Set `IMPULSE_INSTALL_MODE = "repo"` in CONFIG — notebooks then import
# MAGIC from `IMPULSE_REPO_PATH/src` directly and this wheel isn't needed.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

assert os.path.isdir(os.path.join(IMPULSE_REPO_PATH, "src", "impulse_query_engine")), \
    f"Impulse Git folder not found at {IMPULSE_REPO_PATH} -- see 00_instructor_guide, step 1"

repo_version = open(os.path.join(IMPULSE_REPO_PATH, "VERSION")).read().strip()
print(f"Git folder {IMPULSE_REPO_PATH} has Impulse version {repo_version}; CONFIG expects {IMPULSE_VERSION}")
assert repo_version == IMPULSE_VERSION.lstrip("v"), (
    "Version mismatch: check out the tag in the Git folder or update IMPULSE_VERSION in CONFIG")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {SHARED_SCHEMA}.{LIBS_VOLUME_NAME} "
          "COMMENT 'Python wheels for the Impulse workshops (databricks-impulse)'")

# COMMAND ----------

# DBTITLE 1,Build the wheel (from a local copy of the Git folder)
work = tempfile.mkdtemp()
src = os.path.join(work, "impulse")
# Only what the build needs: package sources + metadata files referenced by pyproject.toml.
shutil.copytree(IMPULSE_REPO_PATH, src,
                ignore=shutil.ignore_patterns(".git", "apps", "demos", "docs", "tests", "skills", "*.ipynb"))
out = os.path.join(work, "dist")
build = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--quiet", "-w", out, src],
                       capture_output=True, text=True)
print(build.stdout[-2000:], build.stderr[-2000:])
build.check_returncode()
wheel = glob.glob(os.path.join(out, "*.whl"))[0]
print("built", os.path.basename(wheel))
assert os.path.basename(wheel) == os.path.basename(IMPULSE_WHEEL_PATH), \
    f"unexpected wheel name {os.path.basename(wheel)} (CONFIG expects {os.path.basename(IMPULSE_WHEEL_PATH)})"

# COMMAND ----------

# DBTITLE 1,Smoke-test the wheel in isolation, then publish it to the volume
check_dir = os.path.join(work, "check")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", check_dir, wheel], check=True)
probe = subprocess.run(
    [sys.executable, "-c", "import impulse_query_engine, impulse_reporting.core.report as r; print(r.__file__)"],
    env={**os.environ, "PYTHONPATH": check_dir}, capture_output=True, text=True)
print(probe.stdout, probe.stderr[-1000:])
assert probe.returncode == 0 and check_dir in probe.stdout, "wheel does not import"

shutil.copy(wheel, IMPULSE_WHEEL_PATH)

# The Genie Code skills are Markdown files in the repo, not part of the wheel: publish them next to the
# wheel, under the same version, so wheel-mode workshops never need the Git folder.
skills_target = f"{LIBS_VOLUME_PATH}/skills/{IMPULSE_VERSION}"
if os.path.isdir(skills_target):
    shutil.rmtree(skills_target)
shutil.copytree(os.path.join(IMPULSE_REPO_PATH, "skills"), skills_target,
                ignore=shutil.ignore_patterns(".claude-plugin"))
skills = sorted(d for d in os.listdir(skills_target) if os.path.isfile(os.path.join(skills_target, d, "SKILL.md")))
print(f"published {len(skills)} skills to {skills_target}: {skills}")

# A small manifest next to the wheel: what was built, from which commit, when.
try:
    from databricks.sdk import WorkspaceClient
    ws = WorkspaceClient()
    repo_id = ws.workspace.get_status(IMPULSE_REPO_PATH).object_id      # a Git folder is a "repo" object
    commit = ws.repos.get(repo_id).head_commit_id
except Exception:
    commit = None
manifest = {"package": "databricks-impulse", "version": repo_version, "wheel": IMPULSE_WHEEL_PATH,
            "skills": skills_target, "skill_names": skills,
            "source": IMPULSE_REPO_PATH, "git_commit": commit, "built_at": datetime.now(timezone.utc).isoformat()}
with open(os.path.join(LIBS_VOLUME_PATH, f"databricks_impulse-{repo_version}.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
display(dbutils.fs.ls(LIBS_VOLUME_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Done. Participants need `READ VOLUME` on `impulse_libs` (granted by `04_validate_and_grant`).
# MAGIC Any notebook can now install Impulse with:
# MAGIC ```
# MAGIC %pip install /Volumes/dbx_workshop_dev/shared/impulse_libs/databricks_impulse-0.11.0-py3-none-any.whl
# MAGIC ```
# MAGIC and jobs can list the same path as a dependency in their serverless environment.

# COMMAND ----------

dbutils.notebook.exit(json.dumps(manifest))
