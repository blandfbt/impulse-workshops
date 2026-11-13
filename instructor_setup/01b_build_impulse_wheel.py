# Databricks notebook source
# MAGIC %md
# MAGIC # 01b · Package Impulse as a wheel in a Unity Catalog volume
# MAGIC
# MAGIC **Instructor only — run once per Impulse version.**
# MAGIC
# MAGIC Downloads the Impulse source for tag `IMPULSE_VERSION` from GitHub (`IMPULSE_GITHUB_REPO`) into a
# MAGIC temporary directory — **nothing is written to the workspace** — builds a standard Python wheel and stores
# MAGIC it, together with the **Genie Code skills** (Markdown, same version), in the volume
# MAGIC **`dbx_workshop_dev.shared.impulse_libs`**. (If a Git folder already exists at `IMPULSE_REPO_PATH`,
# MAGIC it is used instead of the download.)
# MAGIC
# MAGIC ```
# MAGIC github.com/databrickslabs/impulse  (tag v0.11.0, downloaded to a temp dir)
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
# MAGIC notebooks change mid-workshop. To upgrade, bump `IMPULSE_VERSION` in CONFIG and re-run this notebook.
# MAGIC
# MAGIC Prefer the Git-folder approach? Set `IMPULSE_INSTALL_MODE = "repo"` in CONFIG and create the Git folder
# MAGIC (guide step 1) — notebooks then import from `IMPULSE_REPO_PATH/src` directly and this wheel isn't needed.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone

work = tempfile.mkdtemp()
commit = None

if os.path.isdir(os.path.join(IMPULSE_REPO_PATH, "src", "impulse_query_engine")):
    # Optional: an existing Git folder (repo mode) is used as the source.
    source_dir, source = IMPULSE_REPO_PATH, IMPULSE_REPO_PATH
    try:
        from databricks.sdk import WorkspaceClient
        ws = WorkspaceClient()
        commit = ws.repos.get(ws.workspace.get_status(IMPULSE_REPO_PATH).object_id).head_commit_id
    except Exception:
        pass
else:
    # Default: download the tagged source archive from GitHub into the temp dir -- nothing in /Workspace.
    url = f"https://github.com/{IMPULSE_GITHUB_REPO}/archive/refs/tags/{IMPULSE_VERSION}.tar.gz"
    archive = os.path.join(work, "impulse.tar.gz")
    print("downloading", url)
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(os.path.join(work, "download"), filter="data")
    (top,) = os.listdir(os.path.join(work, "download"))          # e.g. impulse-0.11.0/
    source_dir, source = os.path.join(work, "download", top), url
    try:
        api = f"https://api.github.com/repos/{IMPULSE_GITHUB_REPO}/commits/{IMPULSE_VERSION}"
        with urllib.request.urlopen(urllib.request.Request(api, headers={"Accept": "application/vnd.github.sha"})) as r:
            commit = r.read().decode().strip()
    except Exception:
        pass

assert os.path.isdir(os.path.join(source_dir, "src", "impulse_query_engine")), f"no Impulse sources in {source}"
repo_version = open(os.path.join(source_dir, "VERSION")).read().strip()
print(f"Impulse source {source} has version {repo_version}; CONFIG expects {IMPULSE_VERSION}")
assert repo_version == IMPULSE_VERSION.lstrip("v"), (
    "Version mismatch: check out the tag in the Git folder or update IMPULSE_VERSION in CONFIG")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {SHARED_SCHEMA}.{LIBS_VOLUME_NAME} "
          "COMMENT 'Python wheels for the Impulse workshops (databricks-impulse)'")

# COMMAND ----------

# DBTITLE 1,Build the wheel (from a local copy of the source)
src = os.path.join(work, "impulse")
# Only what the build needs: package sources + metadata files referenced by pyproject.toml.
shutil.copytree(source_dir, src,
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
shutil.copytree(os.path.join(source_dir, "skills"), skills_target,
                ignore=shutil.ignore_patterns(".claude-plugin"))
skills = sorted(d for d in os.listdir(skills_target) if os.path.isfile(os.path.join(skills_target, d, "SKILL.md")))
print(f"published {len(skills)} skills to {skills_target}: {skills}")

# A small manifest next to the wheel: what was built, from which commit, when.
manifest = {"package": "databricks-impulse", "version": repo_version, "wheel": IMPULSE_WHEEL_PATH,
            "skills": skills_target, "skill_names": skills,
            "source": source, "git_commit": commit, "built_at": datetime.now(timezone.utc).isoformat()}
with open(os.path.join(LIBS_VOLUME_PATH, f"databricks_impulse-{repo_version}.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
display(dbutils.fs.ls(LIBS_VOLUME_PATH))
shutil.rmtree(work, ignore_errors=True)

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
