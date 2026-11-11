# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Download the dataset into a Unity Catalog volume
# MAGIC
# MAGIC **Instructor only — run once per workspace.**
# MAGIC
# MAGIC Copies [`renumics/synthetic-vehicle-fleet-telemetry`](https://huggingface.co/datasets/renumics/synthetic-vehicle-fleet-telemetry)
# MAGIC (~52 GB, 1,000 measurement files + 4 dimension files) from Hugging Face into the volume
# MAGIC `dbx_workshop_dev.shared.impulse_raw`, preserving the repository layout:
# MAGIC
# MAGIC ```
# MAGIC /Volumes/<catalog>/shared/impulse_raw/synthetic-vehicle-fleet-telemetry/assets/
# MAGIC   ├── dim_signal_info/*.parquet       681 signal definitions
# MAGIC   ├── dim_trip_info/*.parquet         100 trips (= Impulse containers)
# MAGIC   ├── dim_vehicle_info/*.parquet      13 vehicle models
# MAGIC   ├── dim_vehicle_state/*.parquet     49 passenger / tire / driver-style configurations
# MAGIC   └── fact_signal_data/trip_hash=*/partition=*/*.parquet   5.09 B signal observations
# MAGIC ```
# MAGIC
# MAGIC * **Pinned** to a dataset commit (`HF_REVISION` in CONFIG) so every workshop sees the same data.
# MAGIC * **Resumable** — files already present with the right size are skipped; just re-run on failure.
# MAGIC * Streams straight from Hugging Face to the volume (needs outbound internet from serverless).
# MAGIC * `max_trips` lets you download a subset (e.g. `10`) for a quick dry run. Default: all 100.
# MAGIC
# MAGIC The dataset is published by Renumics under **CC BY-SA 4.0** (joint work of Volkswagen AG Group
# MAGIC Innovation, IPG Automotive and Renumics). See `NOTICE.md`.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

dbutils.widgets.text("max_trips", "100", "Max trips to download (1-100)")
dbutils.widgets.text("workers", "16", "Parallel downloads")
max_trips = int(dbutils.widgets.get("max_trips"))
workers = int(dbutils.widgets.get("workers"))

dest = f"{RAW_VOLUME_PATH}/{HF_REPO.split('/')[-1]}"
print("destination:", dest)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SHARED_SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {SHARED_SCHEMA}.{RAW_VOLUME_NAME} "
          "COMMENT 'Raw Hugging Face download of renumics/synthetic-vehicle-fleet-telemetry (Impulse workshops)'")

# COMMAND ----------

# DBTITLE 1,List the files of the pinned dataset revision
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


import urllib.error


def hf_open(url, timeout=60, attempts=8):
    """urlopen with back-off on Hugging Face rate limits (HTTP 429) and transient 5xx errors."""
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 1))
            print(f"HTTP {e.code} from Hugging Face, retrying in {wait:.0f} s", flush=True)
            time.sleep(min(wait, 120))


def list_files():
    """All files of the dataset at HF_REVISION (follows the API's pagination)."""
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}?recursive=true"
    files = []
    while url:
        with hf_open(url) as r:
            files += [f for f in json.load(r) if f["type"] == "file"]
            link = r.headers.get("Link", "")
        url = link.split(";")[0].strip("<>") if 'rel="next"' in link else None
    return files


files = list_files()

# Optionally keep only the first `max_trips` trip directories (dimension + doc files are always kept).
trip_hashes = sorted({f["path"].split("trip_hash=")[1].split("/")[0] for f in files if "trip_hash=" in f["path"]})
keep = set(trip_hashes[:max_trips])
files = [f for f in files if "trip_hash=" not in f["path"] or f["path"].split("trip_hash=")[1].split("/")[0] in keep]

total = sum(f["size"] for f in files)
print(f"{len(files)} files, {total / 1e9:.2f} GB ({len(keep)} of {len(trip_hashes)} trips)")

# COMMAND ----------

# DBTITLE 1,Download (parallel, resumable)
def download(f):
    target = os.path.join(dest, f["path"])
    if os.path.exists(target) and os.path.getsize(target) == f["size"]:
        return f["size"]  # already there
    os.makedirs(os.path.dirname(target), exist_ok=True)
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{f['path']}"
    for attempt in range(5):
        try:
            tmp = target + ".part"
            with hf_open(url, timeout=120) as r, open(tmp, "wb") as out:
                while chunk := r.read(8 << 20):
                    out.write(chunk)
            if os.path.getsize(tmp) != f["size"]:
                raise IOError(f"size mismatch for {f['path']}")
            os.replace(tmp, target)
            return f["size"]
        except Exception as e:  # retry with back-off
            print(f"retry {attempt + 1} {f['path']}: {e}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {f['path']}")


start, done, failed = time.time(), 0, []
with ThreadPoolExecutor(workers) as pool:
    futures = {pool.submit(download, f): f for f in files}
    for i, fut in enumerate(as_completed(futures), 1):
        try:
            done += fut.result()
        except Exception as e:
            failed.append(futures[fut]["path"])
            print(e, flush=True)
        if i % 50 == 0 or i == len(files):
            print(f"{i}/{len(files)} files, {done / 1e9:.2f}/{total / 1e9:.2f} GB, "
                  f"{(time.time() - start) / 60:.1f} min", flush=True)

if failed:
    raise RuntimeError(f"{len(failed)} files failed (re-run to resume): {failed[:10]}")
print(f"✅ done — dataset root: {RAW_DATASET_PATH}")

# COMMAND ----------

display(dbutils.fs.ls(RAW_DATASET_PATH))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"files": len(files), "gb": round(total / 1e9, 2), "trips": len(keep)}))
