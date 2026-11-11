# Databricks notebook source
# MAGIC %md
# MAGIC # ⚙️ Workshop CONFIG
# MAGIC
# MAGIC Every notebook in this repo runs this file first with `%run ../CONFIG`, so **all names and
# MAGIC locations live in one place**.
# MAGIC
# MAGIC | Who | What to do |
# MAGIC |---|---|
# MAGIC | **Participant** | Set **`USER_SCHEMA`** below to your own schema (`catalog.schema`). Nothing else. |
# MAGIC | **Instructor** | Check the *shared locations* section matches your workspace (see `instructor_setup/`), incl. `IMPULSE_INSTALL_MODE` (`wheel` or `repo`). |
# MAGIC
# MAGIC > 💡 When notebooks run as a **job** (e.g. deployed with the Databricks Asset Bundle), a
# MAGIC > `user_schema` job parameter overrides the value below, so the file never has to be edited for
# MAGIC > automated runs.

# COMMAND ----------

# =============================================================================================
# 1. YOUR SCHEMA  ← the only value a participant has to change
# =============================================================================================
# Your personal schema, as "catalog.schema". Gold tables, your own silver tables (workshop 2) and
# your dashboards' data all go here. Example: "dbx_workshop_dev.jane_doe"
USER_SCHEMA = ""

# =============================================================================================
# 2. SHARED, INSTRUCTOR-MANAGED LOCATIONS (participants: do not change)
# =============================================================================================
# The single, workspace-wide clone of the Impulse library (a Databricks Git folder created by the
# instructor -- see instructor_setup/00_instructor_guide). It is the source of the wheel, the Genie
# Code skills, and (in "repo" mode) the code the notebooks import.
IMPULSE_REPO_PATH = "/Workspace/Shared/impulse"
IMPULSE_VERSION = "v0.11.0"  # the release tag the workshops were tested with

# How notebooks get Impulse:
#   "wheel" -> %pip install the wheel the instructor built into the UC volume below (01b_build_impulse_wheel)
#   "repo"  -> import straight from the shared Git folder (IMPULSE_REPO_PATH/src), nothing installed
IMPULSE_INSTALL_MODE = "wheel"
LIBS_VOLUME_NAME = "impulse_libs"   # volume (in the shared schema) holding the Impulse wheel

# Where the shared bronze + silver tables live.
SHARED_CATALOG = "dbx_workshop_dev"
SHARED_SCHEMA_NAME = "shared"

# Unity Catalog volume (in the shared schema) holding the raw Hugging Face download.
RAW_VOLUME_NAME = "impulse_raw"

# SQL warehouse for the AI/BI dashboards. Leave empty to auto-pick a running/serverless one.
WAREHOUSE_ID = ""

# =============================================================================================
# 3. DATASET  (renumics/synthetic-vehicle-fleet-telemetry on Hugging Face, CC BY-SA 4.0)
# =============================================================================================
HF_REPO = "renumics/synthetic-vehicle-fleet-telemetry"
HF_REVISION = "017ec8f02e0753d8b092cf9229e1372d92a7b635"  # pinned commit -> reproducible data

# The dataset has 681 signals. The bronze signal table keeps this curated set (what the labs use),
# which keeps bronze/silver small enough for an interactive workshop. Set to ["ALL"] for everything.
BRONZE_SIGNALS = [
    # --- vehicle motion -------------------------------------------------------------------
    "Vhcl.v",                       # vehicle speed [m/s]  (spikes > 2000 m/s in 3 trips -> drop > 100)
    "Car.Gen.vx_1",                 # longitudinal body velocity [m/s] (corroborating speed sensor)
    "Vhcl.Distance",                # odometer [m]
    "Car.ax", "Car.ay",             # longitudinal / lateral acceleration [m/s^2]
    "Vhcl.YawRate",                 # [rad/s]
    "Vhcl.Steer.Ang",               # steering angle [rad]
    "Vhcl.GearNo",                  # current gear (categorical)
    "Vhcl.PoI.GCS.Elev",            # elevation [m]
    "Vhcl.PoI.GCS.Lat",             # latitude  -- labelled deg, logged in RADIANS (x 180/pi)
    "Vhcl.PoI.GCS.Long",            # longitude -- labelled deg, logged in RADIANS (x 180/pi)
    "Car.WheelSpd_FL",              # wheel speed front-left [rad/s]
    # --- driver inputs --------------------------------------------------------------------
    "DM.Gas", "DM.Brake",           # gas / brake pedal demand [0..1]
    "Brake.Pedal", "Brake.Park",
    # --- powertrain -----------------------------------------------------------------------
    "Vhcl.Engine.rotv",             # engine speed [rad/s]
    "PT.Motor.rotv",                # e-motor speed [rad/s]
    "PT.Motor.Trq",                 # e-motor torque [Nm]
    "PT.Motor.PwrElec",             # e-motor electrical power [W] (negative = regeneration)
    "PT.Motor.VoltBatt",            # e-motor battery-side voltage [V]
    "PT.Control.OperationState",    # off / idle / driving / stalling (categorical)
    "PT.Control.Ignition",          # off / accessory / on / start (categorical)
    "PT.Control.OperationError",    # 0 = no error, 1 = error
    # --- high-voltage battery -------------------------------------------------------------
    "PT.BCU.BattHV.SOC",            # state of charge [%] (0 while the BMS initialises -> ignore <= 0)
    "PT.BattHV.Current",            # [A]  (negative = charging / regen)
    "PT.BattHV.Temp",               # labelled degC, logged in KELVIN (- 273.15)
    "PT.BattHV.Volt_oc",            # open-circuit voltage [V]
    "PT.BattHV.Energy",             # energy content [J]
    "PT.BattHV.Pwr_max",            # max available power [W]
    "PT.BCU.BattHV.TempCool_in",    # coolant inlet temperature [degC]
    "PT.Control.Consump.Elec.Act",  # actual electric consumption [kWh]
    # --- environment / losses / diagnostics -----------------------------------------------
    "Env.Temperature",              # mixed units across trips (degC / K) -> not used for analysis
    "Env.AirPressure",
    "PwrL.Total", "PwrL.Brake", "PwrL.Aero",
    "Log.nError", "Log.nWarn",      # diagnostic counters -> become Points-in-Time events in silver
    "Traffic.nObjs",
]

# How the labs handle the known data quirks (see DATA_NOTES.md). Stored data is never modified;
# every quirk is handled in the TSAL signal definition.
VHCL_V_MAX_MPS = 100.0    # Vhcl.v: values above this are spikes (benchmark DQ-01) -> .where(v < 100)
KELVIN_OFFSET = 273.15    # PT.BattHV.Temp is logged in Kelvin -> temp - 273.15
RAD_TO_DEG = 57.29577951308232  # Vhcl.PoI.GCS.Lat/Long are logged in radians -> x 180/pi

# Trips (session_id) that workshop 2 / notebook 1 engineers from bronze -> silver by hand.
# Two of each scenario family -- incl. two combustion cars (94, 95) and trip 95, whose Vhcl.v sensor
# is corrupted -- keeps the hands-on transformation fast while still hitting every interesting case.
W2_ENGINEERING_TRIPS = [0, 19, 66, 79, 94, 95]

# =============================================================================================
# 4. DERIVED NAMES  (no need to edit anything below)
# =============================================================================================
SHARED_SCHEMA = f"{SHARED_CATALOG}.{SHARED_SCHEMA_NAME}"
RAW_VOLUME_PATH = f"/Volumes/{SHARED_CATALOG}/{SHARED_SCHEMA_NAME}/{RAW_VOLUME_NAME}"
RAW_DATASET_PATH = f"{RAW_VOLUME_PATH}/{HF_REPO.split('/')[-1]}/assets"

TABLE_PREFIX = "impulse"

# Bronze: a faithful copy of the source star schema (+ ingestion metadata).
BRONZE = {
    "signal_data":   f"{SHARED_SCHEMA}.{TABLE_PREFIX}_bronze_signal_data",
    "signal_info":   f"{SHARED_SCHEMA}.{TABLE_PREFIX}_bronze_signal_info",
    "trip_info":     f"{SHARED_SCHEMA}.{TABLE_PREFIX}_bronze_trip_info",
    "vehicle_info":  f"{SHARED_SCHEMA}.{TABLE_PREFIX}_bronze_vehicle_info",
    "vehicle_state": f"{SHARED_SCHEMA}.{TABLE_PREFIX}_bronze_vehicle_state",
}

SILVER_TABLE_NAMES = [
    "container_metrics", "container_tags",
    "channel_metrics", "channel_tags", "channels",
    "poi_channels", "channel_mapping", "unit_conversion",
]


def silver_tables(schema: str, prefix: str = f"{TABLE_PREFIX}_silver") -> dict:
    """Fully-qualified names of the Impulse silver tables `<schema>.<prefix>_<table>`."""
    return {t: f"{schema}.{prefix}_{t}" for t in SILVER_TABLE_NAMES}


# Shared silver layer, built by the instructor and read by the labs.
SILVER = silver_tables(SHARED_SCHEMA)


def impulse_source(silver: dict, optional: bool = True) -> dict:
    """The Impulse `config["source"]` block for a set of silver tables.

    `optional=False` returns only the three tables the DefaultSolver strictly needs.
    """
    src = {
        "container_metrics_table": silver["container_metrics"],
        "channel_metrics_table": silver["channel_metrics"],
        "channels_uri": silver["channels"],
    }
    if optional:
        src.update({
            "container_tags_table": silver["container_tags"],
            "channel_tags_table": silver["channel_tags"],
            "poi_channels_uri": silver["poi_channels"],
            "channel_mapping_table": silver["channel_mapping"],
            "unit_conversion_table": silver["unit_conversion"],
        })
    return src


# Gold-table prefixes (written to USER_SCHEMA) and dashboard names.
W1_REPORT_NAME = "impulse_w1_fleet_report"
W1_GOLD_PREFIX = "impulse_w1"
W2_REPORT_NAME = "impulse_w2_fleet_deep_dive"
W2_GOLD_PREFIX = "impulse_w2"
W2_SILVER_PREFIX = "impulse_w2_silver"          # your hand-built silver (workshop 2, notebook 1)
W2_PIPELINE_SILVER_PREFIX = "impulse_w2_sdp"    # the declarative-pipeline version of the same


# =============================================================================================
# 5. HELPERS
# =============================================================================================
def _widget(name: str, default: str) -> str:
    """A job / widget parameter if one is set, else `default`."""
    try:
        value = dbutils.widgets.get(name)  # noqa: F821 - provided by the notebook
        return value.strip() or default
    except Exception:
        return default


USER_SCHEMA = _widget("user_schema", USER_SCHEMA)
IMPULSE_REPO_PATH = _widget("impulse_repo_path", IMPULSE_REPO_PATH)
IMPULSE_INSTALL_MODE = _widget("impulse_install_mode", IMPULSE_INSTALL_MODE).lower()
assert IMPULSE_INSTALL_MODE in ("wheel", "repo"), "IMPULSE_INSTALL_MODE must be 'wheel' or 'repo'"

# What the `%pip install $IMPULSE_PIP_TARGET` cell after `%run ../CONFIG` installs: the wheel (which pulls
# in its only extra dependency, lz4), or -- in repo mode -- just lz4.
LIBS_VOLUME_PATH = f"/Volumes/{SHARED_CATALOG}/{SHARED_SCHEMA_NAME}/{LIBS_VOLUME_NAME}"
IMPULSE_WHEEL_PATH = f"{LIBS_VOLUME_PATH}/databricks_impulse-{IMPULSE_VERSION.lstrip('v')}-py3-none-any.whl"
IMPULSE_PIP_TARGET = IMPULSE_WHEEL_PATH if IMPULSE_INSTALL_MODE == "wheel" else "lz4==4.4.5"

# The Impulse Genie Code skills (Markdown). Wheel mode: the copy 01b_build_impulse_wheel publishes next to the
# wheel (same version); repo mode: the skills/ folder of the shared Git folder.
IMPULSE_SKILLS_PATH = (f"{LIBS_VOLUME_PATH}/skills/{IMPULSE_VERSION}" if IMPULSE_INSTALL_MODE == "wheel"
                       else f"{IMPULSE_REPO_PATH}/skills")
WAREHOUSE_ID = _widget("warehouse_id", WAREHOUSE_ID)


def _find_repo_root():
    """Walk up from the running notebook to the folder holding databricks.yml (this repo's root)."""
    import os
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
    except Exception:
        return None
    d = os.path.dirname(nb if nb.startswith("/Workspace") else "/Workspace" + nb)
    while d not in ("/", "/Workspace", ""):
        if os.path.exists(os.path.join(d, "databricks.yml")):
            return d
        d = os.path.dirname(d)
    return None


REPO_ROOT = _find_repo_root()
if REPO_ROOT:
    import sys as _sys
    if REPO_ROOT not in _sys.path:
        _sys.path.insert(0, REPO_ROOT)  # makes `import common.<module>` work from any notebook


def current_user() -> str:
    return spark.sql("SELECT current_user()").first()[0]  # noqa: F821


def require_user_schema() -> str:
    """Validate USER_SCHEMA, create it if needed, and return it."""
    if not USER_SCHEMA or "." not in USER_SCHEMA:
        suggestion = f"{SHARED_CATALOG}." + current_user().split("@")[0].replace(".", "_").replace("-", "_")
        raise ValueError(
            "\n\n👉 Please open the CONFIG notebook at the root of this repo and set\n"
            f'       USER_SCHEMA = "{suggestion}"\n'
            "   (your own catalog.schema), then re-run this notebook.\n"
        )
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {USER_SCHEMA}")  # noqa: F821
    return USER_SCHEMA


def setup_impulse() -> None:
    """Make Impulse importable (`import impulse_query_engine` / `impulse_reporting`) in the configured mode."""
    import os
    import sys

    if IMPULSE_INSTALL_MODE == "wheel":
        from importlib import metadata
        try:
            version = metadata.version("databricks-impulse")
        except metadata.PackageNotFoundError:
            raise ImportError(
                f"The Impulse wheel is not installed. Run the `%pip install $IMPULSE_PIP_TARGET` cell first; if it "
                f"failed, ask your instructor to run instructor_setup/01b_build_impulse_wheel ({IMPULSE_WHEEL_PATH})."
            ) from None
        print(f"Impulse {version} installed from the wheel {IMPULSE_WHEEL_PATH}")
        return

    src = os.path.join(IMPULSE_REPO_PATH, "src")
    if not os.path.isdir(os.path.join(src, "impulse_query_engine")):
        raise FileNotFoundError(
            f"Impulse was not found at {IMPULSE_REPO_PATH}. Ask your instructor -- the shared Git "
            "folder is created in instructor_setup/00_instructor_guide."
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    print(f"Impulse imported from the shared Git folder {src}")


def save_table(df, name: str, attempts: int = 4) -> None:
    """Overwrite a Delta table, retrying if a concurrent transaction on it wins first.

    Two runs writing the same table at once (e.g. the same notebook open twice, or a job and you), or
    background maintenance on a freshly created table, can make Delta reject one commit with a
    ConcurrentModification / ConflictingMetadataDomain error. Retrying the overwrite is safe.
    """
    import time

    for attempt in range(attempts):
        try:
            df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(name)
            return
        except Exception as e:  # noqa: BLE001
            conflict = any(k in str(e) for k in ("CONFLICTING_METADATA_DOMAIN", "ConcurrentAppend",
                                                 "ConcurrentModification", "DELTA_CONCURRENT", "ConcurrentDeleteRead"))
            if not conflict or attempt == attempts - 1:
                raise
            print(f"concurrent write on {name}, retrying ({attempt + 1}/{attempts - 1}) ...")
            time.sleep(5 * (attempt + 1))


def pick_warehouse_id() -> str:
    """WAREHOUSE_ID if set, else the first running (or any serverless) SQL warehouse."""
    if WAREHOUSE_ID:
        return WAREHOUSE_ID
    from databricks.sdk import WorkspaceClient

    warehouses = list(WorkspaceClient().warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouse found -- set WAREHOUSE_ID in CONFIG.")
    rank = lambda w: (str(w.state) != "State.RUNNING", not w.enable_serverless_compute)
    return sorted(warehouses, key=rank)[0].id


print(f"Shared schema : {SHARED_SCHEMA}")
print(f"Impulse       : {IMPULSE_VERSION} via {IMPULSE_INSTALL_MODE} "
      f"({IMPULSE_WHEEL_PATH if IMPULSE_INSTALL_MODE == 'wheel' else IMPULSE_REPO_PATH})")
print(f"Your schema   : {USER_SCHEMA or '<not set yet -- edit CONFIG>'}")
