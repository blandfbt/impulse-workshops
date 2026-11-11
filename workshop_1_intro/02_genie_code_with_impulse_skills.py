# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1 · Notebook 2 — Genie Code + the Impulse skills
# MAGIC
# MAGIC **⏱ ~30 minutes · 🎯 Goal: teach Genie Code Impulse, then let it write a TSAL query for you.**
# MAGIC
# MAGIC [Genie Code](https://docs.databricks.com/aws/en/genie-code/) is the AI assistant built into the
# MAGIC Databricks notebook. Out of the box it knows Spark and SQL — but not Impulse, which is a young
# MAGIC library. **Skills** fix that: a skill is a folder with a `SKILL.md` file of instructions and examples
# MAGIC that Genie Code loads *only when relevant* to your request.
# MAGIC
# MAGIC The Impulse repository ships ten of them in its [`skills/`](https://github.com/databrickslabs/impulse/tree/main/skills)
# MAGIC folder. Your instructor made them available next to the shared Impulse library — in the
# MAGIC `impulse_libs` volume next to the wheel, or in the shared Git folder — so you just copy (or point to) them.
# MAGIC
# MAGIC | # | Section |
# MAGIC |---|---|
# MAGIC | 1 | Look at the skills |
# MAGIC | 2 | Add the skills to **your** Genie Code |
# MAGIC | 3 | Verify Genie Code sees them |
# MAGIC | 4 | Prompt Genie Code to write a TSAL query (hands-on) |
# MAGIC | 5 | Compare with a reference solution |

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

require_user_schema()
setup_impulse()
import os

SKILLS_DIR = IMPULSE_SKILLS_PATH          # from CONFIG: the volume copy (wheel mode) or the Git folder (repo mode)
print(f"Impulse skills ({IMPULSE_INSTALL_MODE} mode):", SKILLS_DIR)
assert os.path.isdir(SKILLS_DIR), f"No skills at {SKILLS_DIR} -- ask your instructor to run 01b_build_impulse_wheel"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · What's in the Impulse skills?
# MAGIC
# MAGIC Each sub-folder is one skill. The `description` in each `SKILL.md` front matter is what Genie Code
# MAGIC reads to decide *when* to load the skill; the body teaches it *how*.

# COMMAND ----------

import re

rows = []
for name in sorted(os.listdir(SKILLS_DIR)):
    path = os.path.join(SKILLS_DIR, name, "SKILL.md")
    if os.path.isfile(path):
        text = open(path).read()
        desc = re.search(r"description:\s*>?\s*(.+?)\n---", text, re.S)
        rows.append((name, " ".join(desc.group(1).split())[:180] + "…" if desc else ""))
display(spark.createDataFrame(rows, "skill string, what_it_teaches string"))

# COMMAND ----------

# MAGIC %md
# MAGIC Peek at the TSAL skill — it's plain Markdown with rules and runnable examples:

# COMMAND ----------

print(open(os.path.join(SKILLS_DIR, "impulse-tsal", "SKILL.md")).read()[:2500], "\n…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Add the skills to your Genie Code
# MAGIC
# MAGIC Genie Code reads skills from **workspace folders**: the `.assistant/skills/` folders (your personal one
# MAGIC and the workspace-wide one) plus any extra workspace folder you add under **Customizations**. Where
# MAGIC your copy of the skills comes from depends on how your instructor set up Impulse (`IMPULSE_INSTALL_MODE`
# MAGIC in CONFIG — the cell above printed the source):
# MAGIC
# MAGIC | Install mode | Skills source |
# MAGIC |---|---|
# MAGIC | `wheel` (default) | `/Volumes/dbx_workshop_dev/shared/impulse_libs/skills/v0.11.0/` — Markdown files published next to the wheel, same version |
# MAGIC | `repo` | `/Workspace/Shared/impulse/skills` — the shared Git folder |
# MAGIC
# MAGIC Pick **one** option:
# MAGIC
# MAGIC ### Option A — copy them into your personal skills folder (default; works in both modes)
# MAGIC The next cell copies the Impulse skills from the source above into `/Users/<you>/.assistant/skills/`,
# MAGIC where Genie Code finds them automatically. It's a snapshot of the instructor's version: re-run the cell
# MAGIC after an Impulse upgrade. (Set `COPY_TO_MY_SKILLS = False` to skip.)
# MAGIC
# MAGIC ### Option B — your instructor installed them workspace-wide
# MAGIC If `/Workspace/.assistant/skills/impulse*` exists (the next cell checks), you're already done and can
# MAGIC skip the copy.
# MAGIC
# MAGIC ### Option C (`repo` mode only) — point Genie Code at the shared Git folder, nothing copied
# MAGIC 1. Open the **Genie Code** pane (the ✨ assistant icon at the top right of the notebook).
# MAGIC 2. Click the **⋮** menu at the top of the pane → **Customizations**.
# MAGIC 3. In the **Skills** section, add the folder `/Workspace/Shared/impulse/skills` (type the path or
# MAGIC    browse) and make sure it's switched **on**. Add the `skills` **parent** folder, not a single skill.
# MAGIC
# MAGIC (You can also open the Customizations page to *see* which skills Genie Code has loaded after option A or B.)
# MAGIC
# MAGIC > After adding skills, **start a new Genie Code chat** — open chats don't pick up new skills. If the
# MAGIC > skill list looks stale, hard-refresh the browser tab.

# COMMAND ----------

import shutil

COPY_TO_MY_SKILLS = True   # Option A: copy the Impulse skills into your personal Genie Code skills folder

me = current_user()
workspace_wide = "/Workspace/.assistant/skills"
personal = f"/Workspace/Users/{me}/.assistant/skills"
impulse_skills = sorted(d for d in os.listdir(SKILLS_DIR)
                        if d.startswith("impulse") and os.path.isfile(os.path.join(SKILLS_DIR, d, "SKILL.md")))

if COPY_TO_MY_SKILLS:
    os.makedirs(personal, exist_ok=True)
    for name in impulse_skills:
        shutil.copytree(os.path.join(SKILLS_DIR, name), os.path.join(personal, name), dirs_exist_ok=True)
    print(f"Option A — copied {len(impulse_skills)} skills from {SKILLS_DIR}\n           to {personal}")

installed_me = os.path.isdir(personal) and all(os.path.isdir(os.path.join(personal, n)) for n in impulse_skills)
installed_ws = os.path.isdir(workspace_wide) and any(d.startswith("impulse") for d in os.listdir(workspace_wide))
print("In your personal skills folder :", "✅ yes" if installed_me else "no")
print("Installed workspace-wide       :", "✅ yes" if installed_ws else "no")
if IMPULSE_INSTALL_MODE == "repo":
    print("Option C — or add this folder under ⋮ → Customizations → Skills:", SKILLS_DIR.replace("/Workspace", "", 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Verify
# MAGIC
# MAGIC In a **new** Genie Code chat, ask:
# MAGIC
# MAGIC > *List the Impulse skills you can use.*
# MAGIC
# MAGIC You should see `impulse`, `impulse-tsal`, `impulse-analyze`, `impulse-reporting`, … You can also force
# MAGIC a specific skill into a conversation by **@-mentioning** it, e.g. `@impulse-tsal`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Hands-on: let Genie Code write a TSAL query
# MAGIC
# MAGIC **The question.** The fleet's EVs are charged in some trips. *For every trip, how long was the car
# MAGIC charging, and how many SOC percentage points did it gain while charging?*
# MAGIC
# MAGIC **SOC** (*state of charge*) = how full the high-voltage traction battery is, in **% of its usable capacity** (0 % empty, 100 % full) — the EV equivalent of a fuel gauge. Signal: `PT.BCU.BattHV.SOC`.
# MAGIC
# MAGIC A good definition of **charging** in this dataset (from its documentation): battery current
# MAGIC `PT.BattHV.Current` **below −10 A** while the car is **standing still** (`Vhcl.v` below 0.1 m/s), for
# MAGIC **more than 60 s**.
# MAGIC
# MAGIC ### Your turn
# MAGIC 1. Click into the **empty cell below** and open Genie Code (or press the ✨ in the cell).
# MAGIC 2. Paste the prompt from the box below. It gives Genie the context a colleague would need: where the
# MAGIC    data is, how the notebook is set up, what you want back.
# MAGIC 3. Read the code Genie proposes — does it use `channel(...)`, comparisons → `Intervals`, `.where(...)`?
# MAGIC 4. Run it. If it errors, paste the error back to Genie and let it fix it.
# MAGIC
# MAGIC ```text
# MAGIC Using the Impulse skills, write an ad-hoc TSAL query for this notebook.
# MAGIC Context: the variables `db` (an Impulse MeasurementDB) and `solver` (a DefaultSolver) already exist
# MAGIC (see the setup cell). Timestamps are epoch microseconds. Channels are selected by channel_name, e.g.
# MAGIC db.query.channel(channel_name="Vhcl.v").
# MAGIC Task: define "charging" as PT.BattHV.Current < -10 A AND Vhcl.v < 0.1 m/s, keeping only windows
# MAGIC longer than 60 seconds. For every trip return container_id, the charging windows, the total charging
# MAGIC time in minutes, and the SOC (PT.BCU.BattHV.SOC, in %) gained during charging (max - min of SOC
# MAGIC inside the windows). Data quirks: ignore SOC <= 0 and Vhcl.v values above 100 m/s.
# MAGIC Return a Spark DataFrame (use .solve, not .toPandas) sorted by charging time, only trips that charged.
# MAGIC ```
# MAGIC
# MAGIC 💡 **Prompting tips:** name the variables that already exist, state units, describe the output shape,
# MAGIC and say *"using the Impulse skills"* (or `@impulse-tsal`) so the right skill loads.

# COMMAND ----------

# DBTITLE 1,Setup used by the exercise (run me first)
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver

db = MeasurementDB(MeasurementDBConfig(**impulse_source(SILVER), table_locations="unity_catalog"), WorkspaceClient())
solver = DefaultSolver(spark)

# COMMAND ----------

# 👉 Ask Genie Code to fill this cell (see the prompt above), then run it.


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Reference solution
# MAGIC
# MAGIC Compare Genie's answer with this one. Different-looking code is fine as long as the logic is the same.
# MAGIC The key ideas:
# MAGIC
# MAGIC * Two comparisons give two `Intervals`; `&` intersects them — Impulse aligns the two signals (sampled
# MAGIC   at different instants) for you.
# MAGIC * `.filter(60e6)` drops windows shorter than 60 s (**time is in µs**).
# MAGIC * The data quirks from `DATA_NOTES.md` are handled in the signal definitions (speed < 100 m/s, SOC > 0).
# MAGIC   Tell Genie about them too — e.g. add *"ignore SOC <= 0 and Vhcl.v above 100 m/s"* to your prompt.
# MAGIC * `soc.where(charging)` keeps SOC only inside the windows; `.max() - .min()` reduces it per trip.
# MAGIC   (Strictly that's the *range* across all charging windows; good enough for one charge per trip.)

# COMMAND ----------

current = db.query.channel(channel_name="PT.BattHV.Current")
vhcl_v = db.query.channel(channel_name="Vhcl.v")
soc_raw = db.query.channel(channel_name="PT.BCU.BattHV.SOC")
speed = vhcl_v.where(vhcl_v < VHCL_V_MAX_MPS)                     # data quirks: see DATA_NOTES.md
soc = soc_raw.where(soc_raw > 0)

charging = ((current < -10) & (speed < 0.1)).filter(60e6)        # Intervals, each > 60 s
soc_charging = soc.where(charging)                                # SOC only while charging

charge = (
    db.query.select(
        charging.alias("charging_windows"),          # Intervals -> array<array<double>> [[start, end], ...]
        soc_charging.max().alias("soc_max_pct"),
        soc_charging.min().alias("soc_min_pct"),
    )
    .solve(spark, solver=solver)                      # Spark DataFrame, one row per trip
)

# Generic Spark from here: total window length with a higher-order function (µs -> minutes).
result = (
    charge
    .withColumn("charging_min", F.expr("aggregate(charging_windows, 0D, (acc, w) -> acc + (w[1] - w[0])) / 60e6"))
    .withColumn("soc_gained_pct", F.col("soc_max_pct") - F.col("soc_min_pct"))
    .where(F.col("charging_min") > 0)
    .join(spark.table(SILVER["container_metrics"]).select("container_id", "vehicle_key", "scenario"), "container_id")
    .select("container_id", "vehicle_key", "scenario",
            F.round("charging_min", 2).alias("charging_min"), F.round("soc_gained_pct", 2).alias("soc_gained_pct"))
    .orderBy(F.desc("charging_min"))
)
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stretch prompts to try with Genie Code
# MAGIC
# MAGIC * *"Using @impulse-tsal: for each trip, find regenerative braking — PT.Motor.PwrElec below 0 W while
# MAGIC   Vhcl.v > 1 m/s — and return the total regen time in seconds."*
# MAGIC * *"Using @impulse-analyze: plot Vhcl.v and PT.BattHV.Current for trip 0 on two subplots with matplotlib."*
# MAGIC * *"Using @impulse-events: which event type should I use to split every trip into 5-minute windows?"*
# MAGIC
# MAGIC ## ✅ Recap
# MAGIC * Skills are Markdown instructions Genie Code loads on demand; Impulse ships ten.
# MAGIC * Register the **shared** `skills` folder once — one copy for the whole workshop, always in sync.
# MAGIC * Give Genie context (variables, units, output shape) and verify its TSAL by running it.
# MAGIC
# MAGIC **Next → `03_report_and_dashboard`**: write a real report into your schema and chart it.
