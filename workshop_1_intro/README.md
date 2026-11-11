# Workshop 1 · Introduction to Impulse

**~2 hours · no prior Impulse knowledge needed · Python basics helpful**

## Getting started (5 minutes)

1. **Get the repo into your workspace.** *Workspace* → your user folder → **Create → Git folder** → URL
   `https://github.com/blandfbt/impulse-workshops` → **Create**.
   (Your instructor may have shared a copy instead — either works.)
2. **Set your schema.** Open **`CONFIG`** at the repo root and set
   `USER_SCHEMA = "dbx_workshop_dev.<your_name>"`. Nothing else needs changing — the shared data
   (`dbx_workshop_dev.shared`) and the shared Impulse library (a wheel in `dbx_workshop_dev.shared.impulse_libs`,
   built from the Git folder `/Workspace/Shared/impulse`) are already there.
3. **Use Serverless compute.** In each notebook, pick *Serverless* in the compute selector (top right).
4. **Run top to bottom.** Read each markdown cell first — it explains *what* the next cell does and *why*.
   Cells with ⏳ take a minute or two because they process all 100 trips.

If the first code cell says *"Please open the CONFIG notebook…"*, step 2 is missing.

## Notebooks

| # | Notebook | Time | What you learn |
|---|---|---|---|
| 1 | [`01_intro_to_impulse_tsal`](01_intro_to_impulse_tsal.py) | 45 min | the silver data model, TSAL basics, signals → DataFrames for testing, why duration-weighting and alignment matter, a first (in-memory) report |
| 2 | [`02_genie_code_with_impulse_skills`](02_genie_code_with_impulse_skills.py) | 30 min | add the Impulse **skills** to Genie Code (copied from the shared volume; see them under ⋮ → Customizations), prompt Genie to write a TSAL query, check it against a reference |
| 3 | [`03_report_and_dashboard`](03_report_and_dashboard.py) | 45 min | a fleet report (events + histograms + statistics) written to **your** schema, SQL on the gold star schema, an **AI/BI dashboard** — by hand and generated |

## What you'll have at the end

* Gold tables `<your schema>.impulse_w1_*` (measurement/event/histogram/stats facts and dimensions)
* A published dashboard **"Impulse W1 - Fleet overview (<your schema>)"** in your home folder

## Good to know

* Data quirks (Kelvin temperatures, speed spikes, SOC = 0 at start-up, …) are listed in
  [`../DATA_NOTES.md`](../DATA_NOTES.md); the notebooks handle them in the signal definitions.
* All times in Impulse are **epoch microseconds**: 60 s = `60e6`.
