# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2 · Notebook 2a — TSAL deep dive
# MAGIC
# MAGIC **⏱ ~75 minutes · 🎯 Goal: master the TSAL type system, combine signals, use every optional silver
# MAGIC table, convert units, and write your own UDFs — all on real trips.**
# MAGIC
# MAGIC | # | Section | Highlights |
# MAGIC |---|---|---|
# MAGIC | 1 | Setup | a `MeasurementDB` with **all** optional tables |
# MAGIC | 2 | The four result types | `SampleSeries`, `Intervals`, `PointsInTime`, `PointsInTimeSeries` — with real examples |
# MAGIC | 3 | Combining series | multi-rate alignment, derived physics, interval algebra |
# MAGIC | 4 | Selecting trips and channels | container tags, metrics, channel tags |
# MAGIC | 5 | Aliases & unit conversion | `channel_mapping` + `unit_conversion`, fallbacks, K → °C, rad → ° (route map) |
# MAGIC | 6 | Points-in-Time channels | numeric + string event streams, windows around events |
# MAGIC | 7 | Signal processing | resample, rolling statistics, integration |
# MAGIC | 8 | Custom UDFs | the rules, a metadata-aware UDF, a scalar UDF, validating against native |
# MAGIC | 9 | 🔍 Explore interactively | widgets: pick a trip, signals and a condition |
# MAGIC
# MAGIC We stay with the **`DefaultSolver`** throughout. Notebook 2b turns the best of this into a report.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

require_user_schema()
setup_impulse()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver
from common.workshop_helpers import plot_signals, intervals_to_pandas, points_to_pandas, US

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Setup — every silver table, including the optional ones
# MAGIC
# MAGIC `impulse_source(SILVER)` (from CONFIG) now hands Impulse all eight tables:
# MAGIC
# MAGIC | Optional table | Unlocks |
# MAGIC |---|---|
# MAGIC | `container_tags` | `havingTag(...)`, `query.tag(...)`, `tag_filters`, tags injected into UDFs |
# MAGIC | `channel_tags` | `channel(<any tag>=...)` |
# MAGIC | `poi_channels` | `poi_channel(...)` → event streams |
# MAGIC | `channel_mapping` | `channel_with_alias(...)` → logical names with priority fallback |
# MAGIC | `unit_conversion` | automatic unit conversion of aliased channels |

# COMMAND ----------

db = MeasurementDB(MeasurementDBConfig(**impulse_source(SILVER), table_locations="unity_catalog"), WorkspaceClient())
solver = DefaultSolver(spark)
trips_sdf = spark.table(SILVER["container_metrics"])                 # Spark: for joins with Impulse results
trips = trips_sdf.toPandas().set_index("container_id")              # pandas: quick lookups for labels


def solve_trip(trip_id: int, **expressions) -> pd.Series:
    """Solve named TSAL expressions for ONE trip and return them as a pandas row.

    Fleet-wide queries in this notebook use `.solve(...)` -> Spark DataFrame. This helper uses
    `.toPandas(...)` on purpose: it returns signals/intervals as Python objects (SampleSeries, ...)
    that we inspect and plot -- in a Spark DataFrame a signal is a compressed binary column.
    """
    q = db.query
    selected = [expr.alias(name) for name, expr in expressions.items()]
    return q.where(q.metric("container_id") == trip_id).select(*selected).toPandas(spark, solver=solver).iloc[0]


TRIP = 7   # Volvo XC90 T8 plug-in hybrid, mountain drive: an 8-speed gearbox AND an e-motor with regeneration
print(trips.loc[TRIP, ["vehicle_key", "scenario", "driver", "duration_s"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · The four result types
# MAGIC
# MAGIC Every TSAL expression evaluates — per trip — to exactly one of four types. The type decides what you
# MAGIC can do next, and which event / aggregation accepts it.
# MAGIC
# MAGIC |  | carries values | has duration | typical source | spark column in results |
# MAGIC |---|:---:|:---:|---|---|
# MAGIC | **`SampleSeries`** | ✅ | ✅ | `channel()`, arithmetic | binary (deserialized in pandas) |
# MAGIC | **`Intervals`** | ❌ | ✅ | comparisons, `&`, `\|` | `array<array<double>>` `[[start, end], …]` |
# MAGIC | **`PointsInTime`** | ❌ | ❌ | edges, `start_points()` | `array<double>` `[t, …]` |
# MAGIC | **`PointsInTimeSeries`** | ✅ | ❌ | `poi_channel()`, `signal.where(points)` | `array<array<double>>` `[[t, value], …]` |
# MAGIC
# MAGIC ```
# MAGIC                 > < == …                 start_points() / edges
# MAGIC  SampleSeries ───────────────► Intervals ─────────────────────────► PointsInTime
# MAGIC      │  ▲                          │  ▲   expand(w)                     │
# MAGIC      │  └─── .where(Intervals) ────┘  └─────────────────────────────────┘
# MAGIC      └──────────── .where(PointsInTime) ──────────────► PointsInTimeSeries ── > < == ──► PointsInTime
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a · `SampleSeries` — a signal
# MAGIC Three aligned numpy arrays. Each sample is valid on `[tstart, tend)`. Consecutive samples are
# MAGIC *continuous* only if `tstarts[i+1] == tends[i]`; anything else is a **gap** — and Impulse never
# MAGIC interpolates across gaps.

# COMMAND ----------

vhcl_v = db.query.channel(channel_name="Vhcl.v")
speed = vhcl_v.where(vhcl_v < VHCL_V_MAX_MPS)      # DATA_NOTES.md: drop spikes > 100 m/s (used throughout)
r = solve_trip(TRIP, speed=speed)
s = r.speed
dt = np.diff(s.tstarts) / 1e3
print(f"{len(s):,} intervals | median sample period {np.median(dt):.1f} ms | max {dt.max():.0f} ms")
print(f"continuous segments (gap-free stretches): {len(s.continuous_interval_indices)}")
print(f"duration-weighted mean {s.mean():.2f} m/s vs plain mean of values {s.values.mean():.2f} m/s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b · `Intervals` — *when* something is true
# MAGIC
# MAGIC **Goal: find the moments the car recovers energy by regenerative braking.** In regen the e-motor works as a
# MAGIC generator: its electrical power `PT.Motor.PwrElec` turns **negative** while the car is still moving.
# MAGIC A comparison such as `(motor_power < 0) & (speed > 1)` does not return values; it returns **`Intervals`**,
# MAGIC a list of `[start, end)` windows during which the condition holds.
# MAGIC
# MAGIC **The catch: the raw condition fires constantly.** Motor power hovers around zero whenever the driver lifts
# MAGIC off, coasts or makes small pedal corrections, so it dips below 0 for a split second, again and again. On
# MAGIC trip 7 the raw condition gives **273 windows with a median length of 0.03 s**, and **73 % of them are
# MAGIC shorter than 1 s**. These are noise, not braking. Counted as regen events, they would swamp the
# MAGIC fleet statistics with chatter.
# MAGIC
# MAGIC So we **clean the windows** before using them, in two steps:
# MAGIC
# MAGIC ```
# MAGIC raw        ▌▌ ▌▌▌ ▌   ▌        ▌▌▌▌▌ ▌▌▌▌▌▌ ▌▌▌▌▌▌▌▌      ▌     273 windows, mostly < 1 s
# MAGIC merge  2 s ████████   ▌        ████████████████████      ▌      88 windows: neighbours < 2 s apart joined
# MAGIC filter 5 s ████████            ████████████████████             40 events: only those ≥ 5 s remain
# MAGIC ```
# MAGIC
# MAGIC 1. **`.merge_intervals(2e6)`**: one real regen phase often flickers (power briefly crosses 0 and comes
# MAGIC    back). Windows separated by gaps **shorter than 2 s** are joined, so one deceleration counts as
# MAGIC    **one event** rather than many fragments. Trip 7: 273 → **88** windows.
# MAGIC 2. **`.filter(5e6)`**: after merging, windows **shorter than 5 s** are dropped. What is left are the
# MAGIC    **significant** regen events, sustained decelerations such as a descent on the mountain pass or braking
# MAGIC    for a hairpin. Trip 7: 88 → **40 events, median 16 s**.
# MAGIC
# MAGIC Removing those 233 tiny windows loses almost nothing. The 40 significant events still cover **885 s of the
# MAGIC 928 s** of raw regen time (95 %). The short windows were numerous but carried very little time.
# MAGIC
# MAGIC Order matters: merge **first**, so the flickering edges of a long regen phase are kept as part of it. Filter
# MAGIC first and those fragments are thrown away before they can be joined (on trip 7: same 40 events, but 23 s
# MAGIC less regen time).
# MAGIC
# MAGIC The thresholds (2 s, 5 s) are **analysis choices**, not properties of the data; tune them to the question.
# MAGIC The printout below shows the before/after counts and median durations.
# MAGIC
# MAGIC Other interval-cleaning methods, all with durations in **µs** (`1e6` = 1 s):
# MAGIC
# MAGIC * `.filter(d)` drops windows shorter than `d` · `.merge_intervals(gap)` merges windows closer than `gap`
# MAGIC * `.expand(w)` / `.shrink(w)` grow / shrink both sides · `.debounce(d)` suppresses chatter

# COMMAND ----------

motor_power = db.query.channel(channel_name="PT.Motor.PwrElec")         # [W], negative = generating

# Raw condition -> Intervals: every moment the motor generates while the car moves (> 1 m/s, so a
# standing car with tiny sensor noise doesn't count). Trip 7: 273 windows, median 0.03 s.
regen_raw = (motor_power < 0) & (speed > 1)

# Significant regen events only:
#   1. merge_intervals(2e6): join windows separated by < 2 s -> one deceleration = one event
#   2. filter(5e6)         : then drop events shorter than 5 s -> only sustained energy recovery remains
regen = regen_raw.merge_intervals(2e6).filter(5e6)

r = solve_trip(TRIP, speed=speed, motor_power=motor_power, regen_raw=regen_raw, regen=regen)
t0 = r.speed.tstarts[0]
raw_w, clean_w = intervals_to_pandas(r.regen_raw, t0), intervals_to_pandas(r.regen, t0)
print(f"raw regen windows        : {len(raw_w):4d}  (median {raw_w.duration_s.median():.2f} s)")
print(f"significant regen events : {len(clean_w):4d}  (median {clean_w.duration_s.median():.1f} s)")
clean_w.sort_values("duration_s", ascending=False).head()   # the longest significant events

# COMMAND ----------

fig = plot_signals({"speed [m/s]": r.speed, "motor power [W]": r.motor_power}, intervals=r.regen,
                   title=f"Trip {TRIP}: regeneration windows (shaded) after merge + filter")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c · `PointsInTime` — instants
# MAGIC The **start of every brake application** is the start point of each "brake pressed" interval.
# MAGIC `rising_edges()` / `falling_edges()` give instants where a signal goes up / down — e.g. upshifts.

# COMMAND ----------

brake = db.query.channel(channel_name="DM.Brake")
gear = db.query.channel(channel_name="Vhcl.GearNo")
brake_applied = (brake > 0.2).filter(0.5e6).start_points()      # PointsInTime: each firm brake application
upshifts = gear.rising_edges()                                  # PointsInTime: each time the gear number rises

r = solve_trip(TRIP, speed=speed, gear=gear, brake=brake, brake_applied=brake_applied, upshifts=upshifts)
print(f"{len(r.brake_applied)} brake applications, {len(r.upshifts)} upshifts")
fig = plot_signals({"speed [m/s]": r.speed, "gear": r.gear, "brake": r.brake},
                   points={"brake applied": r.brake_applied}, title="Brake applications as PointsInTime")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d · `PointsInTimeSeries` — values at instants
# MAGIC Sampling a signal **at** instants keeps a value but no duration: *"how fast was the car each time
# MAGIC the driver hit the brake?"* Reducers on a `PointsInTimeSeries` are **unweighted** (every instant counts once).

# COMMAND ----------

speed_kmh = speed * 3.6
speed_at_braking = speed_kmh.where(brake_applied)                # PointsInTimeSeries
r = solve_trip(TRIP, speed_at_braking=speed_at_braking, mean_brake_speed=speed_at_braking.mean())
pts = points_to_pandas(r.speed_at_braking, "speed_kmh", t0=t0)
print(f"mean speed at brake application: {r.mean_brake_speed:.1f} km/h (unweighted over {len(pts)} instants)")
pts.describe().round(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Combining series
# MAGIC
# MAGIC ### 3a · Alignment is automatic
# MAGIC `PT.Motor.VoltBatt` and `PT.BattHV.Current` come from different sensors with **different timestamps**.
# MAGIC Both log at roughly 100 Hz (every ~10–11 ms), but on their own clocks, so their samples almost never
# MAGIC land at the same instant. In SQL you'd now have to write an as-of join; in TSAL you just write
# MAGIC `voltage * current`. Impulse first **synchronizes** them onto shared validity intervals (a new interval
# MAGIC wherever *either* input changes, each input's value held until its next sample), then multiplies.
# MAGIC The result covers only their overlap; nothing is interpolated.

# COMMAND ----------

voltage = db.query.channel(channel_name="PT.Motor.VoltBatt")      # [V]
current = db.query.channel(channel_name="PT.BattHV.Current")      # [A], positive = discharging
battery_power_kw = voltage * current / 1000                       # derived physics: P = U·I

r = solve_trip(TRIP, voltage=voltage, current=current, battery_power_kw=battery_power_kw,
               motor_power_kw=motor_power / 1000)
v_t = np.asarray(r.voltage.tstarts, dtype=float)       # interval start times [µs] of each signal
c_t = np.asarray(r.current.tstarts, dtype=float)
p_t = np.asarray(r.battery_power_kw.tstarts, dtype=float)
shared = np.intersect1d(v_t, c_t).size
print(f"voltage: {len(v_t):,} intervals | current: {len(c_t):,} | product: {len(p_t):,}")
print(f"start times shared by voltage and current: {shared:,}  ->  product = union of both time bases")

fig = plot_signals({"voltage [V]": r.voltage, "current [A]": r.current,
                    "battery P=U·I [kW]": r.battery_power_kw, "motor power [kW]": r.motor_power_kw},
                   title="Battery-side vs motor-side power (two independent measurements agree)")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Zoom in: two clocks, one result
# MAGIC Over a whole trip the curves look perfectly aligned. Zoom in to **0.1 s** and the separate time bases
# MAGIC appear. The cell below picks the moment where the voltage swings the most, and shows:
# MAGIC
# MAGIC * **Plot:** each dot is the start of a new interval. Voltage (blue) and current (orange) tick at
# MAGIC   *different* instants. The product (bottom) gets a new interval at **every** tick of **either** input;
# MAGIC   the dotted lines show which input caused it.
# MAGIC * **Table:** the product's intervals in that window: which input changed, the voltage and current value
# MAGIC   **held** at that moment (zero-order hold: valid until the next sample), and the product Impulse computed.
# MAGIC
# MAGIC That is why the product has about as many intervals as voltage and current **combined**. Voltage has
# MAGIC fewer intervals than current because of RLE: while the voltage holds steady, it stays a single interval.

# COMMAND ----------

import matplotlib.pyplot as plt

# Pick a 0.1 s window where both signals visibly move: among the stretches where the voltage changes 8 times
# within 0.1 s, take the one with the biggest voltage swing (a hard acceleration or regen pulls the pack voltage).
v_v = np.asarray(r.voltage.values, dtype=float)
busy = np.where(v_t[8:] - v_t[:-8] < 0.1 * US)[0]
best = busy[np.argmax(np.abs(v_v[busy + 8] - v_v[busy]))]
lo = v_t[best] - 0.01 * US
hi = lo + 0.1 * US

fig, axes = plt.subplots(3, 1, figsize=(13, 6), sharex=True)
for ax, (label, series, color) in zip(axes, [("voltage [V]", r.voltage, "C0"), ("current [A]", r.current, "C1"),
                                             ("U·I [kW]", r.battery_power_kw, "C2")]):
    t = np.asarray(series.tstarts, dtype=float)
    m = (t >= lo - 0.02 * US) & (t < hi)                                 # include the value held at `lo`
    x = (t[m] - lo) / 1e3                                                # ms since window start
    ax.step(x, np.asarray(series.values)[m], where="post", color=color, lw=1)
    ax.plot(x, np.asarray(series.values)[m], "o", color=color, ms=4)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    ax.grid(alpha=0.3)
for tv in v_t[(v_t >= lo) & (v_t < hi)]:                                 # which input caused each product step
    axes[2].axvline((tv - lo) / 1e3, color="C0", ls=":", lw=0.8)
for tc in c_t[(c_t >= lo) & (c_t < hi)]:
    axes[2].axvline((tc - lo) / 1e3, color="C1", ls=":", lw=0.8)
axes[-1].set_xlim(0, 100)
axes[-1].set_xlabel(f"ms after t = {(lo - t0) / US:.2f} s")
fig.suptitle("100 ms zoom: voltage and current tick on their own clocks; the product steps on both")
fig.tight_layout()

# Timeline of the product's intervals in the window, with the input values held at each start.
in_win = (p_t >= lo) & (p_t < hi)
t = p_t[in_win]
iv = np.searchsorted(v_t, t, side="right") - 1                          # last voltage sample at or before t
ic = np.searchsorted(c_t, t, side="right") - 1                          # last current sample at or before t
pd.DataFrame({
    "t [ms]": ((t - lo) / 1e3).round(2),
    "changed": np.where(np.isin(t, v_t), np.where(np.isin(t, c_t), "both", "voltage"), "current"),
    "voltage [V] (held)": np.asarray(r.voltage.values)[iv].round(3),
    "current [A] (held)": np.asarray(r.current.values)[ic].round(2),
    "U·I [kW] (Impulse)": np.asarray(r.battery_power_kw.values)[in_win].round(3),
})

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b · Interval algebra — build precise conditions from simple ones
# MAGIC *"Hard braking at highway speed while the battery is cold"* is three simple conditions combined. `&` is
# MAGIC intersection, `|` union; `Intervals & PointsInTime` keeps only the instants inside the windows.

# COMMAND ----------

batt_temp = db.query.channel(channel_name="PT.BattHV.Temp") - KELVIN_OFFSET   # logged in Kelvin -> °C
accel = db.query.channel(channel_name="Car.ax")

hard_braking = (accel < -3).filter(0.3e6)
highway = speed_kmh > 80
cool_battery = batt_temp < 30
critical = hard_braking & highway & cool_battery
braking_on_highway_instants = highway & brake_applied             # Intervals & PointsInTime -> PointsInTime

r = solve_trip(TRIP, hard=hard_braking, highway=highway, critical=critical, hw_brakes=braking_on_highway_instants)
for k in ["hard", "highway", "critical"]:
    w_ = intervals_to_pandas(r[k])
    print(f"{k:9s}: {len(w_):4d} windows, {w_.duration_s.sum():8.1f} s")
print(f"brake applications above 80 km/h: {len(r.hw_brakes)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Selecting trips and channels
# MAGIC
# MAGIC Filtering *which trips* to process is the biggest performance lever: Impulse resolves trips and
# MAGIC channels from the small metadata tables first and only then reads samples.
# MAGIC
# MAGIC | Filter | Reads | Example |
# MAGIC |---|---|---|
# MAGIC | `havingTag(k=v)` | `container_tags` | `havingTag(powertrain_type="ev")` |
# MAGIC | `where(q.tag(k) == v)` | `container_tags` | `q.tag("traffic", cast_type="int") >= 8` |
# MAGIC | `where(q.metric(col) > x)` | `container_metrics` columns | `q.metric("duration_s") > 2500` |
# MAGIC | `channel(k=v, …)` | `channel_tags` | `channel(channel_name="PT.BattHV.Current", unit="A")` |

# COMMAND ----------

q = db.query
ev_aggressive = (
    q.havingTag(powertrain_type="ev")
     .where(q.tag("driver") == "aggressive", q.metric("duration_s") > 1500)
     .select(db.query.channel(channel_name="PT.BattHV.Current", unit="A").max().alias("peak_current_a"),
             batt_temp.max().alias("max_batt_temp_c"))
     .solve(spark, solver=solver)
)
display(ev_aggressive.join(trips_sdf.select("container_id", "vehicle_key", "scenario", "driver", "duration_s"),
                           "container_id").orderBy(F.desc("peak_current_a")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Aliases and unit conversion (`channel_mapping` + `unit_conversion`)
# MAGIC
# MAGIC The mapping and the unit families the instructor loaded:

# COMMAND ----------

display(spark.table(SILVER["channel_mapping"]))
display(spark.table(SILVER["unit_conversion"]).orderBy("group_id", F.desc("is_base")))

# COMMAND ----------

# MAGIC %md
# MAGIC #### How an alias is resolved
# MAGIC
# MAGIC With `channel_with_alias(channel_alias="drive_unit_speed")` you ask for a **logical** name, not a physical
# MAGIC channel. For **each trip**, Impulse then does three things:
# MAGIC
# MAGIC 1. **Find the candidates.** It takes every `channel_mapping` row for the alias and joins it to that trip's
# MAGIC    `channel_metrics` on `source_channel = channel_name` (and `data_key`). Only physical channels that
# MAGIC    **exist in this trip** survive the join.
# MAGIC 2. **Pick one.** Among the survivors it keeps the **lowest `priority`**. So priority 1 is the preferred
# MAGIC    source, and priority 2 is the fallback used only when priority 1 is missing.
# MAGIC 3. **Convert the unit.** It reads the physical channel's unit from `channel_metrics.unit` (falling back
# MAGIC    to `channel_mapping.source_unit`), looks up both units in `unit_conversion` and multiplies every value
# MAGIC    by `factor(source) / factor(target)`. The factors convert each unit to its family's base unit:
# MAGIC
# MAGIC | alias | source → target | factors (to base unit) | multiplier |
# MAGIC |---|---|---|---|
# MAGIC | `drive_unit_speed` | rad/s → rpm | rad/s = 1, rpm = 2π/60 = 0.1047 | 1 / 0.1047 = **9.549** |
# MAGIC | `vehicle_speed` | m/s → km/h | m/s = 1, km/h = 1/3.6 | **3.6** |
# MAGIC | `battery_energy` | J → kWh | J = 1, kWh = 3.6·10⁶ | **1 / 3.6·10⁶** |
# MAGIC
# MAGIC The analyst never needs to know which signal a vehicle has or which unit it logs. The query reads the
# MAGIC same on every trip.
# MAGIC
# MAGIC #### The priority fallback in action
# MAGIC
# MAGIC `drive_unit_speed` means "the speed of whatever drives the car". It has two mapping rows:
# MAGIC
# MAGIC * priority **1** → `PT.Motor.rotv`, the e-motor;
# MAGIC * priority **2** → `Vhcl.Engine.rotv`, the combustion engine.
# MAGIC
# MAGIC The e-motor channel only exists in the **63 electrified trips** (EVs and hybrids). Those trips resolve to
# MAGIC the motor. Combustion (`generic`) cars have no motor channel, so step 2 falls back to their engine. Both
# MAGIC arrive in **rpm**, so a single column compares drive-unit speed across the whole fleet.
# MAGIC
# MAGIC (A hybrid has *both* channels; priority 1 makes it resolve to the e-motor. If you'd rather have the engine
# MAGIC for hybrids, change the priorities in the mapping table, not the analysis.)
# MAGIC
# MAGIC The next cell runs on the **lap-time trips**, a mix of EVs, hybrids and combustion cars. Next to the aliases
# MAGIC it selects the physical channels **directly** (`channel(channel_name=...)`, no conversion), so you can check
# MAGIC the arithmetic row by row. The `v_alias < 360` filter is the usual speed-spike rule: 100 m/s = 360 km/h,
# MAGIC because the alias already arrives in km/h.

# COMMAND ----------

drive_rpm = db.query.channel_with_alias(channel_alias="drive_unit_speed")  # e-motor, else engine; rad/s -> rpm
v_alias = db.query.channel_with_alias(channel_alias="vehicle_speed")       # m/s  -> km/h
energy_kwh = db.query.channel_with_alias(channel_alias="battery_energy")   # J    -> kWh
motor_rads = db.query.channel(channel_name="PT.Motor.rotv")                # direct selection: no conversion
engine_rads = db.query.channel(channel_name="Vhcl.Engine.rotv")

cmp = (db.query.havingTag(scenario="lap_time")
       .select(speed.max().alias("speed_max_mps"), v_alias.where(v_alias < 360).max().alias("vehicle_speed_max_kmh"),
               motor_rads.max().alias("motor_max_rad_s"), engine_rads.max().alias("engine_max_rad_s"),
               drive_rpm.max().alias("drive_unit_max_rpm"), energy_kwh.max().alias("battery_energy_max_kwh"))
       .solve(spark, solver=solver))
display(cmp.join(trips_sdf.select("container_id", "vehicle_key", "powertrain_type"), "container_id")
        .orderBy("powertrain_type", "container_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC 👀 `vehicle_speed_max_kmh = 3.6 × speed_max_mps`; for EVs and hybrids `drive_unit_max_rpm = 9.549 ×
# MAGIC motor_max_rad_s`, for the `generic` (combustion) cars — which have no e-motor — it is `9.549 ×
# MAGIC engine_max_rad_s`. One logical name, the right physical signal per vehicle, in the unit you asked for.
# MAGIC
# MAGIC ### Units the conversion table can't fix
# MAGIC `unit_conversion` is **multiplicative** and **trusts the declared unit**. Two quirks from `DATA_NOTES.md`
# MAGIC therefore belong in the TSAL definition:
# MAGIC
# MAGIC * `PT.BattHV.Temp` is *declared* `degC` but logged in **Kelvin** → `temp - 273.15` (an offset).
# MAGIC * `Vhcl.PoI.GCS.Lat` / `.Long` are *declared* `deg` but logged in **radians** → `× 180/π`. A
# MAGIC   `rad → deg` alias would do nothing, because Impulse reads the (wrong) declared unit `deg`.
# MAGIC
# MAGIC (`Env.Temperature` is worse: some trips log °C, others Kelvin — we don't use it for analysis.)

# COMMAND ----------

batt_k = db.query.channel(channel_name="PT.BattHV.Temp")
temps = (db.query.havingTag(powertrain_type="ev")
         .select(batt_k.min().alias("min_as_logged"), (batt_k - KELVIN_OFFSET).min().alias("min_celsius"),
                 (batt_k - KELVIN_OFFSET).max().alias("max_celsius"))
         .solve(spark, solver=solver))
display(temps.join(trips_sdf.select("container_id", "vehicle_key", "starting_temperature_c"), "container_id")
        .orderBy("container_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC The battery's minimum in °C sits at or a little above each trip's nominal `starting_temperature_c` —
# MAGIC consistent with Kelvin logging. Now the route: latitude/longitude in degrees, resampled to 1 Hz.

# COMMAND ----------

import matplotlib.pyplot as plt

lat = db.query.channel(channel_name="Vhcl.PoI.GCS.Lat") * RAD_TO_DEG      # radians -> degrees
lon = db.query.channel(channel_name="Vhcl.PoI.GCS.Long") * RAD_TO_DEG
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, trip_id in zip(axes, [TRIP, 94]):
    rr = solve_trip(trip_id, lat=lat.resample(1e6), lon=lon.resample(1e6), speed=speed_kmh.resample(1e6))
    la, lo = rr.lat.synchronized(rr.lon)
    sc = ax.scatter(lo.values, la.values, s=2, c=np.interp(la.tstarts, rr.speed.tstarts, rr.speed.values), cmap="viridis")
    ax.set(title=f"Trip {trip_id}: {trips.loc[trip_id, 'vehicle_key']} · {trips.loc[trip_id, 'scenario']}",
           xlabel="longitude [°]", ylabel="latitude [°]")
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(la.values))))
fig.colorbar(sc, ax=axes, label="speed [km/h]")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC Around 46.4° N / 9.9° E is the Engadin / Bernina pass (mountain drive); 49.3° N / 8.6° E is the
# MAGIC Hockenheimring (lap). Without the `× 180/π`, both would sit at ~0.8° N, 0.17° E — in the Gulf of Guinea.
# MAGIC
# MAGIC ## 6 · Points-in-Time (POI) channels
# MAGIC
# MAGIC `poi_channel(...)` selects from `poi_channels`. Our silver layer has two event streams:
# MAGIC * `Event.GearShift` — numeric (`dtype="double"`, the default): the new gear at each shift.
# MAGIC * `Event.DriverAlert` — **string** (`dtype="string"`): `HARSH_BRAKE`, `HARSH_ACCEL`, `SHARP_TURN`.
# MAGIC
# MAGIC String POI series support only equality (`== "CODE"` → `PointsInTime`) and sampling. The classic
# MAGIC pattern: turn each event into a **window around it** with `.expand(w)` and look at signals there.

# COMMAND ----------

alerts = db.query.poi_channel(channel_name="Event.DriverAlert", dtype="string")
shifts = db.query.poi_channel(channel_name="Event.GearShift")

harsh_brake = alerts == "HARSH_BRAKE"                          # PointsInTime
around_harsh_brake = harsh_brake.expand(5e6)                   # Intervals: ±5 s around each alert
speed_around = speed_kmh.where(around_harsh_brake)             # SampleSeries inside those windows
shifts_to_top = shifts > 5                                     # numeric POI comparison -> PointsInTime

r = solve_trip(TRIP, alerts=alerts, shifts=shifts, harsh_brake=harsh_brake, windows=around_harsh_brake,
               speed=speed_kmh, speed_around=speed_around, high_gear_shifts=shifts_to_top)
# a string PointsInTimeSeries arrives as a list of (tstart, value) structs
codes = points_to_pandas(r.alerts, "code")["code"].value_counts()
print("driver alerts:", codes.to_dict(), "| gear shifts:", len(r.shifts), "| shifts into gear > 5:", len(r.high_gear_shifts))
fig = plot_signals({"speed [km/h]": r.speed, "speed ±5 s\naround alert": r.speed_around},
                   intervals=r.windows, points={"HARSH_BRAKE": r.harsh_brake}, title="Windows around HARSH_BRAKE alerts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Signal processing
# MAGIC
# MAGIC | Method | Does | Note |
# MAGIC |---|---|---|
# MAGIC | `.resample(rate)` | uniform grid, `rate` = period in µs (`1e6` = 1 Hz) | per continuous segment |
# MAGIC | `.rolling_average(w)` / `.rolling_stats(w)` | moving average / (min, max, avg) over `w` µs | smoothing, envelopes |
# MAGIC | `.cumtrapz()` / `.trapz()` | cumulative / total trapezoidal integral | value × **µs** |
# MAGIC | `.diff()` | differences between consecutive samples | per continuous segment |
# MAGIC
# MAGIC **Distance from speed:** ∫ v dt. `cumtrapz()` integrates over µs, so divide by 1e6 for m.
# MAGIC Compare with the odometer signal as a sanity check.

# COMMAND ----------

distance_km = speed.resample(1e6).cumtrapz() / 1e6 / 1000           # m/s · µs -> m -> km
smooth_speed = speed_kmh.rolling_average(10e6)                       # 10 s moving average
odo_km = db.query.channel_with_alias(channel_alias="odometer")      # m -> km via unit_conversion

r = solve_trip(TRIP, distance=distance_km, odo=odo_km, speed=speed_kmh, smooth=smooth_speed)
print(f"integrated distance: {r.distance.values[-1]:.2f} km | odometer: {r.odo.values[-1] - r.odo.values[0]:.2f} km")
fig = plot_signals({"speed [km/h]": r.speed, "10 s rolling\naverage": r.smooth, "∫v dt [km]": r.distance,
                    "odometer [km]": r.odo}, title="Resample, integrate and smooth")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Custom UDFs
# MAGIC
# MAGIC **Prefer native TSAL** — native operations handle empty inputs, gaps and alignment, and future Impulse
# MAGIC versions can optimize them. Write a UDF only for logic TSAL can't express. The rules (from the
# MAGIC `impulse-tsal` skill):
# MAGIC
# MAGIC 1. **Handle empty input.** Every UDF is called once at planning time with *empty* series (and `None`
# MAGIC    metadata) to infer its output type → guard with `if len(series) == 0`.
# MAGIC 2. **Return the same type on every path** (`SampleSeries.empty()` / `np.nan` for the empty case).
# MAGIC 3. **Respect gaps** — use `series.continuous_interval_indices` for neighbour-based logic.
# MAGIC 4. **Synchronize** before combining raw arrays of several series (`a.synchronized(b)`).
# MAGIC
# MAGIC ### 8a · A metadata-aware UDF: power-to-weight ratio
# MAGIC Declaring `container_metrics=["empty_weight_kg"]` makes Impulse inject each trip's vehicle weight.

# COMMAND ----------

from impulse_query_engine.analyze.metadata.time_series_expression import TimeSeriesExpression
from impulse_query_engine.model.series.sample_series import SampleSeries


@TimeSeriesExpression.udf(container_metrics=["empty_weight_kg"], container_tags=["powertrain_type"])
def power_to_weight(power_w, container_metrics, container_tags):
    """Motor power per kg of vehicle [W/kg]; empty for non-electric powertrains."""
    weight = container_metrics.get("empty_weight_kg")
    if len(power_w) == 0 or not weight or container_tags.get("powertrain_type") not in ("ev", "hev_parallel", "hev_axlesplit"):
        return SampleSeries.empty()
    return SampleSeries(power_w.tstarts, power_w.tends, power_w.values / weight)


pw = power_to_weight(motor_power)
res = (db.query.havingTag(scenario="lap_time")
       .select(pw.max().alias("peak_w_per_kg"), (motor_power / 1000).max().alias("peak_motor_kw"))
       .solve(spark, solver=solver))
display(res.join(trips_sdf.select("container_id", "vehicle_key", "powertrain_type", "empty_weight_kg"), "container_id")
        .where(F.col("peak_w_per_kg").isNotNull() & ~F.isnan("peak_w_per_kg"))
        .orderBy(F.desc("peak_w_per_kg")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b · A scalar UDF that respects gaps: recovered energy
# MAGIC *How much energy did regenerative braking return to the battery?* ∫ −P dt over the regen windows.
# MAGIC The UDF integrates each **continuous segment** separately — never across a gap — and returns kWh.
# MAGIC Then we validate it against the native `trapz()`.

# COMMAND ----------

@TimeSeriesExpression.udf
def energy_kwh(power_w):
    """Trapezoidal energy of a power signal [W] in kWh, integrated per continuous segment."""
    if len(power_w) == 0:
        return np.nan
    total_j = 0.0
    for start, end in power_w.continuous_interval_indices:
        t = np.append(power_w.tstarts[start:end + 1], power_w.tends[end]) / US      # s
        p = np.append(power_w.values[start:end + 1], power_w.values[end])
        total_j += np.trapezoid(p, t)
    return total_j / 3.6e6


regen_power = motor_power.where(regen) * -1                   # positive = energy back into the battery (no unary minus in TSAL)
check = (db.query.havingTag(powertrain_type="ev")
         .select(energy_kwh(regen_power).alias("udf_regen_kwh"),
                 (regen_power.trapz() / US / 3.6e6).alias("native_regen_kwh"))
         .solve(spark, solver=solver))
display(check.join(trips_sdf.select("container_id", "vehicle_key", "scenario"), "container_id")
        .orderBy(F.desc("udf_regen_kwh")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC The two columns agree to rounding — so here the native `trapz()` is the better choice (rule zero!).
# MAGIC The UDF pattern pays off when the logic *isn't* native: a custom efficiency model, a lookup table
# MAGIC keyed by `vehicle_key`, a physics formula with per-vehicle parameters. Notebook 2b reuses this energy
# MAGIC calculation as a **custom statistic** inside a report.
# MAGIC
# MAGIC ## 9 · 🔍 Explore interactively
# MAGIC
# MAGIC Use the widgets at the top of the notebook: pick a **trip**, up to three **signals**, and a
# MAGIC **condition** (`<signal 1> > threshold`). Re-run the cell below (or set widgets to *Run notebook
# MAGIC on change* via the ⚙️ next to the widgets). Things to try:
# MAGIC * trip **66** (Hyundai Ioniq 5, charging) — `PT.BCU.BattHV.SOC` (state of charge, % battery full), `PT.BattHV.Current`, `PT.BattHV.Temp`
# MAGIC * trip **94** (Porsche 911 lap) — `Vhcl.v`, `Car.ay`, `Vhcl.Steer.Ang`, condition `Car.ay > 8`
# MAGIC * trip **95** — `Vhcl.v` is gone; try `Car.Gen.vx_1`

# COMMAND ----------

signal_names = sorted(r.channel_name for r in spark.table(SILVER["channel_metrics"])
                      .where("value_type = 'DOUBLE'").select("channel_name").distinct().collect())
trip_choices = [str(c) for c in sorted(trips.index)]
dbutils.widgets.dropdown("explore_trip", "66", trip_choices, "Trip")
dbutils.widgets.dropdown("explore_signal_1", "PT.BattHV.Current", signal_names, "Signal 1 (condition)")
dbutils.widgets.dropdown("explore_signal_2", "PT.BCU.BattHV.SOC", signal_names, "Signal 2")
dbutils.widgets.dropdown("explore_signal_3", "Vhcl.v", signal_names, "Signal 3")
dbutils.widgets.dropdown("explore_op", "<", ["<", ">"], "Condition operator")
dbutils.widgets.text("explore_threshold", "-10", "Condition threshold")

# COMMAND ----------

trip_id = int(dbutils.widgets.get("explore_trip"))
names = [dbutils.widgets.get(f"explore_signal_{i}") for i in (1, 2, 3)]
op, thr = dbutils.widgets.get("explore_op"), float(dbutils.widgets.get("explore_threshold"))

sigs = {n: db.query.channel(channel_name=n) for n in dict.fromkeys(names)}
cond = (sigs[names[0]] < thr) if op == "<" else (sigs[names[0]] > thr)
row = solve_trip(trip_id, cond=cond, **{f"s{i}": e for i, e in enumerate(sigs.values())})
windows = intervals_to_pandas(row.cond, t0=row.s0.tstarts[0] if len(row.s0) else 0)
info = trips.loc[trip_id]
print(f"Trip {trip_id}: {info.vehicle_key} · {info.scenario} · {info.driver} — "
      f"'{names[0]} {op} {thr}' holds in {len(windows)} windows, {windows.duration_s.sum():.0f} s total")
fig = plot_signals({n: row[f"s{i}"] for i, n in enumerate(sigs)}, intervals=row.cond,
                   title=f"Trip {trip_id}: {names[0]} {op} {thr} (shaded)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Recap
# MAGIC * Four result types; the transitions between them are the grammar of TSAL.
# MAGIC * Operators align multi-rate signals automatically; interval methods (`filter`, `merge_intervals`,
# MAGIC   `expand`) turn noisy conditions into meaningful events.
# MAGIC * Optional tables add filtering by any metadata, event streams, logical names with fallback, and units.
# MAGIC * UDFs: guard empties, keep one return type, respect gaps, synchronize — and prefer native first.
# MAGIC
# MAGIC **Next → `02b_report_sql_dashboard`**: put these building blocks into a report.
