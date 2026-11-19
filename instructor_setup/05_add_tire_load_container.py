# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Add a container: 1 kHz tire-load test (race car)
# MAGIC
# MAGIC **Instructor only. Standalone: run it any time after `03_create_silver_tables`.** `03` overwrites the silver
# MAGIC tables, so run this notebook again after every `03` run.
# MAGIC
# MAGIC This notebook adds **one new container** (`container_id = 1000`) to the shared Impulse silver model in
# MAGIC `dbx_workshop_dev.shared`. It is a simulated test session of a GT race car on a circuit, with **vertical tire
# MAGIC load (Fz) on all four wheels logged at 1 kHz**. Workshop 2 / `02c_tire_load_signal_processing` uses it to
# MAGIC show Butterworth filters and FFTs as Impulse UDFs.
# MAGIC
# MAGIC | channel_id | channel | rate | unit | content |
# MAGIC |---|---|---|---|---|
# MAGIC | 20001 | `Car.FzFL` | **1 kHz** | N | vertical load, front-left tire |
# MAGIC | 20002 | `Car.FzFR` | **1 kHz** | N | vertical load, front-right tire |
# MAGIC | 20003 | `Car.FzRL` | **1 kHz** | N | vertical load, rear-left tire |
# MAGIC | 20004 | `Car.FzRR` | **1 kHz** | N | vertical load, rear-right tire |
# MAGIC | 34 | `DM.Brake` | 100 Hz | none | brake pedal demand 0..1 (the same signal as in the fleet trips) |
# MAGIC | 680 | `Vhcl.v` | 100 Hz | m/s | vehicle speed |
# MAGIC | 31 | `Car.ax` | 100 Hz | m/s^2 | longitudinal acceleration |
# MAGIC
# MAGIC The new channels use CarMaker-style names (`Car.FzFL`, ...) and the `CarMaker` data key, so they behave
# MAGIC like every other channel in the model. The brake, speed and acceleration channels **reuse the fleet's
# MAGIC channel ids**, so `channel(channel_name="DM.Brake")` works on this container like on any trip.
# MAGIC
# MAGIC ## What is in the tire-load signal
# MAGIC
# MAGIC Each `Car.Fz**` channel is the sum of a **low-frequency** and a **high-frequency** part:
# MAGIC
# MAGIC ```
# MAGIC  Fz = static weight + aero downforce(v²) ± longitudinal load transfer   ← low frequency, < 5 Hz
# MAGIC     + road-induced tire force (quarter-car model)                         ← high frequency, 5 .. 500 Hz
# MAGIC     + load-cell noise
# MAGIC ```
# MAGIC
# MAGIC **Low frequency (what you filter out):**
# MAGIC * **Static weight.** 1,250 kg with 42 % on the front axle: about 2,575 N per front tire and 3,556 N per rear tire.
# MAGIC * **Aerodynamic downforce.** It grows with v², reaching about +2,700 N per rear tire at 300 km/h.
# MAGIC * **Load transfer under braking.** When the brake is applied, the car's momentum pitches it onto its nose:
# MAGIC   `ΔFz = m · a_x · h_cg / wheelbase`. That is up to about 2,500 N **onto each front tire** and the same amount
# MAGIC   **off each rear tire**. The transfer passes through the body's **pitch mode** (1.5 Hz, damping ratio 0.3),
# MAGIC   so each brake application and each release rings at **about 1.5 Hz** for a second or two. As the car slows,
# MAGIC   downforce falls away, so all four loads drift down during a long stop.
# MAGIC
# MAGIC **High frequency (the grip signal you keep):**
# MAGIC * **Road texture.** A smooth circuit surface (ISO 8608-like roughness, wavelengths 1.7 cm to 2.5 m) is fed
# MAGIC   through a quarter-car model (sprung mass, spring/damper, 45 kg unsprung mass, 280 kN/m tire). The model
# MAGIC   puts the wheel-hop resonance at about 15 Hz. The amplitude scales with √(load), so a loaded tire shows
# MAGIC   more texture: **the fronts get louder and the rears quieter under braking.**
# MAGIC * **The rear follows the front.** Each rear wheel rolls over the same road as the front wheel on its side,
# MAGIC   `wheelbase / v` later (about 30 ms at 300 km/h).
# MAGIC * **Transverse grooves** (0.35 m pitch, 0.6 mm deep) on part of the main straight. They show up as a narrow
# MAGIC   line at `v / 0.35 m`, which **sweeps from about 180 to 230 Hz** as the car accelerates.
# MAGIC * **Kerbs** (4 mm ridges, 0.6 m pitch) on the inside of each corner exit, on alternating sides.
# MAGIC * **Pebbles and debris.** Short 3–9 mm bumps, about one every 120 m, on one wheel track: first a front spike,
# MAGIC   then a rear spike on the same side.
# MAGIC * **Two lock-up / judder bursts (ground truth for the analysis notebook).** On lap 2, at the first brake
# MAGIC   point (event 7), the **front-left** tire shows 85 Hz judder. On lap 3, at the fifth brake point (event 17),
# MAGIC   the **front-right** tire shows 70 Hz judder. Each burst lasts about 0.3 s, roughly 0.45 s after the pedal
# MAGIC   goes down.
# MAGIC
# MAGIC The simulation runs at 4 kHz. It is decimated to 1 kHz with a zero-phase anti-alias FIR filter, like a
# MAGIC real data logger. The circuit is lapped 3 times (≈ 230 s, 18 brake applications, up to ≈ 310 km/h).
# MAGIC
# MAGIC ## How it is written
# MAGIC The samples go through the **same transformation code as the rest of the silver layer**
# MAGIC (`common/impulse_silver.py`: RLE encoding, `channel_metrics`, `channel_tags`, `container_metrics`,
# MAGIC `container_tags`). The container also gets the car's **setup sheet** as tags (`wheelbase_m`, `cg_height_m`,
# MAGIC `front_weight_ratio`, `aero_cl_a_m2`, `aero_front_ratio`), so analyses can read the vehicle parameters per
# MAGIC container instead of hard-coding them. Every table is written with `replaceWhere container_id = 1000`, which is atomic and
# MAGIC idempotent: re-running replaces this container and never touches the 100 fleet trips.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

setup_impulse()
import hashlib
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from scipy import signal

from common import impulse_silver as silver

silver.use_utc(spark)

CONTAINER_ID = 1000                     # well above the fleet's session ids (0..99)
SEED = 42                               # fixed seed -> the same container on every run
T0_US = 1_774_008_000_000_000           # 2026-03-20 12:00:00 UTC: every trip in this model starts here (DATA_NOTES.md)
TRIP_START = datetime(2026, 11, 19, 9, 30, 0)   # the "real" session start, as trip_start_dt
SCENARIO = "tire_load_test"

TIRES = ["FL", "FR", "RL", "RR"]
FZ_CHANNEL_IDS = {"FL": 20001, "FR": 20002, "RL": 20003, "RR": 20004}   # above the fleet (0..680) and POI (10001+) ids
FLEET_CHANNEL_IDS = {"DM.Brake": 34, "Vhcl.v": 680, "Car.ax": 31}       # reuse the fleet's ids for the same signals

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Simulate the session
# MAGIC The simulation has three parts:
# MAGIC
# MAGIC 1. **A driver / longitudinal model.** Full throttle (power- and traction-limited, minus drag), then a brake
# MAGIC    phase down to the corner speed, then a constant-speed corner. The pedal ramps up in 60 ms and is
# MAGIC    released in 120 ms.
# MAGIC 2. **Road profiles** for the left and right wheel tracks, as a function of distance.
# MAGIC 3. **Per-tire vertical load.** The low-frequency part comes from static weight, aero and pitch-filtered load
# MAGIC    transfer. The high-frequency part comes from the quarter-car response to the road under that tire.

# COMMAND ----------

FS = 1000            # logged rate of the tire loads [Hz]
SIM_FS = 4000        # internal simulation rate [Hz]
LOW_RATE_DECIMATION = 10   # DM.Brake / Vhcl.v / Car.ax are logged at FS / 10 = 100 Hz
G, RHO = 9.81, 1.2

CAR = dict(
    mass=1250.0, front_weight=0.42, cg_height=0.33, wheelbase=2.45,     # kg, -, m, m
    cl_a=2.2, aero_front=0.42, cd_a=0.75,                               # downforce / drag areas [m²], aero balance
    power=480e3, mu=1.2, traction=7.0,                                  # W, brake friction, max drive accel [m/s²]
    unsprung=45.0, k_spring=(80e3, 95e3), c_damper=(3200.0, 3600.0),    # per corner, (front, rear)
    k_tire=280e3, c_tire=150.0,                                         # tire vertical stiffness / damping
    pitch_hz=1.5, pitch_zeta=0.3,                                       # body pitch mode
)

# One lap = 6 corners: (straight [s], corner speed [m/s], corner [s], brake pedal 0..1)
LAP = [(12.0, 25.0, 3.0, 1.00), (6.0, 45.0, 2.5, 0.70), (9.0, 33.0, 2.5, 0.90),
       (4.0, 52.0, 2.0, 0.55), (10.0, 28.0, 3.0, 0.95), (5.0, 40.0, 2.5, 0.80)]
N_LAPS = 3
PEDAL_RAMP_UP_S, PEDAL_RAMP_DOWN_S = 0.06, 0.12
# Lock-up / judder bursts (lap index, corner index, tire, frequency [Hz], amplitude [N]) -- ground truth
GRIP_EVENTS = [(1, 0, "FL", 85.0, 650.0), (2, 4, "FR", 70.0, 450.0)]


def drive(rng):
    """Longitudinal driver + vehicle model on the SIM_FS grid -> t, v, ax, pedal, x and phase marks."""
    c, dt = CAR, 1.0 / SIM_FS
    v, x, t = 40.0, 0.0, 0.0                       # flying start at 144 km/h
    T, V, AX, P, X, marks = [], [], [], [], [], []  # marks: (lap, corner, kind, t, x)

    def step(a, pedal):
        nonlocal v, x, t
        T.append(t); V.append(v); AX.append(a); P.append(pedal); X.append(x)
        v = max(v + a * dt, 1.0); x += v * dt; t += dt

    def drag():
        return 0.5 * RHO * c["cd_a"] * v * v / c["mass"]

    def brake_decel(p):                             # friction-limited, downforce adds grip
        return -p * c["mu"] * (G + 0.5 * RHO * c["cl_a"] * v * v / c["mass"]) - drag()

    for lap in range(N_LAPS):
        for k, (straight_s, v_corner, corner_s, pedal) in enumerate(LAP):
            straight_s *= rng.uniform(0.95, 1.05)                       # lap-to-lap variation
            pedal = float(np.clip(pedal * rng.uniform(0.95, 1.03), 0, 1))
            marks.append((lap, k, "straight", t, x))
            for _ in range(int(straight_s * SIM_FS)):                  # full throttle
                step(min(c["traction"], c["power"] / (c["mass"] * v)) - drag(), 0.0)
            marks.append((lap, k, "brake_on", t, x))
            n_up, i = int(PEDAL_RAMP_UP_S * SIM_FS), 0
            while v > v_corner + 1.5 or i < n_up:                       # ramp up + hold
                p = pedal * min(1.0, (i + 1) / n_up)
                step(brake_decel(p), p)
                i += 1
            n_dn = int(PEDAL_RAMP_DOWN_S * SIM_FS)
            for i in range(n_dn):                                       # release
                p = pedal * (1 - (i + 1) / n_dn)
                step(brake_decel(p), p)
            marks.append((lap, k, "brake_off", t, x))
            for _ in range(int(corner_s * SIM_FS)):                     # constant-speed corner
                step(0.0, 0.0)
            marks.append((lap, k, "corner_end", t, x))
    return np.array(T), np.array(V), np.array(AX), np.array(P), np.array(X), marks


def road_profiles(rng, length_m, marks, dx=0.005):
    """Left / right road elevation [m] vs distance: roughness + grooves + kerbs + pebbles."""
    n = int(length_m / dx) + 1
    xs = np.arange(n) * dx

    def rough(gd0):                                  # ISO 8608-style PSD Gd(n) = gd0 (n / 0.1)^-2
        f = np.fft.rfftfreq(n, dx)                   # spatial frequency [cycles/m]
        band = (f > 0.4) & (f < 60)                  # wavelengths 1.7 cm .. 2.5 m: no long undulations on a circuit
        amp = np.zeros_like(f)
        amp[band] = np.sqrt(2 * gd0 * (f[band] / 0.1) ** -2 * (f[1] - f[0]))
        return np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, f.size)), n) * n / np.sqrt(2)

    common = rough(3e-6)                             # shared by both wheel tracks ...
    left, right = common + rough(1.5e-6), common + rough(1.5e-6)   # ... plus a track-specific part

    def add(track, x0, x1, shape):
        m = (xs >= x0) & (xs < x1)
        track[m] += shape(xs[m] - x0)

    for lap, k, kind, _, x in marks:
        if kind == "straight" and k == 0:            # grooved section on the main straight (25 %..65 % of it)
            x_end = next(m[4] for m in marks if m[:3] == (lap, k, "brake_on"))
            for tr in (left, right):
                add(tr, x + 0.25 * (x_end - x), x + 0.65 * (x_end - x),
                    lambda u: 0.0006 * np.sin(2 * np.pi * u / 0.35))
        if kind == "brake_off":                      # kerb on the inside of the corner exit, alternating sides
            add(left if k % 2 == 0 else right, x + 8, x + 20,
                lambda u: 0.004 * np.clip(np.sin(2 * np.pi * u / 0.6), 0, None))
    for xp in np.cumsum(rng.exponential(120.0, int(length_m / 120 * 1.5))):   # pebbles / debris
        if xp > length_m - 1:
            break
        h, w = rng.uniform(0.003, 0.009), rng.uniform(0.03, 0.06)
        add(left if rng.random() < 0.5 else right, xp, xp + w,
            lambda u, h=h, w=w: h * 0.5 * (1 - np.cos(2 * np.pi * u / w)))
    return xs, left, right


def quarter_car_tf(front: bool):
    """Discrete transfer function: road elevation z_r [m] -> dynamic tire force [N] (quarter-car, SIM_FS)."""
    c, i = CAR, (0 if front else 1)
    ms = c["mass"] * (c["front_weight"] if front else 1 - c["front_weight"]) / 2 - c["unsprung"]
    mu, ks, cs, kt, ct = c["unsprung"], c["k_spring"][i], c["c_damper"][i], c["k_tire"], c["c_tire"]
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],                          # states: z_s, z_u, z_s', z_u'
                  [-ks / ms, ks / ms, -cs / ms, cs / ms],
                  [ks / mu, -(ks + kt) / mu, cs / mu, -(cs + ct) / mu]])
    B = np.array([[0], [0], [0], [kt / mu]])
    C = np.array([[0, -kt, 0, -ct]])                                    # F_tire = kt (z_r - z_u) - ct z_u'
    D = np.array([[kt]])
    num, den = signal.ss2tf(A, B, C, D)
    bd, ad, _ = signal.cont2discrete((num[0], den), 1.0 / SIM_FS, method="bilinear")
    return np.ravel(bd), np.ravel(ad)


def simulate(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    c = CAR
    t, v, ax, pedal, x, marks = drive(rng)
    xs, left, right = road_profiles(rng, x[-1] + 5, marks)

    # Longitudinal load transfer through the pitch mode (2nd order, 1.5 Hz, lightly damped)
    target = -c["mass"] * ax * c["cg_height"] / c["wheelbase"]          # + = onto the front axle
    wn, z = 2 * np.pi * c["pitch_hz"], c["pitch_zeta"]
    b, a, _ = signal.cont2discrete(([wn * wn], [1, 2 * z * wn, wn * wn]), 1.0 / SIM_FS, method="bilinear")
    transfer = signal.lfilter(np.ravel(b), np.ravel(a), target)

    aero = 0.5 * RHO * c["cl_a"] * v * v
    static = {"F": c["mass"] * G * c["front_weight"] / 2, "R": c["mass"] * G * (1 - c["front_weight"]) / 2}
    aero_share = {"F": c["aero_front"] / 2, "R": (1 - c["aero_front"]) / 2}
    tf = {"F": quarter_car_tf(True), "R": quarter_car_tf(False)}
    loads, low_parts = {}, {}
    for tire in TIRES:
        axle, side = tire[0], tire[1]
        low = static[axle] + aero * aero_share[axle] + (transfer if axle == "F" else -transfer) / 2
        track = left if side == "L" else right
        pos = x if axle == "F" else x - c["wheelbase"]                  # the rear wheel meets the same road later
        zr = np.interp(pos, xs, track)
        dyn = signal.lfilter(*tf[axle], zr - zr[0])
        dyn *= np.sqrt(np.clip(low, 200, None) / static[axle])          # stiffer contact patch under load
        f = low + dyn
        for lap, k, tname, hz, amp in GRIP_EVENTS:                       # lock-up judder bursts
            if tname == tire:
                t_on = next(m[3] for m in marks if m[:3] == (lap, k, "brake_on"))
                env = np.exp(-0.5 * ((t - (t_on + 0.45)) / 0.12) ** 2)
                f = f + amp * env * np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi))
        loads[tire], low_parts[tire] = f, low

    q = SIM_FS // FS
    n = len(t) // q
    out = {"t_s": np.arange(n) / FS}
    for tire, f in loads.items():                                        # anti-alias + decimate to 1 kHz
        y = signal.decimate(f, q, ftype="fir", zero_phase=True)[:n] + rng.normal(0, 8.0, n)
        out[tire] = np.round(np.clip(y, 0, None), 1)
        out[f"{tire}_low_truth"] = low_parts[tire][::q][:n]
    lo = slice(0, n * q, q * LOW_RATE_DECIMATION)                         # 100 Hz channels
    out["low_t_s"] = np.arange(len(t[lo])) / (FS / LOW_RATE_DECIMATION)
    out["DM.Brake"] = np.round(pedal[lo], 3)
    out["Vhcl.v"] = np.round(v[lo] + rng.normal(0, 0.02, len(t[lo])), 2)
    out["Car.ax"] = np.round(ax[lo] + rng.normal(0, 0.15, len(t[lo])), 2)
    out["marks"] = marks
    return out


sim = simulate()
marks = pd.DataFrame(sim["marks"], columns=["lap", "corner", "kind", "t_s", "x_m"])
brake_truth = (marks.pivot_table(index=["lap", "corner"], columns="kind", values="t_s").reset_index()
               [["lap", "corner", "brake_on", "brake_off"]])
brake_truth.insert(0, "event", range(1, len(brake_truth) + 1))
print(f"{len(sim['t_s']):,} samples per tire = {sim['t_s'][-1]:.1f} s | {len(brake_truth)} brake applications | "
      f"top speed {sim['Vhcl.v'].max() * 3.6:.0f} km/h | distance {marks.x_m.max() / 1000:.2f} km")
display(pd.DataFrame({k: [sim[k].min(), sim[k].mean(), sim[k].max()] for k in TIRES},
                     index=["min [N]", "mean [N]", "max [N]"]).round(0))

# COMMAND ----------

# DBTITLE 1,Sanity check: does the signal make sense?
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
t, tl = sim["t_s"], sim["low_t_s"]
axes[0].plot(tl, sim["Vhcl.v"] * 3.6, lw=1); axes[0].set_ylabel("speed\n[km/h]")
axes[1].plot(tl, sim["DM.Brake"], lw=1, color="C3"); axes[1].set_ylabel("brake\npedal")
for tire, col in zip(TIRES, ["C0", "C1", "C2", "C4"]):
    axes[2].plot(t[::5], sim[tire][::5], lw=0.3, color=col, label=tire)
    axes[3].plot(t, sim[f"{tire}_low_truth"], lw=1, color=col, label=tire)
axes[2].set_ylabel("Fz logged\n[N]"); axes[3].set_ylabel("Fz low-freq.\npart (truth) [N]")
axes[2].legend(ncol=4, loc="upper right", fontsize=8)
axes[-1].set_xlabel("time [s]")
for a in axes:
    a.grid(alpha=0.3)
fig.suptitle(f"Container {CONTAINER_ID}: 3 laps, 18 brake applications. Fronts gain and rears lose load at every stop")
fig.tight_layout()

# Zoom: the hardest stop of lap 2 (event 7) -> 1.5 Hz pitch ringing, plus the FL judder burst
on = brake_truth.loc[brake_truth.event == 7, "brake_on"].item()
m = (t > on - 1) & (t < on + 5)
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for tire, col in zip(TIRES, ["C0", "C1", "C2", "C4"]):
    axes[0].plot(t[m] - on, sim[tire][m], lw=0.4, color=col, label=tire)
    axes[1].plot(t[m] - on, sim[f"{tire}_low_truth"][m], lw=1.2, color=col, label=tire)
axes[0].set_ylabel("Fz logged [N]"); axes[1].set_ylabel("low-freq. part [N]")
axes[1].set_xlabel("s after brake application (event 7)")
for a in axes:
    a.grid(alpha=0.3); a.axvline(0, color="k", lw=0.8); a.legend(ncol=4, fontsize=8)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Samples → the silver tables
# MAGIC The simulation becomes **samples** `(container_id, channel_id, timestamp [epoch µs], value)`, the same shape
# MAGIC `silver.samples(bronze)` produces for the fleet. From there the shared functions take over.

# COMMAND ----------

def channel_samples(channel_id: int, t_s: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"container_id": np.int64(CONTAINER_ID), "channel_id": np.int32(channel_id),
                         "timestamp": T0_US + np.round(t_s * 1e6).astype(np.int64),
                         "value": values.astype(np.float64)})


samples_pdf = pd.concat(
    [channel_samples(FZ_CHANNEL_IDS[tire], sim["t_s"], sim[tire]) for tire in TIRES]
    + [channel_samples(cid, sim["low_t_s"], sim[name]) for name, cid in FLEET_CHANNEL_IDS.items()],
    ignore_index=True)
samples = spark.createDataFrame(samples_pdf,
                                "container_id bigint, channel_id int, timestamp bigint, value double")
print(f"{len(samples_pdf):,} samples")

# Signal metadata: the fleet's rows for DM.Brake / Vhcl.v / Car.ax, plus four new tire-load signals.
fz_info = spark.createDataFrame(
    [(cid, f"Car.Fz{tire}", f"Car vertical tire force {'front' if tire[0] == 'F' else 'rear'}-"
      f"{'left' if tire[1] == 'L' else 'right'} (1 kHz load cell)", None, False, "force", "N", "Car",
      f"Fz{tire}", tire, "Car", "Float64") for tire, cid in FZ_CHANNEL_IDS.items()],
    "signal_id int, signal_name string, signal_description string, signal_mapping string, is_categorical boolean, "
    "physical_quantity string, unit string, category string, subsystem string, component string, `group` string, "
    "dtype string")
signal_info = (spark.table(BRONZE["signal_info"])
               .where(F.col("signal_id").isin(list(FLEET_CHANNEL_IDS.values())))
               .select(*fz_info.columns)
               .unionByName(fz_info))
names = {r.signal_id: r.signal_name for r in signal_info.collect()}
assert all(names[cid] == name for name, cid in FLEET_CHANNEL_IDS.items()), names

# COMMAND ----------

# Container metadata, in the shape of the bronze trip / vehicle / vehicle-state tables.
trip_name = f"{SCENARIO}_scenario_GT_RaceCar_nopax_25C_slick_305_30R18"
trip_info = spark.createDataFrame(
    [(CONTAINER_ID, trip_name, 25, TRIP_START)],
    "session_id int, trip_name string, starting_temperature bigint, trip_start_timestamp timestamp_ntz")
vehicle_info = spark.createDataFrame(
    [(CONTAINER_ID, "GT_RaceCar", "Impulse Labs", "race car", "generic", None, CAR["mass"])],
    "vehicle_id int, vehicle_name string, manufacturer string, vehicle_type string, powertrain_type string, "
    "battery_voltage_oc double, empty_weight double")
vehicle_state = spark.createDataFrame([(CONTAINER_ID, 1, "slick_305_30R18", "race")],
                                      "vehicle_state_id int, number_of_passengers bigint, tires string, driver string")
keys = spark.createDataFrame(
    [(CONTAINER_ID, CONTAINER_ID, CONTAINER_ID, hashlib.sha256(f"{trip_name}|{SEED}".encode()).hexdigest())],
    "session_id int, vehicle_id int, vehicle_state_id int, trip_hash string")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write: replace only container 1000
# MAGIC `replaceWhere` makes each write one atomic transaction that swaps out this container's rows. The other
# MAGIC 100 trips are never read or rewritten.

# COMMAND ----------

def replace_container(df, table: str) -> None:
    target = spark.table(table).schema
    aligned = df.select(*[F.col(f"`{f.name}`").cast(f.dataType).alias(f.name) for f in target.fields])
    for attempt in range(4):        # a concurrent write on the shared table can win first -> retry
        try:
            (aligned.write.mode("overwrite").option("replaceWhere", f"container_id = {CONTAINER_ID}")
             .saveAsTable(table))
            break
        except Exception as e:      # noqa: BLE001
            if attempt == 3 or not any(k in str(e) for k in ("Concurrent", "CONCURRENT", "CONFLICTING")):
                raise
    n = spark.table(table).where(F.col("container_id") == CONTAINER_ID).count()
    print(f"✅ {table}: {n:,} rows for container {CONTAINER_ID}")


channels = silver.rle_encode(samples)
replace_container(channels, SILVER["channels"])

cm = silver.channel_metrics(spark.table(SILVER["channels"]).where(F.col("container_id") == CONTAINER_ID),
                            samples, signal_info)
replace_container(cm, SILVER["channel_metrics"])
cm_new = spark.table(SILVER["channel_metrics"]).where(F.col("container_id") == CONTAINER_ID)
replace_container(silver.channel_tags(cm_new, signal_info), SILVER["channel_tags"])

ctm = silver.container_metrics(cm_new, keys, trip_info, vehicle_info, vehicle_state)
replace_container(ctm, SILVER["container_metrics"])
ctm_new = spark.table(SILVER["container_metrics"]).where(F.col("container_id") == CONTAINER_ID)
# Extra tags: where the data comes from, plus the car's setup sheet (UDFs in 02c read these per container).
SETUP_SHEET = {"wheelbase_m": CAR["wheelbase"], "cg_height_m": CAR["cg_height"],
               "front_weight_ratio": CAR["front_weight"], "aero_cl_a_m2": CAR["cl_a"],
               "aero_front_ratio": CAR["aero_front"]}
extra_tags = [(CONTAINER_ID, "data_origin", "simulation: 1 kHz tire-load test (instructor_setup/05)"),
              (CONTAINER_ID, "tire_load_rate_hz", str(FS))] + [(CONTAINER_ID, k, str(v)) for k, v in SETUP_SHEET.items()]
tags = silver.container_tags(ctm_new).unionByName(
    spark.createDataFrame(extra_tags, "container_id bigint, key string, value string"))
replace_container(tags, SILVER["container_tags"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Validate
# MAGIC Check the new rows, then run an Impulse query against the shared silver model: select the container by
# MAGIC its tag and the channels by name, exactly as an analyst would.

# COMMAND ----------

display(spark.table(SILVER["channel_metrics"]).where(F.col("container_id") == CONTAINER_ID)
        .select("channel_id", "channel_name", "unit", "raw_sample_count", "sample_count", "rle_compression_ratio",
                "sample_rate_hz", "min", "mean", "max", "duration_s").orderBy("channel_id"))
display(spark.table(SILVER["container_metrics"]).where(F.col("container_id") == CONTAINER_ID))
fleet_untouched = spark.table(SILVER["container_metrics"]).where(F.col("container_id") != CONTAINER_ID).count()
print(f"fleet containers untouched: {fleet_untouched}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver

db = MeasurementDB(MeasurementDBConfig(**impulse_source(SILVER), table_locations="unity_catalog"), WorkspaceClient())
q = db.query
fl, rl = q.channel(channel_name="Car.FzFL"), q.channel(channel_name="Car.FzRL")
brake = q.channel(channel_name="DM.Brake")
braking = (brake > 0.05).filter(0.2e6)
check = (q.havingTag(scenario="tire_load_test")
         .select(fl.mean().alias("fl_mean_n"), fl.where(braking).mean().alias("fl_mean_braking_n"),
                 rl.mean().alias("rl_mean_n"), rl.where(braking).mean().alias("rl_mean_braking_n"),
                 braking.alias("brake_windows"))
         .solve(spark, solver=DefaultSolver(spark))
         .withColumn("n_brake_applications", F.size("brake_windows")).drop("brake_windows"))
display(check)
row = check.first()
assert row.container_id == CONTAINER_ID and row.n_brake_applications == 18, row
assert row.fl_mean_braking_n > row.fl_mean_n and row.rl_mean_braking_n < row.rl_mean_n, "load transfer missing"
print("✅ container 1000 is in the silver model and Impulse-ready: braking loads the fronts and unloads the rears")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"container_id": CONTAINER_ID, "n_brake_applications": row.n_brake_applications,
                                  "fl_mean_braking_n": round(row.fl_mean_braking_n, 1),
                                  "rl_mean_braking_n": round(row.rl_mean_braking_n, 1)}))
