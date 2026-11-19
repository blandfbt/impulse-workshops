# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2 · Notebook 2c — Signal processing UDFs: Butterworth filters and FFTs on 1 kHz tire loads
# MAGIC
# MAGIC **🎯 Goal: split a 1 kHz tire-load signal into its low-frequency part (load transfer under braking) and its
# MAGIC high-frequency part (the tire's contact with the road), with SciPy inside Impulse UDFs, and read both with
# MAGIC FFTs.** Every query here is ad hoc: a TSAL expression solved for one container, then plotted. Nothing is
# MAGIC written.
# MAGIC
# MAGIC **The data:** container **1000**, a simulated GT race car lapping a circuit, added to the shared silver model by
# MAGIC `instructor_setup/05_add_tire_load_container`. It logs vertical tire load (`Car.FzFL/FR/RL/RR`, N) at **1 kHz**
# MAGIC and brake / speed / acceleration (`DM.Brake`, `Vhcl.v`, `Car.ax`) at 100 Hz.
# MAGIC
# MAGIC ```
# MAGIC  Fz ─┬─ low-pass  5 Hz ─► static weight + downforce ± load transfer (rings at ~1.5 Hz on every brake edge)
# MAGIC     └─ high-pass 5 Hz ─► road texture, wheel hop (~15 Hz), grooves, kerbs, pebbles, lock-up judder
# MAGIC ```
# MAGIC
# MAGIC | # | Section | TSAL / UDF |
# MAGIC |---|---|---|
# MAGIC | 1 | Setup and the raw signal | `channel()`, `havingTag()` |
# MAGIC | 2 | Edges: brake applied / released | `(brake > x).filter()`, `start_points()`, `end_points()` |
# MAGIC | 3 | Butterworth low-pass / high-pass | UDF `butterworth` (`scipy.signal.butter` + `sosfiltfilt`) |
# MAGIC | 4 | The low band: load transfer and the pitch mode | metadata-aware UDFs, UDF `dominant_frequency` (FFT) |
# MAGIC | 5 | The high band: spectra and the grip signal | UDF `amplitude_spectrum` (Welch FFT) |
# MAGIC | 6 | Time–frequency: lock-up detection, pavement grooves | UDFs `band_rms`, `peak_frequency` (short-time FFT) |
# MAGIC | 7 | Front → rear: the road arrives twice | UDF `xcorr_delay` (FFT cross-correlation) |
# MAGIC | 8 | The same UDFs in a distributed `.solve()` | one summary row per container |
# MAGIC
# MAGIC Prerequisite: `02a_tsal_deep_dive` §8 (the UDF rules). We stay with the `DefaultSolver`.

# COMMAND ----------

# MAGIC %run ../CONFIG

# COMMAND ----------

# MAGIC %pip install -q $IMPULSE_PIP_TARGET

# COMMAND ----------

setup_impulse()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from scipy import signal as sps
from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.analyze.query.solvers.default_solver import DefaultSolver
from impulse_query_engine.analyze.metadata.time_series_expression import TimeSeriesExpression
from impulse_query_engine.model.series.sample_series import SampleSeries
from impulse_query_engine.model.series.points_in_time_series import PointsInTimeSeries
from common.workshop_helpers import plot_signals, intervals_to_pandas, US

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Setup and the raw signal

# COMMAND ----------

db = MeasurementDB(MeasurementDBConfig(**impulse_source(SILVER), table_locations="unity_catalog"), WorkspaceClient())
solver = DefaultSolver(spark)
q = db.query
TRIP = 1000
display(spark.table(SILVER["container_metrics"]).where(F.col("container_id") == TRIP)
        .select("container_id", "vehicle_key", "vehicle_type", "scenario", "tires", "duration_s", "num_channels"))
display(spark.table(SILVER["container_tags"]).where(F.col("container_id") == TRIP).orderBy("key"))


def solve_trip(**expressions) -> pd.Series:
    """Solve named TSAL expressions for container TRIP and return them as one pandas row of Python objects."""
    selected = [expr.alias(name) for name, expr in expressions.items()]
    return q.where(q.metric("container_id") == TRIP).select(*selected).toPandas(spark, solver=solver).iloc[0]


def as_np(series):
    """SampleSeries -> (t [s since container start], values) numpy arrays, for plotting."""
    return (np.asarray(series.tstarts, dtype=float) - T0) / US, np.asarray(series.values, dtype=float)


def points_arrays(p):
    """PointsInTime / PointsInTimeSeries result -> (tstarts [µs or Hz], values or None) numpy arrays."""
    if p is None:
        return np.array([]), np.array([])
    if hasattr(p, "tstarts"):
        vals = getattr(p, "values", None)
        return np.asarray(p.tstarts, dtype=float), (None if vals is None else np.asarray(vals, dtype=float))
    rows = list(p)
    if rows and isinstance(rows[0], dict):
        return np.array([r["tstart"] for r in rows], float), np.array([r["value"] for r in rows], float)
    if rows and np.ndim(rows[0]) > 0:
        a = np.asarray(rows, dtype=float)
        return a[:, 0], a[:, 1]
    return np.asarray(rows, dtype=float), None


TIRES = ["FL", "FR", "RL", "RR"]
fz = {t: q.channel(channel_name=f"Car.Fz{t}") for t in TIRES}     # 1 kHz vertical tire load [N]
brake = q.channel(channel_name="DM.Brake")                          # 100 Hz pedal demand 0..1
speed = q.channel(channel_name="Vhcl.v")                            # 100 Hz [m/s]
ax = q.channel(channel_name="Car.ax")                               # 100 Hz [m/s²]

raw = solve_trip(brake=brake, speed=speed, **{f"fz_{t}": fz[t] for t in TIRES})
T0 = float(raw.speed.tstarts[0])
s = raw.fz_FL
print(f"Car.FzFL: {len(s):,} intervals, median period {np.median(np.diff(s.tstarts)) / 1e3:.3f} ms, "
      f"{len(s.continuous_interval_indices)} continuous segment(s)")
fig = plot_signals({"speed [m/s]": raw.speed, "brake pedal": raw.brake, "Fz FL [N]": raw.fz_FL, "Fz RL [N]": raw.fz_RL},
                   t0=T0, title="Container 1000: the whole session, 3 laps")

# COMMAND ----------

# MAGIC %md
# MAGIC Zoom on one hard stop from about 310 km/h. **Under braking the front tires gain about 2,500 N and the rears
# MAGIC lose about the same.** The change does not arrive as a clean step: the body pitches, overshoots and rings for
# MAGIC a second or two. Underneath, the load keeps jittering at hundreds of Hz: that is the road, and the part we
# MAGIC want to keep.

# COMMAND ----------

ZOOM = (86.5, 93.0)   # s: the first stop of lap 2 (event 7)
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for tire, col in zip(TIRES, ["C0", "C1", "C2", "C4"]):
    tt, vv = as_np(raw[f"fz_{tire}"])
    m = (tt >= ZOOM[0]) & (tt < ZOOM[1])
    axes[0].plot(tt[m], vv[m], lw=0.35, color=col, label=tire)
tb, vb = as_np(raw.brake)
axes[1].step(tb, vb, where="post", color="C3")
axes[0].set_ylabel("Fz [N]"); axes[1].set_ylabel("brake pedal"); axes[1].set_xlabel("time [s]")
axes[0].legend(ncol=4, fontsize=8)
for a in axes:
    a.grid(alpha=0.3)
axes[1].set_xlim(*ZOOM)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Edges: when is the brake applied and released?
# MAGIC
# MAGIC Everything that follows is organised around the brake edges, so find them first, **natively**:
# MAGIC
# MAGIC * `brake > 0.05` gives `Intervals` while the pedal is pressed. `.filter(0.2e6)` drops anything shorter
# MAGIC   than 0.2 s.
# MAGIC * `.start_points()` / `.end_points()` turn the intervals into `PointsInTime`: the **applied** and
# MAGIC   **released** edges.
# MAGIC
# MAGIC `brake.rising_edges()` looks tempting, but it fires on **every** sample where the pedal value rises. The
# MAGIC 60 ms ramp spans several 100 Hz samples, so one application produces several "edges". Thresholding into
# MAGIC intervals gives exactly one edge per application.

# COMMAND ----------

braking = (brake > 0.05).filter(0.2e6)              # Intervals: pedal pressed
applied = braking.start_points()                    # PointsInTime: brake applied
released = braking.end_points()                     # PointsInTime: brake released
speed_at_apply = (speed * 3.6).where(applied)       # PointsInTimeSeries: km/h at each application

r = solve_trip(braking=braking, applied=applied, released=released, speed_at_apply=speed_at_apply,
               rising=brake.rising_edges())
events = intervals_to_pandas(r.braking, T0)
events.insert(0, "event", range(1, len(events) + 1))
events["speed_at_apply_kmh"] = np.round(points_arrays(r.speed_at_apply)[1], 0)
print(f"{len(r.applied)} applications / {len(r.released)} releases (vs {len(r.rising)} raw rising edges)")
display(events[["event", "start_s", "end_s", "duration_s", "speed_at_apply_kmh"]].round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · A Butterworth filter as a UDF
# MAGIC
# MAGIC TSAL has no frequency-selective filter (`rolling_average` is the closest native operation), so this is a
# MAGIC legitimate UDF. `scipy.signal.butter(..., output="sos")` designs the filter. `sosfiltfilt` runs it forward
# MAGIC and backward, so the result has **zero phase delay**: the filtered brake edge stays exactly where it was.
# MAGIC
# MAGIC Applying the 02a UDF rules to a filter:
# MAGIC
# MAGIC | Rule | Here |
# MAGIC |---|---|
# MAGIC | handle empty input | `len(series) == 0` → `SampleSeries.empty()` (the planning call) |
# MAGIC | one return type | always a `SampleSeries` |
# MAGIC | **respect gaps** | filter each `continuous_interval_indices` segment separately: never across a gap |
# MAGIC | **uniform sampling** | a digital filter assumes a fixed sample period, but silver is **RLE-encoded**. Equal neighbouring values were merged (`Car.FzFL` has 228,401 intervals for 228,496 samples), so first `resample(1e6 / fs)` back onto the exact 1 ms grid |
# MAGIC
# MAGIC Non-series arguments (`cutoff_hz`, `btype`, `order`, `fs`) are passed straight through to the function, so
# MAGIC one UDF covers low-, high- and band-pass. `fs` is the rate of the input: 1000 for the tire loads, 100 for
# MAGIC `Car.ax`.

# COMMAND ----------

def uniform_segments(series, fs, min_len=1):
    """Resample onto an exact 1/fs grid, then yield (tstarts, tends, values) per gap-free segment."""
    u = series.resample(1e6 / fs)
    for a, b in u.continuous_interval_indices:
        if b - a + 1 >= min_len:
            yield u.tstarts[a:b + 1], u.tends[a:b + 1], u.values[a:b + 1]


def concat_series(parts) -> SampleSeries:
    if not parts:
        return SampleSeries.empty()
    return SampleSeries(*(np.concatenate([p[i] for p in parts]) for i in range(3)))


@TimeSeriesExpression.udf
def butterworth(series, cutoff_hz, btype="low", order=4, fs=1000.0):
    """Zero-phase Butterworth filter (scipy sosfiltfilt) per continuous segment. cutoff_hz: f or (f_lo, f_hi)."""
    if len(series) == 0:
        return SampleSeries.empty()
    sos = sps.butter(order, cutoff_hz, btype=btype, fs=fs, output="sos")
    padlen = 3 * (2 * len(sos) + 1)                      # sosfiltfilt needs more samples than its padding
    parts = [(ts, te, sps.sosfiltfilt(sos, v)) for ts, te, v in uniform_segments(series, fs, padlen + 1)]
    return concat_series(parts)


CROSSOVER_HZ = 5.0     # above the 1.5 Hz pitch mode, below the ~15 Hz wheel hop
lp = {t: butterworth(fz[t], CROSSOVER_HZ, btype="low") for t in TIRES}
hp = {t: butterworth(fz[t], CROSSOVER_HZ, btype="high") for t in TIRES}

# COMMAND ----------

# MAGIC %md
# MAGIC Solve raw, low-pass and high-pass for a front and a rear tire, plus the native 0.2 s `rolling_average`
# MAGIC for comparison.

# COMMAND ----------

r = solve_trip(fl=fz["FL"], rl=fz["RL"], lp_fl=lp["FL"], lp_rl=lp["RL"], hp_fl=hp["FL"], hp_rl=hp["RL"],
               ma_fl=fz["FL"].rolling_average(0.2e6))

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for name, col in [("fl", "C0"), ("rl", "C2")]:
    for ax_, key, lw, alpha in [(axes[0], name, 0.35, 0.9), (axes[1], f"lp_{name}", 1.4, 1), (axes[2], f"hp_{name}", 0.35, 0.9)]:
        tt, vv = as_np(r[key])
        m = (tt >= ZOOM[0]) & (tt < ZOOM[1])
        ax_.plot(tt[m], vv[m], lw=lw, alpha=alpha, color=col, label=name.upper())
for key, col, lw, lab in [("fl", "0.75", 0.35, "raw FL"), ("ma_fl", "C1", 1.2, "rolling_average 0.2 s (native)"),
                          ("lp_fl", "C0", 1.4, "Butterworth low-pass 5 Hz (UDF)")]:
    tt, vv = as_np(r[key])
    m = (tt >= ZOOM[0]) & (tt < ZOOM[1])
    axes[3].plot(tt[m], vv[m], lw=lw, color=col, label=lab)
for a, lab in zip(axes, ["raw Fz [N]", "low-pass < 5 Hz [N]", "high-pass > 5 Hz [N]", "FL: native vs UDF [N]"]):
    a.set_ylabel(lab); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper right")
for t_on in events.start_s:
    for a in axes:
        a.axvline(t_on, color="C3", lw=0.8, ls="--")
for t_off in events.end_s:
    for a in axes:
        a.axvline(t_off, color="C3", lw=0.8, ls=":")
axes[-1].set_xlim(*ZOOM); axes[-1].set_xlabel("time [s]  (red dashed: brake applied, dotted: released)")
fig.suptitle("One stop, split at 5 Hz: load transfer (low band) and road contact (high band)")
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC 👀 **What the split shows:**
# MAGIC * **Low band.** FL climbs by about 2,500 N and RL drops by the same amount, with the 1.5 Hz pitch overshoot
# MAGIC   on application **and** on release. Both bands also fall slowly during the stop, because the downforce
# MAGIC   (∝ v²) disappears as the car slows from 310 to 90 km/h.
# MAGIC * **High band.** It is centred on zero, with no trace of the 2,500 N step. Its *amplitude* changes, though:
# MAGIC   the loaded front tire gets louder and the unloaded rear quieter. That is the grip signal. The burst on
# MAGIC   FL about 0.4 s after the brake is applied is the lock-up judder that §6 detects automatically.
# MAGIC * **Native vs UDF (bottom panel).** The native `rolling_average` reacts to the brake edge late and bends the
# MAGIC   1.5 Hz overshoot out of shape, because it averages a trailing 0.2 s window. A boxcar is also a weak
# MAGIC   low-pass: its sinc response has sidelobes at only −13 dB. The zero-phase 4th-order Butterworth stays
# MAGIC   aligned with the edge, keeps the overshoot and rolls off at 24 dB/octave above 5 Hz. That is why the UDF
# MAGIC   is justified here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · The low band: load transfer and the pitch mode
# MAGIC
# MAGIC **How much load moved onto the front axle?** Two independent estimates:
# MAGIC
# MAGIC 1. **From the accelerometer (quasi-static physics):** `ΔF = −m · a_x · h_cg / wheelbase`.
# MAGIC 2. **From the tire loads:** front axle minus rear axle, after removing what isn't load transfer (the static
# MAGIC    weight split and the aero balance, which depends on v²).
# MAGIC
# MAGIC Both need the car's parameters. They are stored with the container (`container_metrics.empty_weight_kg` and
# MAGIC the setup-sheet `container_tags`), so the UDFs **declare** them and Impulse injects each container's values.
# MAGIC Nothing is hard-coded, and the same expression would work for a second car with a different setup.

# COMMAND ----------

SETUP = ["wheelbase_m", "cg_height_m", "front_weight_ratio", "aero_cl_a_m2", "aero_front_ratio"]


def _setup(container_metrics, container_tags):
    return float(container_metrics["empty_weight_kg"]), {k: float(container_tags[k]) for k in SETUP}


@TimeSeriesExpression.udf(container_metrics=["empty_weight_kg"], container_tags=SETUP)
def transfer_from_ax(ax_mps2, container_metrics, container_tags):
    """Quasi-static longitudinal load transfer onto the front axle [N]: -m a_x h / L."""
    if len(ax_mps2) == 0 or container_metrics["empty_weight_kg"] is None or container_tags["wheelbase_m"] is None:
        return SampleSeries.empty()
    m, c = _setup(container_metrics, container_tags)
    return SampleSeries(ax_mps2.tstarts, ax_mps2.tends, -m * ax_mps2.values * c["cg_height_m"] / c["wheelbase_m"])


@TimeSeriesExpression.udf(container_metrics=["empty_weight_kg"], container_tags=SETUP)
def axle_baseline(speed_mps, container_metrics, container_tags):
    """Front-minus-rear axle load with NO load transfer [N]: static split + aero balance (∝ v²)."""
    if len(speed_mps) == 0 or container_metrics["empty_weight_kg"] is None or container_tags["wheelbase_m"] is None:
        return SampleSeries.empty()
    m, c = _setup(container_metrics, container_tags)
    static = m * 9.81 * (2 * c["front_weight_ratio"] - 1)
    aero = 0.5 * 1.2 * c["aero_cl_a_m2"] * speed_mps.values ** 2 * (2 * c["aero_front_ratio"] - 1)
    return SampleSeries(speed_mps.tstarts, speed_mps.tends, static + aero)


# Native TSAL combines the 1 kHz tire loads with the 100 Hz speed and acceleration: alignment is automatic.
front_axle, rear_axle = lp["FL"] + lp["FR"], lp["RL"] + lp["RR"]
transfer_tires = (front_axle - rear_axle - axle_baseline(speed)) / 2           # measured at the tires
transfer_accel = transfer_from_ax(butterworth(ax, CROSSOVER_HZ, fs=100.0))     # quasi-static, from Car.ax
pitch_dynamics = transfer_tires - transfer_accel                               # what the body adds: the pitch mode

# Edges found in the TIRE signal itself: "the nose is loaded" = more than 600 N transferred, for at least 0.2 s
nose_loaded = (transfer_tires > 600).filter(0.2e6)

r = solve_trip(transfer_tires=transfer_tires, transfer_accel=transfer_accel, pitch=pitch_dynamics,
               nose_starts=nose_loaded.start_points(), applied=applied)

# COMMAND ----------

pedal_on, _ = points_arrays(r.applied)
nose_on, _ = points_arrays(r.nose_starts)
nxt = np.searchsorted(nose_on, pedal_on)                       # first tire-signal edge at/after each pedal edge
ok = nxt < len(nose_on)
lag_ms = (nose_on[nxt[ok]] - pedal_on[ok]) / 1e3
edge_table = pd.DataFrame({"event": np.arange(1, len(pedal_on) + 1)[ok], "pedal_applied_s": (pedal_on[ok] - T0) / US,
                           "tire_edge_s": (nose_on[nxt[ok]] - T0) / US, "lag_ms": lag_ms})
edge_table["speed_at_apply_kmh"] = events.set_index("event").loc[edge_table.event, "speed_at_apply_kmh"].values
print(f"{len(nose_on)} load-transfer edges found in the tire signal for {len(pedal_on)} pedal applications; "
      f"lag pedal -> load: median {np.median(lag_ms):.0f} ms (range {lag_ms.min():.0f}-{lag_ms.max():.0f} ms)")
display(edge_table.round(2))

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for key, col, lab in [("transfer_accel", "C7", "quasi-static, from Car.ax"), ("transfer_tires", "C0", "measured at the tires")]:
    tt, vv = as_np(r[key])
    m = (tt >= ZOOM[0]) & (tt < ZOOM[1])
    axes[0].plot(tt[m], vv[m], color=col, lw=1.3, label=lab)
tt, vv = as_np(r.pitch)
m = (tt >= ZOOM[0]) & (tt < ZOOM[1])
axes[1].plot(tt[m], vv[m], color="C5", lw=1.2)
for a in axes:
    a.grid(alpha=0.3)
    for t_on in edge_table.pedal_applied_s:
        a.axvline(t_on, color="C3", lw=0.8, ls="--")
    for t_e in edge_table.tire_edge_s:
        a.axvline(t_e, color="C0", lw=0.8, ls=":")
axes[0].set_ylabel("load onto front\naxle [N]"); axes[0].legend(fontsize=8)
axes[1].set_ylabel("tires − accel\n= pitch dynamics [N]")
axes[1].set_xlim(*ZOOM); axes[1].set_xlabel("time [s]  (red dashed: pedal edge, blue dotted: edge in the tire signal)")
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC **The pitch frequency by FFT.** The bottom trace, *tires minus accelerometer*, is the part of the transfer
# MAGIC the quasi-static formula can't explain: the body's pitch oscillation. Its frequency is a property of the
# MAGIC chassis (spring rates, pitch inertia), useful for checking a setup change.
# MAGIC
# MAGIC The UDF `dominant_frequency` takes the series inside each window, removes the mean, applies a Hann window,
# MAGIC zero-pads to 2¹⁶ points for a fine frequency grid, sums the power of all windows and returns the frequency of
# MAGIC the highest peak in `[fmin, fmax]`. The windows are each brake application plus 1.5 s, so they include the
# MAGIC ringing after the pedal edge: `braking.expand_right(1.5e6)`.

# COMMAND ----------

@TimeSeriesExpression.udf
def dominant_frequency(series, fmin, fmax, fs=1000.0, nfft=2**16):
    """Frequency [Hz] of the strongest spectral peak in [fmin, fmax], power summed over all gap-free segments."""
    if len(series) == 0:
        return np.nan
    grid = np.linspace(fmin, fmax, 4000)
    total = np.zeros_like(grid)
    for _, _, v in uniform_segments(series, fs, min_len=16):
        x = (v - v.mean()) * np.hanning(len(v))
        n = max(nfft, len(x))
        power = np.abs(np.fft.rfft(x, n=n)) ** 2
        total += np.interp(grid, np.fft.rfftfreq(n, 1 / fs), power)
    return float(grid[np.argmax(total)]) if total.any() else np.nan


after_braking = braking.expand_right(1.5e6)             # each application + 1.5 s of ringing
r = solve_trip(pitch_hz=dominant_frequency(pitch_dynamics.where(after_braking), 0.5, 5.0),
               rear_lp_hz=dominant_frequency(butterworth(lp["RL"], (0.5, 5.0), btype="band").where(after_braking), 0.5, 5.0))
print(f"pitch mode from (tires - accelerometer): {r.pitch_hz:.2f} Hz | from the rear low band alone: {r.rear_lp_hz:.2f} Hz")

# COMMAND ----------

# MAGIC %md
# MAGIC The residual lands on **≈ 1.5 Hz**, the pitch mode the simulation was built with. The rear tire's low band
# MAGIC **alone** does not, even after a 0.5–5 Hz band-pass (the same UDF with `btype="band"`). In a 2–5 s window,
# MAGIC most of its energy is the load *step* and the downforce decay, not the ringing, so the FFT peak sits near
# MAGIC 1 Hz. Subtracting the quasi-static physics first is what isolates the oscillation. When an FFT answer looks
# MAGIC wrong, check what else is in the window.
# MAGIC
# MAGIC **Edges in the tire signal.** The load-transfer edge (blue, dotted) trails the pedal edge (red, dashed) by
# MAGIC **≈ 85–145 ms**: the time the body needs to pitch. Light stops take longer to cross the 600 N threshold.
# MAGIC Telemetry engineers watch this lag, because a change in it points to a damper or brake-system issue.
# MAGIC
# MAGIC ## 5 · The high band: spectra and the grip signal
# MAGIC
# MAGIC **Which frequencies make up the road signal, and how do they change under braking?** The UDF
# MAGIC `amplitude_spectrum` computes a Welch spectrum (averaged, windowed FFT segments) over every gap-free segment
# MAGIC of its input.
# MAGIC
# MAGIC **What does a spectrum return?** It isn't a function of time, so it fits none of the four result types
# MAGIC exactly. We return it as a **`PointsInTimeSeries` whose "time" axis holds the frequency in Hz** and whose
# MAGIC values hold the amplitude density in N/√Hz. That keeps the result a proper TSAL type: it comes back from
# MAGIC `.solve()` as `array<array<double>>` `[[f, amplitude], …]`, which SQL can explode for a dashboard. Remember
# MAGIC the convention, and don't combine this "time" axis with real timestamps.

# COMMAND ----------

@TimeSeriesExpression.udf
def amplitude_spectrum(series, fmax=500.0, nperseg=1024, fs=1000.0):
    """Welch amplitude spectral density [unit/√Hz] over all gap-free segments -> PointsInTimeSeries(x = f [Hz])."""
    if len(series) == 0:
        return PointsInTimeSeries.empty()
    acc, n_total, f = None, 0, None
    for _, _, v in uniform_segments(series, fs, min_len=nperseg // 4):
        f, p = sps.welch(v, fs=fs, nperseg=min(nperseg, len(v)), nfft=nperseg)
        acc = p * len(v) if acc is None else acc + p * len(v)        # weight segments by their length
        n_total += len(v)
    if acc is None:
        return PointsInTimeSeries.empty()
    keep = (f > 0) & (f <= fmax)
    return PointsInTimeSeries(f[keep], np.sqrt(acc[keep] / n_total))


steady = (brake < 0.01) & (speed > 30)                  # not braking, above 108 km/h
spectra = {f"{t}_{phase}": amplitude_spectrum(hp[t].where(win))
           for t in ("FL", "RL") for phase, win in [("braking", braking), ("steady", steady)]}
r = solve_trip(**spectra)

fig, ax_ = plt.subplots(1, 1, figsize=(13, 5))
for (key, col, ls) in [("FL_braking", "C0", "-"), ("FL_steady", "C0", ":"), ("RL_braking", "C2", "-"), ("RL_steady", "C2", ":")]:
    f_hz, amp = points_arrays(r[key])                 # x axis = frequency [Hz]
    ax_.semilogy(f_hz, amp, color=col, ls=ls, lw=1.1, label=key.replace("_", " · "))
ax_.axvline(15, color="0.5", lw=0.8); ax_.text(16, ax_.get_ylim()[1] * 0.5, "wheel hop ≈ 15 Hz", fontsize=8)
ax_.set(xlabel="frequency [Hz]", ylabel="amplitude density [N/√Hz]", xlim=(0, 500),
        title="High band of Fz: braking (solid) vs steady driving (dotted), front-left vs rear-left")
ax_.grid(alpha=0.3, which="both"); ax_.legend()
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC **The grip signal as one number per tire and phase:** the RMS of the high band. This is all native TSAL:
# MAGIC `(hp * hp).where(window).mean()` is the time-weighted mean square, and the square root is taken in pandas.

# COMMAND ----------

ms = {f"{t}_{phase}": (hp[t] * hp[t]).where(win).mean() for t in TIRES
      for phase, win in [("braking", braking), ("steady", steady)]}
r = solve_trip(**ms)
grip = pd.DataFrame([(t, np.sqrt(r[f"{t}_steady"]), np.sqrt(r[f"{t}_braking"])) for t in TIRES],
                    columns=["tire", "hf_rms_steady_n", "hf_rms_braking_n"])
grip["change_pct"] = (grip.hf_rms_braking_n / grip.hf_rms_steady_n - 1) * 100
display(grip.round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC 👀 Across the band, the **front** spectra rise under braking and the **rear** spectra fall. The table puts
# MAGIC numbers on it: front high-band RMS **up**, rear **down**. A loaded tire presses harder into the surface, so
# MAGIC its contact patch transmits more of the road texture. An unloaded rear tire skims over it.
# MAGIC
# MAGIC Three details worth noting:
# MAGIC * The plateau at **180–230 Hz** appears only in *steady* driving: it is the grooved stretch of the main
# MAGIC   straight (§6b), crossed at 230–310 km/h. The 85 Hz bump in *FL · braking* is the lock-up judder (§6a).
# MAGIC * Both axles share the **wheel-hop hump near 15 Hz**. That is the unsprung mass bouncing between the tire
# MAGIC   and the spring, and it is why the crossover sits at 5 Hz, well below it.
# MAGIC * The raw signal is dominated by the 2,500 N load-transfer swing, so in it these differences of a few
# MAGIC   tens of N are invisible. The high-pass is what makes them measurable.
# MAGIC
# MAGIC ## 6 · Time–frequency: when does something happen in a band?
# MAGIC
# MAGIC A spectrum averages over time. To find **when** something happens in a band, use a short-time FFT: one FFT
# MAGIC per sliding window, which gives one value per window. Both UDFs below return a `SampleSeries`, so the result
# MAGIC goes straight back into native interval algebra.
# MAGIC
# MAGIC * `band_rms(series, f_lo, f_hi, window_s)`: RMS of one band per window (Parseval on the Hann-windowed FFT).
# MAGIC * `peak_frequency(series, f_lo, f_hi, window_s, min_prominence_db)`: the frequency of the strongest line in
# MAGIC   a band. Windows without a clear line (peak less than `min_prominence_db` above the band median) are
# MAGIC   dropped, which leaves **gaps** in the result.

# COMMAND ----------

def _frames(v, nper):
    hop = nper // 2
    return np.lib.stride_tricks.sliding_window_view(v, nper)[::hop], hop


@TimeSeriesExpression.udf
def band_rms(series, f_lo, f_hi, window_s=0.1, fs=1000.0):
    """RMS of the [f_lo, f_hi] band in sliding Hann windows (50 % overlap) -> SampleSeries, one value per hop."""
    if len(series) == 0:
        return SampleSeries.empty()
    nper = int(window_s * fs)
    w = np.hanning(nper)
    f = np.fft.rfftfreq(nper, 1 / fs)
    band = (f >= f_lo) & (f <= f_hi)
    parts = []
    for ts, te, v in uniform_segments(series, fs, min_len=nper):
        frames, hop = _frames(v, nper)
        power = np.abs(np.fft.rfft(frames * w, axis=1)[:, band]) ** 2
        rms = np.sqrt(2 * power.sum(axis=1) / (nper * (w ** 2).sum()))
        starts = ts[np.arange(len(frames)) * hop + nper // 4]         # each value covers the middle half-window
        ends = np.append(starts[1:], starts[-1] + hop * 1e6 / fs)
        parts.append((starts, ends, rms))
    return concat_series(parts)


@TimeSeriesExpression.udf
def peak_frequency(series, f_lo, f_hi, window_s=0.25, min_prominence_db=12.0, fs=1000.0, pad=8):
    """Frequency [Hz] of the strongest line in [f_lo, f_hi] per sliding window; windows without a clear line dropped."""
    if len(series) == 0:
        return SampleSeries.empty()
    nper = int(window_s * fs)
    nfft = pad * nper                                            # zero-padding: a finer frequency grid
    w = np.hanning(nper)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= f_lo) & (f <= f_hi)
    parts = []
    for ts, te, v in uniform_segments(series, fs, min_len=nper):
        frames, hop = _frames(v - v.mean(), nper)
        power = np.abs(np.fft.rfft(frames * w, n=nfft, axis=1)[:, band]) ** 2
        prominence_db = 10 * np.log10(power.max(axis=1) / np.median(power, axis=1))
        peak_hz = f[band][power.argmax(axis=1)]
        starts = ts[np.arange(len(frames)) * hop + nper // 4]
        keep = prominence_db >= min_prominence_db
        if keep.any():                                           # isolated windows -> each its own short interval
            parts.append((starts[keep], starts[keep] + hop * 1e6 / fs, peak_hz[keep]))
    return concat_series(parts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a · Lock-up / judder detection
# MAGIC Front-wheel judder under braking shows up as a burst of energy at **60–120 Hz**. Rule: the band RMS is
# MAGIC above **250 N** (normal road texture in that band is about 70 N), **while braking**, for **at least
# MAGIC 0.15 s**. The last condition rejects pebbles, which are broadband but last only a few milliseconds.
# MAGIC Everything after the UDF is native interval algebra.

# COMMAND ----------

judder_band = {t: band_rms(hp[t], 60.0, 120.0, window_s=0.1) for t in TIRES}
judder = {t: ((judder_band[t] > 250) & braking).filter(0.15e6) for t in TIRES}
r = solve_trip(**{f"band_{t}": judder_band[t] for t in TIRES}, **{f"judder_{t}": judder[t] for t in TIRES})

rows = []
for t in TIRES:
    for _, w in intervals_to_pandas(r[f"judder_{t}"], T0).iterrows():
        ev = events[(events.start_s <= w.start_s) & (events.end_s >= w.start_s)].event
        rows.append((t, int(ev.iloc[0]) if len(ev) else None, w.start_s, w.duration_s))
detected = pd.DataFrame(rows, columns=["tire", "brake_event", "start_s", "duration_s"])
print("expected (simulation ground truth): FL at brake event 7, FR at brake event 17")
display(detected.round(2))

fig = plot_signals({f"60–120 Hz RMS {t} [N]": r[f"band_{t}"] for t in TIRES}, t0=T0, intervals=None,
                   title="Judder band RMS per tire (dashed: 250 N threshold)")
for a in fig.axes:
    a.axhline(250, color="C3", ls="--", lw=0.8)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b · The pavement grooves: an FFT measures the road
# MAGIC Part of the main straight has transverse grooves. A wheel rolling over grooves with pitch *p* at speed *v* is
# MAGIC excited at **f = v / p**, a narrow line that **sweeps upward as the car accelerates**. `peak_frequency` tracks
# MAGIC that line in the 150–260 Hz band. Then plain TSAL arithmetic, `speed / groove_hz`, turns every window into
# MAGIC an estimate of the groove pitch, in metres. The 100 Hz speed is aligned to the windows automatically.

# COMMAND ----------

groove_hz = peak_frequency(hp["FL"], 150.0, 260.0, window_s=0.25, min_prominence_db=20.0)   # 20 dB: only clear lines
groove_pitch_m = speed / groove_hz
r = solve_trip(groove_hz=groove_hz, groove_pitch=groove_pitch_m, pitch_mean=groove_pitch_m.mean(), hp_fl=hp["FL"],
               speed=speed, braking=braking)
gp = np.asarray(r.groove_pitch.values, dtype=float)
print(f"{len(r.groove_hz)} windows with a clear 150-260 Hz line | groove pitch: mean {r.pitch_mean:.3f} m, "
      f"median {np.median(gp):.3f} m (simulated: 0.350 m)")

# COMMAND ----------

# MAGIC %md
# MAGIC **The spectrogram** shows all of it at once. It is display only: `scipy.signal.spectrogram` runs on the
# MAGIC high-pass result the TSAL query returned. Overlaid are the brake edges (§2), the groove track found by the
# MAGIC `peak_frequency` UDF (§6b) and the judder windows (§6a).

# COMMAND ----------

tt, vv = as_np(r.hp_fl)
f_sg, t_sg, S = sps.spectrogram(vv, fs=1000, nperseg=256, noverlap=192)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
axes[0].pcolormesh(t_sg + tt[0], f_sg, 10 * np.log10(S + 1e-6), shading="auto", cmap="magma", vmin=-5, vmax=35)
gt, gv = as_np(r.groove_hz)
axes[0].plot(gt, gv, ".", color="cyan", ms=3, label="peak_frequency UDF: groove line")
for _, w in detected[detected.tire == "FL"].iterrows():
    axes[0].axvspan(w.start_s, w.start_s + w.duration_s, ymin=0.1, ymax=0.25, color="lime", alpha=0.5, label="judder (FL)")
for a in axes:
    for t_on, t_off in zip(events.start_s, events.end_s):
        a.axvline(t_on, color="white" if a is axes[0] else "C3", lw=0.7, ls="--")
        a.axvline(t_off, color="white" if a is axes[0] else "C3", lw=0.7, ls=":")
axes[0].set(ylabel="frequency [Hz]", ylim=(0, 500), title="Front-left high band: spectrogram with brake edges")
axes[0].legend(loc="upper right", fontsize=8)
st, sv = as_np(r.speed)
axes[1].plot(st, sv * 3.6, color="C0"); axes[1].set(ylabel="speed [km/h]", xlabel="time [s]")
axes[1].grid(alpha=0.3)
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC 👀 Reading the spectrogram:
# MAGIC * **Vertical stripes** (broadband) are pebbles and kerb strikes. Kerbs hit just after each corner exit, on
# MAGIC   alternating sides.
# MAGIC * **The rising cyan line** is the grooved section. Its frequency grows with speed, and v / f recovers
# MAGIC   the 0.35 m groove pitch.
# MAGIC * **Under braking** (between the dashed and dotted lines) the front's high band gets brighter.
# MAGIC * **The energy below ~60 Hz** (road texture plus the 15 Hz wheel hop) follows speed: faster means more energy.
# MAGIC
# MAGIC ## 7 · Front → rear: the road arrives twice
# MAGIC Each rear wheel rolls over the same road as the front wheel on its side, `wheelbase / v` later. Cross-correlate
# MAGIC the front and rear high bands and the lag of the correlation peak is that delay. Multiplied by the speed, it
# MAGIC gives the wheelbase, **measured from the tire signals alone**.
# MAGIC
# MAGIC `xcorr_delay(lead, lag)` synchronizes its two inputs (rule 4: never pair raw arrays of two series), then
# MAGIC runs an FFT cross-correlation (`scipy.signal.correlate(method="fft")`) per gap-free segment. It returns a
# MAGIC `PointsInTimeSeries`: **one delay per window**, stamped at the window start. The windows are the
# MAGIC constant-speed corners: no pedal and |a_x| < 0.5 m/s² for at least 1 s.

# COMMAND ----------

@TimeSeriesExpression.udf
def xcorr_delay(lead, lag, max_delay_s=0.3, fs=1000.0):
    """Delay [s] of `lag` behind `lead` per gap-free segment (FFT cross-correlation) -> PointsInTimeSeries."""
    if len(lead) == 0 or len(lag) == 0:
        return PointsInTimeSeries.empty()
    a_s, b_s = lead.synchronized(lag)
    if len(a_s) == 0:
        return PointsInTimeSeries.empty()
    times, delays = [], []
    for ts, _, va in uniform_segments(a_s, fs, min_len=int(0.5 * fs)):
        vb = np.interp(ts, b_s.tstarts, b_s.values)                    # lag on the same uniform grid
        c = sps.correlate(vb - vb.mean(), va - va.mean(), mode="full", method="fft")
        lags = sps.correlation_lags(len(vb), len(va))
        ok = (lags >= 0) & (lags <= max_delay_s * fs)
        times.append(ts[0])
        delays.append(lags[ok][np.argmax(c[ok])] / fs)
    return PointsInTimeSeries(times, delays)


ax_lp = butterworth(ax, CROSSOVER_HZ, fs=100.0)
constant_speed = ((brake < 0.01) & (ax_lp > -0.5) & (ax_lp < 0.5)).filter(1e6)
delay_left = xcorr_delay(hp["FL"].where(constant_speed), hp["RL"].where(constant_speed))
delay_right = xcorr_delay(hp["FR"].where(constant_speed), hp["RR"].where(constant_speed))
r = solve_trip(delay_left=delay_left, delay_right=delay_right,
               wb_left=delay_left * speed, wb_right=delay_right * speed,      # PointsInTimeSeries × SampleSeries
               v_at=speed.where(delay_left.to_points_in_time()))
(t_l, d_l), (_, d_r) = points_arrays(r.delay_left), points_arrays(r.delay_right)
wb = pd.DataFrame({"t_s": (t_l - T0) / US, "speed_kmh": points_arrays(r.v_at)[1] * 3.6,
                   "delay_left_ms": d_l * 1e3, "delay_right_ms": d_r * 1e3,
                   "wheelbase_left_m": points_arrays(r.wb_left)[1], "wheelbase_right_m": points_arrays(r.wb_right)[1]})
print(f"{len(wb)} constant-speed windows | wheelbase from the tires: left {wb.wheelbase_left_m.median():.2f} m, "
      f"right {wb.wheelbase_right_m.median():.2f} m (setup sheet: 2.45 m)")
display(wb.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · The same UDFs in a distributed `.solve()`
# MAGIC So far every query ran for one container with `.toPandas()`. The UDFs are ordinary TSAL expressions, though:
# MAGIC `.solve()` runs them on the Spark executors for **every container that matches**, one row per container.
# MAGIC Here the filter is the scenario tag, and only container 1000 has it. Point it at a fleet of
# MAGIC tire-load containers and the same code scales out.

# COMMAND ----------

summary = (
    q.havingTag(scenario="tire_load_test")
     .select(dominant_frequency(pitch_dynamics.where(after_braking), 0.5, 5.0).alias("pitch_mode_hz"),
             transfer_tires.max().alias("max_load_transfer_n"),
             nose_loaded.alias("nose_loaded_windows"),
             *[(hp[t] * hp[t]).where(steady).mean().alias(f"hf_ms_{t}_steady") for t in TIRES],
             *[(hp[t] * hp[t]).where(braking).mean().alias(f"hf_ms_{t}_braking") for t in TIRES],
             *[judder[t].alias(f"judder_{t}") for t in TIRES],
             groove_pitch_m.mean().alias("groove_pitch_m"),
             (delay_left * speed).mean().alias("wheelbase_from_tires_m"),
             amplitude_spectrum(hp["FL"].where(braking)).alias("spectrum_fl_braking"))
     .solve(spark, solver=solver))
summary = (summary
           .withColumn("n_load_transfer_edges", F.size("nose_loaded_windows"))
           .select("container_id", "pitch_mode_hz", "max_load_transfer_n", "n_load_transfer_edges",
                   *[F.sqrt(f"hf_ms_{t}_steady").alias(f"hf_rms_{t}_steady_n") for t in TIRES],
                   *[F.sqrt(f"hf_ms_{t}_braking").alias(f"hf_rms_{t}_braking_n") for t in TIRES],
                   *[F.size(f"judder_{t}").alias(f"n_judder_{t}") for t in TIRES],
                   "groove_pitch_m", "wheelbase_from_tires_m", "spectrum_fl_braking"))
display(summary.drop("spectrum_fl_braking"))

# The spectrum arrives as array<array<double>> [[f, amplitude], ...]: explode it like any array column.
display(summary.select("container_id", F.explode("spectrum_fl_braking").alias("p"))
        .select("container_id", F.col("p")[0].alias("frequency_hz"), F.col("p")[1].alias("amplitude_n_per_sqrt_hz"))
        .where("frequency_hz <= 300"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Recap
# MAGIC
# MAGIC | Question | How | Result on container 1000 |
# MAGIC |---|---|---|
# MAGIC | When is the brake applied / released? | native: `(brake > 0.05).filter(0.2e6).start_points()` / `.end_points()` | 18 stops |
# MAGIC | Split load transfer from road contact | UDF `butterworth` (SciPy `butter` + `sosfiltfilt`), 5 Hz crossover | clean, zero-phase bands |
# MAGIC | How much load moved, and how fast? | metadata-aware UDFs + native arithmetic + interval edges | ≈ 2,500 N per tire, 85–145 ms behind the pedal |
# MAGIC | What is the pitch mode? | UDF `dominant_frequency` (zero-padded FFT) | ≈ 1.5 Hz |
# MAGIC | Does the grip signal respond to load? | UDF `amplitude_spectrum` + native mean square | fronts up, rears down under braking |
# MAGIC | Did a wheel lock up? | UDF `band_rms` (STFT) + native `&`, `.filter()` | FL in stop 7, FR in stop 17 |
# MAGIC | What is the road like? | UDF `peak_frequency` + native `speed / f` | 0.35 m groove pitch |
# MAGIC | Front → rear delay | UDF `xcorr_delay` (FFT cross-correlation) | the wheelbase, ≈ 2.45 m |
# MAGIC
# MAGIC **UDF rules for signal processing**, on top of 02a's four:
# MAGIC 1. **Resample to a uniform grid first.** Silver is RLE-encoded, but a digital filter or FFT needs a fixed
# MAGIC    sample period.
# MAGIC 2. **One segment at a time.** Filter / FFT each gap-free segment separately (after `.where(...)` every
# MAGIC    window is its own segment). Skip segments shorter than the filter's padding.
# MAGIC 3. **Zero-phase filtering** (`sosfiltfilt`) keeps the edges where they are, so filtered signals still line up
# MAGIC    with native intervals and points.
# MAGIC 4. **Return a TSAL type.** A time-resolved result is a `SampleSeries` (band RMS, peak track). A spectrum is
# MAGIC    a `PointsInTimeSeries` with frequency on the x axis. A single feature is a float.
