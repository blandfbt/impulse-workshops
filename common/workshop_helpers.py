"""Small plotting / formatting helpers for the Impulse workshops.

They only *display* Impulse results -- every computation in the labs is done by Impulse itself.
Time in the silver layer is epoch microseconds; the helpers show seconds since the trip start.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

US = 1e6  # microseconds per second


def _as_pairs(intervals) -> list:
    """Intervals object / nested list / numpy array / None -> list of (start, end)."""
    if intervals is None:
        return []
    if hasattr(intervals, "get_data"):
        intervals = intervals.get_data()
    return [tuple(p) for p in list(intervals)]


def points_to_pandas(points, value_name: str = "value", t0: float = 0.0) -> pd.DataFrame:
    """PointsInTimeSeries result ([[t, value], ...] / object array / structs) -> DataFrame[t_us, t_s, value]."""
    rows = []
    for p in ([] if points is None else list(points)):
        if isinstance(p, dict):
            rows.append((float(p["tstart"]), p["value"]))
        else:
            p = list(p)
            rows.append((float(p[0]), p[1]))
    df = pd.DataFrame(rows, columns=["t_us", value_name])
    df["t_s"] = (df["t_us"] - t0) / US
    return df


def series_to_pandas(series, t0: float | None = None, name: str = "value") -> pd.DataFrame:
    """A SampleSeries -> DataFrame[t_s, <name>] (one row per interval start)."""
    tstarts = np.asarray(series.tstarts, dtype=float)
    t0 = tstarts[0] if t0 is None and len(tstarts) else (t0 or 0.0)
    return pd.DataFrame({"t_s": (tstarts - t0) / US, name: np.asarray(series.values, dtype=float)})


def intervals_to_pandas(intervals, t0: float = 0.0) -> pd.DataFrame:
    """Intervals result (nested [[start, end], ...] list or Intervals object) -> DataFrame."""
    data = _as_pairs(intervals)
    rows = [(float(s), float(e)) for s, e in data]
    df = pd.DataFrame(rows, columns=["start_us", "end_us"])
    df["start_s"] = (df["start_us"] - t0) / US
    df["end_s"] = (df["end_us"] - t0) / US
    df["duration_s"] = df["end_s"] - df["start_s"]
    return df


def plot_signals(signals: dict, title: str = "", t0: float | None = None, intervals=None,
                 points=None, max_points: int = 20_000, height: float = 2.2):
    """Plot one or more SampleSeries on stacked axes sharing the time axis.

    signals   : {"label": SampleSeries}
    intervals : optional Intervals (nested list) to shade on every axis
    points    : optional {"label": [t_us, ...]} instants to mark with vertical lines
    """
    if t0 is None:
        starts = [s.tstarts[0] for s in signals.values() if len(s)]
        t0 = float(min(starts)) if starts else 0.0
    fig, axes = plt.subplots(len(signals), 1, figsize=(13, height * len(signals)), sharex=True, squeeze=False)
    for ax, (label, s) in zip(axes[:, 0], signals.items()):
        if len(s):
            step = max(1, len(s) // max_points)  # thin very long series for plotting only
            t = (np.asarray(s.tstarts[::step], dtype=float) - t0) / US
            ax.step(t, np.asarray(s.values[::step], dtype=float), where="post", lw=0.8)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(alpha=0.3)
        if intervals is not None:
            for start, end in _as_pairs(intervals):
                ax.axvspan((start - t0) / US, (end - t0) / US, color="tab:orange", alpha=0.25, lw=0)
        for i, (plabel, ts) in enumerate((points or {}).items()):
            for j, tp in enumerate(ts):
                ax.axvline((tp - t0) / US, color=f"C{i + 2}", lw=0.8, alpha=0.7,
                           label=plabel if (j == 0 and ax is axes[0, 0]) else None)
    if points:
        axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time since trip start [s]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_histogram(lower, upper, values, title: str = "", xlabel: str = "", ylabel: str = "time [s]"):
    """Bar chart of histogram bins (e.g. rows of a histogram_fact table)."""
    lower, upper, values = (np.asarray(x, dtype=float) for x in (lower, upper, values))
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(lower, values, width=upper - lower, align="edge", edgecolor="white")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
