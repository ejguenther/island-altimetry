"""Shared plotting helpers (matplotlib, Agg; no basemaps or online tiles)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PRODUCT_ORDER = ["atl03", "atl08", "atl09", "atl24", "gedi_l2a"]
PRODUCT_LABEL = {"atl03": "ATL03 r007", "atl08": "ATL08 r007", "atl09": "ATL09 r007",
                 "atl24": "ATL24 v002", "gedi_l2a": "GEDI L2A v002"}
COLORS = {"atl03": "#2a6fdb", "atl08": "#2e9d5b", "atl09": "#8a63c9", "atl24": "#e07b24", "gedi_l2a": "#c43d5a"}


def add_events(ax, events: pd.DataFrame, labels: bool = True):
    """Dashed lines at sourced events; labels stacked in the upper-left of the axes
    (events cluster in time, so inline labels would overlap)."""
    lines = []
    for k, (_, e) in enumerate(events.sort_values("start").iterrows(), 1):
        t = pd.Timestamp(e["start"])
        ax.axvline(t, color="0.35", lw=0.8, ls="--", zorder=0)
        if labels:
            ax.annotate(str(k), (t, 1.0), xycoords=("data", "axes fraction"), xytext=(2 + 7 * (k - 1), -2),
                        textcoords="offset points", va="top", fontsize=6.5, color="0.25")
            lines.append(f"{k}: {e['start'][:10]} {e['id'].replace('_', ' ')} ({e['precision']})")
    if labels and lines:
        ax.text(0.005, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=6.5, color="0.25",
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))


def timeline(inv: dict[str, pd.DataFrame], events: pd.DataFrame, title: str, run_end: str,
             path, kind: str = "candidate granules"):
    """One row per product, one tick per acquisition."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    prods = [p for p in PRODUCT_ORDER if p in inv]
    for i, p in enumerate(prods):
        df = inv[p]
        if df.empty:
            continue
        t = pd.to_datetime(df["acq_start_utc"], utc=True)
        core = df.get("footprint_intersects_core", pd.Series(True, index=df.index)).astype(bool)
        ax.scatter(t[core], [i] * core.sum(), marker="|", s=180, color=COLORS[p], lw=1.4)
        ax.scatter(t[~core], [i] * (~core).sum(), marker="|", s=90, color=COLORS[p], lw=1.0, alpha=0.45)
    ax.set_yticks(range(len(prods)), [f"{PRODUCT_LABEL[p]} (n={len(inv[p])})" for p in prods])
    ax.set_ylim(-0.7, len(prods) - 0.3)
    ax.axvline(pd.Timestamp(run_end), color="k", lw=1)
    add_events(ax, events)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Acquisition date (UTC)")
    ax.set_title(f"{title}\n{kind}; tall ticks = catalog footprint reaches core zone, faint = outer only; "
                 f"black line = run end {run_end}", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
