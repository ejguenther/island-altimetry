#!/usr/bin/env python
"""M6: coverage maps and timelines from cached chunks and M5 tables (no network,
no imagery or map tiles).

  plot_coverage.py --site fukutoku_okanoba [--run-id RUN]

Figures -> outputs/<site>/<run>/figures/; entries appended to plot_index.csv.
Displayed points may be thinned; all counts shown are computed from full data.
"""

from __future__ import annotations

import argparse
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import shape

from island_altimetry import config, manifest, normalize, quality
from island_altimetry.plotting import COLORS, PRODUCT_LABEL, PRODUCT_ORDER, add_events
from scripts_path import load_acquire
from summarize import iter_views  # noqa: E402

acq = load_acquire()
MAX_PTS_PER_GRANULE = 4000
UNIT_LABEL = {"atl03": "photons", "atl08": "segments", "atl09": "profiles", "atl24": "photons", "gedi_l2a": "footprints"}


def plot_index_append(out_dir, rows):
    p = out_dir / "plot_index.csv"
    df = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        df = pd.concat([old[~old["figure"].isin(df["figure"])], df], ignore_index=True)
    df.to_csv(p, index=False)


def zones(m):
    d = config.ROOT / m["aoi_dir"]
    return {z: shape(json.loads((d / f"{z}.geojson").read_text())["features"][0]["geometry"]) for z in ("core", "outer")}


def draw_aoi(ax, Z, cfg, local=False, zoner=None):
    for z, ls in (("core", "-"), ("outer", "--")):
        x, y = Z[z].exterior.xy
        if local:
            x, y = zoner.xy(np.asarray(x), np.asarray(y))
        ax.plot(x, y, color="k", lw=1, ls=ls, label=f"{z} {'search zone' if z == 'core' else 'AOI'} boundary")
    c = cfg["center"]
    cx, cy = (0, 0) if local else (c["lon"], c["lat"])
    ax.plot(cx, cy, marker="*", ms=12, color="gold", mec="k", label="GVP volcano reference point", zorder=5)
    for f in cfg.get("nearby_features", []):
        if "approx_center" in f:
            fx, fy = f["approx_center"]["lon"], f["approx_center"]["lat"]
            if local:
                fx, fy = zoner.xy(np.array([fx]), np.array([fy]))
                fx, fy = float(fx[0]), float(fy[0])
            ax.plot(fx, fy, marker="^", ms=8, color="0.4", mec="k", zorder=5)
            ax.annotate(f"{f['name']} (approx.)", (fx, fy), textcoords="offset points", xytext=(5, 5), fontsize=7)


def collect_points(ctx, zoner, rng):
    """Thinned baseline points per product for maps, plus full counts."""
    pts, counts = {}, {}
    for pid in PRODUCT_ORDER:
        if pid not in ctx["cfg"]["products"]:
            continue
        rows, n_full = [], {"rows": 0, "baseline": 0, "baseline_core": 0, "acquisitions": 0}
        for r, meta, v, _ in iter_views(ctx, pid, zoner):
            v, _ = quality.apply(pid, v)
            b = v[v["q_baseline"]]
            n_full["rows"] += len(v)
            n_full["baseline"] += len(b)
            n_full["baseline_core"] += int((b["zone"] == "core").sum())
            n_full["acquisitions"] += 1
            if len(b) > MAX_PTS_PER_GRANULE:
                b = b.iloc[np.sort(rng.choice(len(b), MAX_PTS_PER_GRANULE, replace=False))]
            lon = b["lon_mean"] if pid == "atl09" else b["lon"]
            lat = b["lat_mean"] if pid == "atl09" else b["lat"]
            rows.append(pd.DataFrame({"lon": lon.values, "lat": lat.values, "x": b["x_local_m"].values,
                                      "y": b["y_local_m"].values, "t": pd.to_datetime(r["acq_start_utc"], utc=True)}))
        pts[pid] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        counts[pid] = n_full
    return pts, counts


def maps(ctx, Z, zoner, pts, counts, fig_dir, run_end):
    cfg = ctx["cfg"]
    figs = []
    prods = [p for p in PRODUCT_ORDER if p in pts]
    for scope, local, lim in (("regional", False, None), ("core", True, cfg["core_radius_m"] * 1.15)):
        fig, axes = plt.subplots(2, 3, figsize=(16, 11), layout="constrained")
        axes = axes.ravel()
        tmin = min(p["t"].min() for p in pts.values() if len(p))
        tmax = max(p["t"].max() for p in pts.values() if len(p))
        norm = plt.Normalize(tmin.year + tmin.dayofyear / 366, tmax.year + tmax.dayofyear / 366)
        for ax, pid in zip(axes, prods):
            p = pts[pid]
            draw_aoi(ax, Z, cfg, local, zoner)
            if len(p):
                xx, yy = (p["x"], p["y"]) if local else (p["lon"], p["lat"])
                dec = p["t"].dt.year + p["t"].dt.dayofyear / 366
                sc = ax.scatter(xx, yy, c=dec, cmap="viridis", norm=norm, s=1.2 if pid in ("atl03", "atl24") else 6,
                                lw=0, rasterized=True)
            n = counts[pid]
            key = "baseline_core" if local else "baseline"
            ax.set_title(f"{PRODUCT_LABEL[pid]}: {n[key]:,} baseline {UNIT_LABEL[pid]} in {'core' if local else 'AOI'}\n"
                         f"{n['acquisitions']} acquisitions with AOI rows", fontsize=9)
            if local:
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xlabel("east of reference point (m)")
                ax.set_ylabel("north of reference point (m)")
            else:
                ax.set_xlabel("longitude (°E, WGS84)")
                ax.set_ylabel("latitude (°N, WGS84)")
                ax.set_aspect(1 / np.cos(np.radians(cfg["center"]["lat"])))
            if local:
                ax.set_aspect("equal")
            ax.grid(alpha=0.3)
        for ax in axes[len(prods):]:
            ax.axis("off")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes.tolist(), shrink=0.6)
        cb.set_label("acquisition date (decimal year)")
        axes[0].legend(loc="lower left", fontsize=6.5)
        fig.suptitle(f"{cfg['site_id']}: {scope} coverage, baseline quality profile, run end {run_end}\n"
                     f"Core circle is a search zone, not an island outline. Display thinned to "
                     f"≤{MAX_PTS_PER_GRANULE} points per acquisition; counts are from full data.", fontsize=10)
        f = fig_dir / f"map_{scope}_coverage.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        figs.append({"figure": f.name, "kind": "map", "scope": scope, "quality_profile": "baseline",
                     "inputs": "cached chunks (all products)", "parameters": f"thin<= {MAX_PTS_PER_GRANULE}/acq"})
    return figs


def timelines(ctx, fig_dir, tables, events, run_end):
    A = pd.read_csv(tables / "coverage_by_acquisition.csv")
    led = pd.read_csv(tables / "acquisition_ledger.csv")
    M = pd.read_csv(tables / "coverage_by_month.csv")
    prods = [p for p in PRODUCT_ORDER if p in ctx["cfg"]["products"]]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for i, pid in enumerate(prods):
        L = led[led["product"] == pid]
        a = A[A["product"] == pid]
        t_all = pd.to_datetime(L["acq_start_utc"], utc=True)
        ax.scatter(t_all, [i + 0.22] * len(t_all), marker="|", s=60, color="0.6", lw=1)
        t_core = pd.to_datetime(a.loc[a["n_core_baseline"] > 0, "acq_start_utc"], utc=True)
        t_aoi = pd.to_datetime(a.loc[(a["n_core_baseline"] == 0) & (a["n_rows"] > 0), "acq_start_utc"], utc=True)
        ax.scatter(t_aoi, [i - 0.12] * len(t_aoi), marker="|", s=110, color=COLORS[pid], alpha=0.4, lw=1.3)
        ax.scatter(t_core, [i - 0.12] * len(t_core), marker="|", s=160, color=COLORS[pid], lw=1.8)
        bad = L[~L["verified_complete"].astype(bool) & (L["status"] != "not_yet_queried")]
        nq = L[L["status"] == "not_yet_queried"]
        if len(bad):
            ax.scatter(pd.to_datetime(bad["acq_start_utc"], utc=True), [i + 0.22] * len(bad), marker="x", color="red", s=25)
        if len(nq):
            ax.scatter(pd.to_datetime(nq["acq_start_utc"], utc=True), [i + 0.22] * len(nq), marker="o", s=14,
                       facecolors="none", edgecolors="0.3")
    ax.set_yticks(range(len(prods)), [f"{PRODUCT_LABEL[p]}" for p in prods])
    ax.set_ylim(-0.7, len(prods) - 0.3)
    add_events(ax, events)
    ax.axvline(pd.Timestamp(run_end), color="k", lw=1)
    ax.set_xlabel("acquisition date (UTC)")
    ax.set_title("Upper grey ticks: candidate granules; lower colored: actual rows in AOI (faint) / baseline rows in core "
                 "search zone (solid); red x: failed/unresolved; open circle: not yet queried", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    f1 = fig_dir / "timeline_candidate_vs_usable.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(prods), 1, figsize=(13, 1.6 * len(prods) + 1), sharex=True)
    for ax, pid in zip(axes, prods):
        g = M[M["product"] == pid].copy()
        t = pd.to_datetime(g["month"] + "-15")
        ax.bar(t, g["candidate_granules"], width=25, color="0.8", label="candidate granules")
        ax.bar(t, g["acq_with_core_baseline"], width=25, color=COLORS[pid], label="acq. with core baseline rows")
        ax.set_ylabel(PRODUCT_LABEL[pid].split()[0], fontsize=8)
        add_events(ax, events, labels=ax is axes[0])
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_title("Monthly acquisitions: candidates vs acquisitions with baseline-quality rows in the core search zone "
                      "(dashed: sourced 2021 eruption events)", fontsize=9)
    axes[-1].set_xlabel("month (UTC)")
    fig.tight_layout()
    f2 = fig_dir / "timeline_monthly_counts.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    return [{"figure": f1.name, "kind": "timeline", "scope": "AOI/core", "quality_profile": "baseline",
             "inputs": "acquisition_ledger.csv, coverage_by_acquisition.csv, events.csv", "parameters": ""},
            {"figure": f2.name, "kind": "timeline", "scope": "core", "quality_profile": "baseline",
             "inputs": "coverage_by_month.csv, events.csv", "parameters": "monthly"}]


def run_site(site_id, run_id):
    ctx = acq.context(site_id, run_id)
    cfg, m = ctx["cfg"], ctx["m"]
    out = manifest.output_dir(cfg, m["run_id"])
    fig_dir, tables = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    c = cfg["center"]
    zoner = normalize.Zoner(c["lon"], c["lat"], cfg["core_radius_m"], cfg["core_radius_m"] + cfg["buffer_m"])
    Z = zones(m)
    events = pd.read_csv(config.ROOT / m["events_table"])
    pts, counts = collect_points(ctx, zoner, np.random.default_rng(0))
    rows = maps(ctx, Z, zoner, pts, counts, fig_dir, m["query_end_utc"])
    rows += timelines(ctx, fig_dir, tables, events, m["query_end_utc"])
    for r in rows:
        r.update(run_id=m["run_id"], aoi_hash=m["aoi_hash"])
    plot_index_append(out, rows)
    print("\n".join(r["figure"] for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--run-id")
    a = ap.parse_args()
    run_site(a.site, a.run_id)


if __name__ == "__main__":
    main()
