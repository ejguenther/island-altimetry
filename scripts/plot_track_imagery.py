#!/usr/bin/env python
"""M9: lidar profiles beside the nearest-in-time PlanetScope image (no network).

  plot_track_imagery.py --site fukutoku_okanoba [--run-id RUN]
                        [--product atl03|gedi_l2a|all] [--granule G ...] [--max-dt-days 7]
                        [--pick nearest|clearest] [--missing-only]

One figure per core-crossing pass (ATL03 granule or GEDI L2A granule). Left:
one height-vs-along-track panel per beam that reaches the core; right: the
downloaded Planet strip closest in time (within --max-dt-days), zoomed to the
core + 1.5 km, with each beam drawn in its profile colour and an x marking its
along-track zero. Passes without imagery still get the map panel (AOI outlines
and tracks only). Heights are ellipsoidal; ATL03 carries its own EGM2008 geoid,
GEDI L2A has none, so its reference line is the site's median ATL03 geoid.

Planet scenes here are test quality without ground control: image/lidar
offsets of tens to hundreds of metres are possible.

Writes outputs/<site>/<run>/figures/track_imagery/{atl03|gedi}_<granule>.png and
tables/track_imagery_pairs.csv; appends to plot_index.csv.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.gridspec import GridSpec
from pyproj import Transformer
from shapely.geometry import shape

from island_altimetry import config, imagery, manifest, normalize, planet_imagery as pim
from plot_coverage import plot_index_append
from plot_profiles import GT_NAME, load
from scripts_path import load_acquire

acq = load_acquire()
MARGIN_M = 1500          # map / profile window beyond the core radius
MAX_GREY = 40000         # thinned diagnostic photons per ATL03 beam
MAX_MAP_PTS = 3000       # thinned track points per beam on the map
BEAM_COLORS = plt.get_cmap("tab10").colors
PREFIX = {"atl03": "atl03", "gedi_l2a": "gedi"}
GEO_NOTE = ("Planet test-quality scenes, no ground control: image/lidar offsets of "
            "tens to hundreds of metres possible")


# ---------------------------------------------------------------- pure helpers

def nearest_strip(t: pd.Timestamp, orders: dict, clear: dict, max_dt_h: float,
                  prefer: str = "nearest") -> dict | None:
    """Downloaded strip within max_dt_h of t (signed dt = image - lidar), or None.
    prefer="nearest": smallest |dt|; "clearest": highest measured core clear
    (unknown lowest), ties -> smallest |dt|."""
    cands = []
    for name, o in orders.items():
        if o.get("download_status") != "ok":
            continue
        dt = (pd.Timestamp(o["acquired_utc"]) - t) / pd.Timedelta(hours=1)
        if abs(dt) <= max_dt_h:
            c = clear.get(name)
            cands.append({"order_name": name, "strip_id": o["strip_id"], "acquired_utc": o["acquired_utc"],
                          "dt_hours": float(dt), "bundle": o["bundle"], "dir": o["dir"],
                          "measured_clear_core": c})
    if not cands:
        return None
    if prefer == "clearest":
        def clear_of(c):
            v = c["measured_clear_core"]
            return -1.0 if v is None or pd.isna(v) else v
        return min(cands, key=lambda c: (-clear_of(c), abs(c["dt_hours"])))
    return min(cands, key=lambda c: abs(c["dt_hours"]))


def pick_tiered(t: pd.Timestamp, orders: dict, clear: dict, tier_days=(7, 32)) -> dict | None:
    """Clearest downloaded strip within the first tier window that has one
    (e.g. +-7 d, then +-32 d); failing all tiers, the temporally closest
    downloaded strip at any distance. The chosen rule is recorded in "rule"."""
    for d in tier_days:
        s = nearest_strip(t, orders, clear, d * 24, prefer="clearest")
        if s:
            return s | {"rule": f"clearest within ±{d:g} d"}
    s = nearest_strip(t, orders, clear, float("inf"))
    return s | {"rule": "nearest available"} if s else None


def gedi_along_track(df: pd.DataFrame) -> pd.Series:
    """Along-track distance (m) per footprint: per beam, local x/y projected on
    the first->last footprint direction (time order), zero at closest approach
    to the site centre."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("beam"):
        g = g.sort_values("time_utc")
        xy = g[["x_local_m", "y_local_m"]].to_numpy()
        d = xy[-1] - xy[0]
        n = np.linalg.norm(d)
        u = d / n if n > 0 else np.array([1.0, 0.0])
        x0 = xy[int(np.argmin(g["r_m"].to_numpy()))]
        out.loc[g.index] = (xy - x0) @ u
    return out


def height_window(h: pd.Series, geoid: float, below: float = 10, cap: float = 150,
                  min_above: float = 20) -> tuple[float, float]:
    """Y-limits anchored to the geoid: geoid-below up to the 99th percentile of
    returns within `cap` m above the geoid (at least min_above), so cloud
    returns hundreds of metres up do not flatten a tens-of-metres island."""
    rel = (h - geoid)[np.isfinite(h)]
    rel = rel[(rel > -below) & (rel < cap)]
    top = max(min_above, float(np.percentile(rel, 99))) if len(rel) else min_above
    return geoid - below, geoid + top + 5


def thin(n: int, k: int, rng) -> np.ndarray:
    return np.arange(n) if n <= k else np.sort(rng.choice(n, k, replace=False))


# ---------------------------------------------------------------- panels

def beam_rows(pid: str, v: pd.DataFrame) -> list[tuple]:
    """[(beam_id, label, df_with_s)] for beams with samples in the core zone,
    with along-track coordinate `s` (m from closest approach)."""
    rows = []
    if pid == "atl03":
        for gt in sorted(v["gt"].unique()):
            b = v[v["gt"] == gt]
            if not (b["zone"] == "core").any():
                continue
            x0 = float(b.loc[b["r_m"].idxmin(), "x_atc"])
            rows.append((gt, GT_NAME.get(gt, str(gt)), b.assign(s=b["x_atc"] - x0)))
    else:
        v = v.assign(s=gedi_along_track(v))
        for beam in sorted(v["beam"].unique()):
            b = v[v["beam"] == beam]
            if (b["zone"] == "core").any():
                rows.append((beam, f"beam {int(beam):04b}", b))
    return rows


def profile_panel(ax, pid, b, color, core_span, win_m, geoid_ref, rng):
    ax.axvspan(*core_span, color="#fff3c4", zorder=0)
    w = b[(b["s"] >= -win_m) & (b["s"] <= win_m)]
    base = w[w["q_baseline"]]
    if pid == "atl03":
        idx = thin(len(w), MAX_GREY, rng)
        ax.scatter(w["s"].iloc[idx], w["height"].iloc[idx], s=0.3, color="0.75", lw=0, rasterized=True,
                   label="ATL03 diagnostic (thinned)")
        ax.scatter(base["s"], base["height"], s=0.8, color=color, lw=0, rasterized=True, label="ATL03 baseline")
        g = w.drop_duplicates("segment_id").sort_values("s")
        ax.plot(g["s"], g["geoid"], color="#2ca25f", lw=0.9, ls=":", label="ATL03 geoid (EGM2008)")
        h = base["height"]
    else:
        rest = w[~w["q_baseline"]]
        ax.scatter(rest["s"], rest["elevation_lm"], s=14, facecolor="none", edgecolor="0.6", lw=0.7,
                   label="GEDI elevation_lm (not baseline)")
        ax.scatter(base["s"], base["elevation_lm"], s=14, color=color, lw=0, label="GEDI elevation_lm (baseline)")
        ax.axhline(geoid_ref, color="#2ca25f", lw=0.9, ls="--", label=f"site median ATL03 geoid ({geoid_ref:.1f} m)")
        h = base["elevation_lm"] if len(base) else w["elevation_lm"]
    lo, hi = height_window(h, geoid_ref)
    ax.set_ylim(lo, hi)
    above = int((h > hi).sum())
    if above:
        ax.text(0.99, 0.02, f"{above:,} baseline returns above view (cloud?)", transform=ax.transAxes,
                fontsize=6.5, ha="right", color="0.35")
    ax.set_xlim(-win_m, win_m)
    ax.grid(alpha=0.3)
    return int(len(base))


def map_panel(ax, rows, pid, strip, cfg, aoi, rng, max_dt_days):
    lonlat = (cfg["center"]["lon"], cfg["center"]["lat"])
    win = cfg["core_radius_m"] + MARGIN_M
    if strip:
        sr = imagery.files(config.ROOT / strip["dir"], "sr")
        img, ext, crs = imagery.rgb_for_strip(sr)
        ax.imshow(img, extent=ext)
        to_map = imagery.to_raster_xy(crs, lonlat)
        wgs_to = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    else:
        ax.set_facecolor("#eef3f7")
        to_map = lambda x, y: (np.asarray(x, float), np.asarray(y, float))  # noqa: E731 - site AEQD metres
        from island_altimetry.aoi import aeqd_crs
        wgs_to = Transformer.from_crs("EPSG:4326", aeqd_crs(*lonlat), always_xy=True)
    for name, g in aoi.items():
        gm = shapely.transform(g, lambda xy: np.column_stack(wgs_to.transform(xy[:, 0], xy[:, 1])))
        x, y = gm.exterior.xy
        ax.plot(x, y, color="white" if strip else "0.4", lw=0.9, ls="-" if name == "core" else "--")
    cx, cy = to_map(0.0, 0.0)
    for k, ((beam, label, b), color) in enumerate(zip(rows, BEAM_COLORS)):
        pts = b[b["q_baseline"] & (b["r_m"] <= win * 1.5)]
        if pid == "atl03":
            pts = pts.iloc[thin(len(pts), MAX_MAP_PTS, rng)]
            x, y = to_map(pts["x_local_m"], pts["y_local_m"])
            ax.plot(x, y, ".", ms=1.2, color=color)
        else:
            x, y = to_map(b["x_local_m"], b["y_local_m"])
            ax.scatter(x, y, s=10, facecolor="none", edgecolor=color, lw=0.8)
        z = b.loc[b["r_m"].idxmin()]
        zx, zy = to_map(z["x_local_m"], z["y_local_m"])
        ax.plot(zx, zy, "x", ms=6, mew=1.5, color=color)
        ax.annotate(label, (zx, zy), xytext=(6, 4) if k % 2 == 0 else (-6, -10), textcoords="offset points",
                    fontsize=7, color=color, fontweight="bold", ha="left" if k % 2 == 0 else "right")
        ahead = b[(b["s"] > 300) & (b["s"] < 1200)]
        if k == 0 and len(ahead) >= 2:   # direction of increasing along-track distance
            a0 = ahead.loc[ahead["s"].idxmin()]
            a1 = ahead.loc[ahead["s"].idxmax()]
            (x0,), (y0,) = to_map([a0["x_local_m"]], [a0["y_local_m"]])
            (x1,), (y1,) = to_map([a1["x_local_m"]], [a1["y_local_m"]])
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops={"arrowstyle": "-|>", "color": "white" if strip else "0.2", "lw": 1.5})
            ax.annotate("+ along-track", (x1, y1), xytext=(5, 0), textcoords="offset points", fontsize=6.5,
                        color="white" if strip else "0.2")
    ax.set_xlim(float(cx) - win, float(cx) + win)
    ax.set_ylim(float(cy) - win, float(cy) + win)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    if strip:
        clear = strip["measured_clear_core"]
        kind = "RGB TOA" if strip["bundle"] == "analytic_3b_udm2" else "SR"
        dt_txt = (f"{strip['dt_hours'] / 24:+.1f} d" if abs(strip["dt_hours"]) >= 72
                  else f"{strip['dt_hours']:+.1f} h")
        rule = f"\n[{strip['rule']}]" if strip.get("rule") else ""
        ax.set_title(f"Planet {pd.Timestamp(strip['acquired_utc']):%Y-%m-%d %H:%M} UTC  ({kind})\n"
                     f"dt = {dt_txt}, core clear "
                     f"{'n/a' if clear is None or pd.isna(clear) else f'{clear:.0%}'}{rule}",
                     fontsize=8, color="#b30000" if strip.get("rule") == "nearest available" else "black")
        ax.set_xlabel(f"{crs} easting (m)", fontsize=7)
    else:
        ax.set_title(f"no downloaded Planet strip within ±{max_dt_days:g} d", fontsize=8)
        ax.set_xlabel("site AEQD x (m)", fontsize=7)


# ---------------------------------------------------------------- figure

def figure(ctx, pid, granule, zoner, strip_for, aoi, geoid_ref, fig_dir, rng, max_dt_days, pick_label=""):
    v, st = load(ctx, pid, granule, zoner)
    if v is None or v.empty:
        print(f"  {granule}: {st}, skipped")
        return None
    cfg = ctx["cfg"]
    win_m = cfg["core_radius_m"] + MARGIN_M
    rows = beam_rows(pid, v)
    if not rows:
        print(f"  {granule}: no beams near the core, skipped")
        return None
    t = pd.Timestamp(v["time_utc"].median())
    strip = strip_for(t)
    n = len(rows)
    row_h = 2.3 if pid == "atl03" else 1.7
    fig = plt.figure(figsize=(14, max(6.0, row_h * n + 1.4)))
    gs = GridSpec(n, 2, figure=fig, width_ratios=[1.5, 1], hspace=0.35, wspace=0.18)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    hlabel = "ellipsoidal height (m)"
    for ax, (beam, label, b), color in zip(axes, rows, BEAM_COLORS):
        core = b[b["zone"] == "core"]["s"]
        span = (core.min(), core.max()) if len(core) else (0, 0)
        nb = profile_panel(ax, pid, b, color, span, win_m, geoid_ref, rng)
        ax.set_ylabel(f"{label}\n{hlabel}", fontsize=8, color=color)
        ax.tick_params(labelsize=7)
        ax.text(0.01, 0.95, f"baseline n={nb:,}", transform=ax.transAxes, fontsize=7, va="top")
    axes[-1].set_xlabel("along-track distance from closest approach to site centre (m)", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h + [plt.Rectangle((0, 0), 1, 1, color="#fff3c4")], l + ["core search zone"],
                   fontsize=6.5, loc="upper right", markerscale=5 if pid == "atl03" else 1, framealpha=0.85)
    ax_map = fig.add_subplot(gs[:, 1])
    ax_map.set_anchor("N")
    map_panel(ax_map, rows, pid, strip, cfg, aoi, rng, max_dt_days)
    if pid == "atl03":
        what = f"ATL03 r007 {granule}  RGT {int(v['rgt'].iloc[0])} cycle {int(v['cycle'].iloc[0])}"
    else:
        what = f"GEDI L2A v002 {granule}"
    fig.suptitle(f"{what}\nlidar pass {t:%Y-%m-%d %H:%M} UTC", fontsize=10)
    fig.text(0.5, 0.005, GEO_NOTE, ha="center", fontsize=7, color="0.35")
    f = fig_dir / "track_imagery" / f"{PREFIX[pid]}_{granule}.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {f.name}: {n} beam(s), " + (f"strip {pd.Timestamp(strip['acquired_utc']):%Y-%m-%d %H:%M} dt {strip['dt_hours']:+.1f} h"
                                         if strip else "no imagery"))
    return {"figure": f"track_imagery/{f.name}", "kind": "track_imagery", "product": pid, "granule": granule,
            "pass_utc": t.isoformat(), "n_beams": n, "strip_id": strip and strip["strip_id"],
            "strip_acquired_utc": strip and strip["acquired_utc"], "dt_hours": strip and round(strip["dt_hours"], 2),
            "measured_clear_core": strip and strip["measured_clear_core"], "bundle": strip and strip["bundle"],
            "pick_rule": strip and strip.get("rule", f"{pick_label}")}


def run_site(site_id, run_id, products, granules, max_dt_days, pick="tiered", missing_only=False,
             tier_days=(7, 32)):
    run_id = run_id or manifest.latest_run_id(config.resolve_site(site_id))
    ctx = acq.context(site_id, run_id)
    cfg, m = ctx["cfg"], ctx["m"]
    c = cfg["center"]
    zoner = normalize.Zoner(c["lon"], c["lat"], cfg["core_radius_m"], cfg["core_radius_m"] + cfg["buffer_m"])
    out = manifest.output_dir(cfg, run_id)
    P = pim.paths(cfg, run_id)
    orders = json.loads(P["orders"].read_text()) if P["orders"].exists() else {}
    qc_f = P["tables"] / "planet_scene_qc.csv"
    clear = (pd.read_csv(qc_f).set_index("order_name")["measured_clear_core"].to_dict() if qc_f.exists() else {})
    aoi = {z: shape(json.loads((config.ROOT / m["aoi_dir"] / f"{z}.geojson").read_text())["features"][0]["geometry"])
           for z in ("core", "outer")}
    cl = manifest.data_dir(cfg) / "analysis" / run_id / "atl03_core_centerlines.parquet"
    geoid_ref = float(pd.read_parquet(cl, columns=["geoid"])["geoid"].median()) if cl.exists() else 0.0
    A = pd.read_csv(out / "tables" / "coverage_by_acquisition.csv")
    cand = A[A["product"].isin(products) & (A["n_core"] > 0)].sort_values("acq_start_utc")
    if granules:
        cand = cand[cand["granule"].isin(granules)]
    pairs = out / "tables" / "track_imagery_pairs.csv"
    if missing_only:
        if not pairs.exists():
            raise SystemExit(f"--missing-only needs {pairs}; run without it first")
        prev = pd.read_csv(pairs)
        cand = cand[cand["granule"].isin(prev.loc[prev["strip_id"].isna(), "granule"])]
    print(f"== {site_id} {run_id}: {len(cand)} passes, {sum(o.get('download_status') == 'ok' for o in orders.values())}"
          f" downloaded strips, window ±{max_dt_days:g} d")
    rng = np.random.default_rng(0)
    if pick == "tiered":
        def strip_for(t):
            return pick_tiered(t, orders, clear, tier_days)
        label = "tiered " + "/".join(f"±{d:g} d" for d in tier_days) + ", then nearest available"
    else:
        def strip_for(t):
            return nearest_strip(t, orders, clear, max_dt_days * 24, pick)
        label = f"{pick} within ±{max_dt_days:g} d"
    print(f"   image rule: {label}")
    rows = []
    for _, r in cand.iterrows():
        res = figure(ctx, r["product"], r["granule"], zoner, strip_for, aoi, geoid_ref,
                     out / "figures", rng, max_dt_days, f"{pick} within ±{max_dt_days:g} d")
        if res:
            res.update(run_id=run_id, aoi_hash=m["aoi_hash"])
            rows.append(res)
    if not rows:
        return
    df = pd.DataFrame(rows)
    if (granules or missing_only) and pairs.exists():
        old = pd.read_csv(pairs)
        df = pd.concat([old[~old["granule"].isin(df["granule"])], df], ignore_index=True)
    df.sort_values("pass_utc").to_csv(pairs, index=False)
    plot_index_append(out, [{k: r[k] for k in ("figure", "kind", "run_id", "aoi_hash")} |
                            {"inputs": r["granule"] + (f";planet:{r['strip_id']}" if r["strip_id"] else ""),
                             "parameters": f"max_dt_days={max_dt_days}; n_beams={r['n_beams']}"} for r in rows])
    print(f"{len(rows)} figures; {int(df['strip_id'].notna().sum())} with imagery -> {pairs.relative_to(config.ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--product", choices=["atl03", "gedi_l2a", "all"], default="all")
    ap.add_argument("--granule", action="append")
    ap.add_argument("--max-dt-days", type=float, default=7.0)
    ap.add_argument("--pick", choices=["tiered", "nearest", "clearest"], default="tiered",
                    help="tiered (default): clearest within each --tier-days window in turn, else the "
                         "temporally closest downloaded strip; nearest/clearest: within --max-dt-days only")
    ap.add_argument("--tier-days", type=float, nargs="+", default=[7, 32])
    ap.add_argument("--missing-only", action="store_true",
                    help="only re-plot passes that have no strip in track_imagery_pairs.csv")
    a = ap.parse_args()
    products = ["atl03", "gedi_l2a"] if a.product == "all" else [a.product]
    run_site(a.site, a.run_id, products, a.granule, a.max_dt_days, a.pick, a.missing_only, a.tier_days)


if __name__ == "__main__":
    main()
