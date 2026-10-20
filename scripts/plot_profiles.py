#!/usr/bin/env python
"""M6: ATL03 photon profiles (height vs along-track distance) for every ICESat-2
pass with photons in the core search zone, with verified-association overlays:
standard ATL08 terrain segments (same acquisition, beam, release) and ATL24
classes (same acquisition and beam; ATL24 v002 derives from ATL03 r006, so the
association is spatial/temporal only). ATL09 cloud context is shown as a
separate strip at its native ~25 Hz profile sampling.

  plot_profiles.py --site fukutoku_okanoba [--run-id RUN] [--granule G ...]

Heights: ATL03 h_ph, ATL08 h_te_best_fit, ATL24 ellipse_h are all ellipsoidal;
the ATL03 geoid (EGM2008) is drawn for reference. Thinned display; full counts
are printed on each panel.
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from island_altimetry import adapters, config, manifest, normalize, quality, status
from island_altimetry.plotting import COLORS
from plot_coverage import plot_index_append
from scripts_path import load_acquire

acq = load_acquire()
MAX_GREY = 60000
CNF_COLORS = {2: "#9ecae1", 3: "#3182bd", 4: "#08306b"}
GT_NAME = {10: "gt1l", 20: "gt1r", 30: "gt2l", 40: "gt2r", 50: "gt3l", 60: "gt3r"}


def load(ctx, pid, granule, zoner):
    parms = adapters.build_parms(pid, ctx["products"][pid], ctx["products"]["atl03"], ctx["poly"])
    store = acq.store_for(ctx, pid, parms)
    m = store.read_meta(granule)
    if not m or m["status"] != status.OK or not store.is_complete(granule):
        return None, (m or {}).get("status", status.NOT_YET_QUERIED)
    p = store.data_path(granule)
    raw = pd.read_parquet(p) if pid == "atl09" else gpd.read_parquet(p)
    v, _ = normalize.view(pid, raw, granule, zoner)
    v, _ = quality.apply(pid, v)
    return v, status.OK


def partners(ctx, granule):
    a = pd.read_csv(ctx["inv_dir"] / "associations.csv", dtype=str).fillna("")
    row = a[a["atl03"] == granule]
    return row.iloc[0].to_dict() if len(row) else {}


def atl08_along_track(v08: pd.DataFrame, v03: pd.DataFrame) -> pd.Series:
    """ATL08 lacks x_atc in the SlideRule output. Place each 100 m segment at the
    mean ATL03 x_atc of its native segment_id_beg..segment_id_end range (same
    acquisition and release); fall back to the nearest ATL03 photon in local
    metric coordinates, flagged by the returned method column."""
    seg = v03.groupby(["gt", "segment_id"])["x_atc"].mean()
    x, method = [], []
    for _, r in v08.iterrows():
        try:
            s = seg.loc[r["gt"]]
            sel = s[(s.index >= r["segment_id_beg"]) & (s.index <= r["segment_id_end"])]
        except KeyError:
            sel = pd.Series(dtype=float)
        if len(sel):
            x.append(float(sel.mean()))
            method.append("segment_id")
        else:
            b = v03[v03["gt"] == r["gt"]]
            if len(b):
                d = np.hypot(b["x_local_m"] - r["x_local_m"], b["y_local_m"] - r["y_local_m"])
                x.append(float(b["x_atc"].iloc[int(np.argmin(d.values))]))
                method.append("nearest_photon")
            else:
                x.append(np.nan)
                method.append("unplaced")
    return pd.Series(x, index=v08.index), pd.Series(method, index=v08.index)


def profile_figure(ctx, granule, zoner, fig_dir, rng):
    v03, st = load(ctx, "atl03", granule, zoner)
    if v03 is None or not (v03["zone"] == "core").any():
        return None
    assoc = partners(ctx, granule)
    v08 = load(ctx, "atl08", assoc.get("atl08"), zoner)[0] if assoc.get("atl08") else None
    if v08 is not None:
        v08["x_atc"], v08["x_atc_method"] = atl08_along_track(v08, v03)
    v24 = load(ctx, "atl24", assoc.get("atl24"), zoner)[0] if assoc.get("atl24") else None
    v09 = load(ctx, "atl09", granule, zoner)[0]
    beams = [g for g in sorted(v03["gt"].unique()) if ((v03["gt"] == g) & (v03["zone"] == "core")).any()]
    fig, axes = plt.subplots(len(beams), 2, figsize=(15, 3.3 * len(beams) + 0.8), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    date = pd.Timestamp(v03["time_utc"].min()).strftime("%Y-%m-%d %H:%M UTC")
    rgt, cyc = int(v03["rgt"].iloc[0]), int(v03["cycle"].iloc[0])
    beam_info = []
    for row, gt in zip(axes, beams):
        b = v03[v03["gt"] == gt]
        x0 = float(b.loc[b["r_m"].idxmin(), "x_atc"])
        xb = b["x_atc"] - x0
        core = b["zone"] == "core"
        cx = xb[core]
        n_diag, n_base, n_strict = len(b), int(b["q_baseline"].sum()), int(b["q_strict"].sum())
        idx = np.arange(len(b)) if len(b) <= MAX_GREY else np.sort(rng.choice(len(b), MAX_GREY, replace=False))
        base = b[b["q_baseline"]]
        for k, ax in enumerate(row):
            ax.axvspan(cx.min(), cx.max(), color="#fff3c4", zorder=0,
                       label="core search zone (not island outline)" if k == 0 else None)
            ax.scatter(xb.iloc[idx], b["height"].iloc[idx], s=0.3, color="0.65", lw=0, rasterized=True,
                       label=f"ATL03 diagnostic (n={n_diag:,}; {len(idx):,} shown)")
            for cnf, col in CNF_COLORS.items():
                s = base[base["atl03_cnf"] == cnf]
                if len(s):
                    ax.scatter(s["x_atc"] - x0, s["height"], s=0.8, color=col, lw=0, rasterized=True,
                               label=f"baseline conf={cnf} (n={len(s):,})")
            g = b.drop_duplicates("segment_id").sort_values("x_atc")
            ax.plot(g["x_atc"] - x0, g["geoid"], color="#2ca25f", lw=0.8, ls=":", label="ATL03 geoid (EGM2008, ellipsoidal)")
            if v08 is not None:
                s8 = v08[(v08["gt"] == gt)]
                if len(s8):
                    ax.scatter(s8["x_atc"] - x0, s8["terrain/h_te_best_fit"], marker="s", s=10, facecolor="none",
                               edgecolor=COLORS["atl08"], lw=0.9, label=f"ATL08 h_te_best_fit (n={len(s8)} segs)")
            if v24 is not None:
                s24 = v24[(v24["gt"] == gt)]
                for cls, mk, lab in ((41, ".", "sea surface"), (40, "o", "bathymetry")):
                    s = s24[s24["class_ph"] == cls]
                    if len(s):
                        ax.scatter(s["x_atc"] - x0, s["ellipse_h"], s=1.5 if cls == 41 else 4, marker=mk,
                                   color=COLORS["atl24"] if cls == 41 else "#a63603", lw=0,
                                   label=f"ATL24 class {cls} {lab} ellipse_h (n={len(s):,})")
        ax_full, ax_det = row
        ax_full.set_ylabel(f"{GT_NAME.get(gt, gt)}\nellipsoidal height (m)")
        if len(base):
            lo, hi = np.nanpercentile(base["height"], [1, 99])
            core_b = base[base["zone"] == "core"]["height"]
            if len(core_b):
                lo, hi = min(lo, np.nanpercentile(core_b, 1)), max(hi, np.nanpercentile(core_b, 99))
            ax_det.set_ylim(lo - 8, hi + 8)
        ax_det.set_xlim(cx.min() - 1500, cx.max() + 1500)
        ax_full.set_title(f"full range: diag {n_diag:,} / baseline {n_base:,} / strict {n_strict:,} photons", fontsize=8)
        ax_det.set_title("surface detail (baseline 1–99th pct ± 8 m), core ± 1.5 km", fontsize=8)
        if v09 is not None:
            s9 = v09[v09["gt"] == gt].sort_values("x_atc_min")
            for _, p in s9.iterrows():
                col = "#bdbdbd" if p["high_rate/layer_flag"] == 0 else "#6a51a3"
                for ax in row:
                    ax.axvspan(p["x_atc_min"] - x0, p["x_atc_max"] - x0, ymin=0.97, ymax=1.0, color=col, lw=0)
        beam_info.append(GT_NAME.get(gt, str(gt)))
        for ax in row:
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("along-track distance from closest approach to reference point (m)")
    h, l = axes[0][1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, markerscale=6, frameon=False)
    a09 = "ATL09 strip (top): grey clear / purple layer_flag=1, native profiles" if v09 is not None else "ATL09: not available"
    fig.suptitle(f"ATL03 r007 {granule} — {date}, RGT {rgt} cycle {cyc}; quality profiles diagnostic/baseline\n"
                 f"overlays: ATL08 r007 {'yes' if v08 is not None else 'none'}; ATL24 v002 "
                 f"{'yes (spatial/temporal association; derived from ATL03 r006)' if v24 is not None else 'none'}; {a09}",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.06 + 0.02 * (len(l) > 8), 1, 0.95))
    f = fig_dir / "profiles" / f"profile_{granule}.png"
    f.parent.mkdir(exist_ok=True)
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return {"figure": f"profiles/{f.name}", "kind": "profile", "scope": "beams with core photons",
            "quality_profile": "diagnostic+baseline", "inputs": ";".join(filter(None, [granule, assoc.get("atl08"),
                                                                                      assoc.get("atl24"), "atl09" if v09 is not None else ""])),
            "parameters": f"beams={','.join(beam_info)}; thin grey<= {MAX_GREY}/beam"}


def run_site(site_id, run_id, granules):
    ctx = acq.context(site_id, run_id)
    cfg, m = ctx["cfg"], ctx["m"]
    c = cfg["center"]
    zoner = normalize.Zoner(c["lon"], c["lat"], cfg["core_radius_m"], cfg["core_radius_m"] + cfg["buffer_m"])
    out = manifest.output_dir(cfg, m["run_id"])
    tables = out / "tables"
    A = pd.read_csv(tables / "coverage_by_acquisition.csv")
    cand = A[(A["product"] == "atl03") & (A["n_core"] > 0)].sort_values("acq_start_utc")["granule"].tolist()
    if granules:
        cand = [g for g in cand if g in granules]
    rng = np.random.default_rng(0)
    rows = []
    for g in cand:
        r = profile_figure(ctx, g, zoner, out / "figures", rng)
        if r:
            r.update(run_id=m["run_id"], aoi_hash=m["aoi_hash"])
            rows.append(r)
            print(r["figure"], flush=True)
    plot_index_append(out, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--granule", action="append")
    a = ap.parse_args()
    run_site(a.site, a.run_id, a.granule)


if __name__ == "__main__":
    main()
