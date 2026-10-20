#!/usr/bin/env python
"""M6: qualified inter-product comparisons (consistency, not accuracy).

  compare_products.py --site fukutoku_okanoba [--run-id RUN]

1. ATL08 vs ATL03 (same pass, beam, release): ATL08 h_te_best_fit vs median of
   strict ATL03 photons in the same native segment_id range. Shared ancestry:
   this checks processing consistency only.
2. ATL24 vs ATL03 (same pass and beam; ATL24 from r006): ATL24 sea-surface
   (class 41) ellipse_h vs ATL03 strict photons within the same 20 m along-track
   bins. Spatial/temporal association; refraction correction differs below water,
   so only sea-surface photons are compared.
3. GEDI vs ICESat-2: only footprint/bin pairs within max_distance_m and
   max_time_offset_days, both ellipsoidal (WGS84 vs ITRF2014 realizations; cm-level
   frame differences are below the comparison resolution). Pairs outside the time
   tolerance are listed as inappropriate (surface may have changed), not differenced.

Writes tables/comparison_*.csv, tables/vertical_compatibility.csv, and a figure.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from island_altimetry import manifest, normalize
from island_altimetry.plotting import COLORS
from plot_coverage import plot_index_append
from plot_profiles import atl08_along_track, load, partners
from scripts_path import load_acquire

acq = load_acquire()

VERTICAL = [
    {"pair": "ATL08 h_te_best_fit vs ATL03 h_ph", "ref_a": "WGS84 ellipsoid (ITRF2014)", "ref_b": "WGS84 ellipsoid (ITRF2014)",
     "corrections": "same geophysical corrections (ATL08 built from ATL03 r007)", "support": "100 m segment vs photons in same segments",
     "timing": "same pass", "compatible": "yes (consistency only; shared ancestry)"},
    {"pair": "ATL24 ellipse_h (class 41) vs ATL03 h_ph", "ref_a": "ITRF2014 ellipsoid", "ref_b": "ITRF2014 ellipsoid",
     "corrections": "ATL24 refraction-corrected (no effect at the surface); ATL24 from ATL03 r006 vs r007",
     "support": "photons in same 20 m bins", "timing": "same pass",
     "compatible": "sea surface only; bathymetry not compared (refraction, no ATL03 equivalent)"},
    {"pair": "GEDI elev_lowestmode vs ATL03 h_ph", "ref_a": "WGS84 ellipsoid", "ref_b": "ITRF2014 ellipsoid",
     "corrections": "different instruments and processing", "support": "~25 m footprint vs 20 m photon bin",
     "timing": "must be within max_time_offset_days",
     "compatible": "only within tolerance; differences may reflect surface change, sea state, or support"},
    {"pair": "ATL24 ortho_h vs any ellipsoidal height", "ref_a": "EGM2008", "ref_b": "ellipsoid",
     "corrections": "", "support": "", "timing": "", "compatible": "no (different vertical datum; not differenced)"},
]


def bins(v03, gt, bin_m):
    b = v03[(v03["gt"] == gt) & v03["q_strict"]]
    b = b.assign(bin=(b["x_atc"] // bin_m).astype("int64"))
    return b.groupby("bin").agg(h=("height", "median"), n=("height", "size"),
                                x=("x_local_m", "median"), y=("y_local_m", "median"), zone=("zone", "first"))


def run_site(site_id, run_id):
    ctx = acq.context(site_id, run_id)
    cfg, m = ctx["cfg"], ctx["m"]
    out = manifest.output_dir(cfg, m["run_id"])
    tables = out / "tables"
    c = cfg["center"]
    zoner = normalize.Zoner(c["lon"], c["lat"], cfg["core_radius_m"], cfg["core_radius_m"] + cfg["buffer_m"])
    tol = cfg["comparison"]
    A = pd.read_csv(tables / "coverage_by_acquisition.csv")
    passes = A[(A["product"] == "atl03") & (A["n_rows"] > 0)]["granule"].tolist()
    c08, c24, ic2_bins = [], [], []
    for g in passes:
        v03, _ = load(ctx, "atl03", g, zoner)
        if v03 is None:
            continue
        assoc = partners(ctx, g)
        date = pd.Timestamp(v03["time_utc"].min())
        for gt in v03["gt"].unique():
            bb = bins(v03, gt, 20.0).reset_index()
            bb = bb.assign(granule=g, gt=gt, time=date)
            ic2_bins.append(bb)
        if assoc.get("atl08"):
            v08, _ = load(ctx, "atl08", assoc["atl08"], zoner)
            if v08 is not None and len(v08):
                v08["x_atc"], v08["x_atc_method"] = atl08_along_track(v08, v03)
                s03 = v03[v03["q_strict"]]
                for _, r in v08[v08["q_baseline"]].iterrows():
                    ph = s03[(s03["gt"] == r["gt"]) & (s03["segment_id"] >= r["segment_id_beg"]) & (s03["segment_id"] <= r["segment_id_end"])]
                    if len(ph) >= 10:
                        c08.append({"atl03": g, "atl08": assoc["atl08"], "date": date.date(), "gt": r["gt"], "zone": r["zone"],
                                    "segment_id_beg": int(r["segment_id_beg"]), "h_te_best_fit": float(r["terrain/h_te_best_fit"]),
                                    "atl03_strict_median": float(ph["height"].median()), "n_atl03": len(ph),
                                    "diff_m": float(r["terrain/h_te_best_fit"] - ph["height"].median())})
        if assoc.get("atl24"):
            v24, _ = load(ctx, "atl24", assoc["atl24"], zoner)
            if v24 is not None and len(v24):
                for gt in v24["gt"].unique():
                    s = v24[(v24["gt"] == gt) & (v24["class_ph"] == 41) & v24["q_baseline"]]
                    if s.empty:
                        continue
                    s = s.assign(bin=(s["x_atc"] // 20.0).astype("int64")).groupby("bin").agg(h24=("ellipse_h", "median"), n24=("ellipse_h", "size"), zone=("zone", "first"))
                    j = s.join(bins(v03, gt, 20.0)[["h", "n"]], how="inner")
                    j = j[(j["n"] >= 5) & (j["n24"] >= 5)]
                    for b, r in j.iterrows():
                        c24.append({"atl03": g, "atl24": assoc["atl24"], "date": date.date(), "gt": gt, "bin": int(b), "zone": r["zone"],
                                    "atl24_sea_surface_ellipse_h": float(r["h24"]), "atl03_strict_median": float(r["h"]),
                                    "diff_m": float(r["h24"] - r["h"])})
        print(f"{g}: atl08 pairs={sum(x['atl03'] == g for x in c08)} atl24 bins={sum(x['atl03'] == g for x in c24)}", flush=True)

    C08, C24 = pd.DataFrame(c08), pd.DataFrame(c24)
    C08.to_csv(tables / "comparison_atl08_vs_atl03.csv", index=False)
    C24.to_csv(tables / "comparison_atl24_vs_atl03.csv", index=False)

    # GEDI vs ICESat-2 within tolerances.
    gedi = []
    for _, r in acq.resources_for(ctx, "gedi_l2a").iterrows():
        v, st = load(ctx, "gedi_l2a", r["granule"], zoner)
        if v is not None:
            gedi.append(v[v["q_baseline"]])
    G = pd.concat(gedi, ignore_index=True) if gedi else pd.DataFrame()
    B = pd.concat(ic2_bins, ignore_index=True) if ic2_bins else pd.DataFrame()
    B = B[B["n"] >= 5] if len(B) else B
    gm, near = [], []
    if len(G) and len(B):
        bx, by = B["x"].to_numpy(), B["y"].to_numpy()
        for _, f in G.iterrows():
            d = np.hypot(bx - f["x_local_m"], by - f["y_local_m"])
            k = np.where(d <= tol["max_distance_m"])[0]
            for i in k:
                dt_days = abs((pd.Timestamp(f["time_utc"]) - B["time"].iloc[i]).total_seconds()) / 86400
                row = {"shot_number": str(f["shot_number"]), "gedi_time": f["time_utc"], "atl03": B["granule"].iloc[i],
                       "gt": B["gt"].iloc[i], "atl03_time": B["time"].iloc[i], "dist_m": round(float(d[i]), 1),
                       "dt_days": round(dt_days, 2), "zone": f["zone"], "gedi_elev_lowestmode": float(f["elevation_lm"]),
                       "atl03_bin_median": float(B["h"].iloc[i])}
                if dt_days <= tol["max_time_offset_days"]:
                    gm.append({**row, "diff_m": row["gedi_elev_lowestmode"] - row["atl03_bin_median"]})
                else:
                    near.append({**row, "reason_not_compared": f"time offset > {tol['max_time_offset_days']} d"})
    pd.DataFrame(gm).to_csv(tables / "comparison_gedi_vs_icesat2_valid.csv", index=False)
    N = pd.DataFrame(near)
    N.to_csv(tables / "comparison_gedi_vs_icesat2_spatial_only.csv", index=False)
    pd.DataFrame(VERTICAL).to_csv(tables / "vertical_compatibility.csv", index=False)

    def stats(df, by="zone"):
        if df.empty:
            return pd.DataFrame()
        return df.groupby(by)["diff_m"].agg(n="size", median="median", mad=lambda s: float(np.median(np.abs(s - s.median()))),
                                            p05=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95)).reset_index()
    summ = []
    for name, df in (("atl08_minus_atl03", C08), ("atl24_sea_surface_minus_atl03", C24), ("gedi_minus_icesat2_valid", pd.DataFrame(gm))):
        s = stats(df)
        for _, r in s.iterrows():
            summ.append({"comparison": name, **r.to_dict()})
        if s.empty:
            summ.append({"comparison": name, "zone": "all", "n": 0})
    S = pd.DataFrame(summ)
    S.to_csv(tables / "comparison_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, df, col) in zip(axes, (("ATL08 h_te_best_fit − ATL03 strict median", C08, COLORS["atl08"]),
                                          ("ATL24 sea surface ellipse_h − ATL03 strict median (20 m bins)", C24, COLORS["atl24"]),
                                          (f"GEDI elev_lowestmode − ATL03 (≤{tol['max_distance_m']} m, ≤{tol['max_time_offset_days']} d)", pd.DataFrame(gm), COLORS["gedi_l2a"]))):
        if len(df):
            for z, ls in (("core", "-"), ("buffer", "--")):
                d = df[df["zone"] == z]["diff_m"]
                if len(d):
                    ax.hist(d.clip(-5, 5), bins=80, histtype="step", color=col, ls=ls, label=f"{z} n={len(d)} median={d.median():+.2f} m")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no valid matches within tolerances" if "GEDI" in name else "no pairs", ha="center",
                    transform=ax.transAxes, fontsize=9)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel("difference (m), clipped to ±5 m")
        ax.grid(alpha=0.3)
    fig.suptitle("Inter-product consistency (not independent accuracy; ATL03/08/24 share measurement ancestry). "
                 "Ellipsoidal heights only.", fontsize=9)
    fig.tight_layout()
    f = out / "figures" / "comparison_histograms.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    plot_index_append(out, [{"figure": f.name, "kind": "comparison", "scope": "AOI", "quality_profile": "strict ATL03 / baseline others",
                             "inputs": "comparison_*.csv", "parameters": str(tol), "run_id": m["run_id"], "aoi_hash": m["aoi_hash"]}])
    print(S.to_string(index=False))
    print(f"GEDI spatial-only (time offset too large) pairs: {len(N)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--run-id")
    a = ap.parse_args()
    run_site(a.site, a.run_id)


if __name__ == "__main__":
    main()
