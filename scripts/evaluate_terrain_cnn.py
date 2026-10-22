#!/usr/bin/env python
"""CNN terrain M6: per-site evaluation of every method on one partition.

  evaluate_terrain_cnn.py --partition val|test [--methods m1,m2,...]

val:  validation blocks of fitting sites (within) + the transfer-validation site (unseen)
test: test blocks of fitting sites (within) + the two transfer-test sites (unseen)

Methods are the prediction tables in outputs/cnn_terrain/predictions/
(baselines and CNN runs). Per site: bias (pred - ref), MAE, RMSE, count,
coverage, prediction std vs reference std (relief recovery), correlation; on
'full' (each method's own coverage) and 'common' (cells every method that
predicts at that site predicts). Equal-site and pooled (area-weighted)
aggregates are kept separate for within-site and unseen-site sets. Paired
per-site CNN-minus-baseline RMSE on common cells. Strata: reference
elevation, within-cell std, nDSM (<= 1 m), sparse-label distance (within only).

Writes outputs/cnn_terrain/eval_<partition>/{metrics,aggregates,paired,strata,
seeds}.csv and figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from island_altimetry import plotting  # noqa: F401 - Agg backend
from island_altimetry.cnn import config as ccfg
from island_altimetry.coastal import metrics

import matplotlib.pyplot as plt  # noqa: E402

Z_BINS = [-np.inf, 0, 1, 2, 3, 5, 8, np.inf]
STD_BINS = [0, 0.05, 0.1, 0.25, 0.5, np.inf]
NDSM_BINS = [-np.inf, 0.25, 0.5, 1.0]
D_BINS = [0, 50, 150, 300, 600, 1500, np.inf]


def eval_set(cfg, roles, part: str) -> pd.DataFrame:
    rows = []
    for site in roles["fit"] + roles["transfer_val"] + roles["transfer_test"]:
        c = pd.read_parquet(ccfg.site_paths(cfg, site)["work"] / "centers.parquet")
        if site in roles["fit"]:
            c = c[c["partition"] == (2 if part == "val" else 3)].assign(group="within")
        elif (site in roles["transfer_val"]) == (part == "val"):
            c = c.assign(group="unseen")
        else:
            continue
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def summarize(g: pd.DataFrame, col: str) -> dict:
    m = metrics.summarize(g[col].to_numpy(), g["z"].to_numpy())
    ok = np.isfinite(g[col].to_numpy())
    if ok.sum() > 2:
        p, z = g[col].to_numpy()[ok], g["z"].to_numpy()[ok]
        m |= {"pred_std": float(p.std()), "ref_std_pred": float(z.std()), "relief_ratio": float(p.std() / max(z.std(), 1e-9)),
              "corr": float(np.corrcoef(p, z)[0, 1]) if p.std() > 0 else np.nan}
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    ap.add_argument("--partition", choices=["val", "test"], required=True)
    ap.add_argument("--methods", help="comma list of prediction tables (default: all)")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    out = ccfg.out_dir(cfg)
    roles = json.loads((out / "site_roles.json").read_text())
    ev = eval_set(cfg, roles, a.partition)
    pdir = out / "predictions"
    methods = a.methods.split(",") if a.methods else sorted(p.stem for p in pdir.glob("*.parquet"))
    for m in methods:
        p = pd.read_parquet(pdir / f"{m}.parquet")
        ev = ev.merge(p[["site", "cell_id", "pred"]].rename(columns={"pred": m}), on=["site", "cell_id"], how="left")
        if m == "idw_sparse" and "d_label" in p:
            ev = ev.merge(p[["site", "cell_id", "d_label"]], on=["site", "cell_id"], how="left")
    ev["common"] = False
    for s, g in ev.groupby("site"):
        sup = [m for m in methods if np.isfinite(g[m]).any()]
        ev.loc[g.index, "common"] = np.isfinite(g[sup].to_numpy()).all(1)
    odir = out / f"eval_{a.partition}"
    odir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in methods:
        for (grp, s), g in ev.groupby(["group", "site"]):
            for sub in ("full", "common"):
                gg = g if sub == "full" else g[g["common"]]
                rows.append({"method": m, "group": grp, "site": s, "subset": sub, **summarize(gg, m)})
    M = pd.DataFrame(rows)
    M.to_csv(odir / "metrics.csv", index=False)

    agg = []
    for (m, grp, sub), g in M.groupby(["method", "group", "subset"]):
        e = ev if sub == "full" else ev[ev["common"]]
        e = e[e["group"] == grp]
        pooled = metrics.summarize(e[m].to_numpy(), e["z"].to_numpy())
        agg.append({"method": m, "group": grp, "subset": sub, "n_sites": int(g["rmse"].notna().sum()),
                    "equal_site_rmse": g["rmse"].mean(), "equal_site_mae": g["mae"].mean(), "equal_site_bias": g["bias"].mean(),
                    "equal_site_relief_ratio": g.get("relief_ratio", pd.Series(dtype=float)).mean(),
                    "pooled_rmse": pooled["rmse"], "pooled_n": pooled["n_pred"], "pooled_coverage": pooled["coverage"]})
    A = pd.DataFrame(agg)
    A.to_csv(odir / "aggregates.csv", index=False)

    cnn = [m for m in methods if m.startswith(("dense_", "sparse_"))]
    base = [m for m in methods if not m.startswith(("dense_", "sparse_"))]
    pr = []
    for c in cnn:
        for b in base:
            if c.split("_")[0] not in b and not b.startswith("idw"):
                continue                                     # compare within the same label regime
            for (grp, s), g in ev[ev["common"]].groupby(["group", "site"]):
                if not (np.isfinite(g[c]).any() and np.isfinite(g[b]).any()):
                    continue
                rc_, rb = metrics.summarize(g[c].to_numpy(), g["z"].to_numpy()), metrics.summarize(g[b].to_numpy(), g["z"].to_numpy())
                pr.append({"cnn": c, "baseline": b, "group": grp, "site": s, "n": rc_["n_pred"], "rmse_cnn": rc_["rmse"],
                           "rmse_base": rb["rmse"], "delta_rmse": rc_["rmse"] - rb["rmse"],
                           "mean_abs_err_diff": float(np.mean(np.abs(g[c] - g["z"]) - np.abs(g[b] - g["z"])))})
    pd.DataFrame(pr).to_csv(odir / "paired.csv", index=False)

    st = []
    ev["z_bin"] = pd.cut(ev["z"], Z_BINS).astype(str)
    ev["std_bin"] = pd.cut(ev["z_std"], STD_BINS, right=False).astype(str)
    ev["ndsm_bin"] = pd.cut(ev["ndsm"], NDSM_BINS).astype(str)
    if "d_label" in ev:
        ev["d_bin"] = pd.cut(ev["d_label"], D_BINS, right=False).astype(str)
    for m in methods:
        for sname in ("z_bin", "std_bin", "ndsm_bin", "d_bin"):
            if sname not in ev:
                continue
            e = ev if sname != "d_bin" else ev[ev["group"] == "within"]
            for (grp, b), g in e.groupby(["group", sname]):
                st.append({"method": m, "stratifier": sname, "group": grp, "bin": b, **metrics.summarize(g[m].to_numpy(), g["z"].to_numpy())})
    pd.DataFrame(st).to_csv(odir / "strata.csv", index=False)

    seeds = []
    for c in cnn:
        stem = c.rsplit("_s", 1)[0]
        seeds.append({"config": stem, "run": c})
    sd = pd.DataFrame(seeds)
    if not sd.empty:
        sm = A[A["method"].isin(sd["run"]) & (A["subset"] == "full")].merge(sd, left_on="method", right_on="run")
        sm.groupby(["config", "group"]).agg(n_seeds=("run", "count"), rmse_mean=("equal_site_rmse", "mean"),
                                            rmse_min=("equal_site_rmse", "min"), rmse_max=("equal_site_rmse", "max")
                                            ).reset_index().to_csv(odir / "seeds.csv", index=False)

    figures(ev, M, methods, odir, cfg, roles, a.partition)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(A[A["subset"] == "common"].pivot_table(index="method", columns="group",
                                                     values=["equal_site_rmse", "pooled_rmse", "equal_site_relief_ratio"]))
        print(M[M["subset"] == "full"].pivot_table(index="site", columns="method", values="rmse"))


def figures(ev, M, methods, odir: Path, cfg, roles, part):
    """Per-site RMSE bars, relief ratio, pooled scatter, error by elevation,
    and reference/prediction/error maps for each evaluated site."""
    from affine import Affine
    show = [m for m in methods if m.startswith(("dense_", "sparse_", "rf_", "idw", "const_dense"))]
    full = M[M["subset"] == "full"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    piv = full.pivot_table(index="site", columns="method", values="rmse")[show]
    piv.plot.bar(ax=axs[0], rot=30, fontsize=7)
    axs[0].set_ylabel("RMSE (m)")
    axs[0].set_title(f"{part}: per-site RMSE (sites: {', '.join(sorted(set(ev['site'])))})", fontsize=8)
    full.pivot_table(index="site", columns="method", values="relief_ratio")[show].plot.bar(ax=axs[1], rot=30, fontsize=7, legend=False)
    axs[1].axhline(1, color="k", lw=0.6)
    axs[1].set_ylabel("prediction std / reference std")
    axs[1].set_title("relief recovery (1 = full relief, 0 = constant surface)", fontsize=8)
    fig.tight_layout()
    fig.savefig(odir / "per_site.png", dpi=110)
    plt.close(fig)

    cnn = [m for m in show if m.startswith(("dense_", "sparse_"))][:2] + [m for m in ("rf_dense", "rf_sparse") if m in show]
    fig, axs = plt.subplots(2, max(1, len(cnn)), figsize=(4 * max(1, len(cnn)), 8), squeeze=False)
    for j, m in enumerate(cnn):
        for i, grp in enumerate(("within", "unseen")):
            g = ev[(ev["group"] == grp) & np.isfinite(ev[m])]
            if g.empty:
                continue
            lim = np.percentile(g["z"], [0.5, 99.5])
            axs[i, j].hexbin(g["z"], g[m], gridsize=60, bins="log", extent=(*lim, *lim), mincnt=1)
            axs[i, j].plot(lim, lim, "w--", lw=0.8)
            axs[i, j].set_title(f"{m} - {grp}", fontsize=8)
            axs[i, j].set_xlabel("reference (m)")
            axs[i, j].set_ylabel("predicted (m)")
    fig.tight_layout()
    fig.savefig(odir / "scatter.png", dpi=110)
    plt.close(fig)

    for site, g in ev.groupby("site"):
        sp = ccfg.site_paths(cfg, site)
        spec = json.loads((sp["work"] / "reference" / "grid_spec.json").read_text())
        H, W = spec["height"], spec["width"]
        r0, r1, c0, c1 = g["row"].min(), g["row"].max() + 1, g["col"].min(), g["col"].max() + 1
        pad = 20
        r0, c0, r1, c1 = max(r0 - pad, 0), max(c0 - pad, 0), min(r1 + pad, H), min(c1 + pad, W)

        def ras(col):
            a = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
            a[g["row"] - r0, g["col"] - c0] = g[col]
            return a
        ref = ras("z")
        lo, hi = np.nanpercentile(ref, [1, 99])
        cols = cnn[:2] + [m for m in ("rf_dense",) if m in g]
        fig, axs = plt.subplots(2, 1 + len(cols), figsize=(3.6 * (1 + len(cols)), 7.5), squeeze=False)
        t = Affine(*spec["transform"])
        ext = (t.c + c0 * t.a, t.c + c1 * t.a, t.f + r1 * t.e, t.f + r0 * t.e)
        axs[0, 0].imshow(ref, extent=ext, cmap="terrain", vmin=lo, vmax=hi, interpolation="nearest")
        axs[0, 0].set_title(f"{site} ({g['group'].iloc[0]}) reference", fontsize=8)
        axs[1, 0].axis("off")
        for j, m in enumerate(cols, 1):
            p = ras(m)
            axs[0, j].imshow(p, extent=ext, cmap="terrain", vmin=lo, vmax=hi, interpolation="nearest")
            axs[0, j].set_title(m, fontsize=8)
            axs[1, j].imshow(p - ref, extent=ext, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
            axs[1, j].set_title(f"{m} error", fontsize=8)
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(odir / f"map_{site}.png", dpi=90)
        plt.close(fig)


if __name__ == "__main__":
    main()
