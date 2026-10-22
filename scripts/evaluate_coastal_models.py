#!/usr/bin/env python
"""Coastal M6: evaluate every method on one frozen partition.

  evaluate_coastal_models.py [--config ...] --partition val|test

Evaluation set: all usable cells of the partition (fixed for every method and
regime). Land and water are scored separately (ambiguous cells separately,
once per domain model). 'full' rows report each method's own coverage;
'common' rows restrict every method to the declared common-supported subset
(cells every method predicts). Strata: reference elevation, nearest
training-label distance, within-cell std, DSM-DTM class, lidar return count,
extrapolation beyond the regime's training elevation range. Reference values
appear only as targets and diagnostic strata.

Writes outputs/.../tables/{metrics,layout_summary,strata}_<partition>.csv and
figures/eval_<partition>/*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine

from island_altimetry import plotting  # sets the Agg backend
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import experiment as ex
from island_altimetry.coastal import grid, masks, metrics, splits

import matplotlib.pyplot as plt  # noqa: E402

METHOD_ORDER = ["B0", "B1", "R1", "R2", "R3", "T1", "T2"]
METHOD_COLORS = {"B0": "0.55", "B1": "#2a6fdb", "R1": "#2e9d5b", "R2": "#e07b24", "R3": "#c43d5a",
                 "T1": "#8a63c9", "T2": "#8a63c9"}
Z_BINS = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, np.inf]
D_BINS = [0, 30, 100, 300, 1000, 3000, np.inf]
STD_BINS = [0, 0.05, 0.1, 0.25, 0.5, np.inf]
SUP_BINS = [0, 4, 16, 64, 256, np.inf]
MAP_LAYOUT = "t06_v0"


def pred_files(pdir: Path) -> list[tuple[str, str, Path]]:
    out = []
    for f in sorted(pdir.glob("*.parquet")):
        m, _, r = f.stem.partition("_")
        if m in METHOD_ORDER:
            out.append((m, r, f))
    return out


def regime_meta(r: str) -> dict:
    if r.startswith("dense"):
        return {"density": "dense", "n_tracks": np.nan, "variant": np.nan}
    n, v = r[1:].split("_v")
    return {"density": f"{int(n)} tracks", "n_tracks": int(n), "variant": int(v)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    ap.add_argument("--partition", choices=["val", "test"], required=True)
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D, W = ccfg.dirs(cfg), ccfg.dirs(cfg)["work"]
    code = splits.PARTITIONS[a.partition]
    pdir = W / "predictions"
    cells = ex.load_cells(W, ["z_std", "ndsm_class", "support"])
    ev = ex.routed(cells[cells["partition"] == code]).reset_index(drop=True)
    key = ["cell_id", "model_domain"]
    files = pred_files(pdir)
    cols = []
    for m, r, f in files:
        p = pd.read_parquet(f, columns=key + ["pred"])
        p["model_domain"] = p["model_domain"].astype(ev["model_domain"].dtype)
        ev = ev.merge(p.rename(columns={"pred": f"{m}|{r}"}), on=key, how="left")
        cols.append((m, r, f"{m}|{r}"))
    regs = sorted({r for _, r, _ in cols})
    for r in regs:
        d = pd.read_parquet(pdir / f"dist_{r}.parquet")
        d["model_domain"] = d["model_domain"].astype(ev["model_domain"].dtype)
        ev = ev.merge(d.rename(columns={"d_label": f"d|{r}"}), on=key, how="left")
    # common-supported subset per (regime, model domain): cells predicted by every method of that
    # regime that predicts anything in the domain, plus the dense references (B0, R1). Unsupported
    # combinations stay visible with coverage 0 in the 'full' rows.
    common_members = {}
    dense_cols = [c for m, r, c in cols if r == "dense"]
    for r in regs:
        ev[f"common|{r}"] = False
        for md, g in ev.groupby("model_domain"):
            members = [c for m, rr, c in cols if (rr == r or c in dense_cols) and np.isfinite(g[c]).any()]
            common_members[f"{r}|{masks.DOMAIN_NAMES[md]}"] = members
            ev.loc[g.index, f"common|{r}"] = np.all(np.isfinite(g[members].to_numpy()), axis=1)

    # training elevation range per regime/domain -> extrapolation flag
    zr = {}
    for r in regs:
        lab = ex.training_labels(cells, W, r)
        zr[r] = lab.groupby("domain")["z"].agg(["min", "max"])

    rows, srows = [], []
    for m, r, c in cols:
        meta = {"method": m, "regime": r, **regime_meta(r)}
        for (dom, md), g in ev.groupby(["domain", "model_domain"]):
            base = meta | {"domain": masks.DOMAIN_NAMES[dom], "model_domain": masks.DOMAIN_NAMES[md]}
            rows.append(base | {"subset": "full", **metrics.summarize(g[c].to_numpy(), g["z"].to_numpy())})
            gc = g[g[f"common|{r}"]]
            rows.append(base | {"subset": "common", **metrics.summarize(gc[c].to_numpy(), gc["z"].to_numpy())})
            if dom != md:
                continue
            lo, hi = (zr[r].loc[dom, "min"], zr[r].loc[dom, "max"]) if dom in zr[r].index else (np.nan, np.nan)
            g = g.assign(z_bin=pd.cut(g["z"], Z_BINS).astype(str), d_bin=pd.cut(g[f"d|{r}"], D_BINS, right=False).astype(str),
                         std_bin=pd.cut(g["z_std"], STD_BINS, right=False).astype(str),
                         support_bin=pd.cut(g["support"], SUP_BINS).astype(str),
                         extrapolation=np.where(np.isnan(lo), "no_training_labels",
                                                np.where((g["z"] < lo) | (g["z"] > hi), "outside_train_range",
                                                         "inside_train_range")))
            for s in ("z_bin", "d_bin", "std_bin", "ndsm_class", "support_bin", "extrapolation"):
                st = metrics.stratified(g, c, s).rename(columns={s: "bin"})
                srows.append(st.assign(stratifier=s, **base))
    M = pd.DataFrame(rows)
    S = pd.concat(srows, ignore_index=True)
    tag = a.partition
    M.to_csv(D["tables"] / f"metrics_{tag}.csv", index=False)
    S.to_csv(D["tables"] / f"strata_{tag}.csv", index=False)
    prim = M[(M["domain"] == M["model_domain"])]
    lay = (prim[prim["density"] != "dense"].groupby(["method", "density", "n_tracks", "domain", "subset"])
           .agg(rmse_mean=("rmse", "mean"), rmse_min=("rmse", "min"), rmse_max=("rmse", "max"),
                mae_mean=("mae", "mean"), bias_mean=("bias", "mean"), coverage_min=("coverage", "min"),
                n_variants=("rmse", "size"), n_supported=("rmse", "count")).reset_index())
    lay.to_csv(D["tables"] / f"layout_summary_{tag}.csv", index=False)
    from island_altimetry import manifest
    manifest.write_json_atomic(D["tables"] / f"common_subset_{tag}.json", {
        "members": common_members, "frac_common": {
            f"{r}|{masks.DOMAIN_NAMES[k]}": float(v) for r in regs
            for k, v in ev[ev["domain"] == ev["model_domain"]].groupby("model_domain")[f"common|{r}"].mean().items()}})
    print(f"{a.partition}: {len(ev)} routed rows")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(lay[lay["subset"] == "common"].pivot_table(index=["method", "n_tracks"], columns="domain",
                                                         values=["rmse_mean", "rmse_min", "rmse_max"]))
        print(prim[(prim["subset"] == "common") & (prim["density"] == "dense")][["method", "regime", "domain", "rmse", "mae", "bias", "n_pred"]])

    fig_dir = D["figures"] / f"eval_{tag}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    layout_plot(prim[prim["subset"] == "common"], fig_dir / "rmse_by_layout.png", tag)
    scatter_plot(ev, fig_dir / "scatter.png")
    dist_plot(S, fig_dir / "error_vs_distance.png")
    spec = json.loads((W / "reference" / "grid_spec.json").read_text())
    maps(cells, pdir, spec, W, fig_dir / "maps.png", code)
    profiles(cells, pdir, spec, W, fig_dir / "profiles.png", code)


def layout_plot(prim: pd.DataFrame, out: Path, tag: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, dom in zip(axs, ("land", "water")):
        d = prim[prim["domain"] == dom]
        for m in ("B0", "B1", "R2", "R3", "T2"):
            g = d[(d["method"] == m) & (d["density"] != "dense")]
            if g.empty:
                continue
            agg = g.groupby("n_tracks")["rmse"].agg(["mean", "min", "max"])
            ax.plot(agg.index, agg["mean"], "o-", color=METHOD_COLORS[m], label=m)
            ax.fill_between(agg.index, agg["min"], agg["max"], color=METHOD_COLORS[m], alpha=0.15)
        for m, ls in (("R1", "--"), ("T1", "-."), ("B0", ":")):
            g = d[(d["method"] == m) & (d["density"] == "dense")]
            if not g.empty:
                ax.axhline(g["rmse"].iloc[0], color=METHOD_COLORS[m], ls=ls, label=f"{m} dense")
        ax.set_xscale("log", base=2)
        ax.set_xticks([3, 6, 12], ["3", "6", "12"])
        ax.set_xlabel("simulated track centerlines")
        ax.set_ylabel("RMSE (m)")
        ax.set_title(f"{dom} ({tag}, common-supported cells; band = variant range)", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def scatter_plot(ev: pd.DataFrame, out: Path) -> None:
    cases = [("R1", "dense"), ("T1", "dense"), ("B1", MAP_LAYOUT), ("R2", MAP_LAYOUT), ("T2", MAP_LAYOUT)]
    fig, axs = plt.subplots(2, len(cases), figsize=(4 * len(cases), 8))
    for j, (m, r) in enumerate(cases):
        c = f"{m}|{r}"
        if c not in ev:
            continue
        for i, dom in enumerate((masks.LAND, masks.WATER)):
            g = ev[(ev["domain"] == dom) & (ev["model_domain"] == dom) & np.isfinite(ev[c])]
            ax = axs[i, j]
            lim = np.percentile(g["z"], [0.5, 99.5]) if len(g) else (0, 1)
            ax.hexbin(g["z"], g[c], gridsize=70, bins="log", extent=(*lim, *lim), cmap="viridis", mincnt=1)
            ax.plot(lim, lim, "w--", lw=0.8)
            ax.set_title(f"{m} {r} - {masks.DOMAIN_NAMES[dom]}", fontsize=9)
            ax.set_xlabel("reference (m)")
            ax.set_ylabel("predicted (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def dist_plot(S: pd.DataFrame, out: Path) -> None:
    d = S[(S["stratifier"] == "d_bin") & ((S["regime"] == MAP_LAYOUT) | (S["regime"] == "dense"))]
    order = [str(pd.Interval(a, b, closed="left")) for a, b in zip(D_BINS[:-1], D_BINS[1:])]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dom in zip(axs, ("land", "water")):
        for (m, r), g in d[d["domain"] == dom].groupby(["method", "regime"]):
            g = g.set_index("bin").reindex(order)
            ax.plot(range(len(order)), g["rmse"], marker="o", color=METHOD_COLORS[m], ls="--" if r == "dense" else "-",
                    label=f"{m} {r}")
        ax.set_xticks(range(len(order)), [o.replace("inf", "∞") for o in order], rotation=30, fontsize=7)
        ax.set_xlabel("distance to nearest same-domain training label (m)")
        ax.set_ylabel("RMSE (m)")
        ax.set_title(dom)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def as_grid(df: pd.DataFrame, col: str, spec: dict) -> np.ndarray:
    a = np.full((spec["height"], spec["width"]), np.nan, np.float32)
    a[df["row"].to_numpy(), df["col"].to_numpy()] = df[col].to_numpy()
    return a


def own_domain_pred(cells: pd.DataFrame, f: Path) -> pd.DataFrame:
    p = pd.read_parquet(f)
    c = cells[cells["domain"].isin(ex.DOMAINS)][["cell_id", "row", "col", "z", "domain", "partition"]]
    p["model_domain"] = p["model_domain"].astype(c["domain"].dtype)
    return c.merge(p, left_on=["cell_id", "domain"], right_on=["cell_id", "model_domain"], how="left")


def maps(cells, pdir, spec, W, out: Path, code: int) -> None:
    cases = [("R1", "dense"), ("T1", "dense"), ("B1", MAP_LAYOUT), ("R2", MAP_LAYOUT), ("T2", MAP_LAYOUT)]
    t = Affine(*spec["transform"])
    ext = (t.c, t.c + spec["width"] * t.a, t.f + spec["height"] * t.e, t.f)
    ref = as_grid(cells[cells["domain"].isin(ex.DOMAINS)], "z", spec)
    part = grid.read_band(W / "splits" / "split.tif").astype(float)
    from rasterio.crs import CRS
    lines = grid.read_geojson(W / "labels" / f"tracks_{MAP_LAYOUT}.geojson", CRS.from_wkt(spec["crs"]))
    fig, axs = plt.subplots(2, 1 + len(cases), figsize=(4.2 * (1 + len(cases)), 13), sharex=True, sharey=True)
    zkw = {"cmap": "terrain", "vmin": -3, "vmax": 6, "interpolation": "nearest"}
    axs[0, 0].imshow(ref, extent=ext, **zkw)
    axs[0, 0].set_title("reference (land+water cells)", fontsize=9)
    pc = np.where(part == 255, np.nan, part)
    axs[1, 0].imshow(pc, extent=ext, cmap=plt.matplotlib.colors.ListedColormap(["0.8", "#2e9d5b", "#e07b24", "#c43d5a"]),
                     vmin=-0.5, vmax=3.5, interpolation="nearest")
    axs[1, 0].set_title("split: grey excl, green train, orange val, red test", fontsize=8)
    for g in getattr(lines, "geoms", [lines]):
        axs[1, 0].plot(*g.xy, color="k", lw=0.5)
    for j, (m, r) in enumerate(cases, 1):
        f = pdir / f"{m}_{r}.parquet"
        if not f.exists():
            continue
        p = own_domain_pred(cells, f)
        pg = as_grid(p, "pred", spec)
        im = axs[0, j].imshow(pg, extent=ext, **zkw)
        axs[0, j].set_title(f"{m} {r} prediction", fontsize=9)
        err = pg - ref
        im2 = axs[1, j].imshow(err, extent=ext, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
        axs[1, j].set_title(f"{m} {r} error (pred - ref)", fontsize=9)
        if r != "dense":
            for g in getattr(lines, "geoms", [lines]):
                axs[1, j].plot(*g.xy, color="k", lw=0.4)
    fig.colorbar(im, ax=axs[0, :].tolist(), shrink=0.5, label="elevation (m NAVD88)")
    fig.colorbar(im2, ax=axs[1, :].tolist(), shrink=0.5, label="error (m)")
    ax0 = axs[0, 0]
    ax0.set_xlim(ext[0], ext[1])
    ax0.set_ylim(ext[2], ext[3])
    fig.savefig(out, dpi=100)
    plt.close(fig)


def profiles(cells, pdir, spec, W, out: Path, code: int, n: int = 4, half_len: float = 1500) -> None:
    """Fixed cross-strip profiles through the centres of the evaluated
    partition's blocks (chosen from split geometry only)."""
    sd = json.loads((W / "splits" / "split.json").read_text())
    ax_ = sd["axis"]
    (cx, cy), (ux, uy) = ax_["centroid"], ax_["direction"]
    edges = np.array(sd["edges_s_m"])
    centres = [(edges[i] + edges[i + 1]) / 2 for i, c in enumerate(sd["block_codes"]) if c == code][:n]
    t = Affine(*spec["transform"])
    cases = [("R1", "dense"), ("T1", "dense"), ("B1", MAP_LAYOUT), ("R2", MAP_LAYOUT), ("T2", MAP_LAYOUT)]
    preds = {f"{m} {r}": own_domain_pred(cells, pdir / f"{m}_{r}.parquet") for m, r in cases
             if (pdir / f"{m}_{r}.parquet").exists()}
    grids = {k: as_grid(v, "pred", spec) for k, v in preds.items()}
    ref = as_grid(cells[cells["domain"].isin(ex.DOMAINS)], "z", spec)
    fig, axs = plt.subplots(len(centres), 1, figsize=(11, 2.8 * len(centres)), squeeze=False)
    for k, s0 in enumerate(centres):
        s = np.arange(-half_len, half_len, 3.0)
        x, y = cx + ux * s0 - uy * s, cy + uy * s0 + ux * s
        col, row = ~t * (x, y)
        col, row = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (row >= 0) & (row < spec["height"]) & (col >= 0) & (col < spec["width"])
        ax = axs[k, 0]
        def take(a):
            v = np.full(len(s), np.nan)
            v[ok] = a[row[ok], col[ok]]
            return v
        ax.plot(s, take(ref), color="k", lw=1.4, label="reference")
        for name, g in grids.items():
            ax.plot(s, take(g), lw=0.8, color=METHOD_COLORS[name.split()[0]], label=name,
                    ls=":" if name.startswith("T2") else "-")
        ax.set_ylabel("z (m)")
        ax.set_title(f"profile {k + 1}: across strip at s = {s0:.0f} m (block centre)", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, ncol=5)
    axs[-1, 0].set_xlabel("distance across strip (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
