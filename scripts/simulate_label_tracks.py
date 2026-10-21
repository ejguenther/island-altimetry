#!/usr/bin/env python
"""Coastal M3b: dense and simulated-track label sets from the frozen split.

  simulate_label_tracks.py [--config configs/coastal_experiment.yaml]

Sparse: for every predeclared layout (count x variant, variant = azimuth +
translation) the unique reference cells crossed by straight centerlines;
lines may span the whole crop for display, but labels come only from usable
land/water cells of the TRAIN partition. Dense: the union of all sparse cells
plus a reproducible spatially distributed sample of train cells up to the
cap, split evenly between land and water where both have enough cells (the
evaluation population is unchanged). A smaller smoke-run dense set is
written alongside.

Writes data/<site>/coastal/labels/{layouts.json,sparse_<id>.parquet,
dense.parquet,dense_smoke.parquet,tracks_<id>.geojson} and
outputs/.../{tables/label_summary.csv,figures/labels/layouts.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from rasterio.crs import CRS

from island_altimetry import manifest, plotting  # plotting sets the Agg backend
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import grid, masks, splits, tracks

import matplotlib.pyplot as plt  # noqa: E402

TRAIN = splits.PARTITIONS["train"]
DOMAINS = (masks.LAND, masks.WATER)


def layouts(cfg: dict) -> list[dict]:
    t = cfg["labels"]["tracks"]
    if len(t["azimuth_deg"]) != len(t["offset_frac"]):
        raise SystemExit("labels.tracks.azimuth_deg and offset_frac define paired variants; lengths differ")
    return [{"layout_id": f"t{n:02d}_v{v}", "count": n, "variant": v, "azimuth_deg": az, "offset_frac": off}
            for n in t["counts"] for v, (az, off) in enumerate(zip(t["azimuth_deg"], t["offset_frac"]))]


def dense_sample(train: pd.DataFrame, must: np.ndarray, cap: int, rng) -> np.ndarray:
    """Union of `must` cell ids plus a random fill to `cap`, balanced across domains."""
    must = np.unique(must)
    if len(must) >= cap:
        return must
    pool = train[~train["cell_id"].isin(must)]
    left = cap - len(must)
    per = {d: int((pool["domain"] == d).sum()) for d in DOMAINS}
    share = {d: left // len(DOMAINS) for d in DOMAINS}
    for d in DOMAINS:                       # hand unused share to the other domain
        o = [x for x in DOMAINS if x != d][0]
        if per[d] < share[d]:
            share[o] += share[d] - per[d]
            share[d] = per[d]
    take = [must]
    for d in DOMAINS:
        ids = pool.loc[pool["domain"] == d, "cell_id"].to_numpy()
        take.append(rng.choice(ids, size=min(share[d], len(ids)), replace=False))
    return np.unique(np.concatenate(take))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D, lc = ccfg.dirs(cfg), cfg["labels"]
    rdir, ldir = D["work"] / "reference", D["work"] / "labels"
    spec = json.loads((rdir / "grid_spec.json").read_text())
    cells = pd.read_parquet(rdir / "cells.parquet", columns=["cell_id", "x", "y", "z", "domain"]).merge(
        pd.read_parquet(D["work"] / "splits" / "split.parquet", columns=["cell_id", "partition"]), on="cell_id")
    train = cells[(cells["partition"] == TRAIN) & cells["domain"].isin(DOMAINS)]
    train_ids = set(train["cell_id"].to_numpy().tolist())
    t = Affine(*spec["transform"])
    bounds = (cells["x"].min(), cells["y"].min(), cells["x"].max(), cells["y"].max())
    crs = CRS.from_wkt(spec["crs"])
    ldir.mkdir(parents=True, exist_ok=True)

    rows, all_sparse, lines_by = [], [], {}
    for L in layouts(cfg):
        lines, spacing = tracks.layout_lines(bounds, L["azimuth_deg"], L["count"], L["offset_frac"])
        crossed = tracks.line_cells(lines, t, (spec["height"], spec["width"]))
        lab = train[train["cell_id"].isin(crossed)]
        assert lab["cell_id"].is_unique and set(lab["cell_id"]) <= train_ids
        lab[["cell_id"]].to_parquet(ldir / f"sparse_{L['layout_id']}.parquet", index=False)
        grid.write_geojson(shapely.MultiLineString(lines), crs, ldir / f"tracks_{L['layout_id']}.geojson", L)
        lines_by[L["layout_id"]] = lines
        all_sparse.append(lab["cell_id"].to_numpy())
        rec = L | {"spacing_m": spacing, "n_cells_crossed_grid": int(len(crossed)),
                   "n_cells_crossed_usable": int(cells["cell_id"].isin(crossed).sum()), "n_labels": len(lab)}
        for d in DOMAINS:
            z = lab.loc[lab["domain"] == d, "z"]
            nm = masks.DOMAIN_NAMES[d]
            rec |= {f"n_{nm}": int(len(z)), f"z_min_{nm}": float(z.min()) if len(z) else np.nan,
                    f"z_max_{nm}": float(z.max()) if len(z) else np.nan,
                    f"frac_train_{nm}": float(len(z) / max(1, (train["domain"] == d).sum()))}
        rows.append(rec)
        print(f"  {L['layout_id']}: spacing {spacing:.0f} m, {len(lab)} labels "
              f"(land {rec['n_land']}, water {rec['n_water']})")
    must = np.concatenate(all_sparse)
    for name, cap in (("dense", lc["dense_cap"]), ("dense_smoke", lc["smoke_cap"])):
        ids = dense_sample(train, must if name == "dense" else np.array([], np.int64), cap,
                           np.random.default_rng(lc["seed"]))
        assert set(ids.tolist()) <= train_ids
        pd.DataFrame({"cell_id": ids}).to_parquet(ldir / f"{name}.parquet", index=False)
        dd = train[train["cell_id"].isin(ids)]
        rows.append({"layout_id": name, "n_labels": len(ids), "includes_all_sparse": name == "dense" and bool(np.isin(must, ids).all()),
                     **{f"n_{masks.DOMAIN_NAMES[d]}": int((dd["domain"] == d).sum()) for d in DOMAINS}})
        print(f"  {name}: {len(ids)} labels")
    summ = pd.DataFrame(rows)
    summ.to_csv(D["tables"] / "label_summary.csv", index=False)
    manifest.write_json_atomic(ldir / "layouts.json", {"config_hash": cfg["config_hash"], "bounds": list(bounds),
                                                        "layouts": layouts(cfg), "train_cells": len(train)})

    fig_dir = D["figures"] / "labels"
    fig_dir.mkdir(parents=True, exist_ok=True)
    counts = cfg["labels"]["tracks"]["counts"]
    fig, axs = plt.subplots(1, len(counts), figsize=(5 * len(counts), 7), sharex=True, sharey=True)
    sub = cells.iloc[::5]
    pc = {0: "0.85", 1: "#b9e3c6", 2: "#f6d2b0", 3: "#f1b6c1"}
    for ax, n in zip(np.atleast_1d(axs), counts):
        ax.scatter(sub["x"], sub["y"], s=0.03, c=sub["partition"].map(pc), rasterized=True)
        for v, col in zip(range(3), ["k", "#2a6fdb", "#8a63c9"]):
            for ln in lines_by.get(f"t{n:02d}_v{v}", []):
                c = ln.intersection(shapely.box(*bounds))
                for g in getattr(c, "geoms", [c]):
                    if not g.is_empty:
                        ax.plot(*g.xy, color=col, lw=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(f"{n} centerlines x 3 variants\n(green train, orange val, pink test)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "layouts.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
