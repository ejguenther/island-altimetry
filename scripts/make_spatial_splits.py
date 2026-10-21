#!/usr/bin/env python
"""Coastal M3a: freeze the spatial train/validation/test split.

  make_spatial_splits.py [--config configs/coastal_experiment.yaml]

Along-strip blocks of equal usable-cell count, whole blocks assigned to
partitions with a recorded seed, guard buffers excluded at every boundary
touching train. Published (maps, summaries) before any label extraction or
fitting. The buffer must exceed the largest feature-window radius; a
train-only empirical semivariogram is recorded as the spatial-dependence
context for the chosen width (descriptive, not used to tune it).

Writes data/<site>/coastal/splits/{split.parquet,split.tif,split.json} and
outputs/.../{tables/split_summary.json,figures/splits/split_map.png}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from island_altimetry import manifest, plotting  # plotting sets the Agg backend
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import features, grid, masks, splits
from island_altimetry.config import fmt_utc

import matplotlib.pyplot as plt  # noqa: E402


def semivariogram(xy: np.ndarray, z: np.ndarray, rng, n: int = 4000, max_d: float = 1500, bin_m: float = 50) -> pd.DataFrame:
    i = rng.choice(len(z), size=min(n, len(z)), replace=False)
    p, v = xy[i], z[i]
    a, b = np.triu_indices(len(i), 1)
    d = np.hypot(*(p[a] - p[b]).T)
    g = 0.5 * (v[a] - v[b]) ** 2
    m = d < max_d
    bins = np.arange(0, max_d + bin_m, bin_m)
    k = np.digitize(d[m], bins) - 1
    df = pd.DataFrame({"bin": k, "g": g[m]}).groupby("bin")["g"].agg(["mean", "size"]).reset_index()
    df["lag_m"] = bins[df["bin"]] + bin_m / 2
    return df[["lag_m", "mean", "size"]].rename(columns={"mean": "semivariance", "size": "n_pairs"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D, sc = ccfg.dirs(cfg), cfg["splits"]
    rdir, sdir = D["work"] / "reference", D["work"] / "splits"
    spec = json.loads((rdir / "grid_spec.json").read_text())
    ref_sum = json.loads((D["tables"] / "reference_summary.json").read_text())
    cells = pd.read_parquet(rdir / "cells.parquet", columns=["cell_id", "row", "col", "x", "y", "z", "domain"])

    radius_m = features.MAX_RADIUS_PX * spec["res"][0] * np.sqrt(2)
    if sc["buffer_m"] <= 2 * radius_m:
        raise SystemExit(f"splits.buffer_m {sc['buffer_m']} must exceed twice the feature window radius ({radius_m:.1f} m)")

    res = splits.make_split(cells, ref_sum["strip_axis"], sc["n_blocks"], sc["fractions"], sc["buffer_m"], sc["seed"])
    cells["block"], cells["partition"], cells["s"] = res["block"].astype(np.int16), res["partition"].astype(np.int8), res["s"]
    sdir.mkdir(parents=True, exist_ok=True)
    tmp = sdir / "split.parquet.tmp"
    cells[["cell_id", "block", "partition", "s"]].to_parquet(tmp, index=False)
    tmp.replace(sdir / "split.parquet")
    ras = np.full((spec["height"], spec["width"]), 255, np.uint8)
    ras[cells["row"], cells["col"]] = cells["partition"]
    grid.write_raster(sdir / "split.tif", {"partition": ras}, spec, nodata=255, dtype="uint8")

    rng = np.random.default_rng(sc["seed"])
    tr = cells[cells["partition"] == splits.PARTITIONS["train"]]
    vg = {masks.DOMAIN_NAMES[d]: semivariogram(g[["x", "y"]].to_numpy(), g["z"].to_numpy(), rng)
          for d, g in tr.groupby("domain") if d in (masks.LAND, masks.WATER) and len(g) > 100}

    names = {v: k for k, v in splits.PARTITIONS.items()} | {splits.EXCLUDED: "excluded"}
    counts = (cells.assign(part=cells["partition"].map(names), dom=cells["domain"].map(masks.DOMAIN_NAMES))
              .groupby(["part", "dom"]).size().unstack(fill_value=0))
    split_def = {"generated_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)), "git_commit": manifest.git_commit(),
                 "config_hash": cfg["config_hash"], "method": "equal-count along-strip blocks, whole-block assignment",
                 "axis": ref_sum["strip_axis"], "edges_s_m": res["edges"].tolist(),
                 "block_codes": res["block_codes"].tolist(), "partition_codes": names,
                 "seed_config": sc["seed"], "seed_used": res["seed_used"], "attempts": res["attempts"],
                 "buffer_m": sc["buffer_m"], "feature_window_radius_m": radius_m,
                 "buffer_rationale": (f"{sc['buffer_m']} m total gap between train and held-out cells: far above the "
                                      f"{radius_m:.1f} m 5x5 window radius, so no context window spans a train and a held-out "
                                      "cell; set a priori, not tuned on test performance. See semivariogram for the "
                                      "spatial-dependence context (correlation persists beyond the buffer, so held-out "
                                      "blocks are only partly independent; block lengths of ~1.5-2 km help)."),
                 "block_length_m": float(np.diff(res["edges"]).mean())}
    manifest.write_json_atomic(sdir / "split.json", split_def)
    summary = split_def | {"cells_by_partition_domain": counts.to_dict(orient="index"), "coverage": res["coverage"],
                           "semivariogram_train": {k: v.round(4).to_dict(orient="list") for k, v in vg.items()}}
    manifest.write_json_atomic(D["tables"] / "split_summary.json", summary)

    fig_dir = D["figures"] / "splits"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    colors = {1: "#2e9d5b", 2: "#e07b24", 3: "#c43d5a", 0: "0.75"}
    sub = cells.iloc[::3]
    for code, g in sub.groupby("partition"):
        axs[0].scatter(g["x"], g["y"], s=0.05, color=colors[int(code)], label=names[int(code)], rasterized=True)
    axs[0].set_aspect("equal")
    axs[0].legend(markerscale=60, fontsize=8)
    axs[0].set_title(f"split (seed {res['seed_used']}, buffer {sc['buffer_m']} m)")
    for (code, dom), g in cells.groupby(["partition", "domain"]):
        if code == 0 or dom not in (1, 2):
            continue
        axs[1].hist(g["z"], bins=80, histtype="step", density=True, color=colors[int(code)],
                    ls="-" if dom == 1 else "--", label=f"{names[int(code)]} {masks.DOMAIN_NAMES[int(dom)]}")
    axs[1].set_xlabel("reference elevation (m NAVD88)")
    axs[1].legend(fontsize=7)
    axs[1].set_title("elevation by partition (solid land, dashed water)")
    for k, v in vg.items():
        axs[2].plot(v["lag_m"], v["semivariance"], "o-", ms=3, label=k)
    axs[2].axvline(sc["buffer_m"], color="k", ls=":", label="buffer")
    axs[2].set_xlabel("lag (m)")
    axs[2].set_ylabel("semivariance (m$^2$)")
    axs[2].legend(fontsize=8)
    axs[2].set_title("train-partition semivariogram")
    fig.tight_layout()
    fig.savefig(fig_dir / "split_map.png", dpi=110)
    plt.close(fig)
    print(counts)
    print(f"seed used {res['seed_used']}; block codes {res['block_codes'].tolist()}")


if __name__ == "__main__":
    main()
