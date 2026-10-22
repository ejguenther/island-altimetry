#!/usr/bin/env python
"""Coastal QA: show selected terrain feature layers over a subset window
(texture / shape / orientation at 3, 9 and 15 px) to check for artefacts at
mask edges and strip seams. No labels or reference values are used.

  plot_terrain_features.py [--config ...] [--center-x X --center-y Y] [--size 600] [--layer nir]

Writes outputs/.../figures/features/terrain_layers.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from island_altimetry import plotting  # noqa: F401 - sets the Agg backend
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import features, grid
from scripts_path import load_train

import matplotlib.pyplot as plt  # noqa: E402

SHOW = ["glcm_contrast", "entropy", "pos_index", "hess_l2", "orient_cos2", "coherence"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    ap.add_argument("--center-x", type=float, help="subset centre (grid CRS); default: usable-cell centroid")
    ap.add_argument("--center-y", type=float)
    ap.add_argument("--size", type=int, default=600, help="subset edge (px)")
    ap.add_argument("--layer", default="nir")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D = ccfg.dirs(cfg)
    rdir = D["work"] / "reference"
    spec = json.loads((rdir / "grid_spec.json").read_text())
    t = Affine(*spec["transform"])
    if a.center_x is None:
        ax_ = json.loads((D["tables"] / "reference_summary.json").read_text())["strip_axis"]["centroid"]
        a.center_x, a.center_y = ax_
    c, r = ~t * (a.center_x, a.center_y)
    r0, c0 = max(int(r) - a.size // 2, 0), max(int(c) - a.size // 2, 0)
    sl = np.s_[r0:r0 + a.size, c0:c0 + a.size]
    with rasterio.open(rdir / "planet_sr.tif") as s:
        sr, names = s.read()[:, sl[0], sl[1]], list(s.descriptions)
    valid = grid.read_band(rdir / "masks.tif", "image_valid").astype(bool)[sl]
    domain = grid.read_band(rdir / "masks.tif", "domain")[sl]
    tc = load_train().terrain_cfg(cfg) | {"layers": [a.layer], "distance_features": False}
    want = {f"{a.layer}_{st}{w}" for st in SHOW for w in tc["windows"]}
    got = {n: x for n, _, x in features.terrain_layers(sr, names, valid, domain, spec["res"][0], tc) if n in want}
    ws = tc["windows"]
    fig, axs = plt.subplots(len(SHOW), len(ws), figsize=(3.3 * len(ws), 3.1 * len(SHOW)))
    for i, st in enumerate(SHOW):
        for j, w in enumerate(ws):
            x = got[f"{a.layer}_{st}{w}"]
            lo, hi = np.nanpercentile(x, [2, 98])
            axs[i, j].imshow(x, cmap="RdBu_r" if st in ("pos_index", "hess_l2", "orient_cos2") else "viridis",
                             vmin=lo, vmax=hi, interpolation="nearest")
            axs[i, j].set_title(f"{a.layer} {st} {w}x{w}", fontsize=8)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    fig.suptitle(f"terrain features, {a.size}x{a.size} px subset at ({a.center_x:.0f}, {a.center_y:.0f})", fontsize=9)
    fig.tight_layout()
    out = D["figures"] / "features" / "terrain_layers.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
