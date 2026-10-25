#!/usr/bin/env python
"""Slide figures for the CNN terrain feasibility study (static PNGs, 16:9-friendly).

  plot_cnn_slides.py [--site florida_east_coast] [--run sparse_large_s0]

Reads the frozen experiment outputs only (no retraining). Writes
outputs/cnn_terrain/slides/fig1_concept.png ... fig7_unseen_sites.png (no fig5).
Colors: categorical slots blue / orange / aqua for CNN / RF / IDW (validated
all-pairs, light surface; aqua is under 3:1 contrast, so every chart labels
methods directly), gray for the constant baseline; elevation uses a one-hue
blue ramp, errors a blue-gray-red diverging ramp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS

from island_altimetry import plotting  # noqa: F401 - Agg backend
from island_altimetry.cnn import config as ccfg
from island_altimetry.coastal import grid

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CNN, RF, IDW, CONST = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
ELEV = LinearSegmentedColormap.from_list("elev", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
LAB = LinearSegmentedColormap.from_list("lab", ["#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("err", ["#184f95", "#6da7ec", "#f0efec", "#ec8a87", "#a32b2a"])
PART = {1: ("train", "#d9e8fb"), 2: ("validation", "#fbe3d6"), 3: ("test", "#d6f2e7")}
NAMES = {"boca_grande_fl": "Boca Grande, FL", "corpus_christi_tx": "Corpus Christi, TX", "duck_nc": "Duck, NC",
         "florida_east_coast": "FL East Coast", "florida_keys": "Florida Keys", "egmont_key_fl": "Egmont Key, FL",
         "florida_panhandle": "FL Panhandle", "breton_island_la": "Breton Island, LA"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titleweight": "bold",
                     "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})


def gray_rgb(sr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Light grayscale backdrop from the SR bands (context only)."""
    b = sr[:3].astype(np.float32).mean(0)
    lo, hi = np.percentile(b[valid], [2, 98]) if valid.any() else (0, 1)
    g = np.clip((b - lo) / max(hi - lo, 1), 0, 1)
    g = 0.55 + 0.4 * g
    g[~valid] = 1.0
    return np.dstack([g, g, g])


def true_rgb(sr: np.ndarray, names: list[str], valid: np.ndarray) -> np.ndarray:
    out = np.zeros((*sr.shape[1:], 3), np.float32)
    for k, b in enumerate(("red", "green", "blue")):
        a = sr[names.index(b)].astype(np.float32)
        lo, hi = np.percentile(a[valid], [2, 98])
        out[..., k] = np.clip((a - lo) / max(hi - lo, 1), 0, 1)
    out[~valid] = 1.0
    return out


def no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, px_m: float, length_m: float, x0: float, y0: float, color=INK):
    """Scale bar in pixel coordinates (image axes)."""
    L = length_m / px_m
    ax.plot([x0, x0 + L], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 - 8, f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m",
            ha="center", va="bottom", fontsize=10, color=color)


class Site:
    def __init__(self, cfg, site, run):
        sp = ccfg.site_paths(cfg, site)
        r = sp["work"] / "reference"
        self.spec = json.loads((r / "grid_spec.json").read_text())
        self.t = Affine(*self.spec["transform"])
        with rasterio.open(r / "planet_sr.tif") as s:
            self.sr, self.names = s.read(), list(s.descriptions)
        self.valid = grid.read_band(r / "masks.tif", "image_valid").astype(bool)
        self.z = grid.read_band(r / "target.tif", "z_target")
        self.elig = grid.read_band(r / "masks.tif", "eligible").astype(bool)
        self.split = grid.read_band(sp["work"] / "split.tif")
        self.pred = grid.read_band(sp["work"] / "pred" / f"{run}.tif")
        self.centers = pd.read_parquet(sp["work"] / "centers.parquet")
        self.tracks = grid.read_geojson(sp["out"] / "tracks.geojson", CRS.from_wkt(self.spec["crs"]))
        out = ccfg.out_dir(cfg)
        self.labels = pd.read_parquet(out / "pools" / "sparse.parquet").query("site == @site")
        P = {}
        for m in (run, "idw_sparse", "rf_sparse"):
            p = pd.read_parquet(out / "predictions" / f"{m}.parquet").query("site == @site")
            P[m] = p.set_index("cell_id")["pred"]
        self.P = P
        self.res = self.spec["res"][0]

    def raster(self, values: pd.Series) -> np.ndarray:
        a = np.full((self.spec["height"], self.spec["width"]), np.nan, np.float32)
        c = self.centers.set_index("cell_id").loc[values.index]
        a[c["row"].to_numpy(), c["col"].to_numpy()] = values.to_numpy()
        return a

    def track_px(self):
        inv = ~self.t
        for ln in getattr(self.tracks, "geoms", [self.tracks]):
            xs, ys = ln.xy
            c, r = inv * (np.asarray(xs), np.asarray(ys))
            yield c - 0.5, r - 0.5


def strip_rotation(mask: np.ndarray):
    """(angle_deg, centre) rotating the mask's principal axis to horizontal in
    image (col, row) display coordinates; the sign that minimises the rotated
    vertical extent is chosen."""
    from matplotlib.transforms import Affine2D
    r, c = np.nonzero(mask)
    xy = np.column_stack([c, r]).astype(float)[:: max(1, len(r) // 50000)]
    mu = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - mu, full_matrices=False)
    base = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
    best = None
    for ang in (-base, base, 180 - base, base - 180):
        t = Affine2D().rotate_deg_around(mu[0], mu[1], ang)
        p = t.transform(xy)
        ext = np.ptp(p[:, 1])
        if best is None or ext < best[0]:
            best = (ext, ang)
    return best[1], mu


def rotated_axes(ax, mask, ang, mu, pad=15):
    """Limits for a rotated view of `mask` and the data transform to use."""
    from matplotlib.transforms import Affine2D
    tr = Affine2D().rotate_deg_around(mu[0], mu[1], ang)
    r, c = np.nonzero(mask)
    p = tr.transform(np.column_stack([c, r]).astype(float))
    ax.set_xlim(p[:, 0].min() - pad, p[:, 0].max() + pad)
    ax.set_ylim(p[:, 1].max() + pad, p[:, 1].min() - pad)
    ax.set_aspect("equal")
    return tr


def bbox(rows, cols, pad, shape):
    return (max(rows.min() - pad, 0), min(rows.max() + pad, shape[0]), max(cols.min() - pad, 0),
            min(cols.max() + pad, shape[1]))


def fig_concept(S: Site, out: Path):
    """How the model is trained (one label per context patch) and applied (whole image at once)."""
    c = S.labels.iloc[len(S.labels) // 2]
    r, col = int(c["row"]), int(c["col"])
    w = 60
    r0, c0 = max(r - w, 0), max(col - w, 0)
    rgb = true_rgb(S.sr, S.names, S.valid)
    fig = plt.figure(figsize=(16, 5.9))
    ax1 = fig.add_axes([0.02, 0.1, 0.25, 0.68])
    ax1.imshow(rgb[r0:r + w, c0:col + w], interpolation="nearest")
    ax1.add_patch(Rectangle((col - c0 - 16.5, r - r0 - 16.5), 33, 33, fill=False, ec="#f5d000", lw=2.5))
    ax1.plot(col - c0, r - r0, "o", ms=9, mfc="#e34948", mec="white", mew=1.5)
    no_ticks(ax1)
    ax1.set_title("1  Training: one label per image patch\n"
                  "33 × 33 px (99 m) context → the lidar elevation\nof its single centre cell (red)",
                  loc="left", fontsize=13)
    fig.text(0.285, 0.43, "→", fontsize=40, color=INK2, ha="center", va="center")
    ax2 = fig.add_axes([0.31, 0.2, 0.15, 0.5])
    no_ticks(ax2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    for i, lab in enumerate(["3×3 conv, dilation 1", "3×3 conv, dilation 2", "3×3 conv, dilation 4",
                             "3×3 conv, dilation 8", "3×3 conv, dilation 1", "1×1 output"]):
        ax2.add_patch(Rectangle((0.02, 0.86 - i * 0.15), 0.96, 0.11, fc="#d9e8fb", ec=CNN, lw=1.2))
        ax2.text(0.5, 0.915 - i * 0.15, lab, ha="center", va="center", fontsize=10, color=INK)
    ax2.set_title("2  Small CNN\nno pooling, no stride", fontsize=13)
    ax2.text(0.5, -0.12, f"output: one elevation\n(compared with {c['z']:.2f} m)", ha="center", va="top",
             fontsize=11, color=INK2, transform=ax2.transAxes)
    r0w, r1w, c0w, c1w, _ = test_window(S, 420)
    ax3 = fig.add_axes([0.5, 0.1, 0.22, 0.68])
    ax3.imshow(rgb[r0w:r1w, c0w:c1w], interpolation="nearest")
    no_ticks(ax3)
    ax3.set_title("3  Inference: any image size\nsame weights, one pass\n(here an area never used for training)",
                  loc="left", fontsize=13)
    fig.text(0.735, 0.43, "→", fontsize=40, color=INK2, ha="center", va="center")
    ax4 = fig.add_axes([0.755, 0.1, 0.22, 0.68])
    land = S.pred[r0w:r1w, c0w:c1w]
    lo, hi = np.nanpercentile(S.z[S.elig], [2, 98])
    ax4.imshow(gray_rgb(S.sr, S.valid)[r0w:r1w, c0w:c1w], interpolation="nearest")
    im = ax4.imshow(land, cmap=ELEV, vmin=lo, vmax=hi, interpolation="nearest")
    no_ticks(ax4)
    ax4.set_title("4  Elevation map, every land pixel\n3 m grid, m NAVD88", loc="left", fontsize=13)
    cax = fig.add_axes([0.78, 0.06, 0.17, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("elevation (m NAVD88)", fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

def fig_training_data(S: Site, n_total_sparse: int, out: Path):
    """Where the sparse labels are: simulated tracks over the training blocks of one site (rotated so
    the coastal strip runs left-right): a full-strip overview plus a zoomed section."""
    ang, mu = strip_rotation(S.elig)
    fig, axs = plt.subplots(2, 1, figsize=(16, 5.2), gridspec_kw={"height_ratios": [1, 2.9], "hspace": 0.22})
    rgb = true_rgb(S.sr, S.names, S.valid)
    part = np.where(S.split == 255, 0, S.split)
    over = np.zeros((*part.shape, 4))
    for code, (_, colr) in PART.items():
        over[part == code] = plt.matplotlib.colors.to_rgba(colr, 0.95)
    lab = S.labels
    lo, hi = np.nanpercentile(S.z[S.elig], [2, 98])
    # overview: blocks + tracks + labels on the whole strip
    ax = axs[0]
    tr = rotated_axes(ax, S.elig, ang, mu)
    ax.imshow(gray_rgb(S.sr, S.valid), interpolation="nearest", transform=tr + ax.transData)
    ax.imshow(over, interpolation="nearest", transform=tr + ax.transData)
    for x, y in S.track_px():
        q = tr.transform(np.column_stack([x, y]))
        ax.plot(q[:, 0], q[:, 1], color="#f5d000", lw=1.2, alpha=0.9)
    p = tr.transform(np.column_stack([lab["col"], lab["row"]]).astype(float))
    ax.scatter(p[:, 0], p[:, 1], c=lab["z"], cmap=LAB, vmin=lo, vmax=hi, s=5, edgecolors="none", zorder=3)
    xl = ax.get_xlim()
    full = xl[1] - xl[0]
    z0, z1 = xl[0] + 0.02 * full, xl[0] + 0.40 * full       # zoom: first track + train/val/test blocks
    yl = ax.get_ylim()
    ax.add_patch(Rectangle((z0, yl[1]), z1 - z0, yl[0] - yl[1], fill=False, ec=INK, lw=1.4, ls="--"))
    ax.set_title(f"{NAMES[S.name]}: whole site, ~11 km of coast. Tracks (yellow) 1 km apart; labels only where "
                 "they cross training blocks", loc="left", fontsize=13)
    scalebar(ax, S.res, 1000, xl[0] + 15, yl[0] - 6)
    # zoom: real image + labelled cells
    ax = axs[1]
    tr = rotated_axes(ax, S.elig, ang, mu, pad=5)
    ax.imshow(rgb, interpolation="nearest", transform=tr + ax.transData)
    ax.imshow(over, interpolation="nearest", transform=tr + ax.transData, alpha=0.55)
    sc = ax.scatter(p[:, 0], p[:, 1], c=lab["z"], cmap=LAB, vmin=lo, vmax=hi, s=34, edgecolors="white",
                    linewidths=0.6, zorder=3)
    ax.set_xlim(z0, z1)
    rr, cc = np.nonzero(part > 0)
    q = tr.transform(np.column_stack([cc, rr]).astype(float)[::50])
    codes = part[rr[::50], cc[::50]]
    yl = ax.get_ylim()
    for code, (name, _) in PART.items():
        xs = q[(codes == code) & (q[:, 0] > z0) & (q[:, 0] < z1), 0]
        if len(xs) > 50:
            note = "training block: labels on the track" if code == 1 else f"{name} block: no labels"
            ax.text(np.median(xs), yl[1] + 0.12 * (yl[0] - yl[1]), note, ha="center", va="center", fontsize=11,
                    color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_title("Zoom (dashed box): PlanetScope image; each dot is one 3 m training label = mean lidar ground "
                 "elevation", loc="left", fontsize=13)
    scalebar(ax, S.res, 500, z0 + 10, ax.get_ylim()[0] - 6, color="white")
    for a in axs:
        no_ticks(a)
    cax = fig.add_axes([0.80, 0.03, 0.18, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_label("label elevation (m NAVD88)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    handles = [Patch(fc=c, ec="none", label=f"{n} blocks") for n, c in PART.values()]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.11, 0.0), ncol=3, frameon=False, fontsize=11)
    n_train = int((S.centers["partition"] == 1).sum())
    fig.text(0.5, -0.03, f"{len(lab):,} labelled cells at this site = {100 * len(lab) / n_train:.1f}% of its "
             f"training area; {n_total_sparse:,} labels over all 5 training sites. Water is image context only.",
             ha="center", fontsize=11, color=INK2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

def test_window(S: Site, max_px: int = 520):
    """Window on the largest test block of the site (chosen by geometry only)."""
    c = S.centers[S.centers["partition"] == 3]
    blk = c["block"].value_counts().index[0]
    c = c[c["block"] == blk]
    r0, r1, c0, c1 = bbox(c["row"].to_numpy(), c["col"].to_numpy(), 20, S.z.shape)
    if r1 - r0 > max_px:
        mid = (r0 + r1) // 2
        r0, r1 = mid - max_px // 2, mid + max_px // 2
    return r0, r1, c0, c1, blk


def fig_difference(S: Site, run: str, out: Path):
    """Held-out test block: reference, sparse-label CNN, and errors of CNN vs interpolation (same labels)."""
    r0, r1, c0, c1, blk = test_window(S)
    test = S.centers[(S.centers["partition"] == 3)]
    ids = test["cell_id"]
    ref = S.raster(test.set_index("cell_id")["z"])
    cnn = S.raster(S.P[run].reindex(ids).dropna())
    idw = S.raster(S.P["idw_sparse"].reindex(ids).dropna())
    lo, hi = np.nanpercentile(ref[r0:r1, c0:c1], [2, 98])
    back = gray_rgb(S.sr, S.valid)[r0:r1, c0:c1]
    win = np.s_[r0:r1, c0:c1]

    def rmse(a):
        e = (a - ref)[win]
        return float(np.sqrt(np.nanmean(e ** 2)))
    panels = [("Lidar reference (held out)", ref, ELEV, (lo, hi), None),
              ("CNN prediction (sparse labels)", np.where(S.elig, S.pred, np.nan), ELEV, (lo, hi), None),
              (f"CNN error  ·  RMSE {rmse(cnn):.2f} m", cnn - ref, DIV, (-2, 2), CNN),
              (f"Interpolation error  ·  RMSE {rmse(idw):.2f} m", idw - ref, DIV, (-2, 2), IDW)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 6.6), gridspec_kw={"wspace": 0.04})
    ims = []
    for ax, (title, a, cmap, (vmin, vmax), mark) in zip(axs, panels):
        ax.imshow(back, interpolation="nearest")
        ims.append(ax.imshow(a[win], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"))
        no_ticks(ax)
        ax.set_title(title, loc="left", fontsize=13)
        if mark:
            ax.plot([0.0, 0.18], [1.0, 1.0], transform=ax.transAxes, color=mark, lw=5, clip_on=False)
    scalebar(axs[0], S.res, 500, 15, r1 - r0 - 18)
    cb1 = fig.colorbar(ims[0], ax=axs[:2].tolist(), orientation="horizontal", fraction=0.05, pad=0.03, aspect=40)
    cb1.set_label("elevation (m NAVD88)")
    cb1.outline.set_visible(False)
    cb2 = fig.colorbar(ims[2], ax=axs[2:].tolist(), orientation="horizontal", fraction=0.05, pad=0.03, aspect=40,
                       extend="both")
    cb2.set_label("error = prediction − lidar (m)   blue: too low · red: too high")
    cb2.outline.set_visible(False)
    fig.suptitle(f"{NAMES[S.name]}, held-out test block: sparse-label CNN vs interpolation of the same track labels",
                 x=0.125, ha="left", y=0.93, fontsize=15, fontweight="bold")
    fig.text(0.125, 0.875, "Test block (no labels inside, ≥150 m from any training cell). "
             "Both use the same 1 km track labels; CNN seed 0.", fontsize=11, color=INK2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return (r0, r1, c0, c1)


def fig_per_site(M: pd.DataFrame, out: Path):
    """Per training site: test RMSE of each sparse-label method (common cells), seeds as a range."""
    f = M[(M["subset"] == "common") & (M["group"] == "within")]
    sites = ["corpus_christi_tx", "boca_grande_fl", "florida_keys", "florida_east_coast", "duck_nc"]
    piv = f.pivot_table(index="site", columns="method", values="rmse").reindex(sites)
    seeds = piv[["sparse_large_s0", "sparse_large_s1", "sparse_large_s2"]]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    y = np.arange(len(sites))[::-1]
    ax.grid(axis="x", color=GRID, lw=0.8)
    for i, s in enumerate(sites):
        vals = [piv.loc[s, "const_sparse"], piv.loc[s, "idw_sparse"], piv.loc[s, "rf_sparse"], seeds.loc[s].mean()]
        ax.plot([min(vals), max(vals)], [y[i]] * 2, color=GRID, lw=2, zorder=1)
    ax.scatter(piv["const_sparse"], y, s=110, marker="|", color=CONST, lw=3, label="constant (mean of labels)", zorder=3)
    ax.scatter(piv["idw_sparse"], y, s=90, color=IDW, edgecolor=SURFACE, lw=2, label="interpolation", zorder=3)
    ax.scatter(piv["rf_sparse"], y, s=90, color=RF, edgecolor=SURFACE, lw=2, label="random forest", zorder=3)
    ax.hlines(y, seeds.min(1), seeds.max(1), color=CNN, lw=6, alpha=0.35, zorder=2)
    ax.scatter(seeds.mean(1), y, s=110, color=CNN, edgecolor=SURFACE, lw=2, label="CNN (3 seeds: dot = mean, bar = range)",
               zorder=4)
    for i, s in enumerate(sites):
        ax.text(seeds.loc[s].mean(), y[i] - 0.3, f"CNN {seeds.loc[s].mean():.2f}", ha="center", fontsize=10, color=INK)
    ax.set_yticks(y, [NAMES[s] for s in sites])
    ax.set_xlabel("test RMSE (m)  ·  lower is better")
    ax.set_xlim(0, None)
    ax.set_ylim(-0.7, len(sites) - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=10)
    ax.set_title("Sparse labels: CNN beats the random forest at all 5 sites and interpolation at 4 of 5", loc="left")
    fig.text(0.01, 0.01, "Held-out test blocks of the 5 training sites; identical evaluation cells for all methods. "
             "Interpolation wins at Boca Grande, where the terrain is flat and close to the track labels.",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_summary(A: pd.DataFrame, out: Path):
    """Equal-site within-site test RMSE: dense vs sparse labels, by method."""
    a = A[(A["subset"] == "full") & (A["group"] == "within")].set_index("method")["equal_site_rmse"]
    groups = [("constant", "const_dense", "const_sparse", CONST), ("interpolation", None, "idw_sparse", IDW),
              ("random forest", "rf_dense", "rf_sparse", RF),
              ("CNN", ["dense_large_s0", "dense_large_s1", "dense_large_s2"],
               ["sparse_large_s0", "sparse_large_s1", "sparse_large_s2"], CNN)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(axis="y", color=GRID, lw=0.8)
    w = 0.36
    for i, (lab, d, s, colr) in enumerate(groups):
        for j, (k, kind) in enumerate(((s, "sparse tracks"), (d, "dense (50k/site)"))):
            if k is None:
                continue
            vals = a[k] if isinstance(k, str) else a[k]
            v = float(np.mean(vals))
            x = i + (j - 0.5) * w
            ax.bar(x, v, w * 0.92, color=colr, alpha=1.0 if j == 0 else 0.45, edgecolor="none")
            if not isinstance(k, str):
                ax.plot([x, x], [float(np.min(vals)), float(np.max(vals))], color=INK, lw=1.4)
            top = v if isinstance(k, str) else max(v, float(np.max(vals)))
            ax.text(x, top + 0.03, f"{v:.2f}", ha="center", fontsize=11, color=INK)
            ax.text(x, 0.04, "sparse" if j == 0 else "dense", ha="center", fontsize=9, color="white" if j == 0 else INK,
                    rotation=90, va="bottom")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel("test RMSE, mean of 5 training sites (m)")
    ax.set_ylim(0, 1.9)
    ax.axhline(float(np.mean(a[["dense_large_s0", "dense_large_s1", "dense_large_s2"]])), color=CNN, lw=1, ls=":")
    ax.set_title("Image context helps: CNN on 1 km tracks beats the random forest on dense labels", loc="left")
    fig.text(0.01, 0.01, "Solid = sparse track labels (6,880 cells over 5 sites); faded = dense labels (250,000 cells). "
             "Error bars on CNN: range over 3 training seeds.", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_unseen(M: pd.DataFrame, out: Path):
    """Caveat slide: sites never seen in training."""
    f = M[(M["subset"] == "full") & (M["group"] == "unseen")]
    piv = f.pivot_table(index="site", columns="method", values=["rmse", "bias"])
    sites = list(piv.index)
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.6))
    for ax, v, xl in ((axs[0], "rmse", "RMSE (m)"), (axs[1], "bias", "bias = mean(prediction − lidar) (m)")):
        y = np.arange(len(sites))[::-1]
        ax.grid(axis="x", color=GRID, lw=0.8)
        seeds = piv[v][["sparse_large_s0", "sparse_large_s1", "sparse_large_s2"]]
        ax.scatter(piv[v]["const_sparse"], y, s=110, marker="|", color=CONST, lw=3, label="constant")
        ax.scatter(piv[v]["rf_sparse"], y, s=90, color=RF, edgecolor=SURFACE, lw=2, label="random forest")
        ax.hlines(y, seeds.min(1), seeds.max(1), color=CNN, lw=6, alpha=0.35)
        ax.scatter(seeds.mean(1), y, s=110, color=CNN, edgecolor=SURFACE, lw=2, label="CNN (3 seeds)")
        ax.set_yticks(y, [NAMES[s] for s in sites])
        ax.set_xlabel(xl)
        if v == "bias":
            ax.axvline(0, color=INK2, lw=1)
    for ax in axs:
        ax.set_ylim(-0.6, len(sites) - 0.4)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False, fontsize=10)
    axs[0].set_title("Unseen sites: not yet better than a constant", loc="left")
    axs[1].set_title("…because the whole-site level is off", loc="left")
    fig.text(0.01, 0.01, "Sites never used for training (sparse-label models). Interpolation needs local labels, so it "
             "has no prediction here. Next step: feed sparse heights in as an input to anchor the level.",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    ap.add_argument("--site", default="florida_east_coast")
    ap.add_argument("--run", default="sparse_large_s0")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    out = ccfg.out_dir(cfg) / "slides"
    out.mkdir(parents=True, exist_ok=True)
    S = Site(cfg, a.site, a.run)
    S.name = a.site
    M = pd.read_csv(ccfg.out_dir(cfg) / "eval_test" / "metrics.csv")
    A = pd.read_csv(ccfg.out_dir(cfg) / "eval_test" / "aggregates.csv")
    n_sparse = len(pd.read_parquet(ccfg.out_dir(cfg) / "pools" / "sparse.parquet"))
    fig_concept(S, out / "fig1_concept.png")
    fig_training_data(S, n_sparse, out / "fig2_sparse_training_data.png")
    fig_difference(S, a.run, out / "fig3_difference_map.png")
    fig_per_site(M, out / "fig4_per_site_rmse.png")
    fig_summary(A, out / "fig6_dense_vs_sparse.png")
    fig_unseen(M, out / "fig7_unseen_sites.png")
    print("\n".join(str(p) for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
