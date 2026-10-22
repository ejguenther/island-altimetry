"""Reusable reference-building steps on an exact PlanetScope grid: SR/UDM2
mosaics as the grid template, LAZ return support / point means, and QA
helpers. Shared by the coastal RF experiment and the CNN terrain experiment."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from ..imagery import UDM_NODATA, files, mosaic, mosaic_udm2
from . import grid
from .lidar_support import bin_returns

# PlanetScope SR band order: SuperDove 8-band (ortho_analytic_8b_sr) and 4-band (ortho_analytic_4b_sr, BGRN)
BANDS_8B = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
BANDS_4B = ["blue", "green", "red", "nir"]


def build_imagery(src_dir: Path, out_dir: Path, bands: list[str] | None = None) -> tuple[dict, dict]:
    """Mosaic a downloaded strip's SR and UDM2 onto the strip's own grid (the
    reference-grid template) and write planet_sr.tif / planet_udm2.tif.
    `bands` keeps only the named bands, in that order."""
    sr_f, udm_f = files(src_dir, "sr"), files(src_dir, "udm2")
    if not sr_f or len(sr_f) != len(udm_f):
        raise SystemExit(f"expected matching SR/UDM2 files in {src_dir}: {len(sr_f)} SR, {len(udm_f)} UDM2")
    info = []
    for f in sr_f:
        with rasterio.open(f) as s:
            info.append({"file": f.name, "crs": str(s.crs), "res": s.res, "count": s.count, "dtype": s.dtypes[0],
                         "origin_mod_res": [s.transform.c % s.res[0], s.transform.f % s.res[1]]})
    if len({i["crs"] for i in info}) != 1 or len({tuple(i["res"]) for i in info}) != 1:
        raise SystemExit(f"strip items differ in CRS/resolution: {info}")
    if len({tuple(np.round(i["origin_mod_res"], 6)) for i in info}) != 1:
        raise SystemExit(f"strip items are not on one pixel lattice (mosaic would resample): {info}")
    sr, tr, crs = mosaic(sr_f)
    udm, utr, _ = mosaic_udm2(udm_f)
    if sr.shape[1:] != udm.shape[1:] or tr != utr:
        raise SystemExit("SR and UDM2 mosaics are on different grids")
    spec = {"crs": crs.to_wkt(), "epsg": crs.to_epsg(), "transform": list(tr)[:6], "width": sr.shape[2],
            "height": sr.shape[1], "res": [tr.a, -tr.e]}
    names = {8: BANDS_8B, 4: BANDS_4B}.get(sr.shape[0], [f"b{i + 1}" for i in range(sr.shape[0])])
    if bands:
        missing = [b for b in bands if b not in names]
        if missing:
            raise SystemExit(f"bands {missing} not in this product ({names}); not fabricating them")
        sr = sr[[names.index(b) for b in bands]]
        names = list(bands)
    grid.write_raster(out_dir / "planet_sr.tif", dict(zip(names, sr)), spec, nodata=0, dtype="uint16")
    grid.write_raster(out_dir / "planet_udm2.tif", {f"udm2_b{i + 1}": b for i, b in enumerate(udm)}, spec,
                      nodata=UDM_NODATA, dtype="uint8")
    return spec, {"items": info, "bands": names, "n_product_bands": info[0]["count"]}


def _bin_window(args):
    """bin_returns cropped to the tile's non-empty window (small pickles)."""
    b = bin_returns(*args)
    r, c = np.nonzero(b["counts"])
    if not len(r):
        return None, b["class_hist"]
    sl = np.s_[r.min():r.max() + 1, c.min():c.max() + 1]
    return (r.min(), c.min(), b["counts"][sl], b["sum_z"][sl], b["sum_z2"][sl]), b["class_hist"]


def support_grid(laz: list[Path], spec: dict, classes, src_crs, workers: int) -> tuple[dict, dict]:
    """Return count, point-mean z and point std per cell (class 2/29 returns)."""
    shape_ = (spec["height"], spec["width"])
    t = Affine(*spec["transform"])
    counts = np.zeros(shape_, np.uint32)
    sz, sz2 = np.zeros(shape_, np.float64), np.zeros(shape_, np.float64)
    hist = np.zeros(256, np.int64)
    args = [(p, t, shape_, classes, src_crs, CRS.from_wkt(spec["crs"])) for p in laz]
    with ProcessPoolExecutor(workers) as ex:
        for win, h in ex.map(_bin_window, args):
            hist += h
            if win is None:
                continue
            r0, c0, n, a, b = win
            sl = np.s_[r0:r0 + n.shape[0], c0:c0 + n.shape[1]]
            counts[sl] += n
            sz[sl] += a
            sz2[sl] += b
    with np.errstate(invalid="ignore", divide="ignore"):
        zp = np.where(counts > 0, sz / counts, np.nan)
        sp = np.sqrt(np.clip(np.where(counts > 0, sz2 / counts - zp * zp, np.nan), 0, None))
    return ({"support": counts, "z_pts": zp.astype(np.float32), "z_pts_std": sp.astype(np.float32)},
            {int(k): int(v) for k, v in enumerate(hist) if v})


def rgb_stretch(sr: np.ndarray, names: list[str], valid: np.ndarray) -> np.ndarray:
    idx = [names.index(b) for b in ("red", "green", "blue")] if "red" in names else [2, 1, 0]
    out = np.zeros((*sr.shape[1:], 3), np.float32)
    for k, i in enumerate(idx):
        b = sr[i].astype(np.float32)
        lo, hi = np.percentile(b[valid], [2, 98]) if valid.any() else (0, 1)
        out[..., k] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~valid] = 1.0
    return out


def strip_axis(mask: np.ndarray, spec: dict) -> tuple:
    """Centroid, unit principal direction and length of a mask (cell centres)."""
    t = Affine(*spec["transform"])
    r, c = np.nonzero(mask)
    x, y = t * (c + 0.5, r + 0.5)
    xy = np.column_stack([x, y])
    mu = xy.mean(0)
    _, _, vt = np.linalg.svd(xy[::max(1, len(xy) // 200000)] - mu, full_matrices=False)
    u = vt[0] / np.linalg.norm(vt[0])
    proj = (xy - mu) @ u
    return (tuple(mu), tuple(u), float(proj.max() - proj.min()))


def registration_check(z: np.ndarray, nir: np.ndarray, land: np.ndarray, max_shift: int = 5) -> dict:
    """Integer-pixel shift maximising correlation between DTM and NIR gradient
    magnitudes over land (stable relief edges: roads, dunes, ditches). A QA
    diagnostic only: never applied, never tuned on prediction errors."""
    from scipy import ndimage
    def grad(a):
        a = np.where(np.isfinite(a), a, np.nanmedian(a))
        return np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))
    gz, gn = grad(z.astype(np.float64)), grad(nir.astype(np.float64))
    mask = ndimage.binary_erosion(land, iterations=max_shift + 2)
    best, scores = None, {}
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            sh = np.roll(np.roll(gn, dy, 0), dx, 1)
            a, b = gz[mask], sh[mask]
            r = float(np.corrcoef(a, b)[0, 1]) if mask.sum() > 100 else np.nan
            scores[(dy, dx)] = r
            if best is None or (np.isfinite(r) and r > scores[best]):
                best = (dy, dx)
    return {"best_shift_px_row_col": list(best), "best_r": scores[best], "r_at_zero": scores[(0, 0)],
            "n_cells": int(mask.sum()), "note": "diagnostic only; no shift applied"}
