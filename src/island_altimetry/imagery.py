"""Raster helpers for downloaded PlanetScope strips: file discovery, mosaics
(analytic and UDM2), display stretch, and site-AEQD -> raster coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.merge import merge

from .aoi import aeqd_crs

RGB_BANDS = {8: (6, 4, 2), 4: (3, 2, 1), 3: (1, 2, 3)}   # 1-based: 8-band, 4-band (BGRN), 3-band (RGB)
UDM_NODATA = 255


def files(dest: Path, kind: str) -> list[Path]:
    tifs = sorted(dest.rglob("*.tif"))
    if kind == "udm2":
        return [f for f in tifs if "udm2" in f.name.lower()]
    return [f for f in tifs if "analytic" in f.name.lower() and "udm" not in f.name.lower()]


def mosaic(paths: list[Path], bands=None):
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, tr = merge(srcs, indexes=bands, nodata=0)
        return arr, tr, srcs[0].crs
    finally:
        for s in srcs:
            s.close()


def mosaic_udm2(paths: list[Path]):
    """Mosaic UDM2 strips. 0 is a valid UDM2 value ("not clear"), so blackfill
    pixels (band 8 bit 0) are recoded to 255 and merged with nodata=255; a
    nodata=0 merge would let one scene's zeros be overwritten by another's."""
    from rasterio.io import MemoryFile
    mems, srcs = [], []
    try:
        for f in paths:
            with rasterio.open(f) as s:
                a, prof = s.read(), s.profile
            a = a.astype("uint8")
            a[:, (a[7] & 1) == 1] = UDM_NODATA
            prof.update(dtype="uint8", nodata=UDM_NODATA)
            m = MemoryFile()
            with m.open(**prof) as d:
                d.write(a)
            mems.append(m)
            srcs.append(m.open())
        arr, tr = merge(srcs, nodata=UDM_NODATA)
        return arr, tr, srcs[0].crs
    finally:
        for s in srcs:
            s.close()
        for m in mems:
            m.close()


def stretch(rgb: np.ndarray) -> np.ndarray:
    """Per-band 2-98th percentile stretch of non-zero pixels to [0, 1] (H, W, 3)."""
    out = np.zeros(rgb.shape[1:] + (3,), dtype=float)
    for i in range(3):
        b = rgb[i].astype(float)
        v = b[b > 0]
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1e-9), 0, 1)
    return out


def rgb_for_strip(sr_paths: list[Path]):
    """Stretched RGB mosaic of a strip: (img HxWx3, extent for imshow, crs)."""
    with rasterio.open(sr_paths[0]) as s:
        nb = s.count
    arr, tr, crs = mosaic(sr_paths, bands=list(RGB_BANDS.get(nb, (3, 2, 1))))
    img = stretch(arr)
    h, w = img.shape[:2]
    return img, (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f), crs


def to_raster_xy(crs, lonlat_center: tuple[float, float]):
    """Function (x_local_m, y_local_m) -> raster (x, y) for site-AEQD coordinates."""
    tr = Transformer.from_crs(aeqd_crs(*lonlat_center), crs, always_xy=True)

    def f(x, y):
        return tr.transform(np.asarray(x, float), np.asarray(y, float))
    return f
