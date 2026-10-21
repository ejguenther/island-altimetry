"""Reference grid handling: GeoJSON I/O in a projected CRS, the grid
specification taken from the Planet template, exact-grid checks, and windowed
area-weighted aggregation of the 1 m DTM onto that grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import mapping, shape

from ..manifest import write_json_atomic


def reproject_geom(geom, src_crs, dst_crs):
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely.transform(geom, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


def write_geojson(geom, crs, path: Path, props: dict | None = None) -> None:
    """Write one geometry as a WGS84 FeatureCollection (repo AOI convention)."""
    g = reproject_geom(geom, crs, "EPSG:4326")
    write_json_atomic(path, {"type": "FeatureCollection",
                             "features": [{"type": "Feature", "properties": props or {}, "geometry": mapping(g)}]})


def read_geojson(path: Path, crs=None):
    """First feature's geometry, in WGS84 or reprojected to `crs`."""
    g = shape(json.loads(Path(path).read_text())["features"][0]["geometry"])
    return g if crs is None else reproject_geom(g, "EPSG:4326", crs)


# ---------------------------------------------------------------- grid spec

def grid_spec(path: Path) -> dict:
    """CRS, transform and shape of a template raster (the Planet SR mosaic)."""
    import rasterio
    with rasterio.open(path) as s:
        return {"crs": s.crs.to_wkt(), "epsg": s.crs.to_epsg(), "transform": list(s.transform)[:6],
                "width": s.width, "height": s.height, "res": list(s.res)}


def assert_same_grid(paths: list[Path], spec: dict, tol: float = 1e-9) -> None:
    """Exact grid equality (CRS, transform, dimensions) for every raster."""
    import rasterio
    from rasterio.crs import CRS
    want = CRS.from_wkt(spec["crs"])
    for p in paths:
        with rasterio.open(p) as s:
            bad = []
            if s.crs != want:
                bad.append(f"crs {s.crs}")
            if (s.width, s.height) != (spec["width"], spec["height"]):
                bad.append(f"shape {(s.width, s.height)}")
            if any(abs(a - b) > tol for a, b in zip(list(s.transform)[:6], spec["transform"])):
                bad.append(f"transform {tuple(s.transform)[:6]}")
            if bad:
                raise AssertionError(f"{p.name} is not on the reference grid: {'; '.join(bad)}")


def write_raster(path: Path, arrays: dict[str, np.ndarray], spec: dict, nodata=None, dtype=None) -> None:
    """Tiled, compressed GeoTIFF on the reference grid, one named band per array;
    written to a temporary file and renamed so partial outputs never appear."""
    import os

    import rasterio
    from affine import Affine
    from rasterio.crs import CRS
    first = next(iter(arrays.values()))
    dtype = dtype or first.dtype
    tmp = path.with_suffix(".tmp.tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp, "w", driver="GTiff", width=spec["width"], height=spec["height"], count=len(arrays),
                       dtype=dtype, crs=CRS.from_wkt(spec["crs"]), transform=Affine(*spec["transform"]),
                       nodata=nodata, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                       predictor=3 if np.dtype(dtype).kind == "f" else 2) as d:
        for i, (name, a) in enumerate(arrays.items(), 1):
            d.write(a.astype(dtype), i)
            d.set_band_description(i, name)
    os.replace(tmp, path)


def read_band(path: Path, name: str | None = None, masked: bool = False) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as s:
        i = 1 if name is None else list(s.descriptions).index(name) + 1
        return s.read(i, masked=masked)


# ---------------------------------------------------------------- aggregation

def owned_box(bounds, km: float = 1000.0):
    """The 1 km tile square a (possibly partial) DTM tile belongs to: tiles are
    cut on a 1 km grid but stored 1001 px wide, so neighbours overlap by one
    pixel; each source pixel is counted only by the tile whose square holds its
    centre."""
    cx, cy = (bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2
    x0, y0 = np.floor(cx / km) * km, np.floor(cy / km) * km
    return x0, y0, x0 + km, y0 + km


def tile_sums(path: Path, nodata: float, dst_transform, dst_shape, dst_crs, own: bool = True) -> tuple:
    """Area-weighted sums of one 1 m tile on a sub-window of the destination
    grid: (window row0, col0, sum_w, sum_wz, sum_wz2) where w is the valid
    source area in source-pixel units. GDAL 'sum' resampling weights every
    source pixel by the fraction of it inside each destination cell, and
    source area outside the tile (or NoData) contributes zero, so dividing by
    the full cell area gives a gap-aware valid fraction."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(path) as s:
        z = s.read(1).astype(np.float64)
        st, scrs, b = s.transform, s.crs, s.bounds
    valid = (z != nodata) & np.isfinite(z)
    if own:
        x0, y0, x1, y1 = owned_box(b)
        cols = st.c + st.a * (np.arange(z.shape[1]) + 0.5)
        rows = st.f + st.e * (np.arange(z.shape[0]) + 0.5)
        valid &= ((rows >= y0) & (rows < y1))[:, None] & ((cols >= x0) & (cols < x1))[None, :]
    w = valid.astype(np.float64)
    wz = np.where(valid, z, 0.0)
    wz2 = wz * wz
    l, bt, r, t = transform_bounds(scrs, dst_crs, *b, densify_pts=21)
    win = from_bounds(l, bt, r, t, Affine(*dst_transform) if not isinstance(dst_transform, Affine) else dst_transform)
    r0 = max(int(np.floor(win.row_off)) - 1, 0)
    c0 = max(int(np.floor(win.col_off)) - 1, 0)
    r1 = min(int(np.ceil(win.row_off + win.height)) + 1, dst_shape[0])
    c1 = min(int(np.ceil(win.col_off + win.width)) + 1, dst_shape[1])
    if r1 <= r0 or c1 <= c0:
        return r0, c0, None, None, None
    dt = Affine(*dst_transform) if not isinstance(dst_transform, Affine) else dst_transform
    sub_t = dt * Affine.translation(c0, r0)
    out = []
    for a in (w, wz, wz2):
        d = np.zeros((r1 - r0, c1 - c0), np.float64)
        reproject(a, d, src_transform=st, src_crs=scrs, dst_transform=sub_t, dst_crs=dst_crs,
                  resampling=Resampling.sum, src_nodata=None, dst_nodata=None, init_dest_nodata=False)
        out.append(d)
    return (r0, c0, *out)


def aggregate_tiles(paths: list[Path], nodata: float, spec: dict, workers: int = 4) -> dict[str, np.ndarray]:
    """Mean elevation, valid-area fraction (of the full destination cell) and
    within-cell standard deviation on the reference grid, tile by tile (no 1 m
    mosaic in memory)."""
    from concurrent.futures import ProcessPoolExecutor

    from affine import Affine
    from rasterio.crs import CRS
    shape_ = (spec["height"], spec["width"])
    dst_t, dst_crs = Affine(*spec["transform"]), CRS.from_wkt(spec["crs"])
    sw, swz, swz2 = (np.zeros(shape_, np.float64) for _ in range(3))
    src_px_area = None
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(tile_sums, p, nodata, dst_t, shape_, dst_crs) for p in paths]
        for f in futs:
            r0, c0, w, wz, wz2 = f.result()
            if w is None:
                continue
            sl = np.s_[r0:r0 + w.shape[0], c0:c0 + w.shape[1]]
            sw[sl] += w
            swz[sl] += wz
            swz2[sl] += wz2
    import rasterio
    with rasterio.open(paths[0]) as s:
        src_px_area = abs(s.transform.a * s.transform.e)
    cell_area = abs(dst_t.a * dst_t.e)
    frac = sw * src_px_area / cell_area
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(sw > 0, swz / sw, np.nan)
        var = np.where(sw > 0, swz2 / sw - mean * mean, np.nan)
    return {"z_mean": mean.astype(np.float32), "valid_frac": frac.astype(np.float32),
            "z_std": np.sqrt(np.clip(var, 0, None)).astype(np.float32)}
