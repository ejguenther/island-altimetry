"""Lidar support: count classified returns per destination cell straight from
the source COPC/LAZ tiles. The gridded DTM is 100% valid even where the survey
recorded no ground/bathymetry return (interpolated fill), so return counts are
the only record of where the reference is measured rather than filled."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer

GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
ADJUSTED_GPS_OFFSET_S = 1e9   # LAS 1.4 "adjusted standard GPS time" (global encoding bit 0)


def gps_to_utc(t: float, adjusted: bool = True) -> dt.datetime:
    """GPS seconds -> UTC (leap seconds ignored: 18 s in 2022, irrelevant here)."""
    return GPS_EPOCH + dt.timedelta(seconds=float(t) + (ADJUSTED_GPS_OFFSET_S if adjusted else 0))


def header_info(path: Path) -> dict:
    import laspy
    with laspy.open(path) as r:
        h = r.header
        crs = h.parse_crs()
    subs = [s.name for s in crs.sub_crs_list] if crs is not None and crs.is_compound else []
    vert = crs.sub_crs_list[1] if crs is not None and crs.is_compound else None
    return {"las_version": str(h.version), "point_format": h.point_format.id, "point_count": int(h.point_count),
            "crs_name": crs.name if crs is not None else None, "sub_crs": subs,
            "vertical_epsg": vert.to_epsg() if vert is not None else None,
            "horizontal_epsg": crs.sub_crs_list[0].to_epsg() if subs else (crs.to_epsg() if crs else None),
            "adjusted_gps": bool(h.global_encoding.gps_time_type),
            "bounds": [float(v) for v in (*h.mins[:2], *h.maxs[:2])], "z_range": [float(h.mins[2]), float(h.maxs[2])]}


def bin_returns(path: Path, transform: Affine, shape: tuple[int, int], classes, src_crs=None, dst_crs=None,
                chunk: int = 5_000_000) -> dict:
    """Count returns of `classes` per cell of a north-up grid (transform, shape),
    optionally reprojecting points src_crs -> dst_crs. Also returns the class
    histogram (all classes), GPS time range of the tile, and per-cell sums of
    z and z^2 over the counted returns (point-mean reference for QA)."""
    import laspy
    if transform.b != 0 or transform.d != 0 or transform.e >= 0:
        raise ValueError("bin_returns needs a north-up, unrotated grid")
    tr = None
    if dst_crs is not None and not CRS.from_user_input(src_crs).equals(CRS.from_user_input(dst_crs)):
        tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    counts = np.zeros(shape, np.uint32)
    sz, sz2 = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    hist = np.zeros(256, np.int64)
    t0, t1 = np.inf, -np.inf
    inv = ~transform
    with laspy.open(path) as r:
        for pts in r.chunk_iterator(chunk):
            c = np.asarray(pts.classification)
            hist += np.bincount(c, minlength=256)
            if len(c):
                g = np.asarray(pts.gps_time)
                t0, t1 = min(t0, g.min()), max(t1, g.max())
            m = np.isin(c, classes)
            if not m.any():
                continue
            x, y, z = np.asarray(pts.x)[m], np.asarray(pts.y)[m], np.asarray(pts.z)[m]
            if tr is not None:
                x, y = tr.transform(x, y)
            col, row = inv * (x, y)
            col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
            ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
            row, col, z = row[ok], col[ok], z[ok]
            if not len(row):
                continue
            # bincount over the chunk's bounding window (a tile is small relative to the grid)
            r0, c0 = row.min(), col.min()
            h, w = row.max() - r0 + 1, col.max() - c0 + 1
            flat = (row - r0) * w + (col - c0)
            counts[r0:r0 + h, c0:c0 + w] += np.bincount(flat, minlength=h * w).reshape(h, w).astype(np.uint32)
            sz[r0:r0 + h, c0:c0 + w] += np.bincount(flat, z, minlength=h * w).reshape(h, w)
            sz2[r0:r0 + h, c0:c0 + w] += np.bincount(flat, z * z, minlength=h * w).reshape(h, w)
    return {"counts": counts, "sum_z": sz, "sum_z2": sz2, "class_hist": hist, "gps_range": (t0, t1)}
