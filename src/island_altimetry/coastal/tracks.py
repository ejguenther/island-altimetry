"""Simulated satellite-lidar-like track layouts: straight parallel centerlines.
A label is a reference cell intersected by a centerline (one per unique cell),
not a synthetic photon or footprint."""

from __future__ import annotations

import numpy as np
from affine import Affine
from shapely.geometry import LineString


def layout_lines(bounds, azimuth_deg: float, count: int, offset_frac: float) -> tuple[list[LineString], float]:
    """`count` parallel lines at azimuth (degrees clockwise from grid north)
    spanning `bounds` (minx, miny, maxx, maxy); spacing = cross-track extent /
    count. Lines sit at (i + 0.5 + offset_frac) * spacing across the extent
    (wrapping), so offset variants translate the whole pattern."""
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    a = np.deg2rad(azimuth_deg)
    u = np.array([np.sin(a), np.cos(a)])     # along-track
    v = np.array([u[1], -u[0]])              # cross-track
    corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]]) - [cx, cy]
    cmin, cmax = (corners @ v).min(), (corners @ v).max()
    half_len = np.abs(corners @ u).max() + 1.0
    spacing = (cmax - cmin) / count
    lines = []
    for i in range(count):
        off = cmin + ((i + 0.5 + offset_frac) % count) * spacing
        p = np.array([cx, cy]) + v * off
        lines.append(LineString([p - u * half_len, p + u * half_len]))
    return lines, float(spacing)


def line_cells(lines: list[LineString], transform: Affine, shape: tuple[int, int], step: float | None = None) -> np.ndarray:
    """Unique flat cell ids (row * width + col) crossed by straight
    centerlines, by dense sampling at <= 1/8 cell (a cell clipped only at a
    corner sliver can be missed; immaterial for a sampling experiment)."""
    step = step or abs(transform.a) / 8
    inv = ~transform
    ids = []
    for ln in lines:
        d = np.arange(0, ln.length + step, step)
        pts = _sample(ln, d)
        col, row = inv * (pts[:, 0], pts[:, 1])
        col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
        ids.append(row[ok] * shape[1] + col[ok])
    return np.unique(np.concatenate(ids)) if ids else np.array([], np.int64)


def _sample(ln: LineString, d: np.ndarray) -> np.ndarray:
    (x0, y0), (x1, y1) = ln.coords[0], ln.coords[-1]
    f = d / ln.length
    return np.column_stack([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f])


def layout_by_spacing(bounds, azimuth_deg: float, spacing_m: float, offset_frac: float) -> list[LineString]:
    """Parallel lines exactly `spacing_m` apart (cross-track) covering `bounds`,
    the first at (0.5 + offset_frac) * spacing from the cross-track minimum."""
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    a = np.deg2rad(azimuth_deg)
    u = np.array([np.sin(a), np.cos(a)])
    v = np.array([u[1], -u[0]])
    corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]]) - [cx, cy]
    cmin, cmax = (corners @ v).min(), (corners @ v).max()
    half_len = np.abs(corners @ u).max() + 1.0
    lines, off = [], cmin + (0.5 + offset_frac) * spacing_m
    while off <= cmax:
        p = np.array([cx, cy]) + v * off
        lines.append(LineString([p - u * half_len, p + u * half_len]))
        off += spacing_m
    return lines
