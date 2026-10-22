"""Sparse-height interpolation baseline: inverse-distance weighting over the k
nearest training labels within a radius (KD-tree, chunked queries; no dense
pairwise distance matrix). Unsupported cells (no label within the radius)
are returned as NaN, never filled."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def idw(train_xy: np.ndarray, train_z: np.ndarray, query_xy: np.ndarray, k: int, power: float,
        max_radius: float, chunk: int = 500_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(prediction, nearest-label distance, neighbours used) per query point."""
    n = len(query_xy)
    pred = np.full(n, np.nan, np.float32)
    dmin = np.full(n, np.inf, np.float32)
    used = np.zeros(n, np.int16)
    if len(train_z) == 0:
        return pred, dmin, used
    tree = cKDTree(train_xy)
    k = min(k, len(train_z))
    for i in range(0, n, chunk):
        q = query_xy[i:i + chunk]
        d, j = tree.query(q, k=k, distance_upper_bound=max_radius)
        d, j = d.reshape(len(q), k), j.reshape(len(q), k)
        ok = np.isfinite(d)
        dmin[i:i + chunk] = np.where(ok[:, 0], d[:, 0], np.inf)
        used[i:i + chunk] = ok.sum(1)
        zz = np.where(ok, train_z[np.where(ok, j, 0)], 0.0)
        exact = ok & (d < 1e-9)
        with np.errstate(divide="ignore"):
            w = np.where(ok, 1.0 / np.maximum(d, 1e-9) ** power, 0.0)
        w = np.where(exact.any(1, keepdims=True), exact.astype(float), w)
        ws = w.sum(1)
        with np.errstate(invalid="ignore"):
            pred[i:i + chunk] = np.where(ws > 0, (w * zz).sum(1) / ws, np.nan)
    return pred, dmin, used


def nearest_distance(train_xy: np.ndarray, query_xy: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    if len(train_xy) == 0:
        return np.full(len(query_xy), np.inf, np.float32)
    tree = cKDTree(train_xy)
    out = np.empty(len(query_xy), np.float32)
    for i in range(0, len(query_xy), chunk):
        out[i:i + chunk] = tree.query(query_xy[i:i + chunk], k=1)[0]
    return out
