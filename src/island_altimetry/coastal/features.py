"""Imagery-only predictors. Reference elevations never enter a feature.

bands:   SR reflectance (scaled to [0, 1], float32), band order as delivered.
context: bands + predefined normalized differences / log ratio + local
         mean/std of a few layers in 3x3 and 5x5 windows. Window statistics
         use only valid image pixels (invalid neighbours are excluded from both
         numerator and count); cells with no valid neighbour get NaN and are
         excluded, never filled with arbitrary values.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

SR_SCALE = 1e-4   # PlanetScope SR digital numbers -> reflectance

NORMALIZED_DIFFS = {          # (a - b) / (a + b)
    "ndwi_g_nir": ("green", "nir"),
    "ndvi": ("nir", "red"),
    "nd_blue_green": ("blue", "green"),
    "nd_coastal_green": ("coastal_blue", "green"),
    "nd_rededge_red": ("rededge", "red"),
}
LOG_RATIOS = {"log_blue_green": ("blue", "green")}   # Stumpf-style ratio for shallow water
WINDOW_LAYERS = ["blue", "green", "nir", "ndwi_g_nir"]
WINDOWS = (3, 5)
MAX_RADIUS_PX = max(WINDOWS) // 2


def _nd(a, b):
    s = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(s) > 1e-4, (a - b) / s, np.nan).astype(np.float32)


def _logratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((a > 1e-4) & (b > 1e-4), np.log(a) - np.log(b), np.nan).astype(np.float32)


def window_stats(a: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware local mean and std over valid pixels of a size x size window."""
    ok = valid & np.isfinite(a)
    v = np.where(ok, a, 0).astype(np.float64)
    n = ndimage.uniform_filter(ok.astype(np.float64), size, mode="constant") * size * size
    s1 = ndimage.uniform_filter(v, size, mode="constant") * size * size
    s2 = ndimage.uniform_filter(v * v, size, mode="constant") * size * size
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0.5, s1 / n, np.nan)
        var = np.where(n > 1.5, s2 / n - mean * mean, np.nan)
    return mean.astype(np.float32), np.sqrt(np.clip(var, 0, None)).astype(np.float32)


def feature_layers(sr: np.ndarray, band_names: list[str], valid: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    """Full-grid feature layers (ordered dict name -> 2-D float32)."""
    refl = {b: np.where(valid, sr[i].astype(np.float32) * SR_SCALE, np.nan) for i, b in enumerate(band_names)}
    out = dict(refl)
    if kind == "bands":
        return out
    if kind != "context":
        raise ValueError(f"unknown feature set {kind}")
    for name, (a, b) in NORMALIZED_DIFFS.items():
        if a in refl and b in refl:
            out[name] = _nd(refl[a], refl[b])
    for name, (a, b) in LOG_RATIOS.items():
        if a in refl and b in refl:
            out[name] = _logratio(refl[a], refl[b])
    for layer in WINDOW_LAYERS:
        if layer not in out:
            continue
        for w in WINDOWS:
            m, s = window_stats(out[layer], valid, w)
            out[f"{layer}_mean{w}"] = m
            out[f"{layer}_std{w}"] = s
    return out


def sample(layers: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.column_stack([layers[k][rows, cols] for k in layers]).astype(np.float32)
