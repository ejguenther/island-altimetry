"""Imagery-only predictors. Reference elevations never enter a feature.

bands:   SR reflectance (scaled to [0, 1], float32), band order as delivered.
context: bands + predefined normalized differences / log ratio + local
         mean/std of a few layers in 3x3 and 5x5 windows. Window statistics
         use only valid image pixels (invalid neighbours are excluded from both
         numerator and count); cells with no valid neighbour get NaN and are
         excluded, never filled with arbitrary values.
terrain: context + texture / shape / orientation of layers nir, bright (mean
         of all SR bands) and ndvi at 3x3, 9x9 and 15x15 (see texture.py and
         TERRAIN_DEFAULTS), plus imagery distance to water and to land.
         texture:     std, range, mean gradient magnitude, entropy (16 levels),
                      GLCM contrast / homogeneity / correlation (1 px, 4 directions)
         shape:       Laplacian of Gaussian, Hessian l1 >= l2 (sigma = w/6),
                      position index (pixel - window mean)
         orientation: structure-tensor gradient orientation (cos 2t, sin 2t),
                      coherence
         distance:    whole-scene distance transforms of the imagery domain
                      mask - a documented exception to the window-guard rule
                      (imagery only, no labels).
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
TERRAIN_DEFAULTS = {"version": 1, "layers": ["nir", "bright", "ndvi"], "windows": [3, 9, 15], "glcm_levels": 16,
                    "glcm_offsets": [[0, 1], [1, 1], [1, 0], [1, -1]], "gaussian_truncate": 3.0,
                    "distance_features": True}
TEXTURE_STATS = ["std", "range", "gradmag", "entropy", "glcm_contrast", "glcm_homogeneity", "glcm_correlation"]
SHAPE_STATS = ["log", "hess_l1", "hess_l2", "pos_index"]
ORIENT_STATS = ["orient_cos2", "orient_sin2", "coherence"]
# largest windowed support (px): 15x15 box radius 7; Gaussian sigma 15/6 truncated at 3 sigma = 7.5
MAX_RADIUS_PX = 7.5


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


def terrain_names(tc: dict) -> list[tuple[str, str]]:
    """(feature name, family) for the terrain additions, in computation order."""
    out = []
    for layer in tc["layers"]:
        for w in tc["windows"]:
            out += [(f"{layer}_{st}{w}", "texture") for st in TEXTURE_STATS]
            out += [(f"{layer}_{st}{w}", "shape") for st in SHAPE_STATS]
            out += [(f"{layer}_{st}{w}", "orientation") for st in ORIENT_STATS]
    if tc["distance_features"]:
        out += [("dist_to_water", "distance"), ("dist_to_land", "distance")]
    return out


def max_support_px(tc: dict) -> float:
    w = max(tc["windows"])
    return max(w // 2, tc["gaussian_truncate"] * w / 6) + 1   # +1: central differences / GLCM offset


def terrain_layers(sr: np.ndarray, band_names: list[str], valid: np.ndarray, domain: np.ndarray, res: float,
                   tc: dict):
    """Yield (name, family, 2-D float32) one layer at a time: the context set
    first (family 'context'), then terrain_names(tc) in order."""
    from . import masks, texture
    for name, a in feature_layers(sr, band_names, valid, "context").items():
        yield name, "context", a
    refl = {b: np.where(valid, sr[i].astype(np.float32) * SR_SCALE, np.nan) for i, b in enumerate(band_names)}
    base = {"nir": refl["nir"], "bright": np.nanmean(np.stack(list(refl.values())), axis=0).astype(np.float32),
            "ndvi": _nd(refl["nir"], refl["red"])}
    lv, trunc = tc["glcm_levels"], tc["gaussian_truncate"]
    for layer in tc["layers"]:
        a = base[layer].astype(np.float64)
        ok = valid & np.isfinite(a)
        gr, gc = texture.gradient(np.where(ok, a, np.nan))
        gmag = np.hypot(gr, gc)
        q = texture.quantize(a, ok, lv)
        for w in tc["windows"]:
            mean, _ = texture.box_mean(a, ok, w)
            m2, _ = texture.box_mean(a * a, ok, w)
            con, hom, cor = texture.glcm_stats(q, w, tc["glcm_offsets"])
            tex = {"std": np.sqrt(np.clip(m2 - mean * mean, 0, None)), "range": texture.local_range(a, ok, w),
                   "gradmag": texture.box_mean(gmag, np.isfinite(gmag), w)[0],
                   "entropy": texture.local_entropy(q, w, lv), "glcm_contrast": con, "glcm_homogeneity": hom,
                   "glcm_correlation": cor}
            for st in TEXTURE_STATS:
                yield f"{layer}_{st}{w}", "texture", _f32(tex[st], ok)
            sm = texture.normalized_gaussian(a, ok, w / 6, trunc)
            log, l1, l2 = texture.hessian_eigs(sm)
            shp = {"log": log, "hess_l1": l1, "hess_l2": l2, "pos_index": a - mean}
            for st in SHAPE_STATS:
                yield f"{layer}_{st}{w}", "shape", _f32(shp[st], ok)
            c2, s2, coh = texture.structure_tensor(a, ok, w, truncate=trunc)
            for st, x in zip(ORIENT_STATS, (c2, s2, coh)):
                yield f"{layer}_{st}{w}", "orientation", _f32(x, ok)
    if tc["distance_features"]:
        yield "dist_to_water", "distance", _f32(texture.distance_to(domain == masks.WATER, res), valid)
        yield "dist_to_land", "distance", _f32(texture.distance_to(domain == masks.LAND, res), valid)


def _f32(a: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Features are defined at valid image pixels only."""
    return np.where(ok, a, np.nan).astype(np.float32)


def sample(layers: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.column_stack([layers[k][rows, cols] for k in layers]).astype(np.float32)
