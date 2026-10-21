"""Imagery-derived validity and surface-domain masks. Land/water routing is
decided from the image alone (never from reference elevation, slope or shore
distance); the reference only decides whether a cell has a usable label."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

LAND, WATER, AMBIGUOUS, NONE = 1, 2, 3, 0
DOMAIN_NAMES = {LAND: "land", WATER: "water", AMBIGUOUS: "ambiguous", NONE: "none"}


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI; NaN where both bands are ~0 (no data)."""
    g, n = green.astype(np.float32), nir.astype(np.float32)
    s = g + n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 1e-6, (g - n) / s, np.nan).astype(np.float32)


def image_valid(sr: np.ndarray, udm2: np.ndarray, udm_nodata: int = 255) -> np.ndarray:
    """All SR bands present, UDM2 band 1 'clear', not blackfill. Water is a
    valid study surface (UDM2 clear includes clear water)."""
    present = np.all(sr > 0, axis=0)
    return present & (udm2[0] == 1) & (udm2[7] != udm_nodata)


def otsu(x: np.ndarray, bins: int = 256) -> float:
    h, e = np.histogram(x, bins=bins)
    p, c = h / h.sum(), (e[:-1] + e[1:]) / 2
    w, mu = np.cumsum(p), np.cumsum(p * c)
    return float(c[np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12))])


def nir_thresholds(nir_valid: np.ndarray, tail_frac: float = 0.05, bin_w: float = 0.0025) -> dict:
    """Image-only NIR thresholds: water_max = upper edge of the dark (water)
    histogram mode, where density first falls below tail_frac of the mode
    peak; land_min = Otsu threshold on log NIR (between the modes). Neither
    uses reference elevations."""
    x = nir_valid[np.isfinite(nir_valid) & (nir_valid > 0)]
    land_min = float(np.exp(otsu(np.log(x))))
    h, e = np.histogram(x[x < land_min], bins=np.arange(0, land_min + bin_w, bin_w))
    k = int(np.argmax(h))
    tail = np.nonzero(h[k:] < tail_frac * h[k])[0]
    water_max = float(e[k + tail[0]]) if len(tail) else land_min
    return {"water_nir_max": min(water_max, land_min), "land_nir_min": land_min, "water_mode_nir": float(e[k] + bin_w / 2)}


def surface_domain_nir(nir: np.ndarray, w: np.ndarray, valid: np.ndarray, water_nir_max: float, land_nir_min: float,
                       water_ndwi_min: float, buffer_px: int) -> np.ndarray:
    """LAND / WATER / AMBIGUOUS / NONE: water is dark in NIR (and not strongly
    vegetated: NDWI above water_ndwi_min, which rejects shadowed vegetation),
    land is bright in NIR; intermediate pixels and a buffer_px band around
    land-water contacts are ambiguous. McFeeters NDWI alone fails here: over
    Gulf water the SR NIR (~0.07) is about equal to green, putting NDWI ~0."""
    water = valid & (nir < water_nir_max) & (w > water_ndwi_min)
    land = valid & (nir > land_nir_min)
    return _with_buffer(water, land, valid, buffer_px)


def _with_buffer(water, land, valid, buffer_px):
    amb = valid & ~water & ~land
    if buffer_px > 0:
        st = ndimage.generate_binary_structure(2, 2)
        contact = (ndimage.binary_dilation(water, st) & land) | (ndimage.binary_dilation(land, st) & water)
        amb |= valid & ndimage.binary_dilation(contact | amb, st, iterations=buffer_px)
    out = np.full(water.shape, NONE, np.uint8)
    out[land] = LAND
    out[water] = WATER
    out[amb] = AMBIGUOUS
    return out


def surface_domain(w: np.ndarray, valid: np.ndarray, water_min: float, land_max: float, buffer_px: int) -> np.ndarray:
    """LAND / WATER / AMBIGUOUS / NONE from NDWI thresholds. Values between the
    thresholds, and a `buffer_px` band around every land-water contact, are
    ambiguous (surf, wet sand, very shallow water)."""
    return _with_buffer(valid & (w > water_min), valid & (w < land_max), valid, buffer_px)
