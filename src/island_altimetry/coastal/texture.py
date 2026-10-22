"""Windowed texture, shape and orientation statistics of one image layer.

Every function takes a 2-D float layer and a boolean valid mask, ignores
invalid pixels (no arbitrary fills) and returns NaN where a window holds no
valid support. Windows are square (w x w); Gaussian operators use
sigma = w / 6 truncated at `truncate` sigma, so their support never exceeds
the box window's radius by more than half a pixel.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-6


def box_mean(a: np.ndarray, valid: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware w x w mean of `a` over valid pixels, and the valid count."""
    ok = valid & np.isfinite(a)
    n = ndimage.uniform_filter(ok.astype(np.float64), w, mode="constant") * w * w
    s = ndimage.uniform_filter(np.where(ok, a, 0).astype(np.float64), w, mode="constant") * w * w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0.5, s / n, np.nan), n


def normalized_gaussian(a: np.ndarray, valid: np.ndarray, sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Gaussian smoothing by normalized convolution: values at invalid pixels
    never contribute; NaN where the kernel support holds (almost) no valid pixel."""
    ok = valid & np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(ok, a, 0).astype(np.float64), sigma, truncate=truncate, mode="constant")
    den = ndimage.gaussian_filter(ok.astype(np.float64), sigma, truncate=truncate, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-3, num / den, np.nan)


def _d(a: np.ndarray, axis: int) -> np.ndarray:
    """Central difference (NaN propagates; edges NaN)."""
    out = np.full(a.shape, np.nan)
    sl_c, sl_p, sl_m = [slice(None)] * 2, [slice(None)] * 2, [slice(None)] * 2
    sl_c[axis], sl_p[axis], sl_m[axis] = slice(1, -1), slice(2, None), slice(None, -2)
    out[tuple(sl_c)] = (a[tuple(sl_p)] - a[tuple(sl_m)]) / 2
    return out


def gradient(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d/drow, d/dcol) by central differences."""
    return _d(a, 0), _d(a, 1)


def hessian_eigs(smooth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(LoG, lambda1, lambda2) of a smoothed layer, lambda1 >= lambda2.
    Ridges (bright lines) have lambda2 << 0 and lambda1 ~ 0; valleys the reverse."""
    gr, gc = gradient(smooth)
    hrr, hrc = gradient(gr)
    _, hcc = gradient(gc)
    tr = hrr + hcc
    disc = np.sqrt(((hrr - hcc) / 2) ** 2 + hrc ** 2)
    return tr, tr / 2 + disc, tr / 2 - disc


def structure_tensor(a: np.ndarray, valid: np.ndarray, w: int, inner_sigma: float = 0.5,
                     truncate: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cos 2theta, sin 2theta, coherence) of the dominant GRADIENT orientation
    theta over a w x w window (theta measured from the +column axis toward
    +row; structures such as ridges run perpendicular to it). Coherence is
    (l1 - l2) / (l1 + l2): ~1 for one dominant orientation, ~0 isotropic."""
    s = normalized_gaussian(a, valid, inner_sigma, truncate)
    gr, gc = gradient(s)
    ok = valid & np.isfinite(gr) & np.isfinite(gc)
    jcc, _ = box_mean(gc * gc, ok, w)
    jrr, _ = box_mean(gr * gr, ok, w)
    jrc, _ = box_mean(gr * gc, ok, w)
    r = np.sqrt((jcc - jrr) ** 2 + 4 * jrc ** 2)
    tr = jcc + jrr
    with np.errstate(invalid="ignore", divide="ignore"):
        c2 = np.where(r > EPS, (jcc - jrr) / r, 0.0)
        s2 = np.where(r > EPS, 2 * jrc / r, 0.0)
        coh = np.where(tr > EPS, r / tr, 0.0)
    nan = ~np.isfinite(tr)
    for x in (c2, s2, coh):
        x[nan] = np.nan
    return c2, s2, coh


def quantize(a: np.ndarray, valid: np.ndarray, levels: int) -> np.ndarray:
    """Integer levels 0..levels-1 from the 1-99 % range of valid values; NaN where invalid."""
    v = a[valid & np.isfinite(a)]
    lo, hi = np.percentile(v, [1, 99]) if len(v) else (0.0, 1.0)
    q = np.floor((a - lo) / max(hi - lo, EPS) * levels)
    q = np.clip(q, 0, levels - 1)
    return np.where(valid & np.isfinite(a), q, np.nan)


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc] (NaN outside)."""
    out = np.full(a.shape, np.nan)
    H, W = a.shape
    rs, re = max(0, -dr), min(H, H - dr)
    cs, ce = max(0, -dc), min(W, W - dc)
    out[rs:re, cs:ce] = a[rs + dr:re + dr, cs + dc:ce + dc]
    return out


def glcm_stats(q: np.ndarray, w: int, offsets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GLCM contrast, homogeneity and correlation in a w x w window, averaged
    over `offsets` [(drow, dcol), ...]. Pairs are (q[p], q[p + offset]) for
    every anchor p in the window with both pixels valid; the GLCM is
    symmetric (pooled marginals for correlation). Computed exactly by box
    filtering pair images instead of building per-window matrices."""
    con = np.zeros(q.shape)
    hom = np.zeros(q.shape)
    cor = np.zeros(q.shape)
    cnt = np.zeros(q.shape)
    for dr, dc in offsets:
        j = _shift(q, dr, dc)
        ok = np.isfinite(q) & np.isfinite(j)
        i_, j_ = np.where(ok, q, 0.0), np.where(ok, j, 0.0)
        n = ndimage.uniform_filter(ok.astype(np.float64), w, mode="constant") * w * w

        def m(x):
            return ndimage.uniform_filter(np.where(ok, x, 0.0), w, mode="constant") * w * w

        d2 = (i_ - j_) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            c = m(d2) / n
            h = m(1.0 / (1.0 + d2)) / n
            mu = (m(i_) + m(j_)) / (2 * n)
            var = (m(i_ * i_) + m(j_ * j_)) / (2 * n) - mu * mu
            r = np.where(var > EPS, (m(i_ * j_) / n - mu * mu) / var, 1.0)
        has = n > 0.5
        con += np.where(has, c, 0)
        hom += np.where(has, h, 0)
        cor += np.where(has, r, 0)
        cnt += has
    with np.errstate(invalid="ignore", divide="ignore"):
        out = [np.where(cnt > 0, x / cnt, np.nan) for x in (con, hom, cor)]
    return tuple(out)


def local_entropy(q: np.ndarray, w: int, levels: int) -> np.ndarray:
    """Shannon entropy (bits) of the quantized values in a w x w window."""
    ok = np.isfinite(q)
    n = ndimage.uniform_filter(ok.astype(np.float64), w, mode="constant") * w * w
    ent = np.zeros(q.shape)
    for lv in range(levels):
        c = ndimage.uniform_filter((q == lv).astype(np.float64), w, mode="constant") * w * w
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(n > 0.5, c / n, 0.0)
            ent -= np.where(p > EPS, p * np.log2(p), 0.0)
    return np.where(n > 0.5, ent, np.nan)


def local_range(a: np.ndarray, valid: np.ndarray, w: int) -> np.ndarray:
    ok = valid & np.isfinite(a)
    hi = ndimage.maximum_filter(np.where(ok, a, -np.inf), w, mode="constant", cval=-np.inf)
    lo = ndimage.minimum_filter(np.where(ok, a, np.inf), w, mode="constant", cval=np.inf)
    return np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, np.nan)


def distance_to(mask: np.ndarray, res: float) -> np.ndarray:
    """Euclidean distance (m) from every pixel to the nearest True pixel of `mask`
    (whole-scene operator; inf-free: no True pixel -> NaN)."""
    if not mask.any():
        return np.full(mask.shape, np.nan)
    return ndimage.distance_transform_edt(~mask) * res
