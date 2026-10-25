"""Fully convolutional raster inference with halo tiles.

For a model with receptive field R (offset o = (R - 1) / 2), an output tile
covering center rows [r0, r1) and cols [c0, c1) needs input rows
[r0 - o, r1 + o) and cols [c0 - o, c1 + o). Only these valid cores are
written; overlapping halos supply context and are never averaged. Centers
closer than o to the raster edge cannot be predicted (NaN). No per-pixel
loop and no patch materialization.
"""

from __future__ import annotations

import numpy as np


def predict_full(model, img, device, amp_dtype=None) -> np.ndarray:
    """One forward pass over a whole (C, H, W) normalized array -> (H, W) with
    NaN on the o-pixel border."""
    import torch
    o = model.offset
    C, H, W = img.shape
    out = np.full((H, W), np.nan, np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        x = torch.as_tensor(img, device=device, dtype=torch.float32)[None]
        y = model(x)[0].float().cpu().numpy()
    out[o:H - o, o:W - o] = y
    return out


def predict_tiled(model, img, device, tile: int = 512, amp_dtype=None) -> np.ndarray:
    """Halo-tiled equivalent of predict_full for arrays that do not fit."""
    import torch
    o = model.offset
    C, H, W = img.shape
    out = np.full((H, W), np.nan, np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        for r0 in range(o, H - o, tile):
            r1 = min(r0 + tile, H - o)
            for c0 in range(o, W - o, tile):
                c1 = min(c0 + tile, W - o)
                x = torch.as_tensor(np.ascontiguousarray(img[:, r0 - o:r1 + o, c0 - o:c1 + o]), device=device,
                                    dtype=torch.float32)[None]
                y = model(x)[0].float().cpu().numpy()
                assert y.shape == (r1 - r0, c1 - c0)
                out[r0:r1, c0:c1] = y
    return out


def predict_patches(model, img, rows, cols, device, batch: int = 2048, amp_dtype=None) -> np.ndarray:
    """Slow reference: independent R x R patches for given centers (tests only)."""
    import torch
    o = model.offset
    res = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        for i in range(0, len(rows), batch):
            p = np.stack([img[:, r - o:r + o + 1, c - o:c + o + 1] for r, c in zip(rows[i:i + batch], cols[i:i + batch])])
            res.append(model(torch.as_tensor(p, device=device, dtype=torch.float32))[:, 0, 0].float().cpu().numpy())
    return np.concatenate(res) if res else np.array([], np.float32)
