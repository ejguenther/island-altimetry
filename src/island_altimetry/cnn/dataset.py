"""Center-label patch access, fixed channel scalers and AOI-balanced sampling.

Site images are small (about 90 MB per 100 km2 as float16 reflectance), so
each is held once as a GPU tensor and R x R patches are gathered by index for a
whole batch at once: no patch arrays are ever materialized on disk or in host
memory. Each example is exactly one image context and one center elevation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR_SCALE = 1e-4


def load_image(path: Path, bands: list[str]) -> np.ndarray:
    """SR reflectance (C, H, W) float32 in the configured band order."""
    import rasterio
    with rasterio.open(path) as s:
        names = list(s.descriptions)
        return np.stack([s.read(names.index(b) + 1).astype(np.float32) * SR_SCALE for b in bands])


def fit_scalers(images: dict[str, np.ndarray], centers: dict[str, np.ndarray], radius: int, max_per_site: int,
                rng: np.random.Generator) -> dict:
    """Per-channel mean/std over pixels of training-center contexts only
    (a random subset of each site's training centers; each context's pixels
    are pooled). The caller passes fitting sites' TRAIN centers only."""
    vals = []
    for s, rc in centers.items():
        img = images[s]
        pick = rc[rng.choice(len(rc), size=min(max_per_site, len(rc)), replace=False)]
        offs = np.arange(-radius, radius + 1, 4)   # strided sub-grid of each context
        grid_r, grid_c = np.meshgrid(offs, offs, indexing="ij")
        rr = (pick[:, 0, None, None] + grid_r[None]).ravel()
        cc = (pick[:, 1, None, None] + grid_c[None]).ravel()
        vals.append(img[:, rr, cc])
    v = np.concatenate(vals, axis=1).astype(np.float64)
    return {"mean": v.mean(1).tolist(), "std": v.std(1).tolist(), "n_pixels": int(v.shape[1])}


class GpuPatches:
    """Normalized site images on one device; `gather(site_idx, rows, cols)`
    returns (N, C, R, R) contexts centred on the given cells."""

    def __init__(self, images: dict[str, np.ndarray], scalers: dict, radius: int, device, dtype=None):
        import torch
        self.sites = list(images)
        self.radius = radius
        mean = torch.tensor(scalers["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(scalers["std"], dtype=torch.float32).view(-1, 1, 1)
        self.imgs = [((torch.from_numpy(images[s]) - mean) / std).to(device=device, dtype=dtype or torch.float32)
                     for s in self.sites]
        self.offs = torch.arange(-radius, radius + 1, device=device)

    def gather(self, site_idx, rows, cols):
        import torch
        out = torch.empty((len(rows), self.imgs[0].shape[0], 2 * self.radius + 1, 2 * self.radius + 1),
                          device=self.offs.device, dtype=self.imgs[0].dtype)
        for k in torch.unique(site_idx).tolist():
            sel = torch.nonzero(site_idx == k).squeeze(1)
            rr = rows[sel, None, None] + self.offs[None, :, None]
            cc = cols[sel, None, None] + self.offs[None, None, :]
            out[sel] = self.imgs[k][:, rr, cc].permute(1, 0, 2, 3)
        return out


def augment(x, k: int, flip: bool):
    """Rotation by k*90 degrees and optional horizontal flip of a batch of
    contexts (center targets unchanged). Alters directional illumination cues."""
    import torch
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


def balanced_indices(site_of: np.ndarray, n_sites: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n example indices, equal share per site, uniform with replacement
    within each site's pool (revisits are expected for small sparse pools)."""
    per = n // n_sites
    out = []
    for k in range(n_sites):
        pool = np.nonzero(site_of == k)[0]
        if len(pool):
            out.append(rng.choice(pool, size=per, replace=True))
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx


def save_json(path: Path, obj) -> None:
    from ..manifest import write_json_atomic
    write_json_atomic(path, obj)


def load_json(path: Path):
    return json.loads(Path(path).read_text())
