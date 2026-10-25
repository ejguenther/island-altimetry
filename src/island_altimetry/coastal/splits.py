"""Spatial train/validation/test split: coherent along-strip blocks with
excluded guard buffers at every boundary that touches the training region.
Defined from cell geometry and the imagery domain only, before any label is
extracted or any model is fitted."""

from __future__ import annotations

import numpy as np
import pandas as pd

PARTITIONS = {"train": 1, "val": 2, "test": 3}
EXCLUDED = 0


def along_axis(x: np.ndarray, y: np.ndarray, centroid, direction) -> np.ndarray:
    return (x - centroid[0]) * direction[0] + (y - centroid[1]) * direction[1]


def block_edges(s: np.ndarray, n_blocks: int) -> np.ndarray:
    """Equal-count block edges along the strip axis (usable cells)."""
    return np.quantile(s, np.linspace(0, 1, n_blocks + 1))


def assign_blocks(n_blocks: int, fractions: dict, rng: np.random.Generator) -> np.ndarray:
    """Partition code per block: whole blocks, counts from the fractions."""
    n_val = max(1, int(round(fractions["val"] * n_blocks)))
    n_test = max(1, int(round(fractions["test"] * n_blocks)))
    codes = np.array([PARTITIONS["train"]] * (n_blocks - n_val - n_test) + [PARTITIONS["val"]] * n_val +
                     [PARTITIONS["test"]] * n_test)
    return rng.permutation(codes)


def partition_cells(s: np.ndarray, edges: np.ndarray, block_codes: np.ndarray, buffer_m: float) -> tuple:
    """(block index, partition code) per cell. A cell within buffer_m / 2 of a
    block boundary where the partitions differ and one side is train is
    EXCLUDED, so the closest train and held-out cells are >= buffer_m apart
    along the strip axis (blocks span the full strip width)."""
    blk = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(block_codes) - 1)
    part = block_codes[blk].copy()
    half = buffer_m / 2
    for i in range(1, len(block_codes)):
        a, b = block_codes[i - 1], block_codes[i]
        if a != b and PARTITIONS["train"] in (a, b):
            part[np.abs(s - edges[i]) < half] = EXCLUDED
    return blk, part


def coverage_ok(cells: pd.DataFrame, part: np.ndarray, min_cells: int, domains=(1, 2)) -> tuple[bool, dict]:
    """Descriptive split-support check: every partition holds >= min_cells of
    each domain, and val/test elevation ranges are mostly inside train's. Uses
    reference elevations only as descriptive support, never for tuning."""
    out, ok = {}, True
    for d in domains:
        sel = cells["domain"].to_numpy() == d
        z = cells["z"].to_numpy()
        tr = z[sel & (part == PARTITIONS["train"])]
        for name, code in PARTITIONS.items():
            zz = z[sel & (part == code)]
            rec = {"n": int(len(zz))}
            if len(zz):
                rec.update(z_p05=float(np.percentile(zz, 5)), z_p95=float(np.percentile(zz, 95)))
                if name != "train" and len(tr):
                    rec["frac_outside_train_range"] = float(((zz < tr.min()) | (zz > tr.max())).mean())
            ok &= len(zz) >= min_cells
            out[f"domain{d}_{name}"] = rec
    return ok, out


def make_split(cells: pd.DataFrame, axis: dict, n_blocks: int, fractions: dict, buffer_m: float, seed: int,
               min_cells: int = 500, max_tries: int = 50, domains=(1, 2)) -> dict:
    """Deterministic: seeds seed, seed+1, ... until the coverage check passes;
    every attempt is recorded."""
    s = along_axis(cells["x"].to_numpy(), cells["y"].to_numpy(), axis["centroid"], axis["direction"])
    edges = block_edges(s, n_blocks)
    tries = []
    for k in range(max_tries):
        rng = np.random.default_rng(seed + k)
        codes = assign_blocks(n_blocks, fractions, rng)
        blk, part = partition_cells(s, edges, codes, buffer_m)
        ok, cov = coverage_ok(cells, part, min_cells, domains)
        tries.append({"seed": seed + k, "block_codes": codes.tolist(), "coverage_ok": bool(ok)})
        if ok:
            return {"s": s, "edges": edges, "block_codes": codes, "block": blk, "partition": part,
                    "seed_used": seed + k, "attempts": tries, "coverage": cov}
    raise RuntimeError(f"no split passed the coverage check in {max_tries} seeds: {tries[-1]}")
