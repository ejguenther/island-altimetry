"""Shared access to the frozen experiment inputs: usable cells joined with
the split, label sets by regime, and the per-domain routing used by every
method (land model for land cells, water model for water cells, and both for
ambiguous cells, which are reported separately)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import masks, splits

DOMAINS = (masks.LAND, masks.WATER)
TRAIN, VAL, TEST = (splits.PARTITIONS[k] for k in ("train", "val", "test"))


def load_cells(work: Path, columns: list[str] | None = None) -> pd.DataFrame:
    cols = ["cell_id", "row", "col", "x", "y", "z", "domain"] + [c for c in (columns or []) if c not in
                                                                  ("cell_id", "row", "col", "x", "y", "z", "domain")]
    c = pd.read_parquet(work / "reference" / "cells.parquet", columns=cols)
    s = pd.read_parquet(work / "splits" / "split.parquet", columns=["cell_id", "partition", "block"])
    return c.merge(s, on="cell_id", validate="one_to_one")


def regimes(work: Path, smoke: bool = False) -> list[str]:
    lay = json.loads((work / "labels" / "layouts.json").read_text())["layouts"]
    sparse = [L["layout_id"] for L in lay]
    return ["dense_smoke", sparse[0]] if smoke else ["dense", *sparse]


def label_ids(work: Path, regime: str) -> np.ndarray:
    f = work / "labels" / (f"{regime}.parquet" if regime.startswith("dense") else f"sparse_{regime}.parquet")
    return pd.read_parquet(f)["cell_id"].to_numpy()


def training_labels(cells: pd.DataFrame, work: Path, regime: str) -> pd.DataFrame:
    """Label rows for a regime; asserts every label is a usable train cell."""
    ids = label_ids(work, regime)
    lab = cells[cells["cell_id"].isin(ids)]
    if len(lab) != len(np.unique(ids)) or not (lab["partition"] == TRAIN).all() or not lab["domain"].isin(DOMAINS).all():
        raise AssertionError(f"{regime}: labels must be unique usable land/water cells of the train partition")
    return lab


def routed(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, model domain): land/water cells once, ambiguous
    cells once per domain model."""
    base = cells[cells["domain"].isin(DOMAINS)].assign(model_domain=lambda d: d["domain"])
    amb = cells[cells["domain"] == masks.AMBIGUOUS]
    parts = [base] + [amb.assign(model_domain=d) for d in DOMAINS]
    return pd.concat(parts, ignore_index=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
