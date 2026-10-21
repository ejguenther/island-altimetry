"""Error metrics (prediction minus reference) with explicit counts and
coverage, so every table row can be reconciled against its evaluation set."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(pred: np.ndarray, ref: np.ndarray) -> dict:
    """bias, MAE, RMSE over predicted cells; coverage = predicted / evaluated."""
    n_eval = len(ref)
    ok = np.isfinite(pred)
    e = pred[ok].astype(np.float64) - ref[ok].astype(np.float64)
    n = int(ok.sum())
    return {"n_eval": n_eval, "n_pred": n, "coverage": n / n_eval if n_eval else np.nan,
            "bias": float(e.mean()) if n else np.nan, "mae": float(np.abs(e).mean()) if n else np.nan,
            "rmse": float(np.sqrt((e * e).mean())) if n else np.nan,
            "median_abs": float(np.median(np.abs(e))) if n else np.nan,
            "ref_std": float(np.std(ref)) if n_eval else np.nan}


def stratified(df: pd.DataFrame, pred_col: str, by: str | list[str], ref_col: str = "z") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True):
        rows.append({**dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,))),
                     **summarize(g[pred_col].to_numpy(), g[ref_col].to_numpy())})
    return pd.DataFrame(rows)
