#!/usr/bin/env python
"""Coastal M4: constant (B0) and sparse-height interpolation (B1) baselines.

  run_interpolation.py [--config configs/coastal_experiment.yaml] [--smoke]

B0: per-domain mean of each regime's training labels.
B1: IDW from the regime's same-domain training-track labels only (KD-tree,
bounded k and radius, chunked). Ambiguous cells are predicted once from each
domain's labels and reported separately. One IDW configuration (k, power) is
selected on VALIDATION cells (mean val RMSE over land and water, all sparse
layouts) from a small predeclared grid; the test partition is not read.
Also writes nearest-training-label distances per regime and model domain,
shared by all methods' diagnostics.

Writes data/<site>/coastal/predictions[_smoke]/{B0_<regime>,B1_<layout>,
dist_<regime>}.parquet and outputs/.../tables/{interp_selection.csv,
baseline_val_metrics.csv,interp_choice.json}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from island_altimetry import manifest
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import experiment as ex
from island_altimetry.coastal import interp, masks, metrics

K_GRID = (4, 12, 32)
POWER_GRID = (1.0, 2.0, 3.0)


def predict_idw(rt: pd.DataFrame, lab: pd.DataFrame, k: int, power: float, radius: float) -> pd.DataFrame:
    out = []
    for d in ex.DOMAINS:
        q = rt[rt["model_domain"] == d]
        L = lab[lab["domain"] == d]
        p, dmin, used = interp.idw(L[["x", "y"]].to_numpy(), L["z"].to_numpy(), q[["x", "y"]].to_numpy(), k, power, radius)
        out.append(pd.DataFrame({"cell_id": q["cell_id"].to_numpy(), "model_domain": np.int8(d), "pred": p,
                                 "n_neighbours": used}))
    return pd.concat(out, ignore_index=True)


def val_metrics(pred: pd.DataFrame, rt: pd.DataFrame) -> list[dict]:
    m = rt[rt["partition"] == ex.VAL][["cell_id", "model_domain", "z", "domain"]].merge(
        pred, on=["cell_id", "model_domain"], how="left")
    rows = []
    for (dom, md), g in m.groupby(["domain", "model_domain"]):
        rows.append({"domain": masks.DOMAIN_NAMES[dom], "model_domain": masks.DOMAIN_NAMES[md],
                     **metrics.summarize(g["pred"].to_numpy(), g["z"].to_numpy())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    ap.add_argument("--smoke", action="store_true", help="dense_smoke + first layout, val cells only")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D, ic = ccfg.dirs(cfg), cfg["interp"]
    W = D["work"]
    pdir = W / ("predictions_smoke" if a.smoke else "predictions")
    cells = ex.load_cells(W)
    rt = ex.routed(cells)
    if a.smoke:
        rt = rt[rt["partition"] == ex.VAL]
    regs = ex.regimes(W, a.smoke)
    sparse = [r for r in regs if not r.startswith("dense")]
    labs = {r: ex.training_labels(cells, W, r) for r in regs}

    # nearest same-domain training label per routed cell, every regime
    for r in regs:
        out = []
        for d in ex.DOMAINS:
            q = rt[rt["model_domain"] == d]
            L = labs[r][labs[r]["domain"] == d]
            out.append(pd.DataFrame({"cell_id": q["cell_id"].to_numpy(), "model_domain": np.int8(d),
                                     "d_label": interp.nearest_distance(L[["x", "y"]].to_numpy(), q[["x", "y"]].to_numpy())}))
        ex.save_parquet(pd.concat(out, ignore_index=True), pdir / f"dist_{r}.parquet")

    rows = []
    for r in regs:   # B0
        means = labs[r].groupby("domain")["z"].mean()
        p = rt[["cell_id", "model_domain"]].assign(pred=rt["model_domain"].map(means).astype(np.float32))
        p["model_domain"] = p["model_domain"].astype(np.int8)
        ex.save_parquet(p, pdir / f"B0_{r}.parquet")
        rows += [{"method": "B0", "regime": r, **x} for x in val_metrics(p, rt)]

    sel = []
    for k, pw in itertools.product(K_GRID, POWER_GRID):   # B1 selection on validation only
        t0 = time.time()
        for r in sparse:
            vm = val_metrics(predict_idw(rt[rt["partition"] == ex.VAL], labs[r], k, pw, ic["max_radius_m"]), rt)
            sel += [{"k": k, "power": pw, "regime": r, **x} for x in vm if x["domain"] == x["model_domain"]]
        print(f"  idw k={k} p={pw}: {time.time() - t0:.1f}s")
    sel = pd.DataFrame(sel)
    score = sel.groupby(["k", "power"])["rmse"].mean().sort_values()
    k_best, p_best = score.index[0]
    choice = {"k": int(k_best), "power": float(p_best), "max_radius_m": ic["max_radius_m"],
              "criterion": "mean validation RMSE over land & water and all sparse layouts",
              "score": float(score.iloc[0]), "grid": {"k": list(K_GRID), "power": list(POWER_GRID)},
              "config_hash": cfg["config_hash"], "smoke": a.smoke}
    print(f"selected IDW k={k_best} power={p_best} (val RMSE {score.iloc[0]:.3f} m)")
    for r in sparse:
        p = predict_idw(rt, labs[r], int(k_best), float(p_best), ic["max_radius_m"])
        ex.save_parquet(p, pdir / f"B1_{r}.parquet")
        rows += [{"method": "B1", "regime": r, **x} for x in val_metrics(p, rt)]

    tag = "_smoke" if a.smoke else ""
    sel.to_csv(D["tables"] / f"interp_selection{tag}.csv", index=False)
    pd.DataFrame(rows).to_csv(D["tables"] / f"baseline_val_metrics{tag}.csv", index=False)
    manifest.write_json_atomic(D["tables"] / f"interp_choice{tag}.json", choice)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(rows).query("domain == model_domain")
              .pivot_table(index=["method", "regime"], columns="domain", values="rmse"))


if __name__ == "__main__":
    main()
