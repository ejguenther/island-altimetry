#!/usr/bin/env python
"""Coastal terrain features: descriptive ablation on the VALIDATION partition.

  ablate_coastal_features.py [--config configs/coastal_experiment.yaml] [--regimes dense,t12_v0,t12_v1,t12_v2]

The frozen T1/T2 feature set is not changed by this script. Two diagnostics:
1. Drop-one-family refits (texture, shape, orientation, distance) with the
   declared RF settings, scored on validation cells of each model's own domain.
2. Permutation importance per family for the saved T1 dense models: every
   column of a family permuted jointly (same row permutation), on a fixed
   50k-cell validation sample per domain, 3 repeats.

Writes outputs/.../tables/{terrain_ablation_val.csv,terrain_permutation_val.csv}
and figures/eval_val/terrain_ablation.png. Test cells are never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from island_altimetry import plotting  # noqa: F401 - sets the Agg backend
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import experiment as ex
from island_altimetry.coastal import masks, metrics
from scripts_path import load_train

import matplotlib.pyplot as plt  # noqa: E402

FAMILIES = ["texture", "shape", "orientation", "distance"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    ap.add_argument("--regimes", default="dense,t12_v0,t12_v1,t12_v2")
    ap.add_argument("--n-perm", type=int, default=50_000)
    a = ap.parse_args()
    tr = load_train()
    cfg = ccfg.load(Path(a.config))
    D, rc = ccfg.dirs(cfg), cfg["rf"]
    W = D["work"]
    cells = ex.load_cells(W)
    pos = pd.Series(np.arange(len(cells)), index=cells["cell_id"].to_numpy())
    X, names = tr.build_features(W, cells, "terrain", tr.terrain_cfg(cfg))
    fam = json.loads(sorted((W / "features").glob("terrain_*.json"))[-1].read_text())["families"]
    fam_of = np.array([fam[n] for n in names])
    val = cells[(cells["partition"] == ex.VAL) & cells["domain"].isin(ex.DOMAINS)]

    rows = []
    for r in a.regimes.split(","):
        lab = ex.training_labels(cells, W, r)
        for d in ex.DOMAINS:
            L = lab[lab["domain"] == d]
            V = val[val["domain"] == d]
            if len(L) < rc["min_labels_per_domain"]:
                continue
            for drop in [None, *FAMILIES]:
                cols = np.nonzero(fam_of != drop)[0] if drop else None
                _, pred, info = tr.fit_predict(X, cells, pos[V["cell_id"].to_numpy()].to_numpy(),
                                               pos[L["cell_id"].to_numpy()].to_numpy(), L["z"].to_numpy(), rc,
                                               rc["n_estimators"], cols)
                m = metrics.summarize(pred, V["z"].to_numpy())
                rows.append({"regime": r, "domain": masks.DOMAIN_NAMES[d], "dropped": drop or "none",
                             "n_features": len(names) if cols is None else len(cols), **m, **info})
                print(f"  {r} {masks.DOMAIN_NAMES[d]} drop={drop}: val RMSE {m['rmse']:.3f}")
    ab = pd.DataFrame(rows)
    full = ab[ab["dropped"] == "none"].set_index(["regime", "domain"])["rmse"]
    ab["delta_rmse_vs_full"] = ab["rmse"] - ab.set_index(["regime", "domain"]).index.map(full)
    ab.to_csv(D["tables"] / "terrain_ablation_val.csv", index=False)

    prow = []
    rng = np.random.default_rng(rc["seed"])
    for d in ex.DOMAINS:
        f = W / "models" / f"T1_dense_{masks.DOMAIN_NAMES[d]}.joblib"
        if not f.exists():
            continue
        rf = joblib.load(f)
        V = val[val["domain"] == d]
        V = V.iloc[rng.choice(len(V), size=min(a.n_perm, len(V)), replace=False)]
        Xv = np.asarray(X[pos[V["cell_id"].to_numpy()].to_numpy()])
        ok = np.isfinite(Xv).all(1)
        Xv, z = Xv[ok], V["z"].to_numpy()[ok]
        base = metrics.summarize(rf.predict(Xv), z)["rmse"]
        for fname in ["context", *FAMILIES]:
            idx = np.nonzero(fam_of == fname)[0]
            for k in range(3):
                Xp = Xv.copy()
                Xp[:, idx] = Xv[rng.permutation(len(Xv))][:, idx]
                prow.append({"domain": masks.DOMAIN_NAMES[d], "family": fname, "repeat": k, "n": len(z),
                             "rmse_base": base, "rmse_permuted": metrics.summarize(rf.predict(Xp), z)["rmse"]})
        del rf
    pm = pd.DataFrame(prow)
    pm["delta_rmse"] = pm["rmse_permuted"] - pm["rmse_base"]
    pm.to_csv(D["tables"] / "terrain_permutation_val.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dom in zip(axs, ("land", "water")):
        g = ab[(ab["domain"] == dom) & (ab["dropped"] != "none")]
        if g.empty:
            continue
        piv = g.pivot_table(index="dropped", columns="regime", values="delta_rmse_vs_full").reindex(FAMILIES)
        piv.plot.bar(ax=ax, rot=0)
        p = pm[pm["domain"] == dom].groupby("family")["delta_rmse"].mean()
        ax.set_title(f"{dom}: val RMSE change when a family is dropped\n"
                     f"(T1 permutation dRMSE: {', '.join(f'{k} {v:+.2f}' for k, v in p.items())})", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("RMSE(drop) - RMSE(full) (m)")
    fig.tight_layout()
    out = D["figures"] / "eval_val" / "terrain_ablation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(ab.pivot_table(index=["domain", "dropped"], columns="regime", values="rmse"))
        print(pm.groupby(["domain", "family"])["delta_rmse"].mean().unstack())


if __name__ == "__main__":
    main()
