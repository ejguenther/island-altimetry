#!/usr/bin/env python
"""Coastal M5: random-forest benchmarks with imagery-only predictors.

  train_coastal_rf.py [--config configs/coastal_experiment.yaml] [--smoke] [--only R1,R2,R3]

R1: bands, dense labels. R2: bands, each sparse layout. R3: context features
(bands + predefined normalized differences/log ratio + 3x3/5x5 local stats),
each sparse layout. Separate land and water forests with the same declared
settings (no hyperparameter search). A domain with fewer than
rf.min_labels_per_domain labels is marked unsupported. Cells with any NaN
feature are not predicted (counted). No coordinates, no reference-derived
covariates. Predictions cover every usable cell (all partitions, for maps);
metrics here are validation only.

Writes data/<site>/coastal/{features/*.npy,models/*.joblib(+json),
predictions[_smoke]/R*_<regime>.parquet} and
outputs/.../tables/{rf_runs.csv,rf_val_metrics.csv}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
import sklearn
from sklearn.ensemble import RandomForestRegressor

from island_altimetry import manifest
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import experiment as ex
from island_altimetry.coastal import features, grid, masks, metrics

EXPERIMENTS = {"R1": ("bands", "dense"), "R2": ("bands", "sparse"), "R3": ("context", "sparse")}


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def build_features(W: Path, cells: pd.DataFrame, kind: str) -> tuple[np.ndarray, list[str]]:
    """Feature matrix aligned with cells.parquet order, cached by input hash."""
    fdir = W / "features"
    sr_f = W / "reference" / "planet_sr.tif"
    key = hashlib.sha256(f"{kind}:{sr_f.stat().st_mtime_ns}:{(W / 'reference' / 'masks.tif').stat().st_mtime_ns}:"
                         f"{len(cells)}".encode()).hexdigest()[:12]
    f, meta = fdir / f"{kind}_{key}.npy", fdir / f"{kind}_{key}.json"
    if f.exists() and meta.exists():
        return np.load(f, mmap_mode="r"), json.loads(meta.read_text())["names"]
    with rasterio.open(sr_f) as s:
        sr, names = s.read(), list(s.descriptions)
    valid = grid.read_band(W / "reference" / "masks.tif", "image_valid").astype(bool)
    layers = features.feature_layers(sr, names, valid, kind)
    del sr
    X = features.sample(layers, cells["row"].to_numpy(), cells["col"].to_numpy())
    fnames = list(layers)
    del layers
    fdir.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npy")
    np.save(tmp, X)
    tmp.replace(f)
    manifest.write_json_atomic(meta, {"names": fnames, "kind": kind, "n": len(cells), "key": key,
                                      "definitions": features.__doc__})
    print(f"  features {kind}: {X.shape}, peak RSS {rss_gb():.1f} GB")
    return X, fnames


def fit_predict(X, cells, rt_idx, lab_idx, lab_z, rc, n_estimators) -> tuple:
    ok = np.isfinite(X[lab_idx]).all(1)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=rc["max_depth"],
                               min_samples_leaf=rc["min_samples_leaf"], n_jobs=rc["n_jobs"], random_state=rc["seed"])
    t0 = time.time()
    rf.fit(X[lab_idx][ok], lab_z[ok])
    t_fit = time.time() - t0
    pred = np.full(len(rt_idx), np.nan, np.float32)
    t0 = time.time()
    for i in range(0, len(rt_idx), 400_000):
        xi = np.asarray(X[rt_idx[i:i + 400_000]])
        good = np.isfinite(xi).all(1)
        if good.any():
            pred[i:i + 400_000][good] = rf.predict(xi[good])
    return rf, pred, {"n_train": int(ok.sum()), "n_train_nan_dropped": int((~ok).sum()), "fit_s": round(t_fit, 1),
                      "predict_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    ap.add_argument("--smoke", action="store_true", help="dense_smoke + first layout, 50 trees, val cells only")
    ap.add_argument("--only", help="comma list of experiments (R1,R2,R3)")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D, rc = ccfg.dirs(cfg), cfg["rf"]
    W = D["work"]
    tag = "_smoke" if a.smoke else ""
    pdir, mdir = W / f"predictions{tag}", W / f"models{tag}"
    cells = ex.load_cells(W)
    pos = pd.Series(np.arange(len(cells)), index=cells["cell_id"].to_numpy())
    rt = ex.routed(cells)
    if a.smoke:
        rt = rt[rt["partition"] == ex.VAL]
    rt_pos = pos[rt["cell_id"].to_numpy()].to_numpy()
    regs = ex.regimes(W, a.smoke)
    n_est = 50 if a.smoke else rc["n_estimators"]
    todo = [e for e in EXPERIMENTS if not a.only or e in a.only.split(",")]
    Xs = {}
    runs, vrows = [], []
    for e in todo:
        kind, regime_type = EXPERIMENTS[e]
        if kind not in Xs:
            Xs[kind] = build_features(W, cells, kind)
        X, fnames = Xs[kind]
        for r in [x for x in regs if x.startswith("dense") == (regime_type == "dense")]:
            lab = ex.training_labels(cells, W, r)
            out = []
            for d in ex.DOMAINS:
                L = lab[lab["domain"] == d]
                sel = (rt["model_domain"] == d).to_numpy()
                run = {"experiment": e, "regime": r, "model_domain": masks.DOMAIN_NAMES[d], "features": kind,
                       "n_features": len(fnames), "n_labels": len(L), "n_estimators": n_est}
                if len(L) < rc["min_labels_per_domain"]:
                    run["status"] = "unsupported_too_few_labels"
                    out.append(pd.DataFrame({"cell_id": rt["cell_id"].to_numpy()[sel], "model_domain": np.int8(d),
                                             "pred": np.float32(np.nan)}))
                    runs.append(run)
                    print(f"  {e} {r} {run['model_domain']}: unsupported ({len(L)} labels)")
                    continue
                rf, pred, info = fit_predict(X, cells, rt_pos[sel], pos[L["cell_id"].to_numpy()].to_numpy(),
                                             L["z"].to_numpy(), rc, n_est)
                name = f"{e}_{r}_{run['model_domain']}"
                mdir.mkdir(parents=True, exist_ok=True)
                joblib.dump(rf, mdir / f"{name}.joblib", compress=3)
                run |= info | {"status": "ok", "model": f"{name}.joblib", "peak_rss_gb": round(rss_gb(), 2),
                               "label_z_range": [float(L["z"].min()), float(L["z"].max())]}
                manifest.write_json_atomic(mdir / f"{name}.json", run | {
                    "feature_names": fnames, "rf_params": rf.get_params(), "sklearn": sklearn.__version__,
                    "config_hash": cfg["config_hash"], "git_commit": manifest.git_commit(),
                    "importances": dict(zip(fnames, np.round(rf.feature_importances_, 5).tolist()))})
                del rf
                out.append(pd.DataFrame({"cell_id": rt["cell_id"].to_numpy()[sel], "model_domain": np.int8(d), "pred": pred}))
                runs.append(run)
                print(f"  {e} {r} {run['model_domain']}: n={info['n_train']} fit {info['fit_s']}s "
                      f"predict {info['predict_s']}s RSS {rss_gb():.1f} GB")
            p = pd.concat(out, ignore_index=True)
            ex.save_parquet(p, pdir / f"{e}_{r}.parquet")
            m = rt[rt["partition"] == ex.VAL][["cell_id", "model_domain", "domain", "z"]].merge(
                p, on=["cell_id", "model_domain"], how="left")
            for (dom, md), g in m.groupby(["domain", "model_domain"]):
                vrows.append({"method": e, "regime": r, "domain": masks.DOMAIN_NAMES[dom],
                              "model_domain": masks.DOMAIN_NAMES[md], **metrics.summarize(g["pred"].to_numpy(), g["z"].to_numpy())})
    runs_df, v = pd.DataFrame(runs), pd.DataFrame(vrows)
    for f, df in ((f"rf_runs{tag}.csv", runs_df), (f"rf_val_metrics{tag}.csv", v)):
        path = D["tables"] / f
        if a.only and path.exists():   # partial re-run: replace only the experiments run now
            old = pd.read_csv(path)
            df = pd.concat([old[~old[old.columns[0]].isin(todo)], df], ignore_index=True)
        df.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(v.query("domain == model_domain").pivot_table(index=["method", "regime"], columns="domain", values="rmse"))


if __name__ == "__main__":
    main()
