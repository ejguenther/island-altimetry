#!/usr/bin/env python
"""CNN terrain M4 baselines on the identical pools and center sets.

  run_cnn_baselines.py [--config configs/cnn_terrain.yaml] [--only const,rf,idw]

const_<regime>: mean elevation of the regime's pooled training labels.
rf_<regime>:    one RandomForestRegressor pooled over the fitting sites'
                exact training pool; features = the 4 common bands plus the
                coastal 'context' recipe restricted to them (normalized
                differences, log ratio, 3x3/5x5 mean/std); imagery only.
idw_sparse:     within-site IDW from that site's sparse training centers
                (configured k / power / radius); transfer sites get no
                prediction (unsupported), never borrowed labels.
Predictions are written for every eligible center of every role site; the
evaluation script decides which partition is scored.

Writes outputs/cnn_terrain/predictions/{const_dense,const_sparse,rf_dense,
rf_sparse,idw_sparse}.parquet and baselines.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor

from island_altimetry import manifest
from island_altimetry.cnn import config as ccfg
from island_altimetry.coastal import features, grid, interp


def site_features(cfg, site) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    sp = ccfg.site_paths(cfg, site)
    rdir = sp["work"] / "reference"
    with rasterio.open(rdir / "planet_sr.tif") as s:
        names = list(s.descriptions)
        sr = s.read()
    valid = grid.read_band(rdir / "masks.tif", "image_valid").astype(bool)
    layers = features.feature_layers(sr, names, valid, "context")
    c = pd.read_parquet(sp["work"] / "centers.parquet", columns=["site", "cell_id", "row", "col", "x", "y", "z", "partition"])
    return features.sample(layers, c["row"].to_numpy(), c["col"].to_numpy()), list(layers), c


def save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    ap.add_argument("--only", default="const,rf,idw")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    todo = a.only.split(",")
    out = ccfg.out_dir(cfg)
    roles = json.loads((out / "site_roles.json").read_text())
    pools = {r: pd.read_parquet(out / "pools" / f"{r}.parquet") for r in ("dense", "sparse")}
    sites = roles["fit"] + roles["transfer_val"] + roles["transfer_test"]
    pdir = out / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    meta = {}
    feats = {}
    if "rf" in todo or "const" in todo or "idw" in todo:
        for s in sites:
            feats[s] = site_features(cfg, s) if "rf" in todo else (None, None, pd.read_parquet(
                ccfg.site_paths(cfg, s)["work"] / "centers.parquet", columns=["site", "cell_id", "row", "col", "x", "y", "z",
                                                                              "partition"]))
    allc = pd.concat([feats[s][2] for s in sites], ignore_index=True)
    if "const" in todo:
        for r, p in pools.items():
            m = float(p["z"].mean())
            save(allc[["site", "cell_id"]].assign(pred=np.float32(m)), pdir / f"const_{r}.parquet")
            meta[f"const_{r}"] = {"value_m": m, "n_labels": len(p)}
    if "idw" in todo:
        ic, rows = cfg["interp"], []
        for s in sites:
            c = feats[s][2]
            lab = pools["sparse"][pools["sparse"]["site"] == s]
            if s not in roles["fit"] or lab.empty:
                rows.append(c[["site", "cell_id"]].assign(pred=np.float32(np.nan)))
                continue
            p, d, _ = interp.idw(lab[["x", "y"]].to_numpy(), lab["z"].to_numpy(), c[["x", "y"]].to_numpy(), ic["k"],
                                 ic["power"], ic["max_radius_m"])
            rows.append(c[["site", "cell_id"]].assign(pred=p, d_label=d))
        save(pd.concat(rows, ignore_index=True), pdir / "idw_sparse.parquet")
        meta["idw_sparse"] = ic | {"transfer": "unsupported (no local labels)"}
    if "rf" in todo:
        rc = cfg["rf"]
        names = feats[sites[0]][1]
        key = {s: pd.Series(np.arange(len(feats[s][2])), index=feats[s][2]["cell_id"].to_numpy()) for s in sites}
        for r, p in pools.items():
            X = np.concatenate([feats[s][0][key[s][g["cell_id"].to_numpy()].to_numpy()] for s, g in p.groupby("site")])
            z = np.concatenate([g["z"].to_numpy() for _, g in p.groupby("site")])
            ok = np.isfinite(X).all(1)
            t0 = time.time()
            rf = RandomForestRegressor(n_estimators=rc["n_estimators"], max_depth=rc["max_depth"],
                                       min_samples_leaf=rc["min_samples_leaf"], n_jobs=rc["n_jobs"],
                                       random_state=rc["seed"]).fit(X[ok], z[ok])
            fit_s = time.time() - t0
            rows = []
            for s in sites:
                Xs = feats[s][0]
                pred = np.full(len(Xs), np.nan, np.float32)
                g = np.isfinite(Xs).all(1)
                for i in range(0, len(Xs), 500_000):
                    sl = slice(i, i + 500_000)
                    if g[sl].any():
                        pred[sl][g[sl]] = rf.predict(Xs[sl][g[sl]])
                rows.append(feats[s][2][["site", "cell_id"]].assign(pred=pred))
            save(pd.concat(rows, ignore_index=True), pdir / f"rf_{r}.parquet")
            meta[f"rf_{r}"] = {"n_train": int(ok.sum()), "features": names, "fit_seconds": round(fit_s, 1),
                               "importances": dict(zip(names, np.round(rf.feature_importances_, 4).tolist()))}
            print(f"  rf_{r}: n={ok.sum()} fit {fit_s:.0f}s")
            del rf
    f = out / "baselines.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    manifest.write_json_atomic(f, old | meta | {"config_hash": cfg["config_hash"]})
    print({k: {kk: vv for kk, vv in v.items() if kk not in ("features", "importances")} if isinstance(v, dict) else v
           for k, v in meta.items()})


if __name__ == "__main__":
    main()
