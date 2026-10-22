#!/usr/bin/env python
"""CNN terrain M2b: training pools, validation sets and fixed image scalers.

  sample_cnn_centers.py [--config configs/cnn_terrain.yaml]

From the frozen splits (make_cnn_splits.py):
- dense:  up to sampling.dense_cap_per_site random TRAIN centers per fitting site
- sparse: unique TRAIN centers crossed by straight track centerlines
          (track_spacing_m apart, azimuth, offset) over each fitting site
- val_within: up to val_cap_per_site random VAL centers per fitting site
- val_transfer: up to val_cap_per_site centers of the transfer-validation site
- diag_val: elevation-stratified validation diagnostic set (deciles), kept
            separate from the area-representative val sets
- scalers: per-channel mean/std over training-center contexts of fitting
           sites only (shared by dense and sparse; no target scaling)
Test blocks and transfer-test sites are never sampled here.

Writes outputs/cnn_terrain/pools/{dense,sparse,val_within,val_transfer,
diag_val}.parquet, scalers.json, pools.json (counts, hashes) and per-site
tracks.geojson.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from rasterio.crs import CRS

from island_altimetry import manifest
from island_altimetry.cnn import config as ccfg
from island_altimetry.cnn import dataset
from island_altimetry.cnn.model import output_offset
from island_altimetry.coastal import grid, tracks
from island_altimetry.config import fmt_utc

COLS = ["site", "cell_id", "row", "col", "x", "y", "z", "partition"]


def h(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df[["site", "cell_id"]], index=False).values.tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    sc = cfg["sampling"]
    out = ccfg.out_dir(cfg)
    pdir = out / "pools"
    pdir.mkdir(parents=True, exist_ok=True)
    roles = json.loads((out / "site_roles.json").read_text())
    pools = {k: [] for k in ("dense", "sparse", "val_within", "val_transfer", "diag_val")}
    images, train_rc, info = {}, {}, {}
    for site in roles["fit"] + roles["transfer_val"]:
        sp = ccfg.site_paths(cfg, site)
        c = pd.read_parquet(sp["work"] / "centers.parquet")
        if site in roles["transfer_val"]:
            v = c.sample(n=min(sc["val_cap_per_site"], len(c)), random_state=sc["seed"])
            pools["val_transfer"].append(v[COLS])
            continue
        tr, va = c[c["partition"] == 1], c[c["partition"] == 2]
        pools["dense"].append(tr.sample(n=min(sc["dense_cap_per_site"], len(tr)), random_state=sc["seed"])[COLS])
        spec = json.loads((sp["work"] / "reference" / "grid_spec.json").read_text())
        b = (c["x"].min(), c["y"].min(), c["x"].max(), c["y"].max())
        lines = tracks.layout_by_spacing(b, sc["track_azimuth_deg"], sc["track_spacing_m"], sc["track_offset_frac"])
        count, spacing = len(lines), float(sc["track_spacing_m"])
        ids = tracks.line_cells(lines, Affine(*spec["transform"]), (spec["height"], spec["width"]))
        sp_cells = tr[tr["cell_id"].isin(ids)]
        pools["sparse"].append(sp_cells[COLS])
        grid.write_geojson(shapely.MultiLineString(lines), CRS.from_wkt(spec["crs"]), sp["out"] / "tracks.geojson",
                           {"spacing_m": spacing, "azimuth_deg": sc["track_azimuth_deg"], "count": count,
                            "offset_frac": sc["track_offset_frac"]})
        pools["val_within"].append(va.sample(n=min(sc["val_cap_per_site"], len(va)), random_state=sc["seed"])[COLS])
        dec = pd.qcut(va["z"], 10, labels=False, duplicates="drop")
        pools["diag_val"].append(va.groupby(dec, group_keys=False).apply(
            lambda g: g.sample(n=min(200, len(g)), random_state=sc["seed"]))[COLS])
        info[site] = {"train_centers": len(tr), "dense": int(min(sc["dense_cap_per_site"], len(tr))),
                      "sparse_unique": len(sp_cells), "track_spacing_m": spacing, "track_count": count,
                      "val_centers": len(va), "dense_z_p05_p95": [float(tr["z"].quantile(.05)), float(tr["z"].quantile(.95))],
                      "sparse_z_range": [float(sp_cells["z"].min()), float(sp_cells["z"].max())] if len(sp_cells) else None}
        images[site] = dataset.load_image(sp["work"] / "reference" / "planet_sr.tif", cfg["imagery"]["bands"])
        train_rc[site] = tr[["row", "col"]].to_numpy()
    o = output_offset(cfg["model"]["dilations"])
    scal = dataset.fit_scalers(images, train_rc, o, 2000, np.random.default_rng(sc["seed"]))
    scal |= {"bands": cfg["imagery"]["bands"], "sites": roles["fit"], "source": "fitting-site TRAIN center contexts"}
    manifest.write_json_atomic(pdir / "scalers.json", scal)
    meta = {"generated_utc": fmt_utc(pd.Timestamp.now(tz="UTC")), "config_hash": cfg["config_hash"],
            "roles": {k: roles[k] for k in ("fit", "transfer_val", "transfer_test")}, "per_site": info}
    for k, parts in pools.items():
        df = pd.concat(parts, ignore_index=True)
        assert df.duplicated(["site", "cell_id"]).sum() == 0
        if k in ("dense", "sparse"):
            assert (df["partition"] == 1).all() and set(df["site"]) <= set(roles["fit"])
        assert not set(df["site"]) & set(roles["transfer_test"])
        tmp = pdir / f"{k}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(pdir / f"{k}.parquet")
        meta[k] = {"n": len(df), "hash": h(df), "by_site": df.groupby("site").size().to_dict()}
    manifest.write_json_atomic(pdir / "pools.json", meta)
    print(json.dumps({k: meta[k]["by_site"] for k in pools}, indent=1))
    print("scalers", {k: scal[k] for k in ("mean", "std", "n_pixels")})


if __name__ == "__main__":
    main()
