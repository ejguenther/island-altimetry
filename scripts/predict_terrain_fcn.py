#!/usr/bin/env python
"""CNN terrain M5: fully convolutional raster inference with a trained run.

  predict_terrain_fcn.py --run dense_large_s0 [--site S ...] [--qa]

For every role site (fitting, transfer-validation, transfer-test): normalize
the image with the run's training scalers, predict with halo tiles
(inference.tile x tile output cores, offset-pixel halos), and write a
full-grid GeoTIFF on the original transform. NoData: the outer offset-pixel
border, centers without a fully valid image context, and anything that is not
imagery land inside the crop. Water imagery remains usable context.
--qa compares full-crop, several halo-tile sizes and independent patches on
a subset (float32) and times FCN vs patch-by-patch inference.

Writes data/<site>/cnn/pred/<run>.tif, outputs/cnn_terrain/predictions/
<run>.parquet (one row per eligible center) and inference_qa.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from island_altimetry import manifest
from island_altimetry.cnn import config as ccfg
from island_altimetry.cnn import dataset, inference, model
from island_altimetry.coastal import grid
from island_altimetry.config import ROOT, fmt_utc


def load_net(path: Path, device):
    st = torch.load(path, map_location=device, weights_only=False)
    net = model.build(len(st["bands"]), st["channels"], st["dilations"]).to(device)
    net.load_state_dict(st["model"])
    return net.eval(), st


def normalized(img: np.ndarray, scal: dict) -> np.ndarray:
    m = np.asarray(scal["mean"], np.float32)[:, None, None]
    s = np.asarray(scal["std"], np.float32)[:, None, None]
    return (img - m) / s


def qa(net, img, device, tiles=(128, 256, 512), size=700, n_patch=2000) -> dict:
    H, W = img.shape[1:]
    r0, c0 = max(0, H // 2 - size // 2), max(0, W // 2 - size // 2)
    sub = np.ascontiguousarray(img[:, r0:r0 + size, c0:c0 + size])
    t = time.time()
    full = inference.predict_full(net, sub, device)
    t_full = time.time() - t
    out = {"crop": [r0, c0, size], "full_seconds": round(t_full, 3)}
    for T in tiles:
        tl = inference.predict_tiled(net, sub, device, tile=T)
        out[f"tile{T}_max_abs_diff"] = float(np.nanmax(np.abs(tl - full)))
        out[f"tile{T}_nan_pattern_equal"] = bool(np.array_equal(np.isnan(tl), np.isnan(full)))
    o = net.offset
    rng = np.random.default_rng(0)
    rr = rng.integers(o, sub.shape[1] - o, n_patch)
    cc = rng.integers(o, sub.shape[2] - o, n_patch)
    t = time.time()
    pp = inference.predict_patches(net, sub, rr, cc, device)
    t_patch = time.time() - t
    out |= {"patch_max_abs_diff": float(np.max(np.abs(pp - full[rr, cc]))), "patch_n": n_patch,
            "patch_seconds": round(t_patch, 3), "fcn_seconds_per_mpx": round(t_full / (sub.shape[1] * sub.shape[2] / 1e6), 4),
            "patch_seconds_per_mpx": round(t_patch / (n_patch / 1e6), 2), "dtype": "float32"}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    ap.add_argument("--run", required=True)
    ap.add_argument("--site", nargs="+")
    ap.add_argument("--qa", action="store_true")
    ap.add_argument("--checkpoint", default="final.pt", help="final.pt (fixed budget) or best.pt (criterion)")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    out = ccfg.out_dir(cfg)
    roles = json.loads((out / "site_roles.json").read_text())
    sites = a.site or roles["fit"] + roles["transfer_val"] + roles["transfer_test"]
    dev = torch.device("cuda")
    # strict float32 (no TF32 tensor-core rounding) so tiled / full / patch outputs agree to fp32 precision
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net, st = load_net(ROOT / cfg["data_root"] / cfg["experiment"] / "runs" / a.run / a.checkpoint, dev)
    tile = cfg["inference"]["tile"]
    rows, qa_out = [], {}
    torch.cuda.reset_peak_memory_stats()
    for site in sites:
        sp = ccfg.site_paths(cfg, site)
        rdir = sp["work"] / "reference"
        spec = json.loads((rdir / "grid_spec.json").read_text())
        img = normalized(dataset.load_image(rdir / "planet_sr.tif", st["bands"]), st["scalers"])
        if a.qa and not qa_out:
            qa_out = {"site": site, **qa(net, img, dev)}
        t = time.time()
        pred = inference.predict_tiled(net, img, dev, tile=tile)
        secs = time.time() - t
        m = {k: grid.read_band(rdir / "masks.tif", k).astype(bool) for k in ("context_valid", "in_crop")}
        land = grid.read_band(rdir / "masks.tif", "domain") == 1
        keep = m["context_valid"] & land & m["in_crop"]
        pred = np.where(keep, pred, np.nan).astype(np.float32)
        grid.write_raster(sp["work"] / "pred" / f"{a.run}{'' if a.checkpoint == 'final.pt' else '_' + Path(a.checkpoint).stem}.tif", {"z_pred": pred}, spec, nodata=np.nan, dtype="float32")
        c = pd.read_parquet(sp["work"] / "centers.parquet", columns=["site", "cell_id", "row", "col"])
        rows.append(c.assign(pred=pred[c["row"], c["col"]])[["site", "cell_id", "pred"]])
        print(f"  {site}: {spec['width']}x{spec['height']} in {secs:.1f}s, centers {len(c)}, "
              f"predicted {np.isfinite(rows[-1]['pred']).mean():.4f}")
    pdir = out / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(rows, ignore_index=True)
    name = a.run + ("" if a.checkpoint == "final.pt" else "_" + Path(a.checkpoint).stem)
    tmp = pdir / f"{name}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(pdir / f"{name}.parquet")
    if qa_out:
        qa_out |= {"run": a.run, "tile": tile, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "generated_utc": fmt_utc(pd.Timestamp.now(tz="UTC"))}
        manifest.write_json_atomic(out / "inference_qa.json", qa_out)
        print(json.dumps(qa_out, indent=1))


if __name__ == "__main__":
    main()
