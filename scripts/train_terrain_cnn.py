#!/usr/bin/env python
"""CNN terrain M4: train the center-label FCN on dense or sparse pools.

  train_terrain_cnn.py --regime dense|sparse [--seed 0] [--context large|small] [--smoke] [--epochs N]

One center elevation (m NAVD88, no target scaling) per R x R image context;
AOI-balanced sampling (equal share per fitting site, with replacement);
identical architecture, scalers, augmentation (random 90-degree rotation and
flip, one random dihedral transform per batch), optimizer and step budget for dense and sparse. Selection
criterion per epoch: w_within * mean per-site VALIDATION RMSE (fitting sites)
+ w_transfer * transfer-validation RMSE, logged every epoch; with
train.selection = fixed_budget every run trains train.fixed_epochs and the last
checkpoint (final.pt) is the model, best.pt the criterion-best sensitivity
checkpoint. Test data are never loaded.

Writes data/cnn_terrain/runs/<run>/{final.pt,best.pt,last.pt} and
outputs/cnn_terrain/runs/<run>/{history.csv,run.json}.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from island_altimetry import manifest
from island_altimetry.cnn import config as ccfg
from island_altimetry.cnn import dataset, model
from island_altimetry.config import ROOT, fmt_utc


def run_dirs(cfg: dict, run: str) -> tuple[Path, Path]:
    return ROOT / cfg["data_root"] / cfg["experiment"] / "runs" / run, ccfg.out_dir(cfg) / "runs" / run


def load_inputs(cfg: dict, sites: list[str]) -> dict[str, np.ndarray]:
    return {s: dataset.load_image(ccfg.site_paths(cfg, s)["work"] / "reference" / "planet_sr.tif", cfg["imagery"]["bands"])
            for s in sites}


def to_tensors(df: pd.DataFrame, site_index: dict, device) -> tuple:
    return (torch.as_tensor(df["site"].map(site_index).to_numpy(), device=device),
            torch.as_tensor(df["row"].to_numpy(), device=device), torch.as_tensor(df["col"].to_numpy(), device=device),
            torch.as_tensor(df["z"].to_numpy(np.float32), device=device))


@torch.no_grad()
def predict(net, gp: dataset.GpuPatches, s, r, c, amp, batch: int = 4096) -> torch.Tensor:
    net.eval()
    out = []
    for i in range(0, len(r), batch):
        x = gp.gather(s[i:i + batch], r[i:i + batch], c[i:i + batch])
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out.append(net(x)[:, 0, 0].float())
    return torch.cat(out)


def site_rmse(pred, z, s, names) -> dict:
    e = (pred - z) ** 2
    return {names[k]: float(torch.sqrt(e[s == k].mean())) for k in torch.unique(s).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    ap.add_argument("--regime", choices=["dense", "sparse"], required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--context", choices=["large", "small"], default="large")
    ap.add_argument("--smoke", action="store_true", help="3 short epochs to measure speed and memory")
    ap.add_argument("--epochs", type=int, help="override train.max_epochs")
    ap.add_argument("--fp32", action="store_true", help="float32 instead of bf16 autocast (reference check)")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    tc = cfg["train"]
    seed = tc["seed"] if a.seed is None else a.seed
    dil = cfg["model"]["dilations"] if a.context == "large" else cfg["model"]["dilations_small"]
    run = f"{a.regime}_{a.context}_s{seed}" + ("_fp32" if a.fp32 else "") + ("_smoke" if a.smoke else "")
    wdir, odir = run_dirs(cfg, run)
    wdir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    out = ccfg.out_dir(cfg)
    roles = json.loads((out / "site_roles.json").read_text())
    pdir = out / "pools"
    scal = json.loads((pdir / "scalers.json").read_text())
    pool = pd.read_parquet(pdir / f"{a.regime}.parquet")
    vw, vt = pd.read_parquet(pdir / "val_within.parquet"), pd.read_parquet(pdir / "val_transfer.parquet")
    assert not set(pool["site"]) - set(roles["fit"]) and (pool["partition"] == 1).all()

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda")
    amp = None if a.fp32 else getattr(torch, tc["amp_dtype"])
    sites = roles["fit"] + roles["transfer_val"]
    sidx = {s: i for i, s in enumerate(sites)}
    radius = model.output_offset(dil)
    gp = dataset.GpuPatches(load_inputs(cfg, sites), scal, radius, dev)
    P = to_tensors(pool, sidx, dev)
    VW, VT = to_tensors(vw, sidx, dev), to_tensors(vt, sidx, dev)
    site_of = pool["site"].map({s: i for i, s in enumerate(roles["fit"])}).to_numpy()

    net = model.build(len(cfg["imagery"]["bands"]), cfg["model"]["channels"], dil).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    fixed = tc.get("selection") == "fixed_budget"
    epochs = 3 if a.smoke else (a.epochs or (tc["fixed_epochs"] if fixed else tc["max_epochs"]))
    per_epoch = 5000 if a.smoke else tc["samples_per_epoch"]
    bs, w = tc["batch_size"], tc["criterion_weights"]
    hist, best, bad, seen = [], np.inf, 0, np.zeros(len(pool), np.int64)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        t0 = time.time()
        idx = dataset.balanced_indices(site_of, len(roles["fit"]), per_epoch, rng)
        np.add.at(seen, idx, 1)
        losses = []
        for i in range(0, len(idx) - bs + 1, bs):
            b = torch.as_tensor(idx[i:i + bs], device=dev)
            x = gp.gather(P[0][b], P[1][b], P[2][b])
            k = rng.integers(0, 8)                      # one of 8 dihedral transforms per batch
            x = dataset.augment(x, int(k % 4), bool(k // 4))
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                y = net(x)[:, 0, 0].float()
            loss = torch.mean((y - P[3][b]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        pw = predict(net, gp, VW[0], VW[1], VW[2], amp)
        pt = predict(net, gp, VT[0], VT[1], VT[2], amp)
        within = site_rmse(pw, VW[3], VW[0], sites)
        transfer = float(torch.sqrt(torch.mean((pt - VT[3]) ** 2)))
        crit = w["within_val"] * float(np.mean(list(within.values()))) + w["transfer_val"] * transfer
        rec = {"epoch": ep, "train_rmse": float(np.sqrt(np.mean(losses))), "val_within_mean_rmse": float(np.mean(list(within.values()))),
               "val_transfer_rmse": transfer, "criterion": crit, "seconds": round(time.time() - t0, 1),
               **{f"val_{k}": v for k, v in within.items()}}
        hist.append(rec)
        state = {"model": net.state_dict(), "optimizer": opt.state_dict(), "epoch": ep, "criterion": crit,
                 "dilations": dil, "channels": cfg["model"]["channels"], "bands": cfg["imagery"]["bands"],
                 "scalers": scal, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
        torch.save(state, wdir / "last.pt")
        if crit < best:
            best, bad = crit, 0
            torch.save(state, wdir / "best.pt")
        else:
            bad += 1
        print(f"  ep {ep:2d} train {rec['train_rmse']:.3f} within {rec['val_within_mean_rmse']:.3f} "
              f"transfer {transfer:.3f} crit {crit:.3f} ({rec['seconds']}s){' *' if bad == 0 else ''}")
        pd.DataFrame(hist).to_csv(odir / "history.csv", index=False)
        if not fixed and bad >= tc["patience"]:
            print(f"  early stop at epoch {ep}")
            break
    torch.save(state, wdir / "final.pt")    # fixed budget: the final model; best.pt = criterion-best (sensitivity)
    meta = {"run": run, "selection": tc.get("selection", "criterion"), "final_checkpoint": "final.pt" if fixed else "best.pt", "regime": a.regime, "context": a.context, "receptive_field_px": model.receptive_field(dil),
            "receptive_field_m": model.receptive_field(dil) * 3, "seed": seed, "amp": str(amp), "epochs_run": len(hist),
            "best_epoch": int(pd.DataFrame(hist)["criterion"].idxmin()) + 1, "best_criterion": best,
            "pool_unique_centers": len(pool), "optimizer_examples": int(seen.sum()),
            "distinct_centers_seen": int((seen > 0).sum()), "max_revisits": int(seen.max()),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "train_minutes": round((time.time() - t_start) / 60, 1),
            "roles": {k: roles[k] for k in ("fit", "transfer_val", "transfer_test")},
            "pool_hash": json.loads((pdir / "pools.json").read_text())[a.regime]["hash"], "config_hash": cfg["config_hash"],
            "git_commit": manifest.git_commit(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
            "finished_utc": fmt_utc(pd.Timestamp.now(tz="UTC")), "checkpoints": str(wdir.relative_to(ROOT))}
    manifest.write_json_atomic(odir / "run.json", meta)
    print(json.dumps({k: meta[k] for k in ("run", "best_epoch", "best_criterion", "optimizer_examples",
                                            "distinct_centers_seen", "peak_vram_gb", "train_minutes")}, indent=1))


if __name__ == "__main__":
    main()
