#!/usr/bin/env python
"""CNN terrain M2a: freeze site roles and within-site spatial splits.

  make_cnn_splits.py [--config configs/cnn_terrain.yaml]

Roles: seeded random draw over the sites that passed M1 (NAVD88, sufficient
eligible centers) -> n_fit fitting, 1 transfer-validation, 2 transfer-test.
Never chosen by model performance. Fitting sites: equal-count blocks along
each site's principal axis (~block_length_m), whole blocks assigned 70/15/15
with a recorded seed; centers within guard_m / 2 of a boundary touching
train are excluded, and the script asserts that no train-center context
(R x R) overlaps any validation/test-center context.

Writes data/<site>/cnn/centers.parquet (every eligible center with its
partition), data/<site>/cnn/split.tif, outputs/cnn_terrain/{site_roles.json,
splits.csv, sites/<site>/split_map.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from scipy import ndimage

from island_altimetry import manifest, plotting  # noqa: F401 - Agg backend
from island_altimetry.cnn import config as ccfg
from island_altimetry.cnn.model import output_offset
from island_altimetry.coastal import grid, splits
from island_altimetry.coastal.reference import strip_axis
from island_altimetry.config import fmt_utc

import matplotlib.pyplot as plt  # noqa: E402

TRAIN, VAL, TEST, EXCL = 1, 2, 3, 0
TRANSFER_VAL, TRANSFER_TEST = 4, 5
CODES = {EXCL: "excluded", TRAIN: "train", VAL: "val", TEST: "test", TRANSFER_VAL: "transfer_val",
         TRANSFER_TEST: "transfer_test"}


def assign_roles(passing: list[str], rc: dict) -> dict:
    order = list(np.random.default_rng(rc["seed"]).permutation(sorted(passing)))
    n_tv, n_tt = rc["n_transfer_val"], rc["n_transfer_test"]
    n_fit = min(rc["n_fit"], len(order) - n_tv - n_tt)
    if n_fit < 2:
        raise SystemExit(f"only {len(order)} passing sites: too few for the role design")
    return {"fit": sorted(order[:n_fit]), "transfer_val": sorted(order[n_fit:n_fit + n_tv]),
            "transfer_test": sorted(order[n_fit + n_tv:n_fit + n_tv + n_tt]),
            "unused": sorted(order[n_fit + n_tv + n_tt:]), "draw_order": order}


def site_centers(cfg: dict, site: str) -> tuple[pd.DataFrame, dict]:
    sp = ccfg.site_paths(cfg, site)
    rdir = sp["work"] / "reference"
    spec = json.loads((rdir / "grid_spec.json").read_text())
    elig = grid.read_band(rdir / "masks.tif", "eligible").astype(bool)
    r, c = np.nonzero(elig)
    t = Affine(*spec["transform"])
    x, y = t * (c + 0.5, r + 0.5)
    ref = {k: grid.read_band(rdir / "target.tif", k) for k in ("z_target", "z_target_std", "ndsm")}
    zd = grid.read_band(rdir / "reference.tif", "z_mean")
    sup = grid.read_band(rdir / "support.tif")
    df = pd.DataFrame({"site": site, "cell_id": r.astype(np.int64) * spec["width"] + c, "row": r.astype(np.int32),
                       "col": c.astype(np.int32), "x": x, "y": y, "z": ref["z_target"][r, c], "z_std": ref["z_target_std"][r, c],
                       "z_dtm": zd[r, c],
                       "ndsm": ref["ndsm"][r, c], "support": sup[r, c], "domain": 1})
    return df, spec


def context_overlap(rows_a, cols_a, rows_b, cols_b, shape, o: int) -> int:
    """Number of pixels covered by both sets' R x R contexts."""
    def cover(rr, cc):
        m = np.zeros(shape, bool)
        m[rr, cc] = True
        return ndimage.binary_dilation(m, np.ones((2 * o + 1, 2 * o + 1), bool))
    return int((cover(rows_a, cols_a) & cover(rows_b, cols_b)).sum())


def split_site(df: pd.DataFrame, spec: dict, sc: dict, mask_shape, o: int) -> tuple[pd.DataFrame, dict]:
    elig = np.zeros(mask_shape, bool)
    elig[df["row"], df["col"]] = True
    axis = strip_axis(elig, spec)
    ax = {"centroid": axis[0], "direction": axis[1]}
    n_blocks = max(7, int(round(axis[2] / sc["block_length_m"])))
    res = splits.make_split(df, ax, n_blocks, sc["fractions"], sc["guard_m"], sc["seed"], min_cells=2000,
                            max_tries=50, domains=(1,))
    part = res["partition"]
    counts = {CODES[c]: int((part == c).sum()) for c in (TRAIN, VAL, TEST, EXCL)}
    out = df.assign(partition=part.astype(np.int8), block=res["block"].astype(np.int16), s=res["s"])
    tr, ho = out[out["partition"] == TRAIN], out[out["partition"].isin([VAL, TEST])]
    ov = context_overlap(tr["row"], tr["col"], ho["row"], ho["col"], mask_shape, o)
    if ov:
        raise SystemExit(f"train and held-out contexts overlap on {ov} pixels; increase splits.guard_m")
    info = {"axis": {**ax, "length_m": axis[2]}, "n_blocks": n_blocks, "edges_s_m": res["edges"].tolist(),
            "block_codes": res["block_codes"].tolist(), "seed_used": res["seed_used"], "attempts": res["attempts"],
            "context_overlap_px": ov, "counts": counts}
    return out, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    out_dir = ccfg.out_dir(cfg)
    o = output_offset(cfg["model"]["dilations"])
    if cfg["splits"]["guard_m"] <= 2 * o * 3.0:
        raise SystemExit("splits.guard_m must exceed the context diameter (2 x offset x 3 m)")
    passing, status = [], {}
    for site in cfg["sites"]:
        sp = ccfg.site_paths(cfg, site)
        f, af = sp["out"] / "site_summary.json", sp["out"] / "audit.json"
        if not f.exists():
            status[site] = "not built"
            continue
        s, au = json.loads(f.read_text()), json.loads(af.read_text())
        ok = au["navd88"] and s["sufficient"]
        status[site] = "pass" if ok else ("not NAVD88" if not au["navd88"] else f"insufficient ({s['cells']['eligible']})")
        if ok:
            passing.append(site)
    roles_f = out_dir / "site_roles.json"
    if roles_f.exists():
        roles = json.loads(roles_f.read_text())
        print(f"roles already frozen ({roles['frozen_utc']}); reusing")
    else:
        roles = assign_roles(passing, cfg["roles"]) | {"status": status, "seed": cfg["roles"]["seed"],
                                                      "frozen_utc": fmt_utc(pd.Timestamp.now(tz="UTC")),
                                                      "config_hash": cfg["config_hash"], "git_commit": manifest.git_commit()}
        manifest.write_json_atomic(roles_f, roles)
    print(json.dumps({k: roles[k] for k in ("fit", "transfer_val", "transfer_test", "unused", "status")}, indent=1))

    rows = []
    for site in roles["fit"] + roles["transfer_val"] + roles["transfer_test"]:
        sp = ccfg.site_paths(cfg, site)
        df, spec = site_centers(cfg, site)
        shape = (spec["height"], spec["width"])
        if site in roles["fit"]:
            df, info = split_site(df, spec, cfg["splits"], shape, o)
            role = "fit"
        else:
            role = "transfer_val" if site in roles["transfer_val"] else "transfer_test"
            df = df.assign(partition=np.int8(TRANSFER_VAL if role == "transfer_val" else TRANSFER_TEST),
                           block=np.int16(-1), s=np.nan)
            info = {"counts": {role: len(df)}}
        df["role"] = role
        tmp = sp["work"] / "centers.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(sp["work"] / "centers.parquet")
        ras = np.full(shape, 255, np.uint8)
        ras[df["row"], df["col"]] = df["partition"]
        grid.write_raster(sp["work"] / "split.tif", {"partition": ras}, spec, nodata=255, dtype="uint8")
        manifest.write_json_atomic(sp["out"] / "split.json", info | {"role": role, "guard_m": cfg["splits"]["guard_m"],
                                                                      "context_offset_px": o})
        for code, g in df.groupby("partition"):
            rows.append({"site": site, "role": role, "partition": CODES[int(code)], "n": len(g),
                         "z_p05": g["z"].quantile(.05), "z_p50": g["z"].median(), "z_p95": g["z"].quantile(.95),
                         "z_std": g["z"].std()})
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        colors = {EXCL: "0.75", TRAIN: "#2e9d5b", VAL: "#e07b24", TEST: "#c43d5a", TRANSFER_VAL: "#2a6fdb",
                  TRANSFER_TEST: "#8a63c9"}
        sub = df.iloc[::max(1, len(df) // 200000)]
        for code, g in sub.groupby("partition"):
            axs[0].scatter(g["x"], g["y"], s=0.05, color=colors[int(code)], label=CODES[int(code)], rasterized=True)
            axs[1].hist(df.loc[df["partition"] == code, "z"], bins=80, histtype="step", density=True,
                        color=colors[int(code)], label=CODES[int(code)])
        axs[0].set_aspect("equal")
        axs[0].legend(markerscale=50, fontsize=7)
        axs[0].set_title(f"{site} ({role})", fontsize=9)
        axs[1].set_xlabel("target elevation (m NAVD88)")
        axs[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(sp["out"] / "split_map.png", dpi=100)
        plt.close(fig)
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "splits.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
