#!/usr/bin/env python
"""Coastal M2: build the aligned reference, lidar support and imagery masks
on the exact grid of the downloaded PlanetScope SR strip.

  prepare_coastal_reference.py [--config configs/coastal_experiment.yaml] [--strip 5730968] [--workers 8]

Grid template: the strip's SR mosaic (CRS, transform, dimensions used as-is).
Reference: per-cell area-weighted mean of the original 1 m DTM tiles (GDAL
'sum' weights, tile-ownership de-duplication), valid-area fraction of the
full cell, within-cell std; DSM aggregated likewise (stratum only).
Support: class 2/29 LAZ returns per cell; cells without measured returns are
'filled' and excluded from labels and evaluation. Domains: imagery NDWI only.

Writes data/<site>/coastal/reference/{planet_sr,planet_udm2,reference,masks}.tif,
cells.parquet, grid_spec.json and outputs/.../{tables/reference_summary.json,
figures/reference/*.png}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from island_altimetry import manifest, plotting  # plotting sets the Agg backend
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal import grid, masks
from island_altimetry.coastal.reference import (build_imagery, registration_check, rgb_stretch, strip_axis,
                                                support_grid)
from island_altimetry.config import fmt_utc

import matplotlib.pyplot as plt  # noqa: E402

NDSM_BINS = [-np.inf, 0.5, 2.0, 5.0, np.inf]
NDSM_LABELS = ["<0.5m", "0.5-2m", "2-5m", ">5m"]


def order_dir(P: dict, strip: str) -> Path:
    d = sorted(P["scenes"].glob(f"*_{strip}"))
    if not d:
        raise SystemExit(f"no downloaded scenes for strip {strip} under {P['scenes']}; run planet_order.py --coastal --download")
    return d[0]


def qa_maps(ref: dict, m: dict, rgb: np.ndarray, spec: dict, crop, out: Path) -> None:
    t = Affine(*spec["transform"])
    ext = (t.c, t.c + spec["width"] * t.a, t.f + spec["height"] * t.e, t.f)
    panels = [("RGB (red, green, blue; 2-98% stretch)", rgb, None, {}),
              ("target elevation (m NAVD88), supported cells", np.ma.masked_where(~m["ref_valid"], ref["z_target"]), "terrain",
               {"vmin": -4, "vmax": 8}),
              ("LAZ class 2/29 returns per cell", np.ma.masked_equal(ref["support"], 0), "viridis", {"norm": "log"}),
              ("Imagery domain (land / water / ambiguous)", np.ma.masked_equal(m["domain"], 0),
               plt.matplotlib.colors.ListedColormap(["#a0784c", "#2a6fdb", "#f2c14e"]), {"vmin": 1, "vmax": 3}),
              ("Filled (DTM present, no lidar return)", np.ma.masked_where(~m["filled"], m["filled"]), "Reds",
               {"vmin": 0, "vmax": 1}),
              ("DTM mean minus point mean (m)", np.ma.masked_where(~m["ref_valid"], ref["z_mean"] - ref["z_pts"]),
               "RdBu_r", {"vmin": -1, "vmax": 1})]
    fig, axs = plt.subplots(2, 3, figsize=(15, 12), sharex=True, sharey=True)
    for ax, (title, a, cmap, kw) in zip(axs.flat, panels):
        im = ax.imshow(a, extent=ext, cmap=cmap, interpolation="nearest", **kw)
        if cmap is not None:
            fig.colorbar(im, ax=ax, shrink=0.5)
        for g in getattr(crop, "geoms", [crop]):
            ax.plot(*g.exterior.xy, color="crimson", lw=0.6)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "reference_maps.png", dpi=110)
    plt.close(fig)


def transects(dtm_paths, ref, m, ndwi, spec, crop_axis, out: Path, n: int = 4, half_len: float = 1500.0) -> None:
    """Cross-strip profiles: 1 m DTM samples vs the 3 m cell means, with NDWI
    and the domain. Checks coordinate order, NoData and the aggregation."""
    from rasterio.warp import transform as wtransform
    t = Affine(*spec["transform"])
    (cx, cy), (ux, uy), length = crop_axis
    vx, vy = -uy, ux
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.6 * n))
    dtm_src = [rasterio.open(p) for p in dtm_paths]
    try:
        for k, ax in enumerate(np.atleast_1d(axs)):
            f = (k + 0.5) / n - 0.5
            s = np.arange(-half_len, half_len, 1.0)
            x0, y0 = cx + ux * f * length, cy + uy * f * length
            x, y = x0 + vx * s, y0 + vy * s
            col, row = ~t * (x, y)
            col, row = np.floor(col).astype(int), np.floor(row).astype(int)
            ok = (row >= 0) & (row < spec["height"]) & (col >= 0) & (col < spec["width"])
            z3 = np.full(len(s), np.nan)
            z3[ok] = np.where(m["ref_valid"][row[ok], col[ok]], ref["z_target"][row[ok], col[ok]], np.nan)
            dx, dy = wtransform(CRS.from_wkt(spec["crs"]), dtm_src[0].crs, list(x), list(y))
            z1 = np.full(len(s), np.nan)
            for d in dtm_src:
                v = np.array([p[0] for p in d.sample(zip(dx, dy))], np.float64)
                b = d.bounds
                inside = (np.array(dx) >= b.left) & (np.array(dx) < b.right) & (np.array(dy) > b.bottom) & (np.array(dy) <= b.top)
                use = inside & (v != d.nodata) & np.isnan(z1)
                z1[use] = v[use]
            ax.plot(s, z1, color="0.6", lw=0.6, label="1 m DTM (incl. filled)")
            ax.plot(s, z3, color="k", lw=1.2, label="3 m target (supported cells)")
            ax.set_ylabel("z (m)")
            ax2 = ax.twinx()
            nd = np.full(len(s), np.nan)
            nd[ok] = ndwi[row[ok], col[ok]]
            ax2.plot(s, nd, color="#2a6fdb", lw=0.6, alpha=0.7)
            ax2.set_ylabel("NDWI", color="#2a6fdb")
            ax.set_title(f"transect {k + 1}: cross-strip at {f:+.2f} of strip length", fontsize=9)
            if k == 0:
                ax.legend(fontsize=7, loc="upper left")
        np.atleast_1d(axs)[-1].set_xlabel("distance across strip (m)")
    finally:
        for d in dtm_src:
            d.close()
    fig.tight_layout()
    fig.savefig(out / "transects.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    ap.add_argument("--strip", help="Planet strip id (default: imagery.planet.strip_id in config)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D, src, rc, mc = ccfg.dirs(cfg), cfg["sources"], cfg["reference"], cfg["masks"]
    strip = a.strip or cfg["imagery"]["planet"]["strip_id"]
    rdir = D["work"] / "reference"
    fig_dir = D["figures"] / "reference"
    P = ccfg.planet_paths(cfg)

    spec, img_info = build_imagery(order_dir(P, strip), rdir)
    if not 2.5 <= spec["res"][0] <= 3.5 or spec["res"][0] != spec["res"][1]:
        raise SystemExit(f"Planet grid resolution {spec['res']} is not ~3 m square; not relabelling it")
    manifest.write_json_atomic(rdir / "grid_spec.json", spec)
    print(f"grid: EPSG:{spec['epsg']} {spec['width']}x{spec['height']} @ {spec['res']} m")

    dtm = sorted(Path(src["dtm_dir"]).glob(src["tile_glob"] + ".tif"))
    dsm = sorted(Path(src["dsm_dir"]).glob(src["tile_glob"] + ".tif"))
    laz = sorted(Path(src["laz_dir"]).glob(src["tile_glob"] + ".laz"))
    ref = grid.aggregate_tiles(dtm, src["dtm_nodata"], spec, workers=a.workers)
    print("DTM aggregated")
    ds = grid.aggregate_tiles(dsm, src["dtm_nodata"], spec, workers=a.workers)
    print("DSM aggregated")
    from island_altimetry.coastal.lidar_support import header_info
    laz_crs = f"EPSG:{header_info(laz[0])['horizontal_epsg']}"
    pts, hist = support_grid(laz, spec, src["support_classes"], laz_crs, a.workers)
    ref.update(pts)
    print("LAZ support binned")
    with np.errstate(invalid="ignore"):
        ref["ndsm"] = (ds["z_mean"] - ref["z_mean"]).astype(np.float32)

    with rasterio.open(rdir / "planet_sr.tif") as s:
        sr = s.read()
        names = list(s.descriptions)
    with rasterio.open(rdir / "planet_udm2.tif") as s:
        udm = s.read()
    crop = grid.read_geojson(D["work"] / "crop.geojson", CRS.from_wkt(spec["crs"]))
    in_crop = ~geometry_mask([mapping(crop)], out_shape=(spec["height"], spec["width"]),
                             transform=Affine(*spec["transform"]))
    img_ok = masks.image_valid(sr, udm) & in_crop
    w = masks.ndwi(sr[names.index("green")], sr[names.index("nir")])
    nir = sr[names.index("nir")] * 1e-4
    if mc["rule"] == "nir":
        th = masks.nir_thresholds(nir[img_ok])
        dom = masks.surface_domain_nir(nir, w, img_ok, th["water_nir_max"], th["land_nir_min"],
                                       mc["water_ndwi_min"], mc["shoreline_buffer_px"])
    else:
        th = {"ndwi_water_min": mc["ndwi_water_min"], "ndwi_land_max": mc["ndwi_land_max"]}
        dom = masks.surface_domain(w, img_ok, mc["ndwi_water_min"], mc["ndwi_land_max"], mc["shoreline_buffer_px"])
    print(f"domain rule {mc['rule']}: {th}")
    measured = ref["support"] >= rc["min_returns"]
    if rc["target"] == "point_mean":
        ref["z_target"], ref["z_target_std"] = ref["z_pts"], ref["z_pts_std"]
        frac_ok = np.isfinite(ref["z_pts"])
    elif rc["target"] == "dtm_mean":
        ref["z_target"], ref["z_target_std"] = ref["z_mean"], ref["z_std"]
        frac_ok = (ref["valid_frac"] >= rc["min_valid_fraction"]) & np.isfinite(ref["z_mean"])
    else:
        raise SystemExit(f"unknown reference.target {rc['target']}")
    dtm_ok = (ref["valid_frac"] >= rc["min_valid_fraction"]) & np.isfinite(ref["z_mean"])
    M = {"in_crop": in_crop, "image_valid": img_ok, "domain": dom,
         "ref_valid": in_crop & frac_ok & measured, "filled": in_crop & dtm_ok & ~measured}
    usable = M["ref_valid"] & img_ok & (dom != masks.NONE)

    grid.write_raster(rdir / "reference.tif", {k: ref[k] for k in ("z_target", "z_target_std", "z_mean", "valid_frac", "z_std", "ndsm", "z_pts", "z_pts_std")},
                      spec, nodata=np.nan, dtype="float32")
    grid.write_raster(rdir / "support.tif", {"support": ref["support"]}, spec, dtype="uint32")
    grid.write_raster(rdir / "masks.tif", {k: v.astype(np.uint8) for k, v in M.items()}, spec, dtype="uint8")
    grid.write_raster(rdir / "ndwi.tif", {"ndwi": w}, spec, nodata=np.nan, dtype="float32")
    grid.assert_same_grid([rdir / f for f in ("planet_sr.tif", "planet_udm2.tif", "reference.tif", "support.tif",
                                              "masks.tif", "ndwi.tif")], spec)

    r, c = np.nonzero(usable)
    t = Affine(*spec["transform"])
    x, y = t * (c + 0.5, r + 0.5)
    cells = pd.DataFrame({"cell_id": (r.astype(np.int64) * spec["width"] + c), "row": r.astype(np.int32),
                          "col": c.astype(np.int32), "x": x, "y": y, "z": ref["z_target"][r, c],
                          "z_std": ref["z_target_std"][r, c], "z_dtm": ref["z_mean"][r, c],
                          "z_dtm_std": ref["z_std"][r, c], "valid_frac": ref["valid_frac"][r, c],
                          "support": ref["support"][r, c], "ndsm": ref["ndsm"][r, c], "domain": dom[r, c],
                          "ndwi": w[r, c]})
    cells["ndsm_class"] = pd.cut(cells["ndsm"], NDSM_BINS, labels=NDSM_LABELS).astype(str)
    cells["dtm_minus_target"] = cells["z_dtm"] - cells["z"]
    tmp = rdir / "cells.parquet.tmp"
    cells.to_parquet(tmp, index=False)
    tmp.replace(rdir / "cells.parquet")

    area = abs(t.a * t.e) / 1e6
    in_c = int(in_crop.sum())
    dsum = {masks.DOMAIN_NAMES[d]: {"n_usable": int((cells["domain"] == d).sum()),
                                    "z_range": [float(cells.loc[cells["domain"] == d, "z"].min()),
                                                float(cells.loc[cells["domain"] == d, "z"].max())]
                                    if (cells["domain"] == d).any() else None,
                                    "z_p05_p50_p95": cells.loc[cells["domain"] == d, "z"].quantile([.05, .5, .95]).round(3).tolist()
                                    if (cells["domain"] == d).any() else None,
                                    "n_z_le_0": int(((cells["domain"] == d) & (cells["z"] <= 0)).sum())}
            for d in (masks.LAND, masks.WATER, masks.AMBIGUOUS)}
    axis = strip_axis(usable, spec)
    reg = registration_check(np.where(M["ref_valid"], ref["z_target"], np.nan), sr[names.index("nir")],
                             usable & (dom == masks.LAND))
    summary = {
        "generated_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)), "git_commit": manifest.git_commit(),
        "config_hash": cfg["config_hash"], "strip_id": strip, "grid": {k: spec[k] for k in ("epsg", "width", "height", "res", "transform")},
        "imagery": img_info, "vertical_reference": src["vertical_datum"], "units": "m",
        "target": {"point_mean": "mean elevation of class 2/29 lidar returns per Planet cell",
                   "dtm_mean": "area-weighted mean DTM elevation per Planet cell"}[rc["target"]],
        "dtm_vs_points": {masks.DOMAIN_NAMES[d]: {
            "median_dtm_minus_points": float(np.nanmedian((ref["z_mean"] - ref["z_pts"])[usable & (dom == d)])),
            "frac_abs_gt_0.25m": float((np.abs(ref["z_mean"] - ref["z_pts"])[usable & (dom == d)] > 0.25).mean())}
            for d in (1, 2, 3)},
        "cells_in_crop": in_c, "km2_in_crop": round(in_c * area, 3),
        "excluded_in_crop": {
            "no_target_value": int((in_crop & ~frac_ok).sum()),
            "dtm_filled_below_min_returns": int(M["filled"].sum()),
            "image_invalid": int((in_crop & ~img_ok).sum()),
            "image_valid_but_domain_none": int((img_ok & (dom == masks.NONE)).sum())},
        "domain_cells_image_valid": {masks.DOMAIN_NAMES[d]: int(((dom == d) & img_ok).sum()) for d in (1, 2, 3)},
        "usable_cells": int(usable.sum()), "usable_by_domain": dsum,
        "filled_by_domain": {masks.DOMAIN_NAMES[d]: int((M["filled"] & (dom == d)).sum()) for d in (1, 2, 3)},
        "z_nonpositive_usable": int((cells["z"] <= 0).sum()),
        "laz_class_hist": hist, "domain_rule": mc | {"derived_thresholds": th},
        "strip_axis": {"centroid": axis[0], "direction": axis[1], "length_m": axis[2]},
        "registration_check": reg,
        "datum_note": "Planet grid is WGS84/UTM; DTM is NAD83(2011)/UTM. PROJ uses a null (ballpark) "
                      "datum transformation, ignoring a ~1 m horizontal NAD83(2011)-ITRF offset.",
    }
    manifest.write_json_atomic(D["tables"] / "reference_summary.json", summary)
    rgb = rgb_stretch(sr, names, img_ok | in_crop)
    qa_maps(ref, M, rgb, spec, crop, fig_dir)
    transects(dtm, ref, M, w, spec, axis, fig_dir)
    print(json.dumps({k: summary[k] for k in ("km2_in_crop", "excluded_in_crop", "domain_cells_image_valid",
                                              "usable_cells", "usable_by_domain", "registration_check")}, indent=1))


if __name__ == "__main__":
    main()
