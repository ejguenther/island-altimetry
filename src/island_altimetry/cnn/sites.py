"""Per-site audit, crop selection, Planet scene rule and reference/mask build
for the CNN terrain experiment. Raw DTM/DSM/LAZ are only read (no statistics
or sidecars are written next to them)."""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
from scipy import ndimage
from shapely.geometry import box, mapping, shape

from .. import manifest
from ..coastal import grid, masks
from ..coastal.lidar_support import bin_returns, gps_to_utc, header_info
from ..config import fmt_utc
from . import config as ccfg


def tile_key(p: Path) -> str:
    return p.name.split(".")[0]


def tiles(sp: dict) -> tuple[list[Path], list[Path], list[Path]]:
    g = sp["glob"]
    return (sorted(sp["dtm"].glob(g + ".tif")), sorted(sp["dsm"].glob(g + ".tif")), sorted(sp["laz"].glob(g + ".laz")))


def _tile_bounds(p: Path) -> dict:
    with rasterio.open(p) as s:
        b = s.bounds
        return {"tile": tile_key(p), "epsg": s.crs.to_epsg() if s.crs else None, "res": s.res[0], "left": b.left,
                "bottom": b.bottom, "right": b.right, "top": b.top}


def _laz_audit(args) -> tuple[dict, tuple | None]:
    p, t, shp, cls = args
    h = header_info(p)
    b = bin_returns(p, t, shp, [cls])
    t0, t1 = b["gps_range"]
    hist = {int(k): int(v) for k, v in enumerate(b["class_hist"]) if v}
    r, c = np.nonzero(b["counts"])
    win = None
    if len(r):
        sl = np.s_[r.min():r.max() + 1, c.min():c.max() + 1]
        win = (int(r.min()), int(c.min()), b["counts"][sl])
    row = {"tile": tile_key(p), "horizontal_epsg": h["horizontal_epsg"], "vertical_epsg": h["vertical_epsg"],
           "crs_name": h["crs_name"], "point_count": h["point_count"], "n_ground": hist.get(cls, 0),
           "class_hist": ";".join(f"{k}:{v}" for k, v in hist.items()),
           "acq_start_utc": fmt_utc(gps_to_utc(t0, h["adjusted_gps"])) if np.isfinite(t0) else None,
           "acq_end_utc": fmt_utc(gps_to_utc(t1, h["adjusted_gps"])) if np.isfinite(t1) else None}
    return row, win


def densest_window(mask: np.ndarray, side: int) -> tuple[int, int]:
    """Top-left (row, col) of the side x side window with the most True cells."""
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = mask.shape
    s_r, s_c = min(side, h), min(side, w)
    tot = ii[s_r:, s_c:] - ii[:-s_r, s_c:] - ii[s_r:, :-s_c] + ii[:-s_r, :-s_c]
    r, c = np.unravel_index(np.argmax(tot), tot.shape)
    return int(r), int(c)


def audit_site(cfg: dict, site: str, workers: int = 8) -> dict:
    """Tile/LAZ audit, 30 m ground-support raster, ground footprint and the
    frozen land-rich crop (crop.geojson is never rewritten once present)."""
    sp, ac = ccfg.site_paths(cfg, site), cfg["audit"]
    dtm, dsm, laz = tiles(sp)
    adir = sp["work"] / "audit"
    adir.mkdir(parents=True, exist_ok=True)
    sp["out"].mkdir(parents=True, exist_ok=True)
    tb = pd.DataFrame([_tile_bounds(p) for p in dtm])
    epsg = sorted(tb["epsg"].dropna().unique().astype(int).tolist())
    if len(epsg) != 1:
        raise SystemExit(f"{site}: DTM tiles in several CRSs {epsg}")
    cell = ac["support_cell_m"]
    x0, y1 = np.floor(tb["left"].min() / cell) * cell, np.ceil(tb["top"].max() / cell) * cell
    x1, y0 = np.ceil(tb["right"].max() / cell) * cell, np.floor(tb["bottom"].min() / cell) * cell
    t = Affine(cell, 0, x0, 0, -cell, y1)
    shp = (int(round((y1 - y0) / cell)), int(round((x1 - x0) / cell)))
    support = np.zeros(shp, np.uint32)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (r, win) in enumerate(ex.map(_laz_audit, [(p, t, shp, cfg["ground_class"]) for p in laz]), 1):
            rows.append(r)
            if win is not None:
                a, b, cnt = win
                support[a:a + cnt.shape[0], b:b + cnt.shape[1]] += cnt
            if i % 100 == 0:
                print(f"  {site}: laz {i}/{len(laz)}")
    lz = pd.DataFrame(rows)
    tab = tb.merge(lz, on="tile", how="outer", indicator="sources")
    tab.to_csv(sp["out"] / "audit_tiles.csv", index=False)
    crs = f"EPSG:{epsg[0]}"
    with rasterio.open(adir / f"support_{cell}m.tif", "w", driver="GTiff", height=shp[0], width=shp[1], count=1,
                       dtype="uint32", crs=crs, transform=t, compress="deflate") as d:
        d.write(support, 1)

    ground = support >= ac["support_min_returns"]
    polys = [shape(g) for g, v in shapes(ground.astype("uint8"), mask=ground, transform=t) if v == 1]
    fp = shapely.union_all(polys).buffer(cell).buffer(-cell).simplify(cell / 2) if polys else shapely.Polygon()
    grid.write_geojson(fp, crs, adir / "ground_footprint.geojson", {"cell_m": cell, "km2": round(fp.area / 1e6, 3)})
    crop_f = sp["work"] / "crop.geojson"
    windowed = False
    if not crop_f.exists():
        if site_strip(cfg, site):
            raise SystemExit(f"{site}: existing scene configured but no crop at {crop_f}; copy the order clip first")
        region = fp
        if fp.area / 1e6 > ac["crop_max_km2"] or cfg["sites"][site].get("force_window"):
            side = int(np.sqrt(ac["crop_max_km2"] * 1e6) / cell)
            r, c = densest_window(ground, side)
            wx0, wy1 = t * (c, r)
            region = fp.intersection(box(wx0, wy1 - side * cell, wx0 + side * cell, wy1))
            windowed = True
        crop = region.buffer(ac["crop_buffer_m"]).simplify(10)
        if windowed:
            crop = crop.intersection(box(*region.bounds).buffer(ac["crop_buffer_m"]))
        grid.write_geojson(crop, crs, crop_f, {"source": "LAZ class-2 footprint", "windowed": windowed,
                                               "km2": round(crop.area / 1e6, 3), "frozen_utc": fmt_utc(_now())})
    crop = grid.read_geojson(crop_f, crs)
    # survey times of tiles touching the crop
    tb_geo = [box(r.left, r.bottom, r.right, r.top) for r in tb.itertuples()]
    inside = tb["tile"][[g.intersects(crop) for g in tb_geo]]
    acq = lz[lz["tile"].isin(inside)][["acq_start_utc", "acq_end_utc"]].dropna()
    st, en = pd.to_datetime(acq["acq_start_utc"]), pd.to_datetime(acq["acq_end_utc"])
    mid = st + (en - st) / 2
    vert = sorted(lz["vertical_epsg"].dropna().astype(int).unique().tolist())
    out = {"site": site, "source": cfg["sites"][site], "generated_utc": fmt_utc(_now()), "git_commit": manifest.git_commit(),
           "n_tiles": {"dtm": len(dtm), "dsm": len(dsm), "laz": len(laz)},
           "tiles_missing_laz": sorted(set(tb["tile"]) - set(lz["tile"])), "dtm_epsg": epsg[0],
           "laz_horizontal_epsg": sorted(lz["horizontal_epsg"].dropna().astype(int).unique().tolist()),
           "vertical_epsg": vert, "navd88": vert == [cfg["vertical_datum_epsg"]],
           "survey_utc": [fmt_utc(st.min()), fmt_utc(en.max())] if len(acq) else None,
           "survey_median_utc": fmt_utc(mid.median()) if len(acq) else None,
           "ground_returns": int(lz["n_ground"].sum()), "ground_footprint_km2": round(fp.area / 1e6, 3),
           "crop_km2": round(crop.area / 1e6, 3), "crop_windowed": windowed, "n_tiles_in_crop": int(len(inside))}
    manifest.write_json_atomic(sp["out"] / "audit.json", out)
    return out


def site_strip(cfg: dict, site: str) -> str | None:
    return cfg["sites"][site].get("strip_id")


def _now():
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------- Planet selection rule

def select_scene(strips: pd.DataFrame, survey: tuple[pd.Timestamp, pd.Timestamp], ic: dict) -> tuple[dict | None, pd.DataFrame]:
    """Predeclared rule: full crop footprint, AOI clear >= aoi_clear_min, an
    allowed SR bundle; then the smallest time gap to the survey interval (0
    inside it); ties -> preferred bundle order, then higher clear."""
    s = strips.copy()
    t0, t1 = survey
    acq = pd.to_datetime(s["acquired"], utc=True)
    s["gap_days"] = np.where(acq < t0, (t0 - acq).dt.total_seconds(), np.where(acq > t1, (acq - t1).dt.total_seconds(), 0)) / 86400
    reasons = []
    for r in s.itertuples():
        why = []
        if r.aoi_footprint < ic["aoi_footprint_min"]:
            why.append(f"footprint {r.aoi_footprint:.2f}")
        if r.aoi_clear is None or not np.isfinite(r.aoi_clear) or r.aoi_clear < ic["aoi_clear_min"]:
            why.append(f"clear {r.aoi_clear}")
        if r.bundle not in ic["bundles"]:
            why.append(f"bundle {r.bundle}")
        reasons.append("; ".join(why) or "eligible")
    s["status"] = reasons
    ok = s[s["status"] == "eligible"].copy()
    if ok.empty:
        return None, s
    ok["bundle_rank"] = ok["bundle"].map({b: i for i, b in enumerate(ic["bundles"])})
    best = ok.sort_values(["gap_days", "bundle_rank", "aoi_clear"], ascending=[True, True, False]).iloc[0]
    s.loc[s["strip_id"] == best["strip_id"], "status"] = "selected"
    return best.to_dict(), s


# ---------------------------------------------------------------- reference build

def context_valid(image_valid: np.ndarray, rf: int) -> np.ndarray:
    """True where every pixel of the rf x rf window centred on the cell is
    image-valid (outside the raster counts as invalid)."""
    bad = (~image_valid).astype(np.float32)
    n = ndimage.uniform_filter(bad, rf, mode="constant", cval=1.0) * rf * rf
    return n < 0.5


def crop_mask(crop, spec: dict) -> np.ndarray:
    return ~geometry_mask([mapping(crop)], out_shape=(spec["height"], spec["width"]), transform=Affine(*spec["transform"]))


def build_site(cfg: dict, site: str, scene_dir: Path, workers: int = 8) -> dict:
    """Exact-grid SR/UDM2 (common bands), DTM mean / valid fraction / std, DSM
    mean, LAZ class-2 support / point mean / point std, imagery domains and
    context validity; then finalize_site() derives the target and center masks."""
    from ..coastal.reference import build_imagery, support_grid
    from .model import receptive_field
    sp, tc = ccfg.site_paths(cfg, site), cfg["target"]
    rdir = sp["work"] / "reference"
    spec, info = build_imagery(scene_dir, rdir, cfg["imagery"]["bands"])
    if not 2.5 <= spec["res"][0] <= 3.5 or spec["res"][0] != spec["res"][1]:
        raise SystemExit(f"{site}: Planet grid resolution {spec['res']} is not ~3 m square")
    manifest.write_json_atomic(rdir / "grid_spec.json", spec)
    dtm, dsm, laz = tiles(sp)
    ref = grid.aggregate_tiles(dtm, cfg["dtm_nodata"], spec, workers=workers)
    ds = grid.aggregate_tiles(dsm, cfg["dtm_nodata"], spec, workers=workers)
    laz_crs = f"EPSG:{header_info(laz[0])['horizontal_epsg']}"
    pts, hist = support_grid(laz, spec, [cfg["ground_class"]], laz_crs, workers)
    with rasterio.open(rdir / "planet_sr.tif") as s:
        sr, names = s.read(), list(s.descriptions)
    with rasterio.open(rdir / "planet_udm2.tif") as s:
        udm = s.read()
    crop = grid.read_geojson(sp["work"] / "crop.geojson", CRS.from_wkt(spec["crs"]))
    in_crop = crop_mask(crop, spec)
    img_ok = masks.image_valid(sr, udm) & in_crop
    nir = sr[names.index("nir")] * 1e-4
    w = masks.ndwi(sr[names.index("green")], sr[names.index("nir")])
    th = masks.nir_thresholds(nir[img_ok])
    dom = masks.surface_domain_nir(nir, w, img_ok, th["water_nir_max"], th["land_nir_min"], tc["water_ndwi_min"],
                                   tc["shoreline_buffer_px"])
    ctx = context_valid(img_ok, receptive_field(cfg["model"]["dilations"]))
    grid.write_raster(rdir / "reference.tif", {"z_mean": ref["z_mean"], "valid_frac": ref["valid_frac"],
                                               "z_std": ref["z_std"], "dsm_mean": ds["z_mean"], "z_pts": pts["z_pts"],
                                               "z_pts_std": pts["z_pts_std"]}, spec, nodata=np.nan, dtype="float32")
    grid.write_raster(rdir / "support.tif", {"support": pts["support"]}, spec, dtype="uint32")
    grid.write_raster(rdir / "base_masks.tif", {"in_crop": in_crop.astype(np.uint8), "image_valid": img_ok.astype(np.uint8),
                                                "domain": dom, "context_valid": ctx.astype(np.uint8)}, spec, dtype="uint8")
    manifest.write_json_atomic(rdir / "build_info.json", {"imagery": info, "land_rule": th, "laz_class_hist": hist,
                                                          "generated_utc": fmt_utc(_now())})
    return finalize_site(cfg, site)


def finalize_site(cfg: dict, site: str) -> dict:
    """Target, nDSM and center eligibility from the stored rasters.
    target.definition 'point_mean' (user decision 2026-10-22): mean of class-2
    ground returns per cell with >= min_ground_returns; 'dtm_mean': the DTM
    area-weighted mean with valid fraction >= min_valid_fraction and the same
    ground support. nDSM = DSM mean - target."""
    from ..coastal.reference import registration_check
    sp, tc = ccfg.site_paths(cfg, site), cfg["target"]
    rdir = sp["work"] / "reference"
    spec = json.loads((rdir / "grid_spec.json").read_text())
    R = {k: grid.read_band(rdir / "reference.tif", k) for k in ("z_mean", "valid_frac", "z_std", "z_pts", "z_pts_std")}
    with rasterio.open(rdir / "reference.tif") as s:
        has_dsm = "dsm_mean" in s.descriptions
    dsm = grid.read_band(rdir / "reference.tif", "dsm_mean") if has_dsm else \
        grid.read_band(rdir / "reference.tif", "ndsm") + R["z_mean"]     # first builds stored DTM-based nDSM
    sup = grid.read_band(rdir / "support.tif")
    bm = rdir / "base_masks.tif" if (rdir / "base_masks.tif").exists() else rdir / "masks.tif"
    B = {k: grid.read_band(bm, k) for k in ("in_crop", "image_valid", "domain", "context_valid")}
    in_crop, img_ok, ctx = (B[k].astype(bool) for k in ("in_crop", "image_valid", "context_valid"))
    dom = B["domain"]
    supported = sup >= tc["min_ground_returns"]
    if tc["definition"] == "point_mean":
        z, zs = R["z_pts"], R["z_pts_std"]
        ref_ok = supported & np.isfinite(z)
    elif tc["definition"] == "dtm_mean":
        z, zs = R["z_mean"], R["z_std"]
        ref_ok = supported & np.isfinite(z) & (R["valid_frac"] >= tc["min_valid_fraction"])
    else:
        raise SystemExit(f"unknown target.definition {tc['definition']}")
    with np.errstate(invalid="ignore"):
        ndsm = (dsm - z).astype(np.float32)
    bare = np.isfinite(ndsm) & (ndsm <= tc["max_ndsm_m"])
    land = dom == masks.LAND
    eligible = in_crop & land & ref_ok & bare & ctx
    M = {"in_crop": in_crop, "image_valid": img_ok, "domain": dom, "context_valid": ctx, "ref_ok": ref_ok,
         "bare": bare, "eligible": eligible}
    grid.write_raster(rdir / "target.tif", {"z_target": z, "z_target_std": zs, "ndsm": ndsm}, spec, nodata=np.nan,
                      dtype="float32")
    if not (rdir / "base_masks.tif").exists():
        grid.write_raster(rdir / "base_masks.tif", {k: (B[k] if k == "domain" else B[k].astype(np.uint8))
                                                    for k in B}, spec, dtype="uint8")
    grid.write_raster(rdir / "masks.tif", {k: v.astype(np.uint8) for k, v in M.items()}, spec, dtype="uint8")
    grid.assert_same_grid([rdir / f for f in ("planet_sr.tif", "planet_udm2.tif", "reference.tif", "support.tif",
                                              "target.tif", "masks.tif")], spec)
    with rasterio.open(rdir / "planet_sr.tif") as s:
        sr, names = s.read(), list(s.descriptions)
    old = json.loads((sp["out"] / "site_summary.json").read_text()) if (sp["out"] / "site_summary.json").exists() else {}
    binfo = json.loads((rdir / "build_info.json").read_text()) if (rdir / "build_info.json").exists() else \
        {k: old.get(k) for k in ("imagery", "land_rule", "laz_class_hist")}
    land_cand = in_crop & land & ref_ok
    ze = z[eligible]
    res_m2 = spec["res"][0] * spec["res"][1]
    d = (R["z_mean"] - R["z_pts"])[land_cand]
    summary = {
        "site": site, "generated_utc": fmt_utc(_now()), "config_hash": cfg["config_hash"],
        "target_definition": tc["definition"], "grid": {k: spec[k] for k in ("epsg", "width", "height", "res")},
        **binfo, "km2_in_crop": round(in_crop.sum() * res_m2 / 1e6, 3),
        "cells": {"image_valid": int(img_ok.sum()), "land": int((land & img_ok).sum()),
                  "land_ref_ok": int(land_cand.sum()), "land_ref_ok_not_bare": int((land_cand & ~bare).sum()),
                  "land_ref_ok_bare_context_invalid": int((land_cand & bare & ~ctx).sum()),
                  "eligible": int(eligible.sum())},
        "eligible_km2": round(eligible.sum() * res_m2 / 1e6, 3),
        "eligible_z": {"min": float(ze.min()), "p05": float(np.percentile(ze, 5)), "p50": float(np.median(ze)),
                       "p95": float(np.percentile(ze, 95)), "max": float(ze.max()), "std": float(ze.std())} if len(ze) else None,
        "sufficient": bool(eligible.sum() >= tc["min_eligible_centers"]),
        "dtm_minus_points_land": {"median": float(np.nanmedian(d)), "frac_abs_gt_0.25": float(np.nanmean(np.abs(d) > 0.25))}
        if land_cand.any() else None,
        "ndsm_land_p50_p95": [float(np.nanpercentile(ndsm[land_cand], 50)), float(np.nanpercentile(ndsm[land_cand], 95))]
        if land_cand.any() else None,
        "registration_check": registration_check(np.where(ref_ok, z, np.nan), sr[names.index("nir")], eligible)
        if eligible.sum() > 1000 else None,
    }
    manifest.write_json_atomic(sp["out"] / "site_summary.json", summary)
    crop = grid.read_geojson(sp["work"] / "crop.geojson", CRS.from_wkt(spec["crs"]))
    qa_figure(sr, names, z, M, spec, crop, sp["out"] / "site_qa.png", site)
    return summary


def qa_figure(sr, names, z, M, spec, crop, out: Path, site: str) -> None:
    from .. import plotting  # noqa: F401 - Agg backend
    import matplotlib.pyplot as plt
    from ..coastal.reference import rgb_stretch
    t = Affine(*spec["transform"])
    ext = (t.c, t.c + spec["width"] * t.a, t.f + spec["height"] * t.e, t.f)
    rgb = rgb_stretch(sr, names, M["in_crop"])
    fig, axs = plt.subplots(1, 4, figsize=(20, 6), sharex=True, sharey=True)
    axs[0].imshow(rgb, extent=ext)
    axs[0].set_title(f"{site}: RGB", fontsize=9)
    im = axs[1].imshow(np.ma.masked_where(~M["ref_ok"] | ~M["in_crop"], z), extent=ext, cmap="terrain", vmin=-2,
                       vmax=float(np.nanpercentile(z[M["eligible"]], 99)) if M["eligible"].any() else 5)
    fig.colorbar(im, ax=axs[1], shrink=0.6, label="target (m NAVD88)")
    axs[1].set_title("target where ground-supported", fontsize=9)
    axs[2].imshow(np.ma.masked_equal(M["domain"], 0), extent=ext, cmap=plt.matplotlib.colors.ListedColormap(
        ["#a0784c", "#2a6fdb", "#f2c14e"]), vmin=1, vmax=3, interpolation="nearest")
    axs[2].set_title("imagery domain: land / water / ambiguous", fontsize=9)
    axs[3].imshow(np.ma.masked_where(~M["eligible"], M["eligible"]), extent=ext, cmap="Greens", vmin=0, vmax=1,
                  interpolation="nearest")
    axs[3].set_title("eligible centers (land, ground-supported, nDSM<=1 m, full context)", fontsize=9)
    for ax in axs:
        for g in getattr(crop, "geoms", [crop]):
            ax.plot(*g.exterior.xy, color="crimson", lw=0.6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
