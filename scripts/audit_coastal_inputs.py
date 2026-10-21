#!/usr/bin/env python
"""Coastal M1a: audit the reference DTM/DSM tiles and source LAZ for the
configured AOI (no network). Raw data are only read.

  audit_coastal_inputs.py [--config configs/coastal_experiment.yaml] [--workers 8]

Per-tile inventory (CRS, grid, NoData, statistics; LAZ CRS incl. vertical
datum, class histogram, acquisition time from GPS time), a coarse
return-support raster and the surveyed-strip footprint polygon (WGS84 GeoJSON,
used as the Planet search/coverage AOI). Writes
data/<site>/coastal/audit/{support_<n>m.tif,footprint.geojson},
outputs/<site>/coastal_rf/{source_manifest.json,tables/audit_tiles.csv,
figures/audit_overview.png}.
"""

from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape

from island_altimetry import manifest, plotting  # plotting sets the Agg backend
from island_altimetry.coastal import config as ccfg
from island_altimetry.coastal.grid import write_geojson
from island_altimetry.coastal.lidar_support import bin_returns, gps_to_utc, header_info
from island_altimetry.config import fmt_utc

import matplotlib.pyplot as plt  # noqa: E402


def tile_key(p: Path) -> str:
    return p.name.split(".")[0]


def raster_row(p: Path, nodata_expected: float) -> dict:
    with rasterio.open(p) as s:
        a = s.read(1, masked=True)
        return {"tile": tile_key(p), "crs_epsg": s.crs.to_epsg() if s.crs else None, "width": s.width,
                "height": s.height, "res_x": s.res[0], "res_y": s.res[1], "left": s.bounds.left,
                "bottom": s.bounds.bottom, "right": s.bounds.right, "top": s.bounds.top, "nodata": s.nodata,
                "nodata_matches": s.nodata == nodata_expected, "dtype": s.dtypes[0],
                "valid_frac": float(1 - np.ma.getmaskarray(a).mean()),
                "z_min": float(a.min()), "z_p50": float(np.ma.median(a)), "z_max": float(a.max()),
                "n_zero": int((a == 0).sum()), "n_negative": int((a < 0).sum())}


def laz_row(args) -> tuple[dict, np.ndarray]:
    p, transform, shp, classes = args
    h = header_info(p)
    b = bin_returns(p, transform, shp, classes)
    t0, t1 = b["gps_range"]
    hist = {int(k): int(v) for k, v in enumerate(b["class_hist"]) if v}
    row = {"tile": tile_key(p), **{k: v for k, v in h.items() if k not in ("bounds", "z_range", "sub_crs")},
           "sub_crs": " + ".join(h["sub_crs"]), "laz_z_min": h["z_range"][0], "laz_z_max": h["z_range"][1],
           "class_hist": ";".join(f"{k}:{v}" for k, v in hist.items()),
           "n_support": int(sum(hist.get(c, 0) for c in classes)),
           "acq_start_utc": fmt_utc(gps_to_utc(t0, h["adjusted_gps"])) if np.isfinite(t0) else None,
           "acq_end_utc": fmt_utc(gps_to_utc(t1, h["adjusted_gps"])) if np.isfinite(t1) else None}
    return row, b["counts"]


def footprint(support: np.ndarray, transform: Affine, min_returns: int, cell: float):
    """Surveyed area: cells with >= min_returns support returns, closed by one
    cell to bridge isolated empty cells, then simplified."""
    m = (support >= min_returns).astype("uint8")
    polys = [shape(g) for g, v in shapes(m, mask=m.astype(bool), transform=transform) if v == 1]
    g = shapely.union_all(polys).buffer(cell).buffer(-cell)
    return g.simplify(cell / 2)


def overview(dtm_paths, support, s_tr, fp, out_png, step=20):
    fig, axs = plt.subplots(1, 2, figsize=(11, 8), sharex=True, sharey=True)
    for p in dtm_paths:
        with rasterio.open(p) as s:
            a = s.read(1, masked=True)[::step, ::step]
            b = s.bounds
            im = axs[0].imshow(a, extent=(b.left, b.right, b.bottom, b.top), cmap="terrain", vmin=-6, vmax=8)
    fig.colorbar(im, ax=axs[0], shrink=0.6, label="DTM elevation (m, NAVD88)")
    h, w = support.shape
    ext = (s_tr.c, s_tr.c + w * s_tr.a, s_tr.f + h * s_tr.e, s_tr.f)
    im2 = axs[1].imshow(np.ma.masked_equal(support, 0), extent=ext, cmap="viridis", norm="log")
    fig.colorbar(im2, ax=axs[1], shrink=0.6, label="class 2/29 returns per cell")
    for ax in axs:
        for gg in getattr(fp, "geoms", [fp]):
            ax.plot(*gg.exterior.xy, color="crimson", lw=0.8)
        ax.set_xlabel("UTM 16N easting (m)")
    axs[0].set_ylabel("northing (m)")
    axs[0].set_title("DTM (1 m, decimated)")
    axs[1].set_title("LAZ ground+bathy support; red = footprint")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.COASTAL))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    D, src = ccfg.dirs(cfg), cfg["sources"]
    audit_dir = D["work"] / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    D["tables"].mkdir(parents=True, exist_ok=True)

    dtm = sorted(Path(src["dtm_dir"]).glob(src["tile_glob"] + ".tif"))
    dsm = sorted(Path(src["dsm_dir"]).glob(src["tile_glob"] + ".tif")) if src.get("dsm_dir") else []
    laz = sorted(Path(src["laz_dir"]).glob(src["tile_glob"] + ".laz"))
    keys = {"dtm": {tile_key(p) for p in dtm}, "dsm": {tile_key(p) for p in dsm}, "laz": {tile_key(p) for p in laz}}
    print(f"tiles: dtm {len(dtm)}, dsm {len(dsm)}, laz {len(laz)}")
    if not dtm:
        raise SystemExit(f"no DTM tiles match {src['tile_glob']} in {src['dtm_dir']}")

    rows = pd.DataFrame([raster_row(p, src["dtm_nodata"]) for p in dtm])
    dsm_rows = pd.DataFrame([raster_row(p, src["dtm_nodata"]) for p in dsm]).add_prefix("dsm_") if dsm else None

    cell = cfg["audit"]["footprint_cell_m"]
    x0, y1 = np.floor(rows["left"].min() / cell) * cell, np.ceil(rows["top"].max() / cell) * cell
    x1, y0 = np.ceil(rows["right"].max() / cell) * cell, np.floor(rows["bottom"].min() / cell) * cell
    s_tr = Affine(cell, 0, x0, 0, -cell, y1)
    shp = (int(round((y1 - y0) / cell)), int(round((x1 - x0) / cell)))
    support = np.zeros(shp, np.uint32)
    lrows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (r, counts) in enumerate(ex.map(laz_row, [(p, s_tr, shp, src["support_classes"]) for p in laz]), 1):
            lrows.append(r)
            support += counts
            if i % 10 == 0 or i == len(laz):
                print(f"  laz {i}/{len(laz)}")
    lz = pd.DataFrame(lrows)
    tiles = rows.merge(lz, on="tile", how="outer", indicator="has_laz")
    if dsm_rows is not None:
        tiles = tiles.merge(dsm_rows.rename(columns={"dsm_tile": "tile"}), on="tile", how="left")
    tiles.to_csv(D["tables"] / "audit_tiles.csv", index=False)

    with rasterio.open(audit_dir / f"support_{cell}m.tif", "w", driver="GTiff", height=shp[0], width=shp[1], count=1,
                       dtype="uint32", crs=f"EPSG:{int(rows['crs_epsg'].iloc[0])}", transform=s_tr, compress="deflate") as d:
        d.write(support, 1)
    fp = footprint(support, s_tr, cfg["audit"]["footprint_min_returns"], cell)
    crs = f"EPSG:{int(rows['crs_epsg'].iloc[0])}"
    write_geojson(fp, crs, audit_dir / "footprint.geojson",
                  {"source": "class 2/29 LAZ returns", "cell_m": cell, "area_km2": round(fp.area / 1e6, 3)})
    overview(dtm, support, s_tr, fp, D["figures"] / "audit_overview.png")

    res_set = sorted({(r.res_x, r.res_y) for r in rows.itertuples()})
    origin_off = np.abs(rows["left"] - rows["left"].round()).max()
    acq = lz[["acq_start_utc", "acq_end_utc"]].dropna()
    findings = {
        "n_tiles": {k: len(v) for k, v in keys.items()},
        "tiles_missing": {k: sorted(keys["dtm"] - v) for k, v in keys.items() if k != "dtm"},
        "dtm_crs_epsg": sorted(rows["crs_epsg"].dropna().astype(int).unique().tolist()),
        "dtm_resolution": res_set, "dtm_tile_shapes": sorted({(r.width, r.height) for r in rows.itertuples()}),
        "dtm_max_origin_offset_m": float(origin_off),
        "dtm_nodata": sorted(rows["nodata"].dropna().unique().tolist()),
        "dtm_valid_frac_min": float(rows["valid_frac"].min()),
        "dtm_z_range": [float(rows["z_min"].min()), float(rows["z_max"].max())],
        "dtm_n_zero": int(rows["n_zero"].sum()), "dtm_n_negative": int(rows["n_negative"].sum()),
        "laz_horizontal_epsg": sorted(lz["horizontal_epsg"].dropna().astype(int).unique().tolist()),
        "laz_vertical_epsg": sorted(lz["vertical_epsg"].dropna().astype(int).unique().tolist()),
        "laz_crs_names": sorted(lz["crs_name"].dropna().unique().tolist()),
        "acquisition_utc": [acq["acq_start_utc"].min(), acq["acq_end_utc"].max()] if len(acq) else None,
        "support_classes": src["support_classes"],
        "support_returns_total": int(lz["n_support"].sum()),
        "footprint_km2": round(fp.area / 1e6, 3),
        "footprint_cells_fraction_of_bbox": float((support >= cfg["audit"]["footprint_min_returns"]).mean()),
    }
    out = {"generated_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)), "git_commit": manifest.git_commit(),
           "config_hash": cfg["config_hash"], "survey": src["survey"], "sources": src, "findings": findings,
           "notes": [
               "DTM GeoTIFFs carry no vertical CRS; the vertical datum is taken from the source LAZ compound CRS.",
               "DTM publication/processing date is not the acquisition date; acquisition is from LAZ GPS time.",
               "DTM tiles report 100% valid pixels including unsurveyed areas: support must come from LAZ returns."]}
    manifest.write_json_atomic(D["out"] / "source_manifest.json", out)
    for k, v in findings.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
