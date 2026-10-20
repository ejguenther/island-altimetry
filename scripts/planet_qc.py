#!/usr/bin/env python
"""M8c: QC downloaded PlanetScope strips (no network).

  planet_qc.py --site fukutoku_okanoba [--run-id RUN] [--no-quicklooks]

For every downloaded order: measured clear fraction from UDM2 band 1 over the
core and outer AOIs (pixels outside the strip footprint excluded), compared
with the pre-order coverage estimate; RGB quicklooks with the matched ATL03
core centerlines / GEDI footprints overlaid. Writes
outputs/<site>/<run>/tables/planet_scene_qc.csv and figures/planet/*.png.
UDM2 can label pumice rafts or glint as haze/cloud: low measured clear
fractions after the 2021 eruption need visual review, not automatic rejection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.merge import merge
from pyproj import Transformer
from shapely.geometry import mapping, shape
import shapely

from island_altimetry import config, manifest, plotting  # plotting also sets the Agg backend
from island_altimetry import planet_imagery as pim
from island_altimetry.aoi import aeqd_crs

import matplotlib.pyplot as plt  # noqa: E402

RGB_BANDS = {8: (6, 4, 2), 4: (3, 2, 1), 3: (1, 2, 3)}   # 1-based: 8-band, 4-band (BGRN), 3-band (RGB)


def files(dest: Path, kind: str) -> list[Path]:
    tifs = sorted(dest.rglob("*.tif"))
    if kind == "udm2":
        return [f for f in tifs if "udm2" in f.name.lower()]
    return [f for f in tifs if "analytic" in f.name.lower() and "udm" not in f.name.lower()]


UDM_NODATA = 255


def mosaic_udm2(paths: list[Path]):
    """Mosaic UDM2 strips. 0 is a valid UDM2 value ("not clear"), so blackfill
    pixels (band 8 bit 0) are recoded to 255 and merged with nodata=255; a
    nodata=0 merge would let one scene's zeros be overwritten by another's."""
    from rasterio.io import MemoryFile
    mems, srcs = [], []
    try:
        for f in paths:
            with rasterio.open(f) as s:
                a, prof = s.read(), s.profile
            a = a.astype("uint8")
            a[:, (a[7] & 1) == 1] = UDM_NODATA
            prof.update(dtype="uint8", nodata=UDM_NODATA)
            m = MemoryFile()
            with m.open(**prof) as d:
                d.write(a)
            mems.append(m)
            srcs.append(m.open())
        arr, tr = merge(srcs, nodata=UDM_NODATA)
        return arr, tr, srcs[0].crs
    finally:
        for s in srcs:
            s.close()
        for m in mems:
            m.close()


def mosaic(paths: list[Path], bands=None):
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, tr = merge(srcs, indexes=bands, nodata=0)
        return arr, tr, srcs[0].crs
    finally:
        for s in srcs:
            s.close()


def clear_fraction(udm, transform, crs, aoi_wgs84) -> tuple[float | None, float]:
    """(clear fraction of AOI pixels inside the footprint, footprint fraction)."""
    to = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    g = shapely.transform(aoi_wgs84, lambda xy: np.column_stack(to.transform(xy[:, 0], xy[:, 1])))
    inside = ~geometry_mask([mapping(g)], out_shape=udm.shape[1:], transform=transform)
    valid = inside & (udm[7] != UDM_NODATA)   # blackfill / outside every scene
    n_in, n_valid = int(inside.sum()), int(valid.sum())
    clear = float((udm[0][valid] == 1).mean()) if n_valid else None
    return clear, (n_valid / n_in if n_in else 0.0)


def stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape[1:] + (3,), dtype=float)
    for i in range(3):
        b = rgb[i].astype(float)
        v = b[b > 0]
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1e-9), 0, 1)
    return out


def quicklook(sr_paths, rec, targets, overlays, lonlat, out_png):
    with rasterio.open(sr_paths[0]) as s:
        nb = s.count
    arr, tr, crs = mosaic(sr_paths, bands=list(RGB_BANDS.get(nb, (3, 2, 1))))
    img = stretch(arr)
    h, w = img.shape[:2]
    ext = (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=ext)
    to = Transformer.from_crs(aeqd_crs(*lonlat), crs, always_xy=True)
    cl, fp = overlays
    for t in targets:
        pid, gran = t.split(":", 1)
        if pid == "atl03" and not cl.empty:
            c = cl[cl["granule"] == gran]
            for gt, cc in c.groupby("gt"):
                x, y = to.transform(cc["x"].to_numpy(), cc["y"].to_numpy())
                ax.plot(x, y, ".", ms=1.5, color=plotting.COLORS["atl03"])
        if pid == "gedi_l2a" and not fp.empty:
            f = fp[fp["granule"] == gran]
            x, y = to.transform(f["x_local_m"].to_numpy(), f["y_local_m"].to_numpy())
            ax.plot(x, y, "o", ms=3, mfc="none", color=plotting.COLORS["gedi_l2a"])
    names = [t.split(":")[1][:28] for t in targets]
    lines = [", ".join(names[i:i + 2]) for i in range(0, len(names), 2)]
    ax.set_title(f"{rec['acquired_utc']}  strip {rec['strip_id']}\n" + "\n".join(lines), fontsize=8)
    ax.set_xlabel(f"{crs} easting (m)")
    ax.set_ylabel("northing (m)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--no-quicklooks", action="store_true")
    a = ap.parse_args()
    for site in (config.site_ids() if a.all else [a.site]):
        cfg = config.resolve_site(site)
        pc = cfg["imagery"]["planet"]
        run_id = a.run_id or manifest.latest_run_id(cfg)
        P = pim.paths(cfg, run_id)
        m = manifest.load(cfg, run_id)
        aoi = {n: shape(json.loads((config.ROOT / m["aoi_dir"] / f"{n}.geojson").read_text())["features"][0]["geometry"])
               for n in ("core", "outer")}
        orders = json.loads(P["orders"].read_text()) if P["orders"].exists() else {}
        plan_f = P["tables"] / "planet_order_plan.csv"
        est = pd.read_csv(plan_f, dtype={"strip_id": str}).set_index("order_name")["aoi_clear"] if plan_f.exists() else {}
        an = manifest.data_dir(cfg) / "analysis" / run_id
        cl = pd.read_parquet(an / "atl03_core_centerlines.parquet") if (an / "atl03_core_centerlines.parquet").exists() else pd.DataFrame()
        fp = pd.read_parquet(an / "gedi_core_baseline_footprints.parquet") if (an / "gedi_core_baseline_footprints.parquet").exists() else pd.DataFrame()
        rows = []
        for name, rec in sorted(orders.items()):
            if rec.get("download_status") != "ok":
                continue
            dest = config.ROOT / rec["dir"]
            udm, sr = files(dest, "udm2"), files(dest, "sr")
            row = {"order_name": name, "strip_id": rec["strip_id"], "acquired_utc": rec["acquired_utc"],
                   "bundle": rec["bundle"], "n_sr_files": len(sr), "n_udm2_files": len(udm),
                   "est_aoi_clear": est.get(name) if hasattr(est, "get") else None, "targets": rec["targets"]}
            if udm:
                arr, tr, crs = mosaic_udm2(udm)
                for n, geom in aoi.items():
                    c, f = clear_fraction(arr, tr, crs, geom)
                    row[f"measured_clear_{n}"] = None if c is None else round(c, 3)
                    row[f"footprint_{n}"] = round(f, 3)
            row["passes_clear_min"] = (row.get(f"measured_clear_{pc.get('match_aoi', pc['clip_aoi'])}") or 0) >= pc["aoi_clear_min"]
            if sr and not a.no_quicklooks:
                png = P["figures"] / f"{name}.png"
                quicklook(sr, rec, rec["targets"].split(";"), (cl, fp),
                          (cfg["center"]["lon"], cfg["center"]["lat"]), png)
                row["quicklook"] = str(png.relative_to(config.ROOT))
            rows.append(row)
            print(f"  {name}: clear outer={row.get('measured_clear_outer')} core={row.get('measured_clear_core')} "
                  f"(est {row['est_aoi_clear']})")
        qc = pd.DataFrame(rows)
        P["tables"].mkdir(parents=True, exist_ok=True)
        qc.to_csv(P["tables"] / "planet_scene_qc.csv", index=False)
        print(f"== {site}: {len(qc)} downloaded strips QC'd; "
              f"{int((~qc['passes_clear_min']).sum()) if not qc.empty else 0} below {pc['aoi_clear_min']:.0%} measured clear")
        manifest.update(cfg, run_id, "planet_qc", {"n_strips": len(qc),
                                                   "table": str((P["tables"] / "planet_scene_qc.csv").relative_to(config.ROOT))})


if __name__ == "__main__":
    main()
