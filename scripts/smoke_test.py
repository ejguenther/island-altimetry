#!/usr/bin/env python
"""M3: retrieve small diagnostic samples through the production acquisition path
and validate schemas, times, vertical references, identifiers, filter overrides,
fill values, empty/error/duplicate/interrupt/resume behaviour, and estimate the
full-run size.

  smoke_test.py --site fukutoku_okanoba

Writes docs/m3/{smoke_test.json, field_dictionary.csv, request_examples.json,
size_estimate.csv} (small, committed) and caches samples in the normal chunk store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

from island_altimetry import adapters, aoi, cache, config, sr_client, status
from scripts_path import load_acquire  # noqa: E402  (scripts/ sibling import)

acq = load_acquire()

ATLAS_EPOCH = pd.Timestamp("2018-01-01T00:00:00Z")
FLOAT32_FILL = 3.4028235e38

# Diagnostic sample: Sept 2021 crossings from the reference study period, a
# pre-emergence pass, post-eruption passes, and the latest pass.
SAMPLES = {
    "atl03": ["ATL03_20210910031707_11991201_007_01", "ATL03_20210921145752_13741207_007_01",
              "ATL03_20210721175359_04291207_007_01", "ATL03_20211209225707_11991301_007_01",
              "ATL03_20260712030748_04293207_007_01"],
    "atl09": ["ATL03_20210910031707_11991201_007_01", "ATL03_20210921145752_13741207_007_01",
              "ATL03_20250304143235_11992601_007_01"],
    "atl08": ["ATL08_20210910031707_11991201_007_01", "ATL08_20210921145752_13741207_007_01",
              "ATL08_20210721175359_04291207_007_01"],
    "atl24": ["ATL24_20210910031707_11991201_006_02_002_01", "ATL24_20210921145752_13741207_006_02_002_01"],
    "gedi_l2a": ["GEDI02_A_2021262155644_O15692_03_T08289_02_003_02_V002"],
}

# Column semantics (native source, units, reference). Verified items are checked below.
SEMANTICS = {
    "atl03": {
        "time_ns": ("gtx/heights/delta_time -> UTC by SlideRule", "UTC datetime64[ns]", ""),
        "geometry": ("gtx/heights/lon_ph, lat_ph", "degrees", "EPSG:7912 (ITRF2014 lon/lat)"),
        "height": ("gtx/heights/h_ph", "m", "WGS84 ellipsoid (ITRF2014), no geoid applied"),
        "x_atc": ("dist_ph_along + segment_dist_x", "m", "along-track"),
        "y_atc": ("dist_ph_across", "m", "across-track"),
        "atl03_cnf": ("gtx/heights/signal_conf_ph (max over surface types, srt=-1)", "class", "-2..4"),
        "quality_ph": ("gtx/heights/quality_ph", "class", "0 nominal,1 afterpulse,2 impulse,3 tep"),
        "ph_index": ("photon index within beam in granule", "count", "exact integer ID"),
        "segment_id": ("gtx/geolocation/segment_id", "count", "exact integer ID"),
        "delta_time": ("gtx/heights/delta_time", "s since 2018-01-01 (ATLAS SDP epoch)", "native time"),
        "geoid": ("gtx/geophys_corr/geoid", "m", "EGM2008 geoid height above ellipsoid (mean-tide)"),
        "geoid_free2mean": ("gtx/geophys_corr/geoid_free2mean", "m", "tide-free to mean-tide conversion"),
        "tide_ocean": ("gtx/geophys_corr/tide_ocean", "m", "provided; NOT applied to h_ph"),
        "dac": ("gtx/geophys_corr/dac", "m", "dynamic atmosphere correction; provided, NOT applied to h_ph"),
        "signal_class_ph": ("gtx/heights/signal_class_ph", "class", "r007 experimental"),
        "weight_ph": ("gtx/heights/weight_ph", "score 0-65535", "r007 photon weight"),
        "surf_type": ("gtx/geolocation/surf_type", "flags[5]", "land, ocean, sea ice, land ice, inland water"),
        "podppd_flag": ("gtx/geolocation/podppd_flag", "flag", "geolocation quality"),
        "sigma_h": ("gtx/geolocation/sigma_h", "m", "height uncertainty estimate"),
        "ref_elev": ("gtx/geolocation/ref_elev", "rad", "elevation angle of unit pointing vector"),
        "solar_elevation": ("gtx/geolocation/solar_elevation", "degrees", ""),
        "background_rate": ("gtx/bckgrd_atlas/bckgrd_rate", "photons/s", "solar background"),
    },
    "atl08": {
        "h_te_median": ("land_segments/terrain/h_te_median", "m", "WGS84 ellipsoid"),
        "terrain/h_te_best_fit": ("land_segments/terrain/h_te_best_fit", "m", "WGS84 ellipsoid"),
        "terrain/h_te_mean": ("land_segments/terrain/h_te_mean", "m", "WGS84 ellipsoid"),
        "h_te_uncertainty": ("land_segments/terrain/h_te_uncertainty", "m", ""),
        "segment_id_beg": ("land_segments/segment_id_beg", "count", "first ATL03 segment of 100 m segment"),
        "delta_time": ("land_segments/delta_time", "s since 2018-01-01", "native time"),
        "segment_watermask": ("land_segments/segment_watermask", "flag", "0 land, 1 water (static mask)"),
        "dem_h": ("land_segments/dem_h", "m", "reference DEM height (ellipsoid); not independent truth"),
    },
    "atl24": {
        "ortho_h": ("gtx/ortho_h", "m", "EGM2008 orthometric, refraction corrected"),
        "ellipse_h": ("gtx/ellipse_h", "m", "ITRF2014 ellipsoidal, refraction corrected"),
        "surface_h": ("gtx/surface_h", "m", "EGM2008 orthometric modeled sea surface"),
        "class_ph": ("gtx/class_ph", "class", "0 unclassified, 40 bathymetry, 41 sea surface"),
        "geometry": ("gtx/lon_ph, lat_ph", "degrees", "refraction corrected"),
        "x_atc": ("gtx/x_atc", "m", "not refraction corrected"),
        "index_ph": ("gtx/index_ph", "count", "photon index in source ATL03 r006 granule (not r007)"),
        "delta_time": ("gtx/delta_time", "s since 2018-01-01", "native time"),
    },
    "gedi_l2a": {
        "elevation_lm": ("BEAMxxxx/elev_lowestmode", "m", "WGS84 ellipsoid"),
        "elevation_hr": ("BEAMxxxx/elev_highestreturn", "m", "WGS84 ellipsoid"),
        "shot_number": ("BEAMxxxx/shot_number", "count", "exact uint64 ID"),
        "flags": ("SlideRule bitfield", "bits", "0x01 degrade, 0x02 l2 quality, 0x04 l4 quality, 0x80 surface"),
        "sensitivity": ("BEAMxxxx/sensitivity", "fraction", ""),
        "delta_time": ("BEAMxxxx/delta_time", "s since 2018-01-01", "native time"),
        "digital_elevation_model": ("BEAMxxxx/digital_elevation_model", "m", "TanDEM-X reference; not independent truth"),
    },
    "atl09": {
        "high_rate/delta_time": ("profile_x/high_rate/delta_time", "s since 2018-01-01", "native profile time"),
        "high_rate/layer_flag": ("profile_x/high_rate/layer_flag", "flag", "consolidated cloud flag"),
        "high_rate/cloud_flag_atm": ("profile_x/high_rate/cloud_flag_atm", "count", "number of layers"),
        "high_rate/layer_top": ("profile_x/high_rate/layer_top[:,0]", "m", "TRUNCATED: first layer only"),
        "high_rate/layer_bot": ("profile_x/high_rate/layer_bot[:,0]", "m", "TRUNCATED: first layer only"),
        "high_rate/surface_height": ("profile_x/high_rate/surface_height", "m", "atmospheric-product surface estimate"),
        "n_atl03_photons": ("derived", "count", "ATL03 photons sharing this profile in the AOI"),
    },
}


def load_chunks(ctx, pid, granules):
    contract = ctx["products"][pid]
    parms = adapters.build_parms(pid, contract, ctx["products"]["atl03"], ctx["poly"])
    store = acq.store_for(ctx, pid, parms)
    frames, metas = {}, {}
    for g in granules:
        m = store.read_meta(g)
        metas[g] = m
        if m and m["status"] == status.OK:
            p = store.data_path(g)
            frames[g] = gpd.read_parquet(p) if pid != "atl09" else pd.read_parquet(p)
    return frames, metas, store, parms


def time_check(df: pd.DataFrame, col: str = "delta_time") -> dict:
    t = pd.to_datetime(df.index if "time_ns" not in df.columns else df["time_ns"], utc=True)
    native = ATLAS_EPOCH + pd.to_timedelta(df[col].astype("float64"), unit="s")
    resid = (t - native).total_seconds() if hasattr(t - native, "total_seconds") else (t - native).dt.total_seconds()
    resid = np.asarray(resid, dtype="float64")
    return {"rule": "UTC = 2018-01-01T00:00:00Z + delta_time (GPS-UTC offset embedded in ATLAS epoch)",
            "residual_s_median": float(np.median(resid)), "residual_s_maxabs": float(np.max(np.abs(resid))),
            "pass": bool(np.max(np.abs(resid)) < 1e-3)}


def fill_report(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        if c == "geometry":
            continue
        s = df[c]
        if s.dtype.kind == "f":
            nan = int(s.isna().sum())
            fill = int((s.abs() >= 1e38).sum())
            if nan or fill:
                out[c] = {"nan": nan, "float_max_fill": fill}
    return out


def dictionary_rows(pid: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    sem = SEMANTICS.get(pid, {})
    cols = (["time_ns"] if isinstance(df.index, pd.DatetimeIndex) else []) + list(df.columns)
    for c in cols:
        s = df.index.to_series() if c == "time_ns" and c not in df.columns else df[c]
        src, units, ref = sem.get(c, ("server default column" if pid != "atl09" else "derived/ATL09", "", ""))
        r = {"product": pid, "column": c, "dtype": str(s.dtype), "native_source": src, "units": units,
             "reference_notes": ref, "n": len(s)}
        if c != "geometry" and s.dtype.kind in "fiu":
            v = s.replace([np.inf, -np.inf], np.nan)
            v = v[v.abs() < 1e38] if s.dtype.kind == "f" else v
            r.update(min=float(v.min()) if len(v.dropna()) else None, max=float(v.max()) if len(v.dropna()) else None,
                     n_nan=int(s.isna().sum()) if s.dtype.kind == "f" else 0,
                     n_float_max_fill=int((s.abs() >= 1e38).sum()) if s.dtype.kind == "f" else 0)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--skip-fetch", action="store_true", help="validate already-cached samples only")
    a = ap.parse_args()
    ctx = acq.context(a.site, a.run_id)
    outer = shape(json.loads((config.ROOT / ctx["m"]["aoi_dir"] / "outer.geojson").read_text())["features"][0]["geometry"])
    core = shape(json.loads((config.ROOT / ctx["m"]["aoi_dir"] / "core.geojson").read_text())["features"][0]["geometry"])
    docs = config.ROOT / "docs" / "m3"
    docs.mkdir(parents=True, exist_ok=True)
    report = {"run_id": ctx["m"]["run_id"], "samples": SAMPLES, "checks": {}}
    C = report["checks"]

    # 1. Fetch samples via the production path.
    fetch = {}
    for pid, gs in SAMPLES.items():
        if not a.skip_fetch:
            fetch[pid] = acq.run_product(ctx, pid, gs, workers=2)
    data, metas = {}, {}
    for pid, gs in SAMPLES.items():
        data[pid], metas[pid], _, _ = load_chunks(ctx, pid, gs)
    report["sample_status"] = {pid: {g: (m or {}).get("status", status.NOT_YET_QUERIED) for g, m in ms.items()}
                               for pid, ms in metas.items()}
    report["sample_rows"] = {pid: {g: (m or {}).get("n_rows") for g, m in ms.items()} for pid, ms in metas.items()}

    # 2. Times.
    tc = {}
    for pid in ("atl03", "atl08", "atl24", "gedi_l2a"):
        for g, df in data[pid].items():
            if "delta_time" in df.columns:
                tc[f"{pid}:{g}"] = time_check(df)
    C["time_conversion"] = tc

    # 3. Spatial: all rows inside the outer AOI; zone counts.
    sp = {}
    for pid in ("atl03", "atl08", "atl24", "gedi_l2a"):
        for g, df in data[pid].items():
            inside = df.geometry.within(outer.buffer(1e-6))
            sp[f"{pid}:{g}"] = {"n": len(df), "outside_outer": int((~inside).sum()),
                                "in_core": int(df.geometry.within(core).sum())}
    C["spatial"] = sp

    # 4. Identifiers.
    ids = {}
    for g, df in data["atl03"].items():
        ids[f"atl03:{g}"] = {"ph_index_dtype": str(df["ph_index"].dtype),
                             "dup_(srcid,gt,ph_index)": int(df.duplicated(["srcid", "gt", "ph_index"]).sum()),
                             "segment_id_dtype": str(df["segment_id"].dtype)}
    for g, df in data["atl08"].items():
        ids[f"atl08:{g}"] = {"dup_(gt,segment_id_beg)": int(df.duplicated(["gt", "segment_id_beg"]).sum())}
    for g, df in data["gedi_l2a"].items():
        ids[f"gedi:{g}"] = {"shot_number_dtype": str(df["shot_number"].dtype),
                            "dup_shot_number": int(df["shot_number"].duplicated().sum()),
                            "example": str(df["shot_number"].iloc[0])}
    for g, df in data["atl24"].items():
        ids[f"atl24:{g}"] = {"index_ph_dtype": str(df["index_ph"].dtype) if "index_ph" in df else None,
                             "dup_(gt,index_ph)": int(df.duplicated(["gt", "index_ph"]).sum()) if "index_ph" in df else None}
    C["identifiers"] = ids

    # 5. Filter overrides took effect (permissive diagnostic subset).
    fo = {}
    for g, df in data["atl03"].items():
        fo[f"atl03:{g}"] = {"atl03_cnf": df["atl03_cnf"].value_counts().sort_index().to_dict(),
                            "quality_ph": df["quality_ph"].value_counts().sort_index().to_dict()}
    for g, df in data["atl24"].items():
        fo[f"atl24:{g}"] = {"class_ph": df["class_ph"].value_counts().sort_index().to_dict(),
                            "low_confidence_flag": df["low_confidence_flag"].value_counts().to_dict()}
    for g, df in data["gedi_l2a"].items():
        f = df["flags"].astype(int)
        fo[f"gedi:{g}"] = {"degrade_set": int((f & 1).astype(bool).sum()), "l2_quality_set": int((f & 2).astype(bool).sum()),
                           "l2_quality_not_set": int((~(f & 2).astype(bool)).sum()), "surface_set": int((f & 0x80).astype(bool).sum())}
    C["filter_overrides"] = fo

    # 6. Vertical reference checks.
    vr = {}
    g0 = SAMPLES["atl03"][0]
    rc = sr_client.run("atl03x", {"poly": aoi.to_sliderule_poly(core), **ctx["products"]["atl03"]["request_parms"],
                                  "atl03_ph_fields": ["h_ph"]}, [g0 + ".h5"])
    if rc.status == status.OK:
        d = (rc.gdf["height"].astype("float64") - rc.gdf["h_ph"].astype("float64")).abs()
        vr["atl03_height_equals_native_h_ph"] = {"n": rc.n, "max_abs_diff_m": float(d.max()), "pass": bool(d.max() < 1e-3)}
    else:
        vr["atl03_height_equals_native_h_ph"] = {"status": rc.status}
    for g, df in data["gedi_l2a"].items():
        if "elev_lowestmode" in df:
            d = (df["elevation_lm"].astype("float64") - df["elev_lowestmode"].astype("float64")).abs()
            vr[f"gedi_elevation_lm_equals_elev_lowestmode:{g}"] = {"max_abs_diff_m": float(d.max()), "pass": bool(d.max() < 1e-3)}
    for g, df in data["atl24"].items():
        # ellipse_h - ortho_h should be a smooth geoid-like offset (EGM2008), not constant zero.
        off = (df["ellipse_h"].astype("float64") - df["ortho_h"].astype("float64"))
        vr[f"atl24_ellipse_minus_ortho:{g}"] = {"median_m": float(off.median()), "range_m": float(off.max() - off.min())}
    for g, df in data["atl03"].items():
        sea = df[(df["atl03_cnf"] >= 3)]
        if len(sea):
            vr[f"atl03_conf>=3_height_minus_geoid_median:{g}"] = {
                "median_height_m": float(sea["height"].median()), "median_geoid_m": float(sea["geoid"].median()),
                "median_height_minus_geoid_m": float((sea["height"] - sea["geoid"]).median()),
                "note": "near zero over open ocean supports ellipsoidal heights with EGM2008 geoid"}
    C["vertical_reference"] = vr

    # 7. ATL09 reduction.
    C["atl09_reduction"] = {g: (m or {}).get("atl09_reduction") for g, m in metas["atl09"].items()}

    # 8. Fill values.
    C["fill_values"] = {f"{pid}:{g}": fill_report(df) for pid in data for g, df in data[pid].items()}

    # 9. Behaviour tests.
    beh = {}
    far = aoi.to_sliderule_poly(aoi.disk(141.481 + 1.5, 24.285, 2000))  # no RGT 1199 c12 photons here
    r = sr_client.run("atl03x", {"poly": far, **ctx["products"]["atl03"]["request_parms"]}, [g0 + ".h5"])
    beh["empty_aoi_response"] = {"status": r.status, "n": r.n, "pass": r.status == status.NO_SAMPLES_IN_AOI}
    r = sr_client.run("atl03x", {"poly": far, **ctx["products"]["atl03"]["request_parms"],
                                 "t0": "2031-01-01T00:00:00Z", "t1": "2031-01-02T00:00:00Z"})
    beh["no_granules_response"] = {"status": r.status, "pass": r.status == status.NO_GRANULES}
    g8 = SAMPLES["atl08"][0] + ".h5"
    single = sr_client.run("atl08x", {"poly": ctx["poly"]}, [g8])
    double = sr_client.run("atl08x", {"poly": ctx["poly"]}, [g8, g8])
    beh["duplicate_resource_in_request"] = {"single_n": single.n, "double_n": double.n,
                                            "note": "pipeline issues one request per unique granule; duplicates in a "
                                                    "request are not de-duplicated by the server" if double.n != single.n else
                                                    "server de-duplicates repeated resources"}
    inv_dups = {pid: int(acq.resources_for(ctx, pid)["granule"].duplicated().sum()) for pid in ctx["cfg"]["products"]}
    beh["inventory_duplicate_granules"] = {"counts": inv_dups, "pass": not any(inv_dups.values())}
    # Interrupted chunk: stray partial file + tampered checksum must not count as complete.
    pid, g = "atl08", SAMPLES["atl08"][0]
    _, _, store, parms = load_chunks(ctx, pid, [g])
    store.tmp_path(g).write_bytes(b"partial")
    meta = store.read_meta(g)
    tampered = {**meta, "sha256": "0" * 64}
    store.meta_path(g).write_text(json.dumps(tampered))
    beh["interrupted_chunk_detected"] = {"is_complete_after_tamper": store.is_complete(g)}
    rer = acq.run_product(ctx, pid, [g], quiet=True)
    beh["interrupted_chunk_refetched"] = {"status": rer[0]["status"], "complete": store.is_complete(g),
                                          "stray_partial_removed": not store.tmp_path(g).exists(),
                                          "pass": store.is_complete(g) and not beh["interrupted_chunk_detected"]["is_complete_after_tamper"]}
    # Resume: everything already complete is skipped (no network fetch).
    res = {p: acq.run_product(ctx, p, gs, quiet=True) for p, gs in SAMPLES.items()}
    skipped = {p: sum(bool(x.get("skipped")) for x in v) for p, v in res.items()}
    complete = {p: sum(store_ok for store_ok in [x["status"] in (status.OK, *status.TERMINAL_EMPTY) for x in v]) for p, v in res.items()}
    beh["resume_skips_complete"] = {"skipped": skipped, "terminal": complete,
                                    "pass": all(skipped[p] == complete[p] for p in skipped)}
    C["behaviour"] = beh

    # 10. Size/runtime estimate.
    led_counts = {pid: len(acq.resources_for(ctx, pid)) for pid in ctx["cfg"]["products"]}
    est = []
    for pid, ms in metas.items():
        ok = [m for m in ms.values() if m and m["status"] == status.OK]
        if not ok:
            continue
        b = np.array([m["bytes"] for m in ok], float)
        t = np.array([m["elapsed_s"] for m in ok], float)
        n = np.array([m["n_rows"] for m in ok], float)
        est.append({"product": pid, "n_sampled": len(ok), "mean_mb": round(b.mean() / 1e6, 2), "max_mb": round(b.max() / 1e6, 2),
                    "mean_rows": int(n.mean()), "mean_request_s": round(t.mean(), 1), "n_resources": led_counts[pid],
                    "est_total_gb": round(b.mean() * led_counts[pid] / 1e9, 2),
                    "est_total_gb_upper": round(b.max() * led_counts[pid] / 1e9, 2),
                    "est_serial_hours": round(t.mean() * led_counts[pid] / 3600, 2)})
    est = pd.DataFrame(est)
    est.to_csv(docs / "size_estimate.csv", index=False)
    report["size_estimate"] = est.to_dict(orient="records")

    # Field dictionary and request examples.
    rows = []
    for pid in data:
        if data[pid]:
            rows += dictionary_rows(pid, next(iter(data[pid].values())))
    pd.DataFrame(rows).to_csv(docs / "field_dictionary.csv", index=False)
    ex = {pid: {"endpoint": adapters.endpoint(pid, ctx["products"][pid]),
                "parms_excluding_poly": {k: v for k, v in adapters.build_parms(pid, ctx["products"][pid], ctx["products"]["atl03"], ctx["poly"]).items() if k != "poly"},
                "poly": "outer AOI from data/<site>/aoi/<aoi_hash>/outer.geojson as SlideRule {lon,lat} list",
                "resources": [SAMPLES[pid][0] + ".h5"]} for pid in SAMPLES}
    (docs / "request_examples.json").write_text(json.dumps(ex, indent=2) + "\n")
    (docs / "smoke_test.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("sample_status",)}, indent=1))
    print(est.to_string(index=False))
    fails = [k for k, v in {**C["behaviour"], **C["vertical_reference"], **C["time_conversion"]}.items()
             if isinstance(v, dict) and v.get("pass") is False]
    print("FAILED CHECKS:", fails or "none")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
