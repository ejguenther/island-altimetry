#!/usr/bin/env python
"""M1: build site AOIs (core search zone, outer buffer, buffer ring), the event
table, and start a run by freezing its UTC end time in a new run manifest.

  build_aoi.py --site fukutoku_okanoba
  build_aoi.py --all
  build_aoi.py --site X --set core_radius_m=3000 --set buffer_m=8000
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from shapely.geometry import mapping

from island_altimetry import aoi, config, manifest, sr_client
from island_altimetry.config import fmt_utc


def build_geometries(cfg: dict) -> dict:
    lon, lat = cfg["center"]["lon"], cfg["center"]["lat"]
    r_core = float(cfg["core_radius_m"])
    r_outer = r_core + float(cfg["buffer_m"])
    core = aoi.disk(lon, lat, r_core)
    outer = aoi.disk(lon, lat, r_outer)
    ring = outer.difference(core)
    checks = {}
    for name, g, r in (("core", core, r_core), ("outer", outer, r_outer)):
        dmin, dmax = aoi.max_geodesic_radius_m(g, lon, lat)
        checks[name] = {"target_radius_m": r, "vertex_dist_min_m": round(dmin, 3),
                        "vertex_dist_max_m": round(dmax, 3), "valid": g.is_valid,
                        "area_km2_nominal": round(3.141592653589793 * r * r / 1e6, 4)}
        if not g.is_valid or abs(dmax - r) > 0.5 or abs(dmin - r) > 0.5:
            raise SystemExit(f"{name} AOI failed radius/validity check: {checks[name]}")
    if not core.contains(aoi.Point(lon, lat)):
        raise SystemExit("center not inside core zone (coordinate order?)")
    return {"core": core, "outer": outer, "buffer": ring, "checks": checks}


def write_aoi(cfg: dict, geoms: dict) -> tuple[Path, str]:
    aoi_hash = aoi.geometry_hash(geoms["outer"].union(geoms["core"].boundary))
    d = manifest.data_dir(cfg) / "aoi" / aoi_hash
    d.mkdir(parents=True, exist_ok=True)
    zone_label = {"core": "core search zone (not an island outline)",
                  "buffer": "outer ocean buffer (not a land/ocean label)",
                  "outer": "outer discovery AOI (core + buffer)"}
    for z in ("core", "buffer", "outer"):
        fc = {"type": "FeatureCollection",
              "features": [{"type": "Feature", "geometry": mapping(geoms[z]),
                            "properties": {"site_id": cfg["site_id"], "zone": z, "label": zone_label[z]}}]}
        (d / f"{z}.geojson").write_text(json.dumps(fc, indent=1) + "\n")
    meta = {
        "site_id": cfg["site_id"], "aoi_hash": aoi_hash, "crs": "EPSG:4326 (lon, lat)",
        "construction": f"geodesic disks built as metric buffers in site-centred AEQD "
                        f"(+proj=aeqd +lat_0={cfg['center']['lat']} +lon_0={cfg['center']['lon']} +datum=WGS84), "
                        f"{aoi.N_VERTICES} vertices, transformed to WGS84",
        "center": cfg["center"], "center_source": cfg.get("center_source"),
        "core_radius_m": cfg["core_radius_m"], "buffer_m": cfg["buffer_m"],
        "outer_radius_m": cfg["core_radius_m"] + cfg["buffer_m"],
        "checks": geoms["checks"], "dateline": "rejected if crossed (not crossed)",
        "geometry_hashes": {z: aoi.geometry_hash(geoms[z]) for z in ("core", "buffer", "outer")},
    }
    (d / "aoi_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return d, aoi_hash


def write_events(cfg: dict, out: Path) -> Path:
    p = out / "tables" / "events.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "start", "precision", "description", "source"])
        w.writeheader()
        for e in cfg.get("events", []):
            w.writerow({k: " ".join(str(e.get(k, "")).split()) for k in w.fieldnames})
    return p


def run_site(site_id: str, overrides: list[str], run_id: str | None) -> str:
    cfg = config.resolve_site(site_id, overrides)
    geoms = build_geometries(cfg)
    aoi_dir, aoi_hash = write_aoi(cfg, geoms)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    run_id = run_id or manifest.new_run_id(site_id, now)
    out = manifest.output_dir(cfg, run_id)
    if (out / "run_manifest.json").exists():
        raise SystemExit(f"run {run_id} already exists; runs are immutable snapshots")
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    events = write_events(cfg, out)
    end = now if cfg["end"] == "latest" else config.parse_utc(cfg["end"])
    contracts = json.loads((config.ROOT / "docs" / "m0" / "product_contracts.json").read_text())
    versions = sr_client.server_version()
    m = {
        "run_id": run_id, "site_id": site_id, "created_utc": fmt_utc(now),
        "query_end_utc": fmt_utc(end), "end_policy": cfg["end"],
        "query_start_utc": {p: cfg["start"][p] for p in cfg["products"]},
        "config_hash": config.config_hash(cfg), "aoi_hash": aoi_hash,
        "aoi_dir": str(aoi_dir.relative_to(config.ROOT)),
        "products": {p: {"sliderule_asset": contracts[p]["sliderule_asset"],
                         "endpoint": contracts[p]["endpoint"],
                         "served_release": contracts[p]["sliderule_served_release"],
                         "cmr_latest_version": contracts[p]["cmr_latest_version"]}
                     for p in cfg["products"]},
        "release_policy": cfg["release_policy"],
        "sliderule": versions, "git_commit": manifest.git_commit(),
        "events_table": str(events.relative_to(config.ROOT)),
    }
    manifest.write_json_atomic(out / "run_manifest.json", m)
    print(f"{site_id}: run {run_id} end={m['query_end_utc']} aoi={aoi_hash} config={m['config_hash']}")
    for z, c in geoms["checks"].items():
        print(f"  {z}: r={c['target_radius_m']} m vertices {c['vertex_dist_min_m']}-{c['vertex_dist_max_m']} m valid={c['valid']}")
    return run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="override key=value (dotted keys)")
    ap.add_argument("--run-id", help="explicit run id (default: <site>_<UTC now>)")
    a = ap.parse_args()
    for s in (config.site_ids() if a.all else [a.site]):
        run_site(s, a.set, a.run_id)


if __name__ == "__main__":
    main()
