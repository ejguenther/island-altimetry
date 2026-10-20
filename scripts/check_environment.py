#!/usr/bin/env python
"""M0: record environment and verify product access contracts against the live
SlideRule server and NASA CMR collection metadata.

Writes docs/m0/environment.json, docs/m0/product_support.csv,
docs/m0/product_contracts.json. No credentials are read or written.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata as md
import json
import platform
import re
import shutil
import sys
from pathlib import Path

import requests
import yaml

from island_altimetry import aoi, sr_client, status

ROOT = Path(__file__).resolve().parents[1]
CMR = "https://cmr.earthdata.nasa.gov/search"

# Probe acquisition: ICESat-2 RGT 1199 cycle 12 (2021-09-10) and GEDI orbit 15692
# (2021-09-19) both cross the Fukutoku-Okanoba core zone (verified in M0).
PROBE = {
    "atl03": "ATL03_20210910031707_11991201_007_01.h5",
    "atl08": "ATL08_20210910031707_11991201_007_01.h5",
    "atl09": "ATL03_20210910031707_11991201_007_01.h5",
    "atl24": "ATL24_20210910031707_11991201_006_02_002_01.h5",
    "gedi_l2a": "GEDI02_A_2021262155644_O15692_03_T08289_02_003_02_V002.h5",
}
PROBE_WINDOW = ("2021-09-01T00:00:00Z", "2021-09-30T23:59:59Z")
PROBE_CENTER = (141.481, 24.285)

SOURCES = {
    "sliderule_docs_icesat2": "https://docs.slideruleearth.io/user-guide/icesat2",
    "sliderule_docs_gedi": "https://docs.slideruleearth.io/user-guide/gedi",
    "sliderule_source": "https://github.com/SlideRuleEarth/sliderule",
    "cmr_api": "https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html",
    "atl03_user_guide": "https://nsidc.org/data/atl03",
    "atl08_user_guide": "https://nsidc.org/data/atl08",
    "atl09_user_guide": "https://nsidc.org/data/atl09",
    "atl24_user_guide": "https://nsidc.org/data/atl24",
    "gedi02a_v002": "https://doi.org/10.5067/GEDI/GEDI02_A.002",
    "gedi02a_v003": "https://lpdaac.usgs.gov/products/gedi02_av003/",
}


def pkg_versions() -> dict:
    names = ["sliderule", "pandas", "numpy", "pyarrow", "geopandas", "shapely", "pyproj",
             "matplotlib", "PyYAML", "requests", "pytest"]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def environment() -> dict:
    rel = Path("/etc/redhat-release")
    du = shutil.disk_usage(ROOT)
    return {
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "os_release": rel.read_text().strip() if rel.exists() else platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "disk_free_gb": round(du.free / 1e9, 1),
        "packages": pkg_versions(),
        "sliderule": sr_client.server_version(),
    }


def cmr_collections(short_name: str) -> list[dict]:
    r = requests.get(f"{CMR}/collections.json", params={"short_name": short_name, "page_size": 50}, timeout=60)
    r.raise_for_status()
    rows = []
    for e in r.json()["feed"]["entry"]:
        rows.append({"version": e.get("version_id"), "concept_id": e.get("id"),
                     "provider": e.get("data_center"), "time_start": e.get("time_start"),
                     "time_end": e.get("time_end")})
    return sorted(rows, key=lambda x: x["version"])


def release_from_name(product: str, name: str) -> dict:
    """Parse release identifiers from a granule filename."""
    if product == "atl24":
        m = re.match(r"ATL24_\d{14}_\d{8}_(\d{3})_(\d{2})_(\d{3})_(\d{2})", name)
        return {"release": m.group(3), "atl03_release": m.group(1)} if m else {}
    if product == "gedi_l2a":
        m = re.search(r"_(\d{3})_(\d{2})_V(\d{3})", name)
        return {"release": m.group(3), "pge_version": m.group(1)} if m else {}
    m = re.match(r"ATL\d\d_\d{14}_\d{8}_(\d{3})_(\d{2})", name)
    return {"release": m.group(1)} if m else {}


def _returned(field: str, columns) -> bool:
    """A requested field counts only if it comes back as a column."""
    return any(c == field or c.split("/")[-1] == field.split("/")[-1] for c in columns)


def probe_fields(api: str, base: dict, field_key: str, fields: list[str],
                 resource: str) -> tuple[list[str], list[str], dict]:
    """Request all fields together; on failure, test each individually. A field
    the server accepts but does not return is reported as unsupported."""
    def parms_with(fs):
        p = json.loads(json.dumps(base))
        *nest, key = field_key.split(".")
        d = p
        for k in nest:
            d = d.setdefault(k, {})
        d[key] = fs
        return p

    res = sr_client.run(api, parms_with(fields), [resource])
    if res.status == status.OK:
        cols = list(res.gdf.columns)
        ok = [f for f in fields if _returned(f, cols)]
        bad = [f for f in fields if f not in ok]
        return ok, bad, {"status": res.status, "n": res.n, "columns": cols,
                         "silently_ignored": bad}
    ok, bad = [], []
    for f in fields:
        r1 = sr_client.run(api, parms_with([f]), [resource])
        good = r1.status == status.OK and _returned(f, r1.gdf.columns)
        (ok if good else bad).append(f)
        if r1.status not in (status.OK, status.FIELD_UNSUPPORTED):
            print(f"  field {f}: {r1.status} {r1.error or ''}", file=sys.stderr)
    return ok, bad, {"status": res.status, "n": res.n, "alerts": [a["text"] for a in res.alerts if a["code"] < 0][:5]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "m0"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = environment()
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print(f"server {env['sliderule']['server']['version']} client {env['sliderule']['client']['version']}")

    prods = yaml.safe_load((ROOT / "configs" / "products.yaml").read_text())["products"]
    core = aoi.to_sliderule_poly(aoi.disk(*PROBE_CENTER, 2000))
    outer = aoi.to_sliderule_poly(aoi.disk(*PROBE_CENTER, 7000))
    access_date = env["recorded_utc"][:10]
    contracts, rows = {}, []

    for pid, p in prods.items():
        print(f"== {pid}")
        c = {"product": pid, "title": p["title"], "cmr_short_name": p["cmr_short_name"],
             "sliderule_asset": p["sliderule_asset"], "endpoint": p["endpoint"],
             "record_unit": p["record_unit"], "access_date": access_date, "sources": SOURCES}
        c["cmr_collections"] = cmr_collections(p["cmr_short_name"])
        c["cmr_latest_version"] = c["cmr_collections"][-1]["version"] if c["cmr_collections"] else None

        # What the SlideRule asset actually resolves to.
        try:
            names = sr_client.search(p["sliderule_asset"], outer, *PROBE_WINDOW)
            c["sliderule_resolved_granules"] = sorted(names)
            rels = sorted({json.dumps(release_from_name(pid, n), sort_keys=True) for n in names})
            c["sliderule_served_release"] = [json.loads(r) for r in rels]
        except Exception as e:
            c["sliderule_resolved_granules"] = []
            c["sliderule_served_release"] = []
            c["discovery_error"] = repr(e)

        # Endpoint and field probe on the core zone of a known crossing.
        api = "atl03x" if pid == "atl09" else p["endpoint"]
        base = {"poly": core, **p.get("request_parms", {})}
        if pid == "atl09":
            base.update(prods["atl03"]["request_parms"])
        supported, unsupported, probe = {}, {}, {}
        for key, fields in p.get("anc_fields", {}).items():
            ok, bad, info = probe_fields(api, base, key, fields, PROBE[pid])
            supported[key], unsupported[key], probe[key] = ok, bad, info
        if not p.get("anc_fields"):
            r = sr_client.run(api, base, [PROBE[pid]])
            probe["base"] = {"status": r.status, "n": r.n}
        c.update({"probe_resource": PROBE[pid], "probe_parms": base, "fields_supported": supported,
                  "fields_unsupported": unsupported, "probe": probe})
        contracts[pid] = c

        served = ";".join(sorted({s.get("release", "?") for s in c["sliderule_served_release"]}))
        n_bad = sum(len(v) for v in unsupported.values())
        any_ok = any(v.get("status") == status.OK for v in probe.values()) or any(supported.values())
        rows.append({
            "product": pid, "cmr_short_name": p["cmr_short_name"],
            "cmr_latest_version": c["cmr_latest_version"], "sliderule_served_release": served,
            "release_matches_latest": served == c["cmr_latest_version"],
            "record_unit": p["record_unit"], "access_route": f"{api} ({p['sliderule_asset']})",
            "route_status": "verified" if any_ok else "blocked",
            "fields_requested": sum(len(v) for v in p.get("anc_fields", {}).values()),
            "fields_unsupported": ";".join(f for v in unsupported.values() for f in v),
            "n_unsupported": n_bad,
        })
        print(f"   served={served} latest={c['cmr_latest_version']} unsupported={n_bad}")

    (out / "product_contracts.json").write_text(json.dumps(contracts, indent=2, default=str) + "\n")
    with open(out / "product_support.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
