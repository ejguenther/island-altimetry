#!/usr/bin/env python
"""CNN terrain M1: per-site audit, crop, Planet scene selection, ordering and
exact-grid reference/mask build for the eight configured AOIs.

  prepare_cnn_sites.py --stage audit    [--site S ...]   # LAZ/DTM audit, ground footprint, frozen crop
  prepare_cnn_sites.py --stage search   [--site S ...]   # metadata only (no quota): rule-based pick per site
  prepare_cnn_sites.py --stage order    [--submit]       # dry run unless --submit (uses quota)
  prepare_cnn_sites.py --stage download
  prepare_cnn_sites.py --stage build    [--site S ...]   # rasters, masks, site_summary.json, site_qa.png
  prepare_cnn_sites.py --stage finalize [--site S ...]   # target/nDSM/eligibility from stored rasters
  prepare_cnn_sites.py --stage summary                   # outputs/cnn_terrain/sites.csv

Scene rule (configs/cnn_terrain.yaml imagery.*): full crop footprint, UDM2
clear over the crop >= aoi_clear_min, SR bundle; nearest to the survey
interval of the crop's tiles; windows widen (14 -> 45 -> 120 d) only when
nothing passes. Every candidate's status/reason is kept in the ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import pandas as pd

from island_altimetry import manifest
from island_altimetry.cnn import config as ccfg
from island_altimetry.cnn import sites as S
from island_altimetry.coastal.grid import read_geojson
from island_altimetry.config import ROOT


def stage_audit(cfg, names, workers):
    for site in names:
        sp = ccfg.site_paths(cfg, site)
        if S.site_strip(cfg, site) and not (sp["work"] / "crop.geojson").exists():
            src = ROOT / cfg["data_root"] / site / "coastal" / "crop.geojson"   # clip of the existing order
            sp["work"].mkdir(parents=True, exist_ok=True)
            shutil.copy(src, sp["work"] / "crop.geojson")
        a = S.audit_site(cfg, site, workers)
        print(f"{site}: navd88={a['navd88']} survey={a['survey_utc']} ground {a['ground_footprint_km2']} km2, "
              f"crop {a['crop_km2']} km2 (windowed={a['crop_windowed']})")


def stage_search(cfg, names, refresh):
    from planet_search import search_window
    ic = cfg["imagery"]
    pc = cfg["imagery"]["planet"] | {"scene_cloud_max": ic["scene_cloud_max"], "bundles": ic["bundles"],
                                     "strip_footprint_prefilter": ic["aoi_footprint_min"]}
    for site in names:
        sp, P = ccfg.site_paths(cfg, site), ccfg.planet_paths(cfg, site)
        if S.site_strip(cfg, site):
            print(f"{site}: existing scene {S.site_strip(cfg, site)} (no search)")
            continue
        a = json.loads((sp["out"] / "audit.json").read_text())
        if not a["navd88"]:
            print(f"{site}: not NAVD88 {a['vertical_epsg']} -> incompatible, no search")
            continue
        aoi = read_geojson(sp["work"] / "crop.geojson")
        s0, s1 = (pd.Timestamp(t) for t in a["survey_utc"])
        ref = pd.Timestamp(a["survey_median_utc"])
        best, ledger = None, []
        for w in ic["windows_days"]:
            t0, t1 = (s0 - pd.Timedelta(days=w)).to_pydatetime(), (s1 + pd.Timedelta(days=w)).to_pydatetime()
            items, strips = asyncio.run(search_window(pc, P, aoi, t0, t1, ref, refresh, cfg["retries"],
                                                      cfg["retry_backoff_s"]))
            if strips.empty:
                continue
            best, led = S.select_scene(strips, (s0, s1), ic)
            ledger = led.assign(window_days=w)
            if best:
                break
        sp["out"].mkdir(parents=True, exist_ok=True)
        if len(ledger):
            ledger.drop(columns=[c for c in ("geometry",) if c in ledger]).sort_values("gap_days").to_csv(
                sp["out"] / "planet_candidates.csv", index=False)
        sel = {"site": site, "survey_utc": a["survey_utc"], "window_days": int(ledger["window_days"].iloc[0]) if len(ledger) else None,
               "selected": {k: (str(v) if k == "acquired" else v) for k, v in best.items()
                            if k in ("strip_id", "acquired", "item_ids", "n_items", "instrument", "bundle", "aoi_clear",
                                     "aoi_footprint", "gap_days", "dt_hours", "sun_elevation", "view_angle_max",
                                     "clip_km2", "quality_category")} if best else None}
        manifest.write_json_atomic(sp["out"] / "scene_selection.json", sel)
        if best:
            print(f"{site}: {best['strip_id']} {str(best['acquired'])[:16]} gap {best['gap_days']:.1f} d, "
                  f"clear {best['aoi_clear']:.2f}, {best['bundle']}")
        else:
            print(f"{site}: no eligible scene within {ic['windows_days'][-1]} d")


def selection_table(cfg) -> pd.DataFrame:
    rows = []
    for site in cfg["sites"]:
        sp = ccfg.site_paths(cfg, site)
        f = sp["out"] / "scene_selection.json"
        if S.site_strip(cfg, site):
            rows.append({"site": site, "strip_id": S.site_strip(cfg, site), "status": "existing order"})
            continue
        if not f.exists():
            rows.append({"site": site, "status": "not searched"})
            continue
        s = json.loads(f.read_text())
        if not s["selected"]:
            rows.append({"site": site, "status": "no eligible scene", "survey_utc": s["survey_utc"]})
            continue
        rows.append({"site": site, "status": "selected", "survey_utc": " .. ".join(x[:10] for x in s["survey_utc"]),
                     "window_days": s["window_days"], **s["selected"]})
    return pd.DataFrame(rows)


def order_rows(cfg) -> tuple[list, dict]:
    out = []
    for r in selection_table(cfg).itertuples():
        if r.status != "selected":
            continue
        out.append({"site": r.site, "order_name": f"{r.site}_{pd.Timestamp(r.acquired):%Y%m%d}_{r.strip_id}",
                    "strip_id": r.strip_id, "acquired_utc": r.acquired, "item_ids": r.item_ids, "n_items": int(r.n_items),
                    "bundle": r.bundle, "targets": cfg["experiment"], "est_km2": round(float(r.clip_km2), 1)})
    return out


def stage_order(cfg, submit):
    from planet_order import submit as submit_orders
    rows = order_rows(cfg)
    tab = pd.DataFrame(rows)
    with pd.option_context("display.width", 220):
        print(tab.drop(columns=["item_ids", "targets"]).to_string(index=False))
    total = float(tab["est_km2"].sum()) if len(tab) else 0.0
    print(f"total ~{total:.0f} km2 (budget {cfg['imagery']['order_budget_km2']} km2)")
    if total > cfg["imagery"]["order_budget_km2"]:
        raise SystemExit("selection exceeds imagery.order_budget_km2")
    if not submit:
        print("dry run: add --submit to place these orders")
        return
    pc = cfg["imagery"]["planet"]
    for r in rows:
        sp, P = ccfg.site_paths(cfg, r["site"]), ccfg.planet_paths(cfg, r["site"])
        clip = read_geojson(sp["work"] / "crop.geojson")
        asyncio.run(submit_orders(pd.DataFrame([r]), clip, pc, P))


def stage_download(cfg):
    from planet_order import download
    for site in cfg["sites"]:
        P = ccfg.planet_paths(cfg, site)
        if P["orders"].exists():
            asyncio.run(download(P, None, cfg["retries"]))


def scene_dir(cfg, site) -> Path:
    P = ccfg.planet_paths(cfg, site)
    strip = S.site_strip(cfg, site)
    if not strip:
        sel = json.loads((ccfg.site_paths(cfg, site)["out"] / "scene_selection.json").read_text())["selected"]
        strip = sel["strip_id"]
    d = sorted(P["scenes"].glob(f"*_{strip}"))
    if not d:
        raise SystemExit(f"{site}: strip {strip} not downloaded under {P['scenes']}")
    return d[0]


def stage_build(cfg, names, workers):
    for site in names:
        s = S.build_site(cfg, site, scene_dir(cfg, site), workers)
        print(f"{site}: eligible {s['cells']['eligible']} ({s['eligible_km2']} km2), z {s['eligible_z']}, "
              f"sufficient={s['sufficient']}")


def stage_finalize(cfg, names):
    """Recompute target / nDSM / eligibility from stored rasters (no LAZ re-read)."""
    for site in names:
        s = S.finalize_site(cfg, site)
        print(f"{site}: eligible {s['cells']['eligible']} ({s['eligible_km2']} km2), sufficient={s['sufficient']}, "
              f"z p05/p50/p95 {s['eligible_z']['p05']:.2f}/{s['eligible_z']['p50']:.2f}/{s['eligible_z']['p95']:.2f}"
              if s["eligible_z"] else f"{site}: no eligible centers")


def stage_summary(cfg):
    rows = []
    for site in cfg["sites"]:
        sp = ccfg.site_paths(cfg, site)
        a = json.loads((sp["out"] / "audit.json").read_text()) if (sp["out"] / "audit.json").exists() else {}
        b = json.loads((sp["out"] / "site_summary.json").read_text()) if (sp["out"] / "site_summary.json").exists() else {}
        sel = selection_table(cfg).set_index("site").loc[site].to_dict()
        rows.append({"site": site, "navd88": a.get("navd88"), "survey": a.get("survey_utc"), "crop_km2": a.get("crop_km2"),
                     "scene": sel.get("strip_id"), "scene_acquired": sel.get("acquired"), "gap_days": sel.get("gap_days"),
                     "instrument": sel.get("instrument"), "eligible": b.get("cells", {}).get("eligible"),
                     "eligible_km2": b.get("eligible_km2"), "z": b.get("eligible_z"), "sufficient": b.get("sufficient")})
    df = pd.DataFrame(rows)
    out = ccfg.out_dir(cfg) / "sites.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ccfg.CNN))
    ap.add_argument("--stage", required=True, choices=["audit", "search", "order", "download", "build", "finalize", "summary"])
    ap.add_argument("--site", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    cfg = ccfg.load(Path(a.config))
    names = a.site or list(cfg["sites"])
    unknown = set(names) - set(cfg["sites"])
    if unknown:
        raise SystemExit(f"unknown sites {sorted(unknown)}")
    {"audit": lambda: stage_audit(cfg, names, a.workers), "search": lambda: stage_search(cfg, names, a.refresh),
     "order": lambda: stage_order(cfg, a.submit), "download": lambda: stage_download(cfg),
     "build": lambda: stage_build(cfg, names, a.workers), "summary": lambda: stage_summary(cfg),
     "finalize": lambda: stage_finalize(cfg, names)}[a.stage]()


if __name__ == "__main__":
    main()
