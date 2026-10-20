#!/usr/bin/env python
"""M2: inventory every candidate granule per product over the outer AOI from
each product's start through the run's frozen end.

  inventory.py --site fukutoku_okanoba [--run-id RUN]
  inventory.py --all

Outputs (data/<site>/inventory/<run_id>/ and outputs/<site>/<run_id>/tables/):
raw discovery responses, per-product inventories, discarded revisions,
completeness ledger, ATL03 association table, monthly/annual counts, latency.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon, shape

from island_altimetry import config, inventory as inv, manifest
from island_altimetry.config import fmt_utc, parse_utc

CMR_SHORT = {"atl03": "ATL03", "atl08": "ATL08", "atl09": "ATL09", "atl24": "ATL24", "gedi_l2a": "GEDI02_A"}


def load_zones(m: dict) -> dict:
    d = config.ROOT / m["aoi_dir"]
    return {z: shape(json.loads((d / f"{z}.geojson").read_text())["features"][0]["geometry"])
            for z in ("core", "outer")}


def to_poly(g) -> list[dict]:
    from island_altimetry.aoi import to_sliderule_poly
    return to_sliderule_poly(g)


def footprint(meta: dict):
    try:
        return Polygon([(float(p["lon"]), float(p["lat"])) for p in meta.get("poly", [])]).buffer(0)
    except Exception:
        return None


def run_product(pid: str, cfg: dict, m: dict, zones: dict, raw_dir: Path) -> dict:
    p = m["products"][pid]
    asset = p["sliderule_asset"]
    release = sorted({r["release"] for r in p["served_release"]})
    t0, t1 = m["query_start_utc"][pid], m["query_end_utc"]
    poly = to_poly(zones["outer"])

    ledger = []
    found = {}
    for a, b in inv.windows(t0, t1, cfg["inventory_window_days"]):
        f, rows = inv.sliderule_discover(asset, poly, a, b, cfg["discovery_cap"], cfg["retries"], cfg["retry_backoff_s"])
        found.update(f)
        ledger += [{"product": pid, **r} for r in rows]
    (raw_dir / f"{pid}_sliderule.json").write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")

    cmr_rows, cmr_led = [], []
    for v in release:
        e, row = inv.cmr_discover(CMR_SHORT[pid], v, poly, t0, t1, cfg["retries"])
        cmr_rows += e
        cmr_led.append({"product": pid, **row})
    (raw_dir / f"{pid}_cmr_check.json").write_text(json.dumps(cmr_rows, indent=1) + "\n")
    ledger += cmr_led

    rows = []
    for name, meta in sorted(found.items()):
        r = {"product": pid, **inv.parse_name(pid, name), "resource": name}
        r["acq_start_utc"] = meta.get("t0") or r["acq_start_name_utc"]
        r["acq_end_utc"] = meta.get("t1")
        fp = footprint(meta)
        r["footprint_intersects_outer"] = bool(fp is not None and fp.intersects(zones["outer"]))
        r["footprint_intersects_core"] = bool(fp is not None and fp.intersects(zones["core"]))
        r["discovery_route"] = "sliderule"
        rows.append(r)
    selected, discarded = inv.dedupe(rows) if rows else ([], [])

    # CMR completeness comparison (granule base names; CMR titles may carry suffixes).
    def base(n):
        return n.split(".h5")[0]
    sr = {base(n) for n in found}
    cmr = {base(e.get("producer_granule_id") or e["title"]) for e in cmr_rows}
    cmr_ok = all(r["status"] in ("ok", "no_granules") for r in cmr_led)
    sr_ok = all(r["status"] in ("ok", "no_granules", "split_cap_exceeded") for r in ledger if r["route"] == "sliderule")
    summary = {
        "product": pid, "asset": asset, "release": ";".join(release), "t0": t0, "t1": t1,
        "sliderule_status": "complete" if sr_ok else "retrieval_failed",
        "cmr_check_status": "complete" if cmr_ok else "retrieval_failed",
        "n_sliderule": len(sr), "n_cmr_check": len(cmr) if cmr_ok else None,
        "only_in_sliderule": ";".join(sorted(sr - cmr)) if cmr_ok else "",
        "only_in_cmr": ";".join(sorted(cmr - sr)) if cmr_ok else "",
        "n_selected": len(selected), "n_discarded_revisions": len(discarded),
        "cap_splits": sum(r["status"] == "split_cap_exceeded" for r in ledger),
    }
    if cmr_ok and sr_ok:
        summary["completeness"] = "matched" if sr == cmr else "mismatch"
    else:
        summary["completeness"] = "unresolved"
    return {"selected": selected, "discarded": discarded, "ledger": ledger, "summary": summary}


def complete_lineage(sel: dict[str, pd.DataFrame], m: dict, retries: int) -> list[dict]:
    """Cross-product completion: catalog footprints differ per product, so an
    acquisition found for one ICESat-2 product may be missed for another. Look
    up missing partners by exact name in CMR (read-only) and add them, labeled."""
    ledger = []
    along = [p for p in ("atl03", "atl08", "atl24") if p in sel]
    keys = {}
    for p in along:
        for _, r in sel[p].iterrows():
            keys.setdefault(r["lineage_key"], r)
    for p in along:
        have = set(sel[p]["lineage_key"]) if not sel[p].empty else set()
        rel = sorted({r["release"] for r in m["products"][p]["served_release"]})[0]
        add = []
        for k, ref in sorted(keys.items()):
            if k in have:
                continue
            acq, ident = k.split("_")
            pat = (f"ATL24_{acq}_{ident}_*_{rel}_*" if p == "atl24" else f"{CMR_SHORT[p]}_{acq}_{ident}_{rel}_*")
            hits, row = inv.cmr_name_lookup(CMR_SHORT[p], rel, pat, retries)
            ledger.append({"product": p, **row})
            for h in hits:
                r = {"product": p, **inv.parse_name(p, h["title"]), "resource": h["title"].split(".h5")[0] + ".h5",
                     "acq_start_utc": h["time_start"], "acq_end_utc": h["time_end"],
                     "footprint_intersects_outer": False, "footprint_intersects_core": False,
                     "discovery_route": f"cmr_name_lookup (partner of {ref['granule']})"}
                add.append(r)
        if add:
            sel[p] = pd.concat([sel[p], pd.DataFrame(add)], ignore_index=True)
    # ATL09 partners for ATL03 acquisitions (orbit-segment granules, different naming).
    if "atl09" in sel and "atl03" in sel:
        rel = sorted({r["release"] for r in m["products"]["atl09"]["served_release"]})[0]
        add = []
        for _, r in sel["atl03"].iterrows():
            df = sel["atl09"]
            if not df.empty and ((df["rgt"] == r["rgt"]) & (df["cycle"] == r["cycle"])).any():
                continue
            hits, row = inv.cmr_name_lookup("ATL09", rel, f"ATL09_*_{r['rgt']:04d}{r['cycle']:02d}*_{rel}_*", retries)
            ledger.append({"product": "atl09", **row})
            for h in hits:
                add.append({"product": "atl09", **inv.parse_name("atl09", h["title"]),
                            "resource": h["title"].split(".h5")[0] + ".h5",
                            "acq_start_utc": h["time_start"], "acq_end_utc": h["time_end"],
                            "footprint_intersects_outer": False, "footprint_intersects_core": False,
                            "discovery_route": f"cmr_name_lookup (partner of {r['granule']})"})
        if add:
            sel["atl09"] = pd.concat([sel["atl09"], pd.DataFrame(add)], ignore_index=True).drop_duplicates("granule")
    return ledger


def associations(sel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ATL03-anchored associations using documented identifiers (acquisition
    time, RGT, cycle, region), not filename substitution."""
    a03 = sel.get("atl03")
    if a03 is None or a03.empty:
        return pd.DataFrame()
    out = []
    for _, r in a03.iterrows():
        row = {"atl03": r["granule"], "acq_start_utc": r["acq_start_utc"], "rgt": r["rgt"], "cycle": r["cycle"], "region": r["region"]}
        for pid in ("atl08", "atl24"):
            df = sel.get(pid)
            hit = df[df["lineage_key"] == r["lineage_key"]] if df is not None and not df.empty else df
            row[pid] = ";".join(hit["granule"]) if hit is not None and len(hit) else ""
        row["atl08_basis"] = "same acquisition id (time, RGT, cycle, region); same release 007"
        row["atl24_basis"] = ("same acquisition id; ATL24 v002 derived from ATL03 r006: "
                              "spatial/temporal association only, no photon-index joins")
        df = sel.get("atl09")
        if df is not None and not df.empty:
            t0, t1 = parse_utc(r["acq_start_utc"]), parse_utc(r["acq_end_utc"] or r["acq_start_utc"])
            cand = df[(df["rgt"] == r["rgt"]) & (df["cycle"] == r["cycle"])]
            mask = [parse_utc(s) <= t1 and parse_utc(e if isinstance(e, str) else s) >= t0
                    for s, e in zip(cand["acq_start_utc"], cand["acq_end_utc"])]
            cand = cand.loc[mask] if len(cand) else cand
            row["atl09"] = ";".join(cand["granule"])
        else:
            row["atl09"] = ""
        row["atl09_basis"] = "same RGT and cycle with overlapping acquisition interval (ATL09 granules span larger orbit segments)"
        out.append(row)
    return pd.DataFrame(out)


def counts(sel: dict[str, pd.DataFrame], run_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly, annual, lat = [], [], []
    for pid, df in sel.items():
        if df.empty:
            lat.append({"product": pid, "n_candidates": 0, "earliest_utc": None, "latest_utc": None, "latency_days": None})
            continue
        t = pd.to_datetime(df["acq_start_utc"], utc=True)
        for k, v in t.dt.strftime("%Y-%m").value_counts().sort_index().items():
            monthly.append({"product": pid, "month": k, "n_candidate_granules": int(v)})
        for k, v in t.dt.year.value_counts().sort_index().items():
            annual.append({"product": pid, "year": int(k), "n_candidate_granules": int(v)})
        end = pd.Timestamp(run_end)
        lat.append({"product": pid, "n_candidates": len(df), "earliest_utc": fmt_utc(t.min()),
                    "latest_utc": fmt_utc(t.max()), "latency_days": round((end - t.max()).total_seconds() / 86400, 1)})
    return pd.DataFrame(monthly), pd.DataFrame(annual), pd.DataFrame(lat)


def run_site(site_id: str, run_id: str | None):
    cfg = config.resolve_site(site_id)
    m = manifest.load(cfg, run_id)
    run_id = m["run_id"]
    zones = load_zones(m)
    inv_dir = manifest.data_dir(cfg) / "inventory" / run_id
    raw_dir = inv_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables = manifest.output_dir(cfg, run_id) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    sel, ledger, summaries, discarded = {}, [], [], []
    for pid in cfg["products"]:
        print(f"== {pid}")
        res = run_product(pid, cfg, m, zones, raw_dir)
        df = pd.DataFrame(res["selected"])
        df.to_csv(inv_dir / f"inventory_{pid}.csv", index=False)
        sel[pid] = df
        ledger += res["ledger"]
        discarded += res["discarded"]
        summaries.append(res["summary"])
        s = res["summary"]
        print(f"   sliderule={s['n_sliderule']} cmr_check={s['n_cmr_check']} selected={s['n_selected']} "
              f"discarded={s['n_discarded_revisions']} completeness={s['completeness']}")

    ledger += complete_lineage(sel, m, cfg["retries"])
    for pid, df in sel.items():
        df.to_csv(inv_dir / f"inventory_{pid}.csv", index=False)
        added = int((df["discovery_route"] != "sliderule").sum()) if not df.empty else 0
        next(s for s in summaries if s["product"] == pid)["n_added_by_name_lookup"] = added
        if added:
            print(f"   {pid}: +{added} by cross-product name lookup")
    pd.DataFrame(ledger).to_csv(inv_dir / "completeness_ledger.csv", index=False)
    pd.DataFrame(discarded).to_csv(inv_dir / "discarded_revisions.csv", index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(tables / "inventory_summary.csv", index=False)
    assoc = associations(sel)
    assoc.to_csv(inv_dir / "associations.csv", index=False)
    mon, ann, lat = counts(sel, m["query_end_utc"])
    mon.to_csv(tables / "inventory_monthly_counts.csv", index=False)
    ann.to_csv(tables / "inventory_annual_counts.csv", index=False)
    lat.to_csv(tables / "inventory_latency.csv", index=False)
    for f in ("associations.csv",):
        (tables / f).write_bytes((inv_dir / f).read_bytes())

    from island_altimetry import plotting
    fig_dir = manifest.output_dir(cfg, run_id) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(config.ROOT / m["events_table"])
    plotting.timeline(sel, events, f"{site_id}: preliminary acquisition timeline", m["query_end_utc"],
                      fig_dir / "inventory_timeline.png")

    manifest.update(cfg, run_id, "inventory", {
        "dir": str(inv_dir.relative_to(config.ROOT)),
        "completed_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)),
        "spatial_search": "granule footprint vs outer AOI polygon (candidate coverage only)",
        "discovery_routes": {"primary": "sliderule earthdata.search (with_meta)",
                             "check": "NASA CMR granules.json, read-only metadata, same polygon/time"},
        "dedupe_rule": "one granule per (release, acquisition lineage key); highest revision kept; discarded listed",
        "lineage_completion": "partners missing from AOI discovery looked up by exact name in CMR (read-only), "
                              "labeled discovery_route=cmr_name_lookup; actual AOI intersection decided at retrieval",
        "summary": summaries,
    })
    print(lat.to_string(index=False))
    print(f"wrote {inv_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--run-id")
    a = ap.parse_args()
    for s in (config.site_ids() if a.all else [a.site]):
        run_site(s, a.run_id)


if __name__ == "__main__":
    main()
