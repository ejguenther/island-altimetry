#!/usr/bin/env python
"""M8a: find mostly cloud-free PlanetScope strips coincident with every
AOI-crossing ATL03 and GEDI L2A acquisition. Metadata only: Data API search and
per-AOI coverage estimates do not use download quota.

  planet_search.py --site fukutoku_okanoba [--run-id RUN] [--refresh]

Outputs (outputs/<site>/<run>/tables/): planet_targets.csv, planet_candidates.csv
(strips within the widest tier of any target), planet_matches.csv (one row per
target), planet_order_plan.csv (unique strips to order). Raw search results and
the coverage cache live under data/<site>/imagery/planet/.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import pandas as pd
import shapely
from shapely.geometry import shape

from island_altimetry import adapters, config, manifest
from island_altimetry import planet_imagery as pim
from island_altimetry.config import fmt_utc
from scripts_path import load_acquire

acq = load_acquire()


def load_targets(ctx: dict) -> pd.DataFrame:
    cfg, inv_dir = ctx["cfg"], ctx["inv_dir"]
    invs, stores = {}, {}
    for pid in pim.TARGET_PRODUCTS:
        if pid not in cfg["products"]:
            continue
        invs[pid] = pd.read_csv(inv_dir / f"inventory_{pid}.csv", dtype=str)
        parms = adapters.build_parms(pid, ctx["products"][pid], ctx["products"]["atl03"], ctx["poly"])
        stores[pid] = acq.store_for(ctx, pid, parms)

    def subset_path(pid, granule):
        return stores[pid].data_path(granule) if stores[pid].is_complete(granule) else None

    cov_f = manifest.output_dir(cfg, ctx["run_id"]) / "tables" / "coverage_by_acquisition.csv"
    cov = pd.read_csv(cov_f, dtype={"granule": str}) if cov_f.exists() else None
    return pim.build_targets(invs, subset_path, cov)


def aoi_geom(ctx: dict, name: str):
    return shape(json.loads((config.ROOT / ctx["m"]["aoi_dir"] / f"{name}.geojson").read_text())["features"][0]["geometry"])


async def search(ctx: dict, targets: pd.DataFrame, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    from planet import Session
    cfg, pc = ctx["cfg"], ctx["cfg"]["imagery"]["planet"]
    P = pim.paths(cfg, ctx["run_id"])
    search_aoi = aoi_geom(ctx, pc["clip_aoi"])          # what gets ordered
    aoi = aoi_geom(ctx, pc.get("match_aoi", pc["clip_aoi"]))   # where coverage/clear are judged
    proj = pim.Projector(cfg["center"]["lon"], cfg["center"]["lat"])
    pad = pd.Timedelta(hours=max(pc["tiers_hours"].values()))
    t0 = (targets["overpass_utc"].min() - pad).floor("s").to_pydatetime()
    t1 = (targets["overpass_utc"].max() + pad).ceil("s").to_pydatetime()

    async with Session() as sess:
        client = sess.client("data")
        print(f"searching {pc['item_type']} {t0:%Y-%m-%d}..{t1:%Y-%m-%d} (scene cloud <= {pc['scene_cloud_max']})")
        raw = await pim.fetch_catalog(client, pc["item_type"], search_aoi, t0, t1, pc["scene_cloud_max"],
                                      P["catalog"], refresh, pc.get("standard_quality_only", False))
        items = pd.DataFrame([pim.item_row(i) for i in raw]).drop_duplicates("item_id")
        if items.empty:
            return items, pd.DataFrame()
        items["item_footprint"] = [pim.footprint_fraction(g, aoi, proj) for g in items["geometry"]]
        near = pim.near_any_target(items["acquired"], targets["overpass_utc"], max(pc["tiers_hours"].values()))
        # Coverage only for items near a target that overlap the AOI at all.
        need = items[near & (items["item_footprint"] > 0)]
        print(f"{len(items)} items; {len(need)} within widest tier of a target -> coverage estimates")
        cov = await pim.estimate_coverage(client, pc["item_type"], list(need["item_id"]), aoi, P["coverage"],
                                          workers=pc.get("coverage_workers", 4), retries=cfg["retries"],
                                          backoff_s=cfg["retry_backoff_s"], mode=pc.get("coverage_mode", "UDM2"))
    counts = pim.coverage_counts(cov)
    print(f"coverage: {counts}")
    if counts["incomplete"] or counts["failed"]:
        print("  WARNING: items without a coverage result are treated as unknown clear (never matched); "
              "re-run planet_search.py to resolve them")
    cdf = pd.DataFrame.from_dict(cov, orient="index").rename_axis("item_id").reset_index()
    items = items[items["item_id"].isin(need["item_id"])].merge(cdf, on="item_id", how="left")
    strips = pim.group_strips(items, aoi, proj)
    if not strips.empty:
        geoms = items.groupby("strip_id")["geometry"].agg(lambda g: shapely.union_all(list(g)))
        strips["clip_km2"] = [pim.area_km2(geoms[s].intersection(search_aoi), proj) for s in strips["strip_id"]]
    strips.attrs["coverage_counts"] = counts
    return items, strips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--refresh", action="store_true", help="re-query cached search windows")
    a = ap.parse_args()
    for site in (config.site_ids() if a.all else [a.site]):
        cfg = config.resolve_site(site)
        run_id = a.run_id or manifest.latest_run_id(cfg)
        ctx = acq.context(site, run_id) | {"run_id": run_id}
        pc = cfg["imagery"]["planet"]
        P = pim.paths(cfg, run_id)
        P["tables"].mkdir(parents=True, exist_ok=True)

        targets = load_targets(ctx)
        print(f"== {site} {run_id}: {len(targets)} AOI-crossing targets "
              f"({targets['product'].value_counts().to_dict()}; "
              f"{(targets['time_source'] == 'granule_start').sum()} timed from granule start)")
        targets.to_csv(P["tables"] / "planet_targets.csv", index=False)

        try:
            items, strips = asyncio.run(search(ctx, targets, a.refresh))
        except Exception as e:
            if type(e).__name__ in ("InvalidAPIKey", "AuthException", "ClientError") and "credential" in str(e).lower():
                raise SystemExit("Planet auth failed: run `.venv/bin/planet auth login` (or export PL_API_KEY)") from None
            raise
        matches = pim.match_targets(targets, strips, pc)
        clip_km2 = strips.set_index("strip_id")["clip_km2"].to_dict() if not strips.empty else {}
        plan = pim.order_plan(matches, items, pc, clip_km2, site) if not items.empty else pd.DataFrame()

        strips.to_csv(P["tables"] / "planet_candidates.csv", index=False)
        matches.to_csv(P["tables"] / "planet_matches.csv", index=False)
        plan.to_csv(P["tables"] / "planet_order_plan.csv", index=False)

        tiers = matches["match_tier"].fillna("unmatched").value_counts().reindex(
            [*pc["tiers_hours"], "unmatched"], fill_value=0)
        no_bundle = int(plan["bundle"].isna().sum()) if not plan.empty else 0
        print("matches by tier:", tiers.to_dict())
        print(f"order plan: {len(plan)} unique strips, ~{plan['est_km2'].sum() if not plan.empty else 0:.0f} km2 "
              f"({no_bundle} strips with no downloadable bundle)")
        manifest.update(cfg, run_id, "planet_search", {
            "completed_utc": fmt_utc(pd.Timestamp.now(tz="UTC")), "n_targets": len(targets),
            "n_items_considered": len(items), "n_strips": len(strips),
            "coverage_counts": strips.attrs.get("coverage_counts"),
            "matches_by_tier": tiers.to_dict(), "n_order_strips": len(plan),
            "est_order_km2": float(plan["est_km2"].sum()) if not plan.empty else 0.0,
            "criteria": {k: pc.get(k) for k in ("item_type", "coverage_mode", "match_aoi", "clip_aoi", "scene_cloud_max", "standard_quality_only", "aoi_clear_min",
                                             "aoi_footprint_min", "tiers_hours")},
            "tables": [str((P["tables"] / f).relative_to(config.ROOT)) for f in
                       ("planet_targets.csv", "planet_candidates.csv", "planet_matches.csv", "planet_order_plan.csv")]})


if __name__ == "__main__":
    main()
