#!/usr/bin/env python
"""Gap-fill reference imagery for track/imagery figures that have none.

  planet_gapfill.py --site fukutoku_okanoba [--max-dt-days 32] [--budget-km2 3000] [--submit] [--download]

For every pass in tables/track_imagery_pairs.csv without a strip, choose the
clearest strip within +-max-dt-days (core footprint >= aoi_footprint_min, else
the largest footprint; then highest UDM2 core clear, unknown lowest; then
smallest |dt|). These are reference images, so there is no clear threshold.
Strips already downloaded are reused; new strips are ordered closest-in-time
first until the budget is reached, through planet_order.py's submit/download
(same orders.json bookkeeping, never resubmitted). Dry run by default.

Needs a current planet_search.py run (planet_candidates.csv with UDM2 clear).
Writes tables/planet_gapfill_plan.csv (one row per pass).
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json

import pandas as pd
from shapely.geometry import shape

from island_altimetry import config, manifest
from island_altimetry import planet_imagery as pim
from planet_order import download, load_orders, submit


def catalog_permissions(P: dict) -> dict:
    perms = {}
    for f in glob.glob(str(P["catalog"] / "*.json")):
        for i in json.loads(open(f).read())["items"]:
            perms[i["id"]] = ";".join(i.get("_permissions", []))
    return perms


def plan_gapfill(missing: pd.DataFrame, strips: pd.DataFrame, orders: dict, perms: dict, pc: dict,
                 window_h: float, site_id: str) -> pd.DataFrame:
    """One row per pass: chosen strip, dt, clear, footprint, action (reuse/new/none)."""
    have = {o["strip_id"]: o.get("download_status") for o in orders.values()}
    rows = []
    for _, r in missing.iterrows():
        t = pd.Timestamp(r["pass_utc"])
        row = {"product": r["product"], "granule": r["granule"], "pass_utc": r["pass_utc"]}
        got = pim.gapfill_choice(t, strips, window_h, pc["aoi_footprint_min"])
        if got is None:
            rows.append(row | {"action": "none"})
            continue
        s, dt_h = got
        ids = s["item_ids"].split(";")
        action = "reuse" if have.get(s["strip_id"]) == "ok" else "ordered" if s["strip_id"] in have else "new"
        rows.append(row | {
            "action": action, "strip_id": s["strip_id"], "acquired_utc": pd.Timestamp(s["acquired"]).isoformat(),
            "dt_hours": round(dt_h, 2), "aoi_clear": s["aoi_clear"], "aoi_footprint": round(float(s["aoi_footprint"]), 3),
            "item_ids": s["item_ids"], "n_items": len(ids),
            "bundle": pim.choose_bundle([set(perms.get(i, "").split(";")) for i in ids], pc["bundles"]),
            "est_km2": round(float(s["clip_km2"]), 1),
            "order_name": f"{site_id}_{pd.Timestamp(s['acquired']):%Y%m%d}_{s['strip_id']}"})
    return pd.DataFrame(rows)


def new_orders(plan: pd.DataFrame, budget_km2: float | None) -> pd.DataFrame:
    """Unique new strips with a bundle, closest-in-time first, within budget."""
    if plan.empty or "action" not in plan or not (plan["action"] == "new").any():
        return pd.DataFrame()
    n = plan[(plan["action"] == "new") & plan["bundle"].notna()].copy()
    n["adt"] = n["dt_hours"].abs()
    g = (n.sort_values("adt").groupby("strip_id", sort=False)
         .agg(order_name=("order_name", "first"), acquired_utc=("acquired_utc", "first"),
              item_ids=("item_ids", "first"), n_items=("n_items", "first"), bundle=("bundle", "first"),
              est_km2=("est_km2", "first"), adt=("adt", "min"),
              targets=("granule", lambda g: ";".join(f"gapfill:{x}" for x in g)))
         .reset_index().sort_values("adt"))
    g["n_items"] = g["n_items"].astype(int)   # NaN rows (action none) made the column float
    if budget_km2 is not None:
        g = g[g["est_km2"].cumsum() <= budget_km2]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--max-dt-days", type=float)
    ap.add_argument("--budget-km2", type=float)
    ap.add_argument("--submit", action="store_true", help="place the new orders (uses quota)")
    ap.add_argument("--download", action="store_true", help="wait for and download gap-fill orders")
    a = ap.parse_args()
    cfg = config.resolve_site(a.site)
    pc = cfg["imagery"]["planet"]
    run_id = a.run_id or manifest.latest_run_id(cfg)
    P = pim.paths(cfg, run_id)
    m = manifest.load(cfg, run_id)
    days = a.max_dt_days or pc.get("gapfill_max_dt_days", 32)

    pairs = pd.read_csv(P["tables"] / "track_imagery_pairs.csv")
    missing = pairs[pairs["strip_id"].isna()]
    strips = pd.read_csv(P["tables"] / "planet_candidates.csv", dtype={"strip_id": str})
    strips["acquired"] = pd.to_datetime(strips["acquired"], utc=True)
    orders = load_orders(P["orders"])
    plan = plan_gapfill(missing, strips, orders, catalog_permissions(P), pc, days * 24, a.site)
    plan.to_csv(P["tables"] / "planet_gapfill_plan.csv", index=False)

    sel = new_orders(plan, a.budget_km2)
    counts = plan["action"].value_counts().to_dict()
    n_new = int(plan.loc[plan["action"] == "new", "strip_id"].nunique()) if "strip_id" in plan else 0
    print(f"== {a.site} {run_id}: {len(missing)} passes without imagery, window ±{days:g} d -> {counts}")
    print(f"new strips: {n_new} needed, {len(sel)} selected, ~{sel['est_km2'].sum() if not sel.empty else 0:.0f} km2"
          + (f" (budget {a.budget_km2:.0f})" if a.budget_km2 else ""))
    if not sel.empty:
        print(sel[["order_name", "adt", "bundle", "est_km2", "targets"]].to_string(index=False))
    if a.submit:
        if a.budget_km2 is None:
            raise SystemExit("--submit requires --budget-km2")
        clip = shape(json.loads((config.ROOT / m["aoi_dir"] / f"{pc['clip_aoi']}.geojson").read_text())
                     ["features"][0]["geometry"])
        if not sel.empty:
            asyncio.run(submit(sel.assign(best_tier="gapfill"), clip, pc, P))
    elif not a.download:
        print("dry run: add --submit to place these orders")
    if a.download:
        asyncio.run(download(P, list(sel["order_name"]) if not sel.empty else [], cfg["retries"]))


if __name__ == "__main__":
    main()
