#!/usr/bin/env python
"""M8b: order and download PlanetScope strips selected by planet_search.py.
Orders use download quota, so the default is a dry run that only prints the
selection and its estimated clipped area.

  planet_order.py --site fukutoku_okanoba                        # dry run, all strips
  planet_order.py --site X --max-tier d1 --budget-km2 2000       # dry run, narrowed
  planet_order.py --site X --max-tier d1 --budget-km2 2000 --submit
  planet_order.py --site X --download                            # wait + download placed orders

One order per strip (name <site>_<YYYYMMDD>_<strip_id>), clipped to the
configured AOI, SR bundles harmonized to Sentinel-2. Placed orders are recorded
in data/<site>/imagery/planet/orders.json *before* polling; re-runs reuse
recorded or same-named existing orders and never resubmit them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from shapely.geometry import shape

from island_altimetry import config, manifest, status
from island_altimetry import planet_imagery as pim
from island_altimetry.config import fmt_utc

FAILED_STATES = ("failed", "cancelled")


def now() -> str:
    return fmt_utc(pd.Timestamp.now(tz="UTC"))


def load_orders(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def scene_dir(P: dict, row: dict) -> Path:
    return P["scenes"] / f"{pd.Timestamp(row['acquired_utc']):%Y%m%d}_{row['strip_id']}"


async def existing_order(client, name: str) -> dict | None:
    async for o in client.list_orders(name=name):
        if o.get("name") == name and o.get("state") not in FAILED_STATES:
            return o
    return None


async def submit(sel: pd.DataFrame, clip_geom, pc: dict, P: dict) -> None:
    from planet import Session
    orders = load_orders(P["orders"])
    async with Session() as sess:
        client = sess.client("orders")
        for _, r in sel.iterrows():
            rec = orders.get(r["order_name"])
            if rec and rec.get("state") not in FAILED_STATES:
                print(f"  {r['order_name']}: already placed ({rec['id']}, {rec.get('state')})")
                continue
            found = await existing_order(client, r["order_name"])
            if found:
                print(f"  {r['order_name']}: found existing order {found['id']} ({found['state']})")
                o = found
            else:
                req = pim.build_order_request(r.to_dict(), clip_geom, pc, pc["item_type"])
                o = await client.create_order(req)
                print(f"  {r['order_name']}: placed {o['id']} ({r['bundle']}, {r['n_items']} item(s))")
            orders[r["order_name"]] = {"id": o["id"], "state": o.get("state"), "strip_id": r["strip_id"],
                                       "acquired_utc": r["acquired_utc"], "bundle": r["bundle"],
                                       "item_ids": r["item_ids"], "targets": r["targets"],
                                       "placed_utc": now(), "download_status": status.NOT_YET_QUERIED}
            manifest.write_json_atomic(P["orders"], orders)   # record before any polling


async def download(P: dict, names: list[str] | None, retries: int) -> None:
    from planet import Session
    orders = load_orders(P["orders"])
    todo = {k: v for k, v in orders.items() if (names is None or k in names) and v.get("download_status") != status.OK}
    if not todo:
        print("nothing to download")
        return
    async with Session() as sess:
        client = sess.client("orders")

        async def one(name, rec):
            try:
                state = await client.wait(rec["id"], delay=30, max_attempts=240)
                rec["state"] = state
                if state not in ("success", "partial"):
                    rec["download_status"] = status.RETRIEVAL_FAILED
                    return
                dest = scene_dir(P, rec)
                dest.mkdir(parents=True, exist_ok=True)
                for attempt in range(retries + 1):
                    try:
                        files = await client.download_order(rec["id"], directory=dest, overwrite=False)
                        client.validate_checksum(dest / rec["id"], "MD5")
                        break
                    except Exception:  # noqa: BLE001
                        if attempt == retries:
                            raise
                rec.update(download_status=status.OK, downloaded_utc=now(), dir=str(dest.relative_to(config.ROOT)),
                           files=[str(Path(f).relative_to(dest)) for f in files])
            except Exception as e:  # noqa: BLE001 - recorded, not swallowed silently
                rec.update(download_status=status.RETRIEVAL_FAILED, error=f"{type(e).__name__}: {e}")
            finally:
                orders[name] = rec
                manifest.write_json_atomic(P["orders"], orders)
                print(f"  {name}: {rec.get('state')} / {rec['download_status']}")

        await asyncio.gather(*(one(k, v) for k, v in todo.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--max-tier", help="only strips whose best match is this tier or closer (e.g. d1)")
    ap.add_argument("--usable-only", action="store_true", help="only strips matched to a usable-core lidar pass")
    ap.add_argument("--targets", nargs="+", help="granule ids (substring match) to order for")
    ap.add_argument("--budget-km2", type=float, help="hard cap on estimated clipped area (overrides config)")
    ap.add_argument("--submit", action="store_true", help="place orders (uses quota)")
    ap.add_argument("--download", action="store_true", help="wait for and download recorded orders")
    a = ap.parse_args()
    for site in (config.site_ids() if a.all else [a.site]):
        cfg = config.resolve_site(site)
        pc = cfg["imagery"]["planet"]
        run_id = a.run_id or manifest.latest_run_id(cfg)
        P = pim.paths(cfg, run_id)
        m = manifest.load(cfg, run_id)
        plan_f = P["tables"] / "planet_order_plan.csv"
        if not plan_f.exists() or plan_f.stat().st_size < 2:
            raise SystemExit(f"no order plan at {plan_f}; run scripts/planet_search.py first")
        plan = pd.read_csv(plan_f, dtype={"strip_id": str})
        sel = pim.select_orders(plan, pc, a.max_tier, a.usable_only, a.targets)
        budget = a.budget_km2 if a.budget_km2 is not None else pc.get("order_budget_km2")
        print(f"== {site} {run_id}: {len(sel)} of {len(plan)} strips selected "
              f"(tiers {sel['best_tier'].value_counts().to_dict()}), ~{sel['est_km2'].sum():.0f} km2"
              + (f", budget {budget:.0f} km2" if budget else ""))
        if a.submit:
            if budget is None:
                raise SystemExit("--submit requires a budget (--budget-km2 or imagery.planet.order_budget_km2)")
            pim.check_budget(sel, budget)
            clip = shape(json.loads((config.ROOT / m["aoi_dir"] / f"{pc['clip_aoi']}.geojson").read_text())
                         ["features"][0]["geometry"])
            asyncio.run(submit(sel, clip, pc, P))
        elif not a.download:
            if budget:
                pim.check_budget(sel, budget)
            print(sel[["order_name", "best_tier", "min_abs_dt_hours", "aoi_clear", "bundle", "products"]]
                  .to_string(index=False))
            print("dry run: add --submit to place these orders")
        if a.download:
            names = list(sel["order_name"]) if (a.max_tier or a.usable_only or a.targets) else None
            asyncio.run(download(P, names, cfg["retries"]))
        orders = load_orders(P["orders"])
        if orders:
            manifest.update(cfg, run_id, "planet_orders", {
                "updated_utc": now(), "orders_file": str(P["orders"].relative_to(config.ROOT)),
                "n_orders": len(orders),
                "download_status": pd.Series([o.get("download_status") for o in orders.values()]).value_counts().to_dict()})


if __name__ == "__main__":
    main()
