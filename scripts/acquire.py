#!/usr/bin/env python
"""M4: acquire AOI subsets for every selected inventory resource, with restart
support. Re-running skips verified-complete chunks (checksum-checked); changed
request parameters, AOI, or processing version land in a new cache namespace.

  acquire.py --site fukutoku_okanoba                 # all products, all resources
  acquire.py --site X --product atl03 --limit 3      # partial
  acquire.py --site X --granule ATL03_2021...        # specific granules
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from island_altimetry import adapters, cache, config, manifest, sr_client, status
from island_altimetry.aoi import to_sliderule_poly
from island_altimetry.config import fmt_utc
from shapely.geometry import shape


def context(site_id: str, run_id: str | None) -> dict:
    cfg = config.resolve_site(site_id)
    m = manifest.load(cfg, run_id)
    outer = shape(json.loads((config.ROOT / m["aoi_dir"] / "outer.geojson").read_text())["features"][0]["geometry"])
    return {"cfg": cfg, "m": m, "products": config.load_products(), "poly": to_sliderule_poly(outer),
            "inv_dir": config.ROOT / m["inventory"]["dir"]}


def resources_for(ctx: dict, pid: str) -> pd.DataFrame:
    """ATL09 is retrieved per ATL03 granule (ancillary route)."""
    src = "atl03" if pid == "atl09" else pid
    df = pd.read_csv(ctx["inv_dir"] / f"inventory_{src}.csv", dtype=str)
    return df[["granule", "resource", "acq_start_utc"]].sort_values("acq_start_utc").reset_index(drop=True)


def store_for(ctx: dict, pid: str, parms: dict) -> cache.ChunkStore:
    release = ";".join(sorted({r["release"] for r in ctx["m"]["products"][pid]["served_release"]}))
    return cache.ChunkStore(manifest.data_dir(ctx["cfg"]), pid, release, ctx["m"]["aoi_hash"], cache.request_hash(parms))


def acquire_one(pid: str, contract: dict, parms: dict, store: cache.ChunkStore, granule: str,
                resource: str, cfg: dict) -> dict:
    """Bounded exponential retries for transient failures; unsupported fields are
    dropped (and recorded) rather than discarding the granule's base data."""
    tmp = store.tmp_path(granule)
    dropped: list[str] = []
    attempts = 0
    p = parms
    while True:
        attempts += 1
        st, meta, data = adapters.retrieve(pid, contract, p, granule, resource, tmp)
        if st == status.FIELD_UNSUPPORTED and attempts <= 2:
            bad = sr_client.failed_fields(meta["alerts"], sr_client.requested_fields(p))
            dropped += sorted(bad)
            p = _without_fields(p, bad)
            continue
        if st == status.RETRIEVAL_FAILED and attempts <= cfg["retries"]:
            time.sleep(cfg["retry_backoff_s"] * (2 ** (attempts - 1)) * (0.5 + random.random()))
            continue
        break
    meta.update(attempts=attempts, fields_dropped=dropped,
                sliderule_server=ctx_server_version())
    return store.commit(granule, meta, data)


def _without_fields(parms: dict, bad: set[str]) -> dict:
    out = {}
    for k, v in parms.items():
        if k.endswith("_fields") and isinstance(v, list):
            out[k] = [f for f in v if f not in bad]
        elif isinstance(v, dict):
            out[k] = _without_fields(v, bad)
        else:
            out[k] = v
    return out


_SERVER = None


def ctx_server_version() -> str:
    global _SERVER
    if _SERVER is None:
        _SERVER = sr_client.server_version()["server"]["version"]
    return _SERVER


def run_product(ctx: dict, pid: str, granules: list[str] | None = None, limit: int | None = None,
                workers: int | None = None, quiet: bool = False) -> list[dict]:
    cfg = ctx["cfg"]
    contract = ctx["products"][pid]
    parms = adapters.build_parms(pid, contract, ctx["products"]["atl03"], ctx["poly"])
    store = store_for(ctx, pid, parms)
    res = resources_for(ctx, pid)
    if granules:
        res = res[res["granule"].isin(granules) | res["resource"].isin(granules)]
    if limit:
        res = res.head(limit)
    todo, done = [], []
    for _, r in res.iterrows():
        (done if store.is_complete(r["granule"]) else todo).append(r)
    if not quiet:
        print(f"== {pid}: {len(res)} resources, {len(done)} verified complete, {len(todo)} to fetch -> {store.dir}")
    out = [store.read_meta(r["granule"]) | {"skipped": True} for r in done]
    with ThreadPoolExecutor(max_workers=workers or cfg["workers"]) as ex:
        futs = {ex.submit(acquire_one, pid, contract, parms, store, r["granule"], r["resource"], cfg): r for r in todo}
        for f in as_completed(futs):
            r = futs[f]
            try:
                meta = f.result()
            except Exception as e:  # programming/IO error: record, never a zero
                meta = {"granule": r["granule"], "status": status.RETRIEVAL_FAILED, "error": repr(e), "n_rows": None}
            out.append(meta)
            if not quiet:
                print(f"   {r['granule']}: {meta['status']} n={meta.get('n_rows')} "
                      f"t={meta.get('elapsed_s')}s attempts={meta.get('attempts')}"
                      + (f" dropped={meta['fields_dropped']}" if meta.get("fields_dropped") else ""), flush=True)
    return out


def alerts_summary(meta: dict) -> str:
    """Distinct server error alerts, beam names collapsed (progress notices dropped)."""
    texts = {re.sub(r"gt[123][lr]", "gtX", a["text"]) for a in meta.get("alerts", [])
             if a.get("level", 3) >= 3 and "running" not in a["text"]}
    return " | ".join(sorted(texts))[:400]


def write_ledger(ctx: dict, products: list[str]) -> pd.DataFrame:
    """Acquisition ledger over all inventory resources (not_yet_queried if absent)."""
    rows = []
    for pid in products:
        contract = ctx["products"][pid]
        parms = adapters.build_parms(pid, contract, ctx["products"]["atl03"], ctx["poly"])
        store = store_for(ctx, pid, parms)
        for _, r in resources_for(ctx, pid).iterrows():
            m = store.read_meta(r["granule"])
            complete = store.is_complete(r["granule"])
            rows.append({"product": pid, "granule": r["granule"], "acq_start_utc": r["acq_start_utc"],
                         "status": m["status"] if m else status.NOT_YET_QUERIED,
                         "verified_complete": complete, "n_rows": m.get("n_rows") if m else None,
                         "bytes": m.get("bytes") if m else None, "sha256": m.get("sha256") if m else None,
                         "fields_dropped": ";".join(m.get("fields_dropped", [])) if m else "",
                         "attempts": m.get("attempts") if m else None,
                         "committed_utc": m.get("committed_utc") if m else None,
                         "chunk_dir": str(store.dir.relative_to(config.ROOT)),
                         "error": (m.get("error") or "")[:200] if m else "",
                         "alerts_summary": alerts_summary(m) if m and m["status"] != status.OK else ""})
    df = pd.DataFrame(rows)
    out = manifest.output_dir(ctx["cfg"], ctx["m"]["run_id"]) / "tables" / "acquisition_ledger.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--product", action="append", help="repeatable; default all configured")
    ap.add_argument("--granule", action="append", help="restrict to these granules")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    for site in (config.site_ids() if a.all else [a.site]):
        ctx = context(site, a.run_id)
        prods = a.product or ctx["cfg"]["products"]
        t0 = time.time()
        for pid in prods:
            run_product(ctx, pid, a.granule, a.limit, a.workers)
        led = write_ledger(ctx, ctx["cfg"]["products"])
        summ = led.groupby(["product", "status"]).size().unstack(fill_value=0)
        print(summ.to_string())
        manifest.update(ctx["cfg"], ctx["m"]["run_id"], "acquisition", {
            "updated_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)),
            "ledger": "tables/acquisition_ledger.csv",
            "status_counts": {p: {k: int(v) for k, v in row.items()} for p, row in summ.iterrows()},
            "processing_version": cache.PROCESSING_VERSION,
            "elapsed_s_last_invocation": round(time.time() - t0, 1)})
        unresolved = led[~led["verified_complete"] & (led["status"] != status.NOT_YET_QUERIED)]
        if len(unresolved):
            print(f"{len(unresolved)} unresolved chunks (re-run to retry)", file=sys.stderr)


if __name__ == "__main__":
    main()
