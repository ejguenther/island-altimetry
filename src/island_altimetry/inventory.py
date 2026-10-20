"""M2 discovery: per-product granule inventory via SlideRule (primary) and a
labeled read-only NASA CMR completeness check. Granule footprint intersection
is candidate coverage only."""

from __future__ import annotations

import datetime as dt
import re
import time

import requests

from . import sr_client
from .config import fmt_utc, parse_utc

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"

ICESAT2_RE = re.compile(
    r"^(?P<product>ATL\d\d)_(?P<acq>\d{14})_(?P<rgt>\d{4})(?P<cycle>\d{2})(?P<region>\d{2})_"
    r"(?P<rel>\d{3})_(?P<rev>\d{2})(?:_(?P<rel2>\d{3})_(?P<rev2>\d{2}))?")
GEDI_RE = re.compile(
    r"^GEDI02_A_(?P<acq>\d{13})_O(?P<orbit>\d{5})_(?P<sub>\d{2})_T(?P<track>\d{5})_(?P<ppds>\d{2})_"
    r"(?P<pge>\d{3})_(?P<rev>\d{2})_V(?P<rel>\d{3})")


def parse_name(product: str, name: str) -> dict:
    """Identifiers from a granule filename. Times are nominal granule start (UTC)."""
    base = name.split(".h5")[0]
    if product == "gedi_l2a":
        m = GEDI_RE.match(base)
        if not m:
            raise ValueError(f"unparsed GEDI name {name}")
        d = m.groupdict()
        t = dt.datetime.strptime(d["acq"], "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc)
        return {"granule": base, "acq_start_name_utc": fmt_utc(t), "orbit": int(d["orbit"]),
                "sub_orbit": int(d["sub"]), "track": int(d["track"]), "release": d["rel"],
                "pge_version": d["pge"], "revision": d["rev"],
                "lineage_key": f"O{d['orbit']}_{d['sub']}"}
    m = ICESAT2_RE.match(base)
    if not m:
        raise ValueError(f"unparsed ICESat-2 name {name}")
    d = m.groupdict()
    t = dt.datetime.strptime(d["acq"], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    out = {"granule": base, "acq_start_name_utc": fmt_utc(t), "rgt": int(d["rgt"]),
           "cycle": int(d["cycle"]), "region": int(d["region"])}
    if d["rel2"]:  # ATL24: <atl03 release>_<atl03 rev>_<atl24 release>_<atl24 rev>
        out.update(release=d["rel2"], revision=d["rev2"], atl03_release=d["rel"], atl03_revision=d["rev"])
    else:
        out.update(release=d["rel"], revision=d["rev"])
    out["lineage_key"] = f"{d['acq']}_{d['rgt']}{d['cycle']}{d['region']}"
    return out


def windows(t0: str, t1: str, days: int) -> list[tuple[str, str]]:
    a, b = parse_utc(t0), parse_utc(t1)
    out = []
    while a < b:
        c = min(a + dt.timedelta(days=days), b)
        out.append((fmt_utc(a), fmt_utc(c)))
        a = c
    return out


def _split(t0: str, t1: str) -> list[tuple[str, str]]:
    a, b = parse_utc(t0), parse_utc(t1)
    mid = a + (b - a) / 2
    return [(fmt_utc(a), fmt_utc(mid)), (fmt_utc(mid), fmt_utc(b))]


def sliderule_discover(asset: str, poly: list[dict], t0: str, t1: str, cap: int,
                       retries: int = 3, backoff_s: float = 10) -> tuple[dict, list[dict]]:
    """Return ({granule: meta}, ledger rows). Splits windows that exceed the cap;
    failures are recorded as retrieval_failed, never as zero granules."""
    found, ledger = {}, []
    queue = [(t0, t1)]
    while queue:
        a, b = queue.pop(0)
        row = {"route": "sliderule", "asset": asset, "t0": a, "t1": b,
               "query_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)), "attempts": 0}
        for attempt in range(1, retries + 1):
            row["attempts"] = attempt
            try:
                res = sr_client.search(asset, poly, a, b, with_meta=True, max_resources=cap)
                if isinstance(res, list) and not res:
                    res = {}  # server returns [] (not {}) when a window has no granules
                if not isinstance(res, dict):
                    raise RuntimeError(f"unexpected discovery response type {type(res).__name__}")
                found.update(res)
                row.update(status="ok" if res else "no_granules", n=len(res), error="")
                break
            except Exception as e:
                msg = repr(e)
                if "exceeded maximum allowed" in msg:
                    row.update(status="split_cap_exceeded", n=None, error=msg[:300])
                    queue[:0] = _split(a, b)
                    break
                row.update(status="retrieval_failed", n=None, error=msg[:300])
                time.sleep(backoff_s * attempt)
        ledger.append(row)
    return found, ledger


def cmr_discover(short_name: str, version: str, poly: list[dict], t0: str, t1: str,
                 retries: int = 3) -> tuple[list[dict], dict]:
    """Read-only CMR granule metadata search (discovery check only; no downloads)."""
    coords = ",".join(f"{p['lon']},{p['lat']}" for p in poly)
    params = {"short_name": short_name, "version": version, "polygon": coords,
              "temporal": f"{t0},{t1}", "page_size": 2000, "sort_key": "start_date"}
    entries, headers, row = [], {}, {"route": "cmr_check", "short_name": short_name, "version": version,
                                      "t0": t0, "t1": t1, "query_utc": fmt_utc(dt.datetime.now(dt.timezone.utc))}
    while True:
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(CMR_GRANULES, params=params, headers=headers, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == retries:
                    row.update(status="retrieval_failed", n=None, error=repr(e)[:300])
                    return entries, row
                time.sleep(5 * attempt)
        page = r.json()["feed"]["entry"]
        entries.extend({"title": e["title"], "concept_id": e["id"], "time_start": e.get("time_start"),
                        "time_end": e.get("time_end"), "updated": e.get("updated"),
                        "producer_granule_id": e.get("producer_granule_id")} for e in page)
        hits = int(r.headers.get("CMR-Hits", len(entries)))
        sa = r.headers.get("CMR-Search-After")
        if not page or len(entries) >= hits or not sa:
            break
        headers = {"CMR-Search-After": sa}
    row.update(status="ok" if entries else "no_granules", n=len(entries), cmr_hits=hits, error="")
    return entries, row


def dedupe(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """One record per lineage key (same acquisition/granule); keep highest revision.
    Returns (selected, discarded) with discarded rows pointing at their replacement."""
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["release"], r["lineage_key"]), []).append(r)
    selected, discarded = [], []
    for key, group in sorted(by_key.items()):
        group = sorted(group, key=lambda r: (r["revision"], r["granule"]))
        keep = group[-1]
        selected.append(keep)
        for d in group[:-1]:
            discarded.append({**d, "superseded_by": keep["granule"], "dedupe_rule": "highest_revision"})
    return selected, discarded


def cmr_name_lookup(short_name: str, version: str, pattern: str, retries: int = 3) -> tuple[list[dict], dict]:
    """Read-only CMR lookup of granules by readable-name pattern (wildcards * ?)."""
    params = {"short_name": short_name, "version": version, "readable_granule_name": pattern,
              "options[readable_granule_name][pattern]": "true", "page_size": 100}
    row = {"route": "cmr_name_lookup", "short_name": short_name, "version": version, "pattern": pattern,
           "query_utc": fmt_utc(dt.datetime.now(dt.timezone.utc))}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(CMR_GRANULES, params=params, timeout=120)
            r.raise_for_status()
            page = r.json()["feed"]["entry"]
            row.update(status="ok" if page else "no_granules", n=len(page), error="")
            return [{"title": e["title"], "time_start": e.get("time_start"), "time_end": e.get("time_end")}
                    for e in page], row
        except Exception as e:
            if attempt == retries:
                row.update(status="retrieval_failed", n=None, error=repr(e)[:300])
                return [], row
            time.sleep(5 * attempt)
