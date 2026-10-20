"""M8: PlanetScope scenes temporally coincident with lidar overpasses.

Pure selection logic (targets, strip grouping, tier matching, order planning)
plus thin async wrappers around the Planet SDK. Searching and coverage
estimates do not draw down download quota; only orders do (planet_order.py).

Times are UTC throughout. "Same day" means |dt| <= 12 h from the overpass,
not the same calendar date.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from shapely.geometry import mapping, shape

from . import manifest
from .aoi import aeqd_crs
from .config import ConfigError, fmt_utc, parse_utc

TARGET_PRODUCTS = ("atl03", "gedi_l2a")
COVERAGE_WINDOW_H = 720   # coverage is only estimated for items within the widest tier

# (bundle, permission that must be present on every item), best first.
BUNDLE_PERMISSION = {
    "analytic_8b_sr_udm2": "assets.ortho_analytic_8b_sr:download",
    "analytic_sr_udm2": "assets.ortho_analytic_4b_sr:download",
    "analytic_udm2": "assets.ortho_analytic_4b:download",
    "analytic_3b_udm2": "assets.ortho_analytic_3b:download",   # Dove-Classic RGB-only scenes (TOA)
}
SR_BUNDLES = ("analytic_8b_sr_udm2", "analytic_sr_udm2")


# ---------------------------------------------------------------- config

def validate_imagery(pc: dict) -> None:
    """Validate defaults.imagery.planet."""
    for k in ("scene_cloud_max", "aoi_clear_min", "aoi_footprint_min"):
        if not 0 <= pc[k] <= 1:
            raise ConfigError(f"imagery.planet.{k} must be a fraction in [0, 1]")
    tiers = pc["tiers_hours"]
    hours = list(tiers.values())
    if not hours or hours != sorted(hours) or hours[0] <= 0:
        raise ConfigError("imagery.planet.tiers_hours must be positive and increasing")
    if hours[-1] > COVERAGE_WINDOW_H:
        raise ConfigError(f"widest tier exceeds coverage window ({COVERAGE_WINDOW_H} h)")
    if not isinstance(pc.get("standard_quality_only", False), bool):
        raise ConfigError("imagery.planet.standard_quality_only must be true or false")
    if pc.get("coverage_mode", "UDM2") not in ("UDM2", "estimate"):
        raise ConfigError("imagery.planet.coverage_mode must be UDM2 or estimate")
    for k in ("clip_aoi", "match_aoi"):
        if pc.get(k, "outer") not in ("core", "outer"):
            raise ConfigError(f"imagery.planet.{k} must be core or outer")
    bad = [b for b in pc["bundles"] if b not in BUNDLE_PERMISSION]
    if bad:
        raise ConfigError(f"unknown Planet bundles {bad}")
    b = pc.get("order_budget_km2")
    if b is not None and not (isinstance(b, (int, float)) and b > 0):
        raise ConfigError("imagery.planet.order_budget_km2 must be null or > 0")


def paths(cfg: dict, run_id: str) -> dict:
    d = manifest.data_dir(cfg) / "imagery" / "planet"
    return {"work": d / run_id, "catalog": d / run_id / "catalog", "coverage": d / "coverage_cache.json",
            "orders": d / "orders.json", "scenes": d / "scenes",
            "tables": manifest.output_dir(cfg, run_id) / "tables",
            "figures": manifest.output_dir(cfg, run_id) / "figures" / "planet"}


# ---------------------------------------------------------------- targets

def overpass_time(parquet: Path) -> pd.Timestamp | None:
    """Median sample time of an AOI subset (subsets are already clipped to the
    outer AOI, so this is the AOI crossing, not the granule start)."""
    if not parquet.exists():
        return None
    t = pq.read_table(parquet, columns=["time_ns"]).column("time_ns").to_pandas()
    if t.empty:
        return None
    return pd.Timestamp(t.median()).tz_localize("UTC") if t.dt.tz is None else pd.Timestamp(t.median())


def build_targets(inventories: dict[str, pd.DataFrame], subset_path, coverage: pd.DataFrame | None) -> pd.DataFrame:
    """One row per AOI-crossing ATL03/GEDI granule. `subset_path(pid, granule)`
    returns the cached subset parquet path (or None)."""
    rows = []
    for pid, inv in inventories.items():
        inv = inv[inv["footprint_intersects_outer"].astype(str).str.lower() == "true"]
        for _, r in inv.iterrows():
            p = subset_path(pid, r["granule"])
            t = overpass_time(p) if p is not None else None
            src = "subset_median"
            if t is None:
                t, src = pd.Timestamp(parse_utc(r["acq_start_utc"])), "granule_start"
            rows.append({"target_id": f"{pid}:{r['granule']}", "product": pid, "granule": r["granule"],
                         "overpass_utc": t, "time_source": src,
                         "track": r.get("rgt") if pid == "atl03" else r.get("track")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if coverage is not None and not coverage.empty:
        cov = coverage[["product", "granule", "n_core", "usable_surface_core"]]
        df = df.merge(cov, on=["product", "granule"], how="left")
    else:
        df["n_core"], df["usable_surface_core"] = pd.NA, pd.NA
    df["usable_surface_core"] = df["usable_surface_core"].astype(str).str.lower() == "true"
    return df.sort_values("overpass_utc").reset_index(drop=True)


# ---------------------------------------------------------------- geometry

class Projector:
    """Site AEQD (metres) for footprint and area math."""

    def __init__(self, lon: float, lat: float):
        self._tr = Transformer.from_crs("EPSG:4326", aeqd_crs(lon, lat), always_xy=True)

    def __call__(self, geom):
        return shapely.transform(geom, lambda xy: np.column_stack(self._tr.transform(xy[:, 0], xy[:, 1])))


def footprint_fraction(footprint, aoi, proj: Projector) -> float:
    a = proj(aoi)
    return float(proj(footprint).intersection(a).area / a.area) if a.area else 0.0


def area_km2(geom, proj: Projector) -> float:
    return float(proj(geom).area / 1e6)


# ---------------------------------------------------------------- catalog

def item_row(item: dict) -> dict:
    p = item["properties"]
    return {"item_id": item["id"], "strip_id": str(p.get("strip_id", item["id"])),
            "acquired": pd.Timestamp(parse_utc(p["acquired"])), "satellite_id": p.get("satellite_id"),
            "instrument": p.get("instrument"), "cloud_cover": p.get("cloud_cover"),
            "clear_percent_scene": p.get("clear_percent"), "sun_elevation": p.get("sun_elevation"),
            "view_angle": p.get("view_angle"), "quality_category": p.get("quality_category"),
            "ground_control": p.get("ground_control"), "publishing_stage": p.get("publishing_stage"),
            "permissions": ";".join(item.get("_permissions", [])),
            "geometry": shape(item["geometry"])}


def year_windows(t0: dt.datetime, t1: dt.datetime):
    y = t0.year
    while True:
        a = max(t0, dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc))
        b = min(t1, dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc))
        if a >= b:
            return
        yield a, b
        y += 1


def search_filter(aoi, t0, t1, scene_cloud_max: float, standard_quality_only: bool) -> dict:
    """Open-ocean sites can have only `test`-quality scenes (no ground control
    points to tie the orthorectification to), so the standard-quality filter is
    optional; quality_category and ground_control are recorded per item."""
    from planet import data_filter
    f = [data_filter.geometry_filter(mapping(aoi)),
         data_filter.date_range_filter("acquired", gte=t0, lt=t1),
         data_filter.range_filter("cloud_cover", lte=scene_cloud_max),
         data_filter.permission_filter()]
    if standard_quality_only:
        f.append(data_filter.std_quality_filter())
    return data_filter.and_filter(f)


async def fetch_catalog(client, item_type: str, aoi, t0, t1, scene_cloud_max: float, cache_dir: Path,
                        refresh: bool = False, standard_quality_only: bool = False) -> list[dict]:
    """Yearly Data API searches, each cached as raw JSON. Cached windows whose
    end is in the past are reused; the open (current) window is re-queried."""
    items: list[dict] = []
    now = dt.datetime.now(dt.timezone.utc)
    for a, b in year_windows(t0, t1):
        flt = search_filter(aoi, a, b, scene_cloud_max, standard_quality_only)
        fhash = hashlib.sha256(json.dumps(flt, sort_keys=True, default=str).encode()).hexdigest()[:10]
        f = cache_dir / f"{item_type}_{a:%Y%m%d}_{b:%Y%m%d}_{fhash}.json"
        if f.exists() and not refresh and b < now - dt.timedelta(days=30):
            got = json.loads(f.read_text())["items"]
        else:
            got = [i async for i in client.search([item_type], search_filter=flt, sort="acquired asc", limit=0)]
            manifest.write_json_atomic(f, {"window": [fmt_utc(a), fmt_utc(b)], "filter": flt,
                                           "queried_utc": fmt_utc(now), "items": got})
        print(f"  {a:%Y-%m-%d}..{b:%Y-%m-%d}: {len(got)} items")
        items.extend(got)
    return items


def near_any_target(acquired: pd.Series, targets: pd.Series, hours: float) -> pd.Series:
    """Boolean mask: acquisition within `hours` of at least one target time."""
    t = targets.sort_values().to_numpy()
    a = acquired.to_numpy()
    i = np.searchsorted(t, a)
    best = np.full(len(a), np.inf)
    for j in (i - 1, i):
        ok = (j >= 0) & (j < len(t))
        d = np.abs((a[ok] - t[j[ok]]) / np.timedelta64(1, "h"))
        best[ok] = np.minimum(best[ok], d)
    return pd.Series(best <= hours, index=acquired.index)


# ---------------------------------------------------------------- coverage

def coverage_row(resp: dict) -> dict:
    """Normalize a coverage response to fractions in [0, 1]."""
    def pct(k):
        v = resp.get(k)
        return None if v is None else float(v) / 100.0
    return {"aoi_clear": pct("clear_percent"), "aoi_cloud": pct("cloud_percent"),
            "aoi_haze": pct("haze_percent") if "haze_percent" in resp else
            (None if resp.get("light_haze_percent") is None else
             (float(resp.get("light_haze_percent", 0)) + float(resp.get("heavy_haze_percent", 0))) / 100.0),
            "aoi_shadow": pct("shadow_percent"), "aoi_snow": pct("snow_ice_percent"),
            "coverage_status": resp.get("status")}


async def estimate_coverage(client, item_type: str, item_ids: list[str], aoi, cache_file: Path,
                            workers: int = 4, retries: int = 3, backoff_s: float = 20,
                            mode: str = "UDM2") -> dict:
    """Per-AOI clear coverage, cached by AOI + mode + item id. Only finished
    ('complete') results are cached; failures are returned but not stored.

    mode="UDM2" computes from the full-resolution UDM2 and matched downloaded
    masks to within a few percent; mode="estimate" is faster but was far too
    optimistic under haze (e.g. 91% vs 0% measured clear), so it is not the default."""
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    from .aoi import geometry_hash
    akey = f"{geometry_hash(aoi)}:{mode}"
    todo = [i for i in item_ids if f"{akey}:{i}" not in cache]
    sem = asyncio.Semaphore(workers)
    out = {i: cache[f"{akey}:{i}"] for i in item_ids if f"{akey}:{i}" in cache}

    async def one(iid):
        async with sem:
            for attempt in range(retries + 1):
                try:
                    r = await client.get_item_coverage(item_type, iid, mapping(aoi), mode=mode)
                    return iid, r, None
                except Exception as e:  # noqa: BLE001 - recorded, never a scientific zero
                    if attempt == retries:
                        return iid, None, f"{type(e).__name__}: {e}"
                    await asyncio.sleep(backoff_s * 2 ** attempt)

    done = 0
    for fut in asyncio.as_completed([one(i) for i in todo]):
        iid, r, err = await fut
        done += 1
        if r is not None:
            row = coverage_row(r)
            out[iid] = row
            if row["aoi_clear"] is not None:
                cache[f"{akey}:{iid}"] = row
        else:
            out[iid] = {"coverage_status": "failed", "error": err}
        if done % 50 == 0 or done == len(todo):
            print(f"  coverage {done}/{len(todo)}")
            manifest.write_json_atomic(cache_file, cache)
    return out


# ---------------------------------------------------------------- strips

def group_strips(items: pd.DataFrame, aoi, proj: Projector) -> pd.DataFrame:
    """One row per strip (adjacent scenes of one capture). Strip AOI-clear is the
    footprint-area-weighted mean of its items' AOI clear fractions."""
    rows = []
    a = proj(aoi)
    for sid, g in items.groupby("strip_id"):
        geoms = [proj(x) for x in g["geometry"]]
        union = shapely.union_all(geoms)
        w = [x.intersection(a).area for x in geoms]
        clear = list(g["aoi_clear"]) if "aoi_clear" in g else [None] * len(g)
        ok = [(wi, c) for wi, c in zip(w, clear) if c is not None and wi > 0]
        wsum = sum(wi for wi, _ in ok)

        def wmean(col):
            vals = [(wi, v) for wi, v in zip(w, g[col]) if v is not None and pd.notna(v) and wi > 0] if col in g else []
            s = sum(wi for wi, _ in vals)
            return sum(wi * v for wi, v in vals) / s if s else None
        rows.append({"strip_id": sid, "acquired": g["acquired"].min(),
                     "item_ids": ";".join(sorted(g["item_id"])), "n_items": len(g),
                     "satellite_id": g["satellite_id"].iloc[0], "instrument": g["instrument"].iloc[0],
                     "scene_cloud_cover_max": g["cloud_cover"].max(),
                     "quality_category": ";".join(sorted({str(q) for q in g.get("quality_category", [])})) or None,
                     "ground_control": bool(g["ground_control"].all()) if "ground_control" in g else None,
                     "aoi_footprint": float(union.intersection(a).area / a.area),
                     "aoi_clear": sum(wi * c for wi, c in ok) / wsum if wsum else None,
                     "aoi_cloud": wmean("aoi_cloud"), "aoi_haze": wmean("aoi_haze"),
                     "coverage_complete": wsum > 0 and len(ok) == sum(1 for wi in w if wi > 0)})
    return pd.DataFrame(rows).sort_values("acquired").reset_index(drop=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------- matching

def tier_of(dt_hours: float, tiers: dict[str, float]) -> str | None:
    for name, h in tiers.items():
        if abs(dt_hours) <= h:
            return name
    return None


def match_targets(targets: pd.DataFrame, strips: pd.DataFrame, pc: dict) -> pd.DataFrame:
    """Closest qualifying strip per target, walking tiers in order (closest
    |dt| wins; ties -> higher AOI clear). The best rejected strip within the
    widest tier is kept as a near miss for manual review."""
    tiers = pc["tiers_hours"]
    widest = max(tiers.values())
    out = []
    for _, t in targets.iterrows():
        row = t.to_dict()
        row.update(match_tier=None, dt_hours=None, quality_category=None, ground_control=None,
                   strip_id=None, item_ids=None, satellite_id=None,
                   acquired_utc=None, aoi_clear=None, aoi_cloud=None, aoi_haze=None, aoi_footprint=None,
                   near_miss_strip_id=None, near_miss_dt_hours=None, near_miss_aoi_clear=None,
                   near_miss_aoi_footprint=None, near_miss_reason=None, n_candidates=0)
        if strips.empty:
            out.append(row)
            continue
        d = (strips["acquired"] - t["overpass_utc"]) / pd.Timedelta(hours=1)
        c = strips.assign(dt_hours=d, adt=d.abs())
        c = c[c["adt"] <= widest]
        row["n_candidates"] = len(c)
        good = c[(c["aoi_footprint"] >= pc["aoi_footprint_min"]) &
                 c["aoi_clear"].notna() & (c["aoi_clear"] >= pc["aoi_clear_min"])]
        if not good.empty:
            b = good.assign(nc=-good["aoi_clear"]).sort_values(["adt", "nc"]).iloc[0]
            row.update(match_tier=tier_of(b["dt_hours"], tiers), dt_hours=round(float(b["dt_hours"]), 2),
                       quality_category=b.get("quality_category"), ground_control=b.get("ground_control"),
                       strip_id=b["strip_id"], item_ids=b["item_ids"], satellite_id=b["satellite_id"],
                       acquired_utc=fmt_utc(b["acquired"]), aoi_clear=round(float(b["aoi_clear"]), 3),
                       aoi_cloud=b["aoi_cloud"], aoi_haze=b["aoi_haze"],
                       aoi_footprint=round(float(b["aoi_footprint"]), 3))
            closer = c[(c["adt"] < b["adt"]) & ~c.index.isin(good.index)]
        else:
            closer = c[~c.index.isin(good.index)]
        if not closer.empty:
            # Near miss: closest rejected strip that is closer in time than the match.
            n = closer.assign(cl=-closer["aoi_clear"].fillna(-1)).sort_values(["adt", "cl"]).iloc[0]
            reason = ("footprint" if n["aoi_footprint"] < pc["aoi_footprint_min"] else
                      "no_estimate" if pd.isna(n["aoi_clear"]) else "cloud")
            row.update(near_miss_strip_id=n["strip_id"], near_miss_dt_hours=round(float(n["dt_hours"]), 2),
                       near_miss_aoi_clear=None if pd.isna(n["aoi_clear"]) else round(float(n["aoi_clear"]), 3),
                       near_miss_aoi_footprint=round(float(n["aoi_footprint"]), 3), near_miss_reason=reason)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- ordering

def choose_bundle(permissions: list[set[str]], bundles: list[str]) -> str | None:
    """Best bundle every item in a strip can deliver (lowest common tier)."""
    for b in bundles:
        if all(BUNDLE_PERMISSION[b] in p for p in permissions):
            return b
    return None


def order_plan(matches: pd.DataFrame, items: pd.DataFrame, pc: dict, clip_km2: float | dict, site_id: str) -> pd.DataFrame:
    """Unique strips across all matched targets, with bundle and quota estimate.
    `clip_km2` is one area for every strip, or {strip_id: clipped area}."""
    m = matches[matches["strip_id"].notna()]
    if m.empty:
        return pd.DataFrame()
    perms = items.set_index("item_id")["permissions"].to_dict()
    tier_rank = {k: i for i, k in enumerate(pc["tiers_hours"])}
    rows = []
    for sid, g in m.groupby("strip_id"):
        ids = g["item_ids"].iloc[0].split(";")
        bundle = choose_bundle([set(str(perms.get(i, "")).split(";")) for i in ids], pc["bundles"])
        best = g.loc[g["match_tier"].map(tier_rank).idxmin()]
        date = pd.Timestamp(g["acquired_utc"].iloc[0]).strftime("%Y%m%d")
        rows.append({"order_name": f"{site_id}_{date}_{sid}", "strip_id": sid, "acquired_utc": g["acquired_utc"].iloc[0],
                     "item_ids": ";".join(ids), "n_items": len(ids), "bundle": bundle,
                     "best_tier": best["match_tier"], "min_abs_dt_hours": float(g["dt_hours"].abs().min()),
                     "aoi_clear": g["aoi_clear"].iloc[0], "targets": ";".join(g["target_id"]),
                     "products": ";".join(sorted(set(g["product"]))), "n_targets": len(g),
                     "any_usable_core": bool(g["usable_surface_core"].any()), "est_km2": round(clip_km2[sid] if isinstance(clip_km2, dict) else clip_km2, 1)})
    return pd.DataFrame(rows).sort_values("acquired_utc").reset_index(drop=True)


def select_orders(plan: pd.DataFrame, pc: dict, max_tier: str | None = None, usable_only: bool = False,
                  targets: list[str] | None = None) -> pd.DataFrame:
    sel = plan[plan["bundle"].notna()]
    if max_tier:
        names = list(pc["tiers_hours"])
        if max_tier not in names:
            raise ValueError(f"--max-tier must be one of {names}")
        sel = sel[sel["best_tier"].map(names.index) <= names.index(max_tier)]
    if usable_only:
        sel = sel[sel["any_usable_core"]]
    if targets:
        sel = sel[sel["targets"].apply(lambda s: any(t in s for t in targets))]
    return sel


def check_budget(sel: pd.DataFrame, budget_km2: float | None) -> float:
    total = float(sel["est_km2"].sum()) if not sel.empty else 0.0
    if budget_km2 is not None and total > budget_km2:
        raise RuntimeError(f"selection needs ~{total:.0f} km2 > budget {budget_km2:.0f} km2; "
                           "narrow with --max-tier/--usable-only/--targets or raise --budget-km2")
    return total


def build_order_request(row: dict, clip_geom, pc: dict, item_type: str) -> dict:
    from planet import order_request
    tools = [order_request.clip_tool(mapping(clip_geom))]
    if pc.get("harmonize") and row["bundle"] in SR_BUNDLES:
        tools.append(order_request.harmonize_tool("Sentinel-2"))
    product = order_request.product(row["item_ids"].split(";"), row["bundle"], item_type)
    return order_request.build_request(row["order_name"], products=[product], tools=tools)
