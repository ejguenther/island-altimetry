"""Product adapters behind a common request/status interface.

Each adapter builds a permissive request for one granule over the outer AOI,
runs it through sr_client (alert-aware), and post-processes into the chunk
representation. Raw server output is kept as returned, except ATL09, whose
per-photon ancillary values are reduced (losslessly, after a constancy check)
to one row per native profile."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import sr_client, status

ATL09_KEY = "high_rate/delta_time"


def build_parms(pid: str, contract: dict, atl03_contract: dict | None, poly: list[dict]) -> dict:
    parms = {"poly": poly, **copy.deepcopy(contract.get("request_parms", {}))}
    if pid == "atl09":
        # ATL09 is requested separately so that missing ATL09 can never remove ATL03 photons.
        parms.update(copy.deepcopy(atl03_contract["request_parms"]))
    for key, fields in contract.get("anc_fields", {}).items():
        *nest, k = key.split(".")
        d = parms
        for n in nest:
            d = d.setdefault(n, {})
        d[k] = list(fields)
    return parms


def endpoint(pid: str, contract: dict) -> str:
    return "atl03x" if pid == "atl09" else contract["endpoint"]


def reduce_atl09(gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per (beam, ATL09 profile time). Verifies ATL09 values are constant
    within each profile (otherwise the reduction would lose information)."""
    a09 = [c for c in gdf.columns if c.startswith("high_rate/")]
    keys = ["srcid", "rgt", "cycle", "region", "gt", "spot", ATL09_KEY]
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["time_ns"] = gdf.index
    df["lon"], df["lat"] = gdf.geometry.x, gdf.geometry.y
    scalar = [c for c in a09 if c != ATL09_KEY and not isinstance(df[c].iloc[0], (list, tuple)) and not hasattr(df[c].iloc[0], "__len__")]
    arrays = [c for c in a09 if c not in scalar and c != ATL09_KEY]
    for c in arrays:  # length-1 arrays (truncated multi-layer fields, see docs/m0)
        df[c] = df[c].map(lambda v: float(v[0]) if len(v) else float("nan"))
    vals = scalar + arrays
    g = df.groupby(keys, sort=True, dropna=False)
    nunique = g[vals].nunique(dropna=False)
    inconsistent = {c: int((nunique[c] > 1).sum()) for c in vals if (nunique[c] > 1).any()}
    out = g.agg(**{c: (c, "first") for c in vals},
                n_atl03_photons=("time_ns", "size"),
                photon_time_min=("time_ns", "min"), photon_time_max=("time_ns", "max"),
                lon_mean=("lon", "mean"), lat_mean=("lat", "mean"),
                x_atc_min=("x_atc", "min"), x_atc_max=("x_atc", "max")).reset_index()
    info = {"n_photons_in": len(df), "n_profiles": len(out), "inconsistent_within_profile": inconsistent,
            "truncated_array_fields": arrays,
            "note": "ATL09 values attached per ATL03 photon by SlideRule; reduced to one row per "
                    "(beam, high_rate/delta_time). lon/lat are means of the ATL03 photons sharing the profile."}
    return out, info


def retrieve(pid: str, contract: dict, parms: dict, granule: str, resource: str, tmp: Path) -> tuple[str, dict, Path | None]:
    """Run one granule request. Returns (status, meta, data_file_or_None)."""
    api = endpoint(pid, contract)
    # The client truncates long output paths (observed at ~160 chars), so the server
    # writes to a short temp path that is then moved into the chunk directory.
    fd, short = tempfile.mkstemp(prefix="sr_", suffix=".parquet")
    os.close(fd)
    try:
        res = sr_client.run(api, parms, [resource], out_path=short)
        if res.status == status.OK and pid != "atl09":
            shutil.move(short, tmp)
    finally:
        if os.path.exists(short):
            os.remove(short)
    meta = {"endpoint": api, "resource": resource, "status": res.status, "n_rows": res.n,
            "elapsed_s": round(res.elapsed_s, 2), "error": res.error,
            "alerts": [a for a in res.alerts if a["code"] < 0][:50],
            "server_meta": {k: v for k, v in (res.attrs.get("meta") or {}).items() if k != "request"},
            "request": {k: v for k, v in parms.items() if k != "poly"}}
    if res.status != status.OK:
        return res.status, meta, None
    if pid == "atl09":
        prof, info = reduce_atl09(res.gdf)
        meta["atl09_reduction"] = info
        meta["n_rows"] = len(prof)
        prof.to_parquet(tmp, index=False)
    return status.OK, meta, tmp
