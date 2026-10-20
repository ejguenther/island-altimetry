"""Thin SlideRule request runner that keeps server alerts, so that empty
results can be told apart from failures."""

from __future__ import annotations

import copy
import os
import re
import tempfile
import time
from dataclasses import dataclass, field

import geopandas as gpd
import pyarrow.parquet as pq
from sliderule import sliderule

from . import status

_session = None

# Alert text patterns observed on server v5.6.x (see docs/m0).
_NO_RESOURCES = re.compile(r"has no resources to process")
_MISSING_SCRIPT = re.compile(r"Failed to load script")
# ATL03 request whose ATL09 ancillary partner is absent from the catalog (observed v5.6.3).
_NO_ATL09_PARTNER = re.compile(r"Invalid number of resources for ATL09 CMR request.*: 0|Failed to find atmospheric data")
_RESOURCE_FAILURE = re.compile(r"Failure on resource|read failure|Failed to send data|invalid dataframe|failed to write dataframe|timed? ?out", re.I)
_READ_FAILURE_PATH = re.compile(r"read failure on (\S+)")


def init(url: str | None = None, timeout: tuple[int, int] | None = None):
    """Connect to the public cluster once per process."""
    global _session
    if _session is None:
        kwargs = {"verbose": False}
        if url:
            kwargs["url"] = url
        sliderule.init(**kwargs)
        _session = sliderule.checksession(None)
        if timeout:
            sliderule.set_rqst_timeout(timeout)
    return _session


def server_version() -> dict:
    init()
    return sliderule.get_version()


@dataclass
class RunResult:
    status: str
    gdf: gpd.GeoDataFrame | None
    alerts: list[dict] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0
    attrs: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return 0 if self.gdf is None else len(self.gdf)


def requested_fields(parms: dict) -> set[str]:
    """Ancillary field names requested in parms (leaf names, group prefixes kept)."""
    out = set()
    for k, v in parms.items():
        if k.endswith("_fields") and isinstance(v, list):
            out.update(str(f).rstrip("%") for f in v)
        elif isinstance(v, dict):
            out |= requested_fields(v)
    return out


def failed_fields(alerts: list[dict], fields: set[str]) -> set[str]:
    """Requested fields whose H5 reads failed on the server."""
    bad = set()
    for a in alerts:
        m = _READ_FAILURE_PATH.search(a["text"])
        if m:
            path = m.group(1)
            bad |= {f for f in fields if path.endswith("/" + f)}
    return bad


def classify(alerts: list[dict], error: str | None, gdf, fields: set[str] = frozenset()) -> str:
    texts = [a["text"] for a in alerts] + ([error] if error else [])
    joined = "\n".join(texts)
    if _MISSING_SCRIPT.search(joined):
        return status.SERVICE_UNSUPPORTED
    if failed_fields(alerts, set(fields)):
        return status.FIELD_UNSUPPORTED
    if _NO_ATL09_PARTNER.search(joined):
        return status.NO_GRANULES
    if any(_RESOURCE_FAILURE.search(t) for t in texts) or (error and gdf is None and not _NO_RESOURCES.search(joined)):
        return status.RETRIEVAL_FAILED
    if gdf is not None and len(gdf):
        return status.OK
    if _NO_RESOURCES.search(joined):
        return status.NO_GRANULES
    if gdf is not None:
        return status.NO_SAMPLES_IN_AOI
    return status.RETRIEVAL_FAILED


def run(api: str, parms: dict, resources: list[str] | None = None,
        out_path: str | None = None) -> RunResult:
    """Execute an x-series endpoint, writing GeoParquet to out_path (or a temp file)."""
    session = init()
    parms = copy.deepcopy(parms)
    if resources is not None:
        parms["resources"] = list(resources)
    tmp = out_path is None
    path = out_path or tempfile.mktemp(suffix=".parquet")
    if os.path.exists(path):
        os.remove(path)
    parms["output"] = {"path": path, "format": "geoparquet", "open_on_complete": False}

    alerts: list[dict] = []

    def _alert(rec, _session):
        alerts.append({"code": int(rec.get("code", 0)), "level": int(rec.get("level", 0)), "text": str(rec.get("text", ""))})

    t0 = time.time()
    error = None
    try:
        rsps = session.source(api, {"parms": parms}, stream=True, callbacks={"exceptrec": _alert}, rethrow=True)
        for rsp in rsps or []:
            if isinstance(rsp, dict) and rsp.get("__rectype") == "arrowrec.remote":
                error = "server returned a remote output url; local output expected"
    except Exception as e:  # network/HTTP errors: never a scientific zero
        error = repr(e)

    gdf, attrs = None, {}
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            gdf = gpd.read_parquet(path)
            md = pq.read_metadata(path).metadata or {}
            for k, v in md.items():
                if k in (b"ARROW:schema", b"pandas", b"geo"):
                    continue
                try:
                    import json
                    attrs[k.decode()] = json.loads(v.decode().rstrip("\x00"))
                except Exception:
                    attrs[k.decode()] = v.decode(errors="replace")
        except Exception as e:
            error = (error + "; " if error else "") + f"unreadable output: {e!r}"
            gdf = None
    res = RunResult(classify(alerts, error, gdf, requested_fields(parms)), gdf, alerts, error, time.time() - t0, attrs)
    if tmp and os.path.exists(path):
        os.remove(path)
    return res


def search(asset: str, poly: list[dict], t0: str, t1: str, **extra) -> list[str]:
    """Server-side granule discovery (raises on failure; never returns [] for errors)."""
    from sliderule import earthdata
    init()
    return earthdata.search({"asset": asset, "poly": poly, "t0": t0, "t1": t1, **extra})
