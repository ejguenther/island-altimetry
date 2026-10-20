"""Chunk cache: one immutable GeoParquet file per (product, granule, AOI, request),
with a JSON sidecar recording status, request, alerts, and checksum. A chunk is
complete only if its sidecar has a terminal status and (for data) the file's
checksum matches. Writes are atomic (temp file + os.replace)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from . import status
from .config import fmt_utc

PROCESSING_VERSION = "1"   # bump when chunk content semantics change


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def request_hash(parms: dict) -> str:
    """Hash of request parameters that define chunk content (poly via AOI hash)."""
    p = {k: v for k, v in parms.items() if k not in ("poly", "output", "resources")}
    p["_processing_version"] = PROCESSING_VERSION
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:12]


class ChunkStore:
    def __init__(self, root: Path, product: str, release: str, aoi_hash: str, req_hash: str):
        self.dir = root / "subsets" / product / release / f"aoi_{aoi_hash}" / f"req_{req_hash}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.product, self.release, self.aoi_hash, self.req_hash = product, release, aoi_hash, req_hash

    def data_path(self, granule: str) -> Path:
        return self.dir / f"{granule}.parquet"

    def meta_path(self, granule: str) -> Path:
        return self.dir / f"{granule}.json"

    def tmp_path(self, granule: str) -> Path:
        return self.dir / f".{granule}.parquet.partial"

    def read_meta(self, granule: str) -> dict | None:
        p = self.meta_path(granule)
        return json.loads(p.read_text()) if p.exists() else None

    def is_complete(self, granule: str) -> bool:
        """Verified complete: terminal status and, for data, matching checksum."""
        m = self.read_meta(granule)
        if not m or m["status"] not in (status.OK, *status.TERMINAL_EMPTY):
            return False
        if m["status"] == status.OK:
            p = self.data_path(granule)
            return p.exists() and p.stat().st_size == m["bytes"] and sha256(p) == m["sha256"]
        return True

    def commit(self, granule: str, meta: dict, tmp_file: Path | None) -> dict:
        """Atomically publish data (if any) then its sidecar."""
        meta = {**meta, "product": self.product, "release": self.release, "aoi_hash": self.aoi_hash,
                "request_hash": self.req_hash, "processing_version": PROCESSING_VERSION,
                "granule": granule, "committed_utc": fmt_utc(dt.datetime.now(dt.timezone.utc))}
        if meta["status"] == status.OK:
            if tmp_file is None or not tmp_file.exists():
                raise RuntimeError("ok chunk without data file")
            meta["sha256"] = sha256(tmp_file)
            meta["bytes"] = tmp_file.stat().st_size
            os.replace(tmp_file, self.data_path(granule))
        else:
            if tmp_file is not None and tmp_file.exists():
                tmp_file.unlink()
            stale = self.data_path(granule)
            if stale.exists():
                stale.unlink()
        mp = self.meta_path(granule)
        tmp_meta = mp.with_suffix(".json.partial")
        tmp_meta.write_text(json.dumps(meta, indent=1, default=str) + "\n")
        os.replace(tmp_meta, mp)
        return meta

    def all_meta(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.dir.glob("*.json"))]
