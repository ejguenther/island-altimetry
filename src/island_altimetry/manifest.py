"""Run identity and manifest: frozen UTC end, config/AOI hashes, versions."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

from .config import ROOT, fmt_utc


def new_run_id(site_id: str, now: dt.datetime) -> str:
    return f"{site_id}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def output_dir(cfg: dict, run_id: str) -> Path:
    return ROOT / cfg["output_root"] / cfg["site_id"] / run_id


def data_dir(cfg: dict) -> Path:
    return ROOT / cfg["data_root"] / cfg["site_id"]


def manifest_path(cfg: dict, run_id: str) -> Path:
    return output_dir(cfg, run_id) / "run_manifest.json"


def git_commit() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def latest_run_id(cfg: dict) -> str:
    base = ROOT / cfg["output_root"] / cfg["site_id"]
    runs = sorted(p.name for p in base.glob(f"{cfg['site_id']}_*") if (p / "run_manifest.json").exists())
    if not runs:
        raise FileNotFoundError(f"no runs under {base}; run scripts/build_aoi.py first")
    return runs[-1]


def load(cfg: dict, run_id: str | None = None) -> dict:
    run_id = run_id or latest_run_id(cfg)
    return json.loads(manifest_path(cfg, run_id).read_text())


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def update(cfg: dict, run_id: str, section: str, content) -> dict:
    """Add or replace one section of the run manifest."""
    m = load(cfg, run_id)
    m[section] = content
    m["updated_utc"] = fmt_utc(dt.datetime.now(dt.timezone.utc))
    write_json_atomic(manifest_path(cfg, run_id), m)
    return m
