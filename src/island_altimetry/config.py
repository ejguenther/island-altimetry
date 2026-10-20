"""Configuration loading, validation, site selection, overrides, and hashing."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ISLANDS = ROOT / "configs" / "islands.yaml"
PRODUCTS = ROOT / "configs" / "products.yaml"
KNOWN_PRODUCTS = ("atl03", "atl08", "atl09", "atl24", "gedi_l2a")


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def parse_utc(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ConfigError(f"timestamp must carry a UTC offset: {s}")
    return t.astimezone(dt.timezone.utc)


def fmt_utc(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _set_dotted(d: dict, key: str, value):
    *path, last = key.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = yaml.safe_load(value) if isinstance(value, str) else value


def load_products() -> dict:
    return yaml.safe_load(PRODUCTS.read_text())["products"]


def site_ids(path: Path = ISLANDS) -> list[str]:
    return list(yaml.safe_load(path.read_text())["sites"])


def resolve_site(site_id: str, overrides: list[str] | None = None, path: Path = ISLANDS) -> dict:
    """Merge defaults with the site entry, apply `key=value` overrides, validate."""
    raw = yaml.safe_load(path.read_text())
    if site_id not in raw["sites"]:
        raise ConfigError(f"unknown site {site_id!r}; configured: {list(raw['sites'])}")
    cfg = _merge(raw["defaults"], raw["sites"][site_id])
    cfg["site_id"] = site_id
    for ov in overrides or []:
        if "=" not in ov:
            raise ConfigError(f"override must be key=value: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, v)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    c = cfg.get("center")
    if "geometry" not in cfg:
        if not c or not {"lon", "lat"} <= set(c):
            raise ConfigError("site needs center {lon, lat} or a supplied geometry")
        if not (-180 <= c["lon"] <= 180 and -90 <= c["lat"] <= 90):
            raise ConfigError(f"center out of range: {c}")
    for k in ("core_radius_m", "buffer_m"):
        if not (isinstance(cfg[k], (int, float)) and cfg[k] > 0):
            raise ConfigError(f"{k} must be a positive number of meters")
    bad = [p for p in cfg["products"] if p not in KNOWN_PRODUCTS]
    if bad:
        raise ConfigError(f"unknown products {bad}")
    for p in cfg["products"]:
        if p not in cfg["start"]:
            raise ConfigError(f"no discovery start for {p}")
        parse_utc(cfg["start"][p])
    if cfg["end"] != "latest":
        parse_utc(cfg["end"])
    for e in cfg.get("events", []):
        if not e.get("source"):
            raise ConfigError(f"event {e.get('id')} has no source")
        parse_utc(e["start"])
    if cfg["workers"] < 1 or cfg["retries"] < 0:
        raise ConfigError("workers >= 1 and retries >= 0 required")
    if "planet" in cfg.get("imagery", {}):
        from .planet_imagery import validate_imagery
        validate_imagery(cfg["imagery"]["planet"])


def config_hash(cfg: dict) -> str:
    """Hash of the resolved configuration (independent of run time)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
