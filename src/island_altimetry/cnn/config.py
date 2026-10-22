"""Configuration and per-site paths for the CNN terrain experiment
(configs/cnn_terrain.yaml merged over shared defaults in configs/islands.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..config import ISLANDS, ROOT, ConfigError, _merge, config_hash

CNN = ROOT / "configs" / "cnn_terrain.yaml"
SHARED_KEYS = ("data_root", "output_root", "retries", "retry_backoff_s", "imagery")


def load(path: Path = CNN) -> dict:
    shared = yaml.safe_load(ISLANDS.read_text())["defaults"]
    cfg = _merge({k: shared[k] for k in SHARED_KEYS}, yaml.safe_load(Path(path).read_text()))
    r = cfg["roles"]
    if r["n_fit"] + r["n_transfer_val"] + r["n_transfer_test"] > len(cfg["sites"]):
        raise ConfigError("roles need more sites than configured")
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def site_paths(cfg: dict, site: str) -> dict[str, Path]:
    s, p = cfg["sites"][site], cfg["paths"]
    data = ROOT / cfg["data_root"] / site
    return {"dtm": Path(p["dem_root"]) / s["dir"] / f"dtm_{s['epoch']}_1m",
            "dsm": Path(p["dem_root"]) / s["dir"] / f"dsm_{s['epoch']}_1m",
            "laz": Path(p["laz_root"]) / s["dir"] / f"tiles_{s['epoch']}",
            "glob": s.get("tile_glob", "*"), "data": data, "work": data / "cnn", "planet": data / "imagery" / "planet",
            "out": out_dir(cfg) / "sites" / site}


def out_dir(cfg: dict) -> Path:
    return ROOT / cfg["output_root"] / cfg["experiment"]


def planet_paths(cfg: dict, site: str) -> dict[str, Path]:
    """Shared per-site imagery store plus this experiment's catalog cache."""
    sp = site_paths(cfg, site)
    return {"catalog": sp["planet"] / cfg["experiment"] / "catalog", "coverage": sp["planet"] / "coverage_cache.json",
            "orders": sp["planet"] / "orders.json", "scenes": sp["planet"] / "scenes",
            "thumbs": sp["planet"] / cfg["experiment"] / "thumbs", "tables": sp["out"], "figures": sp["out"]}
