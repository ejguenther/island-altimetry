"""Experiment configuration: configs/coastal_experiment.yaml merged over the
shared defaults in configs/islands.yaml, plus the experiment's directory layout."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..config import ISLANDS, ROOT, ConfigError, _merge, _set_dotted, config_hash

COASTAL = ROOT / "configs" / "coastal_experiment.yaml"
SHARED_KEYS = ("data_root", "output_root", "retries", "retry_backoff_s", "imagery")


def load(path: Path = COASTAL, overrides: list[str] | None = None) -> dict:
    shared = yaml.safe_load(ISLANDS.read_text())["defaults"]
    cfg = _merge({k: shared[k] for k in SHARED_KEYS}, yaml.safe_load(Path(path).read_text()))
    for ov in overrides or []:
        if "=" not in ov:
            raise ConfigError(f"override must be key=value: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, v)
    validate(cfg)
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def validate(cfg: dict) -> None:
    for k in ("dtm_dir", "laz_dir", "tile_glob"):
        if not cfg["sources"].get(k):
            raise ConfigError(f"sources.{k} is required")
    if cfg["reference"].get("target", "dtm_mean") not in ("point_mean", "dtm_mean"):
        raise ConfigError("reference.target must be point_mean or dtm_mean")
    if not 0 < cfg["reference"]["min_valid_fraction"] <= 1:
        raise ConfigError("reference.min_valid_fraction must be in (0, 1]")
    f = cfg["splits"]["fractions"]
    if set(f) != {"train", "val", "test"} or abs(sum(f.values()) - 1) > 1e-9:
        raise ConfigError("splits.fractions must be train/val/test summing to 1")
    if cfg["masks"].get("rule", "ndwi") not in ("ndwi", "nir"):
        raise ConfigError("masks.rule must be ndwi or nir")
    if cfg["masks"]["ndwi_land_max"] > cfg["masks"]["ndwi_water_min"]:
        raise ConfigError("masks.ndwi_land_max must be <= masks.ndwi_water_min")


def dirs(cfg: dict) -> dict[str, Path]:
    """Data (cached, large, git-ignored) and output (tables/figures) locations.
    Planet imagery uses the shared data/<site>/imagery/planet layout."""
    data = ROOT / cfg["data_root"] / cfg["site_id"]
    out = ROOT / cfg["output_root"] / cfg["site_id"] / cfg["experiment"]
    return {"data": data, "work": data / "coastal", "planet": data / "imagery" / "planet",
            "out": out, "tables": out / "tables", "figures": out / "figures"}


def planet_paths(cfg: dict) -> dict[str, Path]:
    """Planet locations: the shared site imagery store (coverage cache, order
    ledger, scenes) plus this experiment's catalog cache, tables and figures."""
    D = dirs(cfg)
    return {"catalog": D["planet"] / cfg["experiment"] / "catalog", "coverage": D["planet"] / "coverage_cache.json",
            "orders": D["planet"] / "orders.json", "scenes": D["planet"] / "scenes",
            "thumbs": D["planet"] / cfg["experiment"] / "thumbs", "tables": D["tables"],
            "figures": D["figures"] / "planet"}
