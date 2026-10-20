import pytest

from island_altimetry import aoi, config
from scripts_loader import load_script

build_aoi = load_script("build_aoi")


def test_radius_and_coordinate_order():
    lon, lat = 141.481, 24.285
    g = aoi.disk(lon, lat, 2000)
    dmin, dmax = aoi.max_geodesic_radius_m(g, lon, lat)
    assert abs(dmin - 2000) < 0.5 and abs(dmax - 2000) < 0.5
    # lon/lat order: a disk at (141.5E, 24.3N) has x-bounds near 141.5
    minx, miny, maxx, maxy = g.bounds
    assert 141.4 < minx < maxx < 141.6 and 24.2 < miny < maxy < 24.4
    poly = aoi.to_sliderule_poly(g)
    assert poly[0] == poly[-1] and set(poly[0]) == {"lon", "lat"}


def test_swapped_center_rejected():
    with pytest.raises(ValueError):
        aoi.disk(24.285, 141.481, 2000)  # lat out of range


def test_antimeridian_rejected():
    with pytest.raises(ValueError):
        aoi.disk(179.99, 10.0, 5000)


def test_aoi_and_config_hash_deterministic():
    cfg1 = config.resolve_site("fukutoku_okanoba")
    cfg2 = config.resolve_site("fukutoku_okanoba")
    assert config.config_hash(cfg1) == config.config_hash(cfg2)
    g1, g2 = build_aoi.build_geometries(cfg1), build_aoi.build_geometries(cfg2)
    for z in ("core", "outer", "buffer"):
        assert aoi.geometry_hash(g1[z]) == aoi.geometry_hash(g2[z])


def test_override_changes_hash_and_radius():
    cfg = config.resolve_site("fukutoku_okanoba", ["buffer_m=8000"])
    assert cfg["buffer_m"] == 8000
    assert config.config_hash(cfg) != config.config_hash(config.resolve_site("fukutoku_okanoba"))
    g = build_aoi.build_geometries(cfg)
    assert g["checks"]["outer"]["target_radius_m"] == 10000


def test_events_require_sources(tmp_path):
    p = tmp_path / "islands.yaml"
    p.write_text(config.ISLANDS.read_text().replace(
        "source: https://doi.org/10.3390/geosciences12010040\n    #", "source: ''\n    #"))
    with pytest.raises(config.ConfigError):
        config.resolve_site("fukutoku_okanoba", path=p)
