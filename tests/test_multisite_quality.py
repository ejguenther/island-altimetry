import numpy as np
import pandas as pd
import pytest

from island_altimetry import adapters, aoi, config, normalize, quality
from scripts_loader import load_script

build_aoi = load_script("build_aoi")

SYNTHETIC = """
  synthetic_test_island:
    names: [Synthetic Test Island]
    center: {lon: -26.0, lat: 38.5}
    center_source: synthetic entry for multi-site validation (no data downloaded)
    core_radius_m: 1500
    buffer_m: 3000
    products: [atl03, atl08, gedi_l2a]
    events:
      - {id: synthetic_event, start: "2020-01-01T00:00:00Z", precision: day, description: test, source: https://example.org}
"""


@pytest.fixture
def two_site_config(tmp_path):
    p = tmp_path / "islands.yaml"
    p.write_text(config.ISLANDS.read_text() + SYNTHETIC)
    return p


def test_second_site_needs_only_configuration(two_site_config):
    assert config.site_ids(two_site_config) == ["fukutoku_okanoba", "synthetic_test_island"]
    cfg = config.resolve_site("synthetic_test_island", path=two_site_config)
    assert cfg["workers"] == 2 and cfg["products"] == ["atl03", "atl08", "gedi_l2a"]  # defaults merged
    g = build_aoi.build_geometries(cfg)
    assert g["checks"]["outer"]["target_radius_m"] == 4500
    poly = aoi.to_sliderule_poly(g["outer"])
    products = config.load_products()
    for pid in cfg["products"]:
        parms = adapters.build_parms(pid, products[pid], products["atl03"], poly)
        assert parms["poly"] == poly  # same acquisition logic, site-specific AOI only
    assert config.config_hash(cfg) != config.config_hash(config.resolve_site("fukutoku_okanoba", path=two_site_config))


def test_zoner_matches_aoi_construction():
    z = normalize.Zoner(141.481, 24.285, 2000, 7000)
    core = aoi.disk(141.481, 24.285, 2000)
    xs, ys = core.exterior.xy
    x, y = z.xy(np.asarray(xs), np.asarray(ys))
    assert np.allclose(np.hypot(x, y), 2000, atol=0.5)
    df = z.apply(pd.DataFrame(index=range(3)), [141.481, 141.481 + 0.03, 141.481 + 0.075], [24.285] * 3)
    assert list(df["zone"]) == ["core", "buffer", "outer_margin"]


def test_quality_rejections_add_up_and_strict_implies_baseline():
    df = pd.DataFrame({"height": [1.0, np.nan, 2.0, 3.0, 4.0], "atl03_cnf": [4, 4, 1, 2, 4],
                       "quality_ph": [0, 0, 0, 0, 1], "podppd_flag": [0, 0, 0, 0, 0]})
    out, rej = quality.apply("atl03", df)
    assert list(out["q_baseline"]) == [True, False, False, True, False]
    assert list(out["q_strict"]) == [True, False, False, False, False]
    assert sum(r["rejected"] for r in rej) == (~out["q_strict"]).sum()
    assert (out["q_strict"] <= out["q_baseline"]).all()
    assert out.loc[3, "reject_reason"] == "strict:signal_conf_ge_medium"


def test_fill_masking_preserves_raw_columns_names():
    df = pd.DataFrame({"h": np.array([1.0, 3.4028235e38], dtype="float32"), "i": [1, 2]})
    counts = normalize.mask_fills(df)
    assert counts == {"h": 1} and np.isnan(df["h"].iloc[1]) and list(df.columns) == ["h", "i"]
