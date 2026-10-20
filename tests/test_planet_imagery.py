import copy

import pandas as pd
import pytest
from shapely.geometry import box

from island_altimetry import aoi, config
from island_altimetry import planet_imagery as pim
from scripts_loader import load_script

LON, LAT = 141.481, 24.285
OUTER = aoi.disk(LON, LAT, 7000)
PROJ = pim.Projector(LON, LAT)
T0 = pd.Timestamp("2021-09-10T03:20:00Z")
ALL_PERMS = ";".join(pim.BUNDLE_PERMISSION.values())


def pc(**kw):
    p = copy.deepcopy(config.resolve_site("fukutoku_okanoba")["imagery"]["planet"])
    p.update(kw)
    return p


def strip(sid, hours, clear=0.9, footprint=1.0):
    return {"strip_id": sid, "acquired": T0 + pd.Timedelta(hours=hours), "item_ids": f"{sid}_a",
            "n_items": 1, "satellite_id": "x", "instrument": "PSB.SD", "scene_cloud_cover_max": 0.1,
            "aoi_footprint": footprint, "aoi_clear": clear, "aoi_cloud": 1 - (clear or 0), "aoi_haze": 0.0,
            "coverage_complete": clear is not None}


def target(tid="atl03:G1", t=T0, usable=True):
    return {"target_id": tid, "product": tid.split(":")[0], "granule": tid.split(":")[1], "overpass_utc": t,
            "time_source": "subset_median", "track": 1199, "n_core": 10, "usable_surface_core": usable}


def test_overpass_time_from_subset_and_fallback(tmp_path):
    t = pd.date_range("2021-09-10T03:19:00", periods=5, freq="10s")
    pd.DataFrame({"time_ns": t, "height": range(5)}).set_index("time_ns").to_parquet(tmp_path / "G1.parquet")
    inv = pd.DataFrame({"granule": ["G1", "G2", "G3"], "acq_start_utc": ["2021-09-10T03:17:07Z"] * 3,
                        "footprint_intersects_outer": ["True", "True", "False"], "rgt": ["1199"] * 3})
    df = pim.build_targets({"atl03": inv}, lambda pid, g: tmp_path / f"{g}.parquet", None)
    assert list(df["granule"]) == ["G2", "G1"]          # G3 outside AOI; sorted by time
    g1 = df.set_index("granule").loc["G1"]
    assert g1["time_source"] == "subset_median"
    assert g1["overpass_utc"] == pd.Timestamp("2021-09-10T03:19:20Z")
    assert df.set_index("granule").loc["G2", "time_source"] == "granule_start"


def test_strip_grouping_unions_footprints():
    minx, miny, maxx, maxy = OUTER.bounds
    midx = (minx + maxx) / 2
    items = pd.DataFrame([
        {"item_id": "a", "strip_id": "s1", "acquired": T0, "satellite_id": "x", "instrument": "i",
         "cloud_cover": 0.1, "geometry": box(minx - 1, miny - 1, midx, maxy + 1), "aoi_clear": 1.0,
         "aoi_cloud": 0.0, "aoi_haze": 0.0},
        {"item_id": "b", "strip_id": "s1", "acquired": T0 + pd.Timedelta(seconds=2), "satellite_id": "x",
         "instrument": "i", "cloud_cover": 0.2, "geometry": box(midx, miny - 1, maxx + 1, maxy + 1),
         "aoi_clear": 0.5, "aoi_cloud": 0.5, "aoi_haze": 0.0},
    ])
    s = pim.group_strips(items, OUTER, PROJ).iloc[0]
    assert s["n_items"] == 2 and s["aoi_footprint"] == pytest.approx(1.0, abs=1e-3)
    assert s["aoi_clear"] == pytest.approx(0.75, abs=0.01)   # area-weighted, halves
    assert pim.footprint_fraction(items["geometry"][0], OUTER, PROJ) == pytest.approx(0.5, abs=0.01)


def test_closest_qualifying_strip_wins_and_near_miss_recorded():
    strips = pd.DataFrame([strip("cloudy", 1, clear=0.5), strip("clear_d1", -20), strip("clear_d3", 50),
                           strip("partial", 2, footprint=0.6)])
    m = pim.match_targets(pd.DataFrame([target()]), strips, pc()).iloc[0]
    assert m["strip_id"] == "clear_d1" and m["match_tier"] == "d1" and m["dt_hours"] == -20
    assert m["near_miss_strip_id"] == "cloudy" and m["near_miss_reason"] == "cloud"


def test_tie_break_on_clear_and_threshold_enforced():
    strips = pd.DataFrame([strip("a", 3, clear=0.85), strip("b", -3, clear=0.95)])
    assert pim.match_targets(pd.DataFrame([target()]), strips, pc()).iloc[0]["strip_id"] == "b"
    m = pim.match_targets(pd.DataFrame([target()]), strips, pc(aoi_clear_min=0.99)).iloc[0]
    assert pd.isna(m["strip_id"]) and m["near_miss_strip_id"] in ("a", "b")


def test_outside_widest_tier_unmatched():
    m = pim.match_targets(pd.DataFrame([target()]), pd.DataFrame([strip("far", 24 * 40)]), pc()).iloc[0]
    assert pd.isna(m["match_tier"]) and m["n_candidates"] == 0


def test_order_plan_dedupes_and_falls_back_bundle():
    strips = pd.DataFrame([strip("s1", 2)])
    targets = pd.DataFrame([target("atl03:G1"), target("gedi_l2a:GEDI1", T0 + pd.Timedelta(hours=30), usable=False)])
    matches = pim.match_targets(targets, strips, pc())
    perms = ";".join(v for k, v in pim.BUNDLE_PERMISSION.items() if k != "analytic_8b_sr_udm2")
    items = pd.DataFrame({"item_id": ["s1_a"], "permissions": [perms]})
    plan = pim.order_plan(matches, items, pc(), 154.0, "site")
    assert len(plan) == 1
    r = plan.iloc[0]
    assert r["n_targets"] == 2 and r["bundle"] == "analytic_sr_udm2" and r["best_tier"] == "same_day"
    assert r["products"] == "atl03;gedi_l2a" and r["any_usable_core"]


def test_choose_bundle_none_when_no_analytic():
    assert pim.choose_bundle([{"assets.ortho_visual:download"}], pc()["bundles"]) is None
    assert pim.choose_bundle([set(ALL_PERMS.split(";"))] * 2, pc()["bundles"]) == "analytic_8b_sr_udm2"


def test_select_and_budget():
    plan = pd.DataFrame({"best_tier": ["same_day", "d3", "d30"], "bundle": ["b", "b", None],
                         "any_usable_core": [True, False, True], "targets": ["atl03:A", "gedi_l2a:B", "atl03:C"],
                         "est_km2": [154.0, 154.0, 154.0]})
    assert len(pim.select_orders(plan, pc())) == 2                         # no-bundle strip dropped
    assert len(pim.select_orders(plan, pc(), max_tier="d1")) == 1
    assert len(pim.select_orders(plan, pc(), usable_only=True)) == 1
    with pytest.raises(RuntimeError):
        pim.check_budget(pim.select_orders(plan, pc()), 200)
    assert pim.check_budget(pim.select_orders(plan, pc()), 400) == 308


def test_imagery_config_validation():
    pim.validate_imagery(pc())
    with pytest.raises(config.ConfigError):
        pim.validate_imagery(pc(aoi_clear_min=80))
    with pytest.raises(config.ConfigError):
        pim.validate_imagery(pc(tiers_hours={"d3": 72, "d1": 24}))
    with pytest.raises(config.ConfigError):
        pim.validate_imagery(pc(bundles=["visual"]))
    with pytest.raises(config.ConfigError):
        config.resolve_site("fukutoku_okanoba", ["imagery.planet.scene_cloud_max=2"])


def test_near_any_target():
    acq = pd.Series(pd.to_datetime(["2021-09-01T00:00Z", "2021-09-10T05:00Z", "2021-12-01T00:00Z"]))
    tg = pd.Series(pd.to_datetime(["2021-09-10T03:00Z", "2021-09-12T00:00Z"]))
    assert list(pim.near_any_target(acq, tg, 72)) == [False, True, False]


def test_order_script_refuses_submit_without_budget(monkeypatch, tmp_path):
    order = load_script("planet_order")
    monkeypatch.setattr("sys.argv", ["planet_order.py", "--site", "fukutoku_okanoba", "--submit"])
    plan = pd.DataFrame([{"order_name": "o", "strip_id": "s1", "acquired_utc": "2021-09-10T03:00:00Z",
                          "item_ids": "a", "n_items": 1, "bundle": "analytic_sr_udm2", "best_tier": "same_day",
                          "min_abs_dt_hours": 1.0, "aoi_clear": 0.9, "targets": "atl03:G", "products": "atl03",
                          "n_targets": 1, "any_usable_core": True, "est_km2": 154.0}])
    tables = tmp_path / "tables"
    tables.mkdir()
    plan.to_csv(tables / "planet_order_plan.csv", index=False)
    monkeypatch.setattr(order.manifest, "latest_run_id", lambda cfg: "run")
    monkeypatch.setattr(order.manifest, "load", lambda cfg, run_id=None: {"aoi_dir": "x"})
    monkeypatch.setattr(order.pim, "paths", lambda cfg, run_id: {"tables": tables, "orders": tmp_path / "orders.json"})
    called = []
    monkeypatch.setattr(order, "submit", lambda *a: called.append(a))
    with pytest.raises(SystemExit, match="budget"):
        order.main()
    monkeypatch.setattr("sys.argv", ["planet_order.py", "--site", "fukutoku_okanoba", "--submit", "--budget-km2", "100"])
    with pytest.raises(RuntimeError, match="budget"):
        order.main()
    assert not called and not (tmp_path / "orders.json").exists()


class FakeCoverageClient:
    """get_item_coverage answers 'activating' for the first `activating[id]` calls."""

    def __init__(self, activating, clear=90):
        self.left, self.clear, self.calls = dict(activating), clear, 0

    async def get_item_coverage(self, item_type, iid, geometry, mode=None):
        self.calls += 1
        if self.left.get(iid, 0) > 0:
            self.left[iid] -= 1
            return {"status": "activating"}
        return {"status": "complete", "clear_percent": self.clear}


def test_estimate_coverage_polls_activating_and_reports_incomplete(tmp_path):
    import asyncio
    import json
    cache = tmp_path / "cov.json"
    client = FakeCoverageClient({"a": 0, "b": 2, "c": 99})
    out = asyncio.run(pim.estimate_coverage(client, "PSScene", ["a", "b", "c"], OUTER, cache,
                                            poll_s=0, max_rounds=4))
    assert out["a"]["aoi_clear"] == 0.9 and out["b"]["aoi_clear"] == 0.9
    assert out["c"]["coverage_status"] == "incomplete"
    assert pim.coverage_counts(out) == {"complete": 2, "incomplete": 1, "failed": 0}
    cached = json.loads(cache.read_text())
    assert len(cached) == 2 and not any(k.endswith(":c") for k in cached)
    # cached items are not re-requested
    client2 = FakeCoverageClient({})
    asyncio.run(pim.estimate_coverage(client2, "PSScene", ["a", "b"], OUTER, cache, poll_s=0))
    assert client2.calls == 0


def test_gapfill_choice_prefers_clear_then_near_with_footprint_fallback():
    strips = pd.DataFrame([strip("near_cloudy", 24, clear=0.3), strip("far_clear", -500, clear=0.95),
                           strip("partial_clearest", 5, clear=1.0, footprint=0.5), strip("unknown", 2, clear=None)])
    s, dt_h = pim.gapfill_choice(T0, strips, 768, 0.95)
    assert s["strip_id"] == "far_clear" and dt_h == -500
    s, _ = pim.gapfill_choice(T0, strips[strips["strip_id"].isin(["partial_clearest"])], 768, 0.95)
    assert s["strip_id"] == "partial_clearest"                     # footprint fallback
    assert pim.gapfill_choice(T0, strips, 1, 0.95) is None         # outside window
    s, _ = pim.gapfill_choice(T0, strips[strips["strip_id"].isin(["near_cloudy", "unknown"])], 768, 0.95)
    assert s["strip_id"] == "near_cloudy"                          # unknown clear ranks lowest
