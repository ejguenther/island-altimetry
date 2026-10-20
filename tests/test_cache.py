import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from island_altimetry import adapters, cache, status


def _store(tmp_path):
    return cache.ChunkStore(tmp_path, "atl03", "007", "aoihash", "req")


def _write(tmp, content=b"data"):
    tmp.write_bytes(content)
    return tmp


def test_commit_ok_then_complete_and_checksum_guard(tmp_path):
    s = _store(tmp_path)
    s.commit("G1", {"status": status.OK, "n_rows": 3}, _write(s.tmp_path("G1")))
    assert s.is_complete("G1") and not s.tmp_path("G1").exists()
    s.data_path("G1").write_bytes(b"datX")  # corrupted after commit
    assert not s.is_complete("G1")


def test_failed_and_unqueried_are_not_complete(tmp_path):
    s = _store(tmp_path)
    assert not s.is_complete("G2")  # not yet queried
    s.commit("G2", {"status": status.RETRIEVAL_FAILED, "n_rows": None}, None)
    assert not s.is_complete("G2")
    s.commit("G3", {"status": status.NO_SAMPLES_IN_AOI, "n_rows": 0}, None)
    assert s.is_complete("G3") and not s.data_path("G3").exists()


def test_partial_without_sidecar_is_incomplete(tmp_path):
    s = _store(tmp_path)
    _write(s.tmp_path("G4"))
    assert not s.is_complete("G4")


def test_request_hash_ignores_poly_but_tracks_filters():
    a = cache.request_hash({"poly": [1], "cnf": -2})
    assert a == cache.request_hash({"poly": [2], "cnf": -2})
    assert a != cache.request_hash({"poly": [1], "cnf": 0})


def test_atl09_reduction_lossless_and_counts():
    n = 6
    gdf = gpd.GeoDataFrame({
        "srcid": 0, "rgt": 1199, "cycle": 12, "region": 1, "gt": [10] * n, "spot": 1,
        "x_atc": np.arange(n, dtype=float),
        "high_rate/delta_time": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "high_rate/layer_flag": [0, 0, 0, 1, 1, 1],
        "high_rate/layer_top": [np.array([5.0], dtype="float32")] * 3 + [np.array([7.0], dtype="float32")] * 3,
    }, geometry=[Point(141.48, 24.28 + i * 1e-4) for i in range(n)],
        index=pd.date_range("2021-09-10", periods=n, freq="ms", tz="UTC"))
    out, info = adapters.reduce_atl09(gdf)
    assert len(out) == 2 and list(out["n_atl03_photons"]) == [3, 3]
    assert list(out["high_rate/layer_flag"]) == [0, 1] and list(out["high_rate/layer_top"]) == [5.0, 7.0]
    assert info["inconsistent_within_profile"] == {}
    assert info["truncated_array_fields"] == ["high_rate/layer_top"]
