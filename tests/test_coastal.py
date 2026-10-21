import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from island_altimetry.coastal import experiment as ex
from island_altimetry.coastal import features, grid, interp, masks, metrics, splits, tracks
from scripts_loader import load_script

UTM = CRS.from_epsg(6345)


def _tile(path, x0, y0, z):
    with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
                       crs=UTM, transform=Affine(1, 0, x0, 0, -1, y0), nodata=-9999) as s:
        s.write(z.astype("float32"), 1)


@pytest.fixture
def two_tiles(tmp_path):
    a = np.full((1001, 1001), 2.0)
    a[:, 500:] = -1.0                      # valid negative elevations
    a[:, 400] = 0.0                        # valid zero column
    a[100:110, 100:110] = -9999            # 100 m2 gap
    _tile(tmp_path / "a.tif", 658000.0004, 3301000.0, a)
    _tile(tmp_path / "b.tif", 659000.0004, 3301000.0, np.full((1001, 1001), 5.0))   # 1-px overlap with a
    spec = {"crs": UTM.to_wkt(), "transform": [3, 0, 657999.5, 0, -3, 3301000.0], "width": 700, "height": 340}
    return [tmp_path / "a.tif", tmp_path / "b.tif"], spec


def test_aggregation_conserves_area_and_handles_gaps_overlap(two_tiles):
    paths, spec = two_tiles
    o = grid.aggregate_tiles(paths, -9999, spec, workers=1)
    f = o["valid_frac"]
    # overlap counted once, the gap counts as missing area of the full cell
    assert f.max() <= 1 + 1e-6
    assert f.sum() * 9 == pytest.approx(2000 * 1000 - 100, abs=1)
    assert f[50, 0] == pytest.approx(2.4996 / 3, abs=1e-3)          # partial edge cell
    assert f[34, 34] == 0 and np.isnan(o["z_mean"][34, 34])          # cell entirely inside the gap
    assert f[33, 33] == pytest.approx(1 - 1.4996 * 2 / 9, abs=1e-3)  # cell overlapping the gap corner
    assert o["z_mean"][33, 33] == pytest.approx(2.0)                 # gap ignored in the mean, not zero-filled
    assert o["z_mean"][50, 333] == pytest.approx(2.0, abs=1e-3)      # (-1 * 1.5 + 5 * 1.5) / 3 at the seam
    # partial-pixel weighting and within-cell std at a step (2.5 m of 2.0, 0.5 m of -1.0)
    assert o["z_mean"][50, 166] == pytest.approx(1.5, abs=1e-3)
    assert o["z_std"][50, 166] == pytest.approx(3 * np.sqrt(5 / 36), abs=1e-3)
    assert np.isnan(o["z_mean"][-1, -1]) and f[-1, -1] == 0          # outside all tiles


def test_zero_elevation_survives(two_tiles):
    paths, spec = two_tiles
    o = grid.aggregate_tiles(paths, -9999, spec, workers=1)
    # source column x=658400..658401 is 0.0; cell 133 covers 658398.5..658401.5 -> (2*1.5 + 0*1 + 2*0.5)/3
    assert o["z_mean"][50, 133] == pytest.approx((2 * 1.4996 + 0 * 1 + 2 * 0.5004) / 3, abs=1e-3)
    assert o["valid_frac"][50, 133] == pytest.approx(1.0)


def test_assert_same_grid(tmp_path):
    spec = {"crs": UTM.to_wkt(), "transform": [3, 0, 0, 0, -3, 30], "width": 10, "height": 10}
    grid.write_raster(tmp_path / "a.tif", {"x": np.zeros((10, 10), np.float32)}, spec)
    grid.assert_same_grid([tmp_path / "a.tif"], spec)
    with pytest.raises(AssertionError):
        grid.assert_same_grid([tmp_path / "a.tif"], spec | {"transform": [3, 0, 1.5, 0, -3, 30]})


def test_surface_domain_uses_imagery_and_buffers_contacts():
    w = np.full((20, 20), -0.5, np.float32)
    w[:, 10:] = 0.5
    valid = np.ones_like(w, bool)
    valid[0, 0] = False
    d = masks.surface_domain(w, valid, 0.0, -0.1, buffer_px=1)
    assert d[0, 0] == masks.NONE
    assert (d[5, 8:12] == masks.AMBIGUOUS).all()
    assert d[5, 2] == masks.LAND and d[5, 17] == masks.WATER


def test_split_buffer_separates_train_from_heldout():
    s = np.linspace(0, 10000, 20001)
    edges = splits.block_edges(s, 10)
    codes = np.array([1, 1, 2, 1, 3, 1, 1, 2, 3, 1])
    _, part = splits.partition_cells(s, edges, codes, 150)
    tr, ho = s[part == 1], s[(part == 2) | (part == 3)]
    gap = np.min(np.abs(tr[:, None] - ho[None, ::20]))
    assert gap >= 150 - 1.0
    assert (part == 0).any()
    # windows of train cells (<= 5x5 at 3 m) cannot reach a held-out cell
    assert features.MAX_RADIUS_PX * 3 * np.sqrt(2) < 150 / 2


def test_track_cells_unique_and_labels_train_only(tmp_path):
    t = Affine(3, 0, 0, 0, -3, 300)
    lines, spacing = tracks.layout_lines((0, 0, 300, 300), 0, 3, 0.0)
    ids = tracks.line_cells(lines, t, (100, 100))
    assert len(ids) == len(np.unique(ids)) == 300          # 3 N-S lines x 100 rows, no duplicates
    assert spacing == pytest.approx(100)
    cells = pd.DataFrame({"cell_id": ids, "partition": np.where(ids % 2, 1, 2), "domain": 1, "x": 0.0, "y": 0.0,
                          "z": 0.0})
    (tmp_path / "labels").mkdir()
    pd.DataFrame({"cell_id": ids}).to_parquet(tmp_path / "labels" / "sparse_x.parquet")
    with pytest.raises(AssertionError):
        ex.training_labels(cells, tmp_path, "x")


def test_idw_uses_only_training_labels_and_reports_unsupported():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1000, (200, 2))
    z = rng.normal(size=200)
    q = np.array([[500.0, 500.0], xy[0], [1e6, 1e6]])
    p, dmin, used = interp.idw(xy, z, q, k=8, power=2, max_radius=300)
    assert p[1] == pytest.approx(z[0])                  # exact at a label
    assert np.isnan(p[2]) and used[2] == 0 and np.isinf(dmin[2])


def test_idw_predictions_invariant_to_heldout_elevations():
    mod = load_script("run_interpolation")
    rng = np.random.default_rng(1)
    n = 400
    cells = pd.DataFrame({"cell_id": np.arange(n), "x": rng.uniform(0, 1000, n), "y": rng.uniform(0, 1000, n),
                          "z": rng.normal(size=n), "domain": np.where(np.arange(n) % 2, 1, 2),
                          "partition": np.where(np.arange(n) < 200, 1, 2)})
    rt = ex.routed(cells)
    lab = cells[cells["partition"] == 1]
    a = mod.predict_idw(rt, lab, 8, 2.0, 2000)
    rt2 = rt.copy()
    rt2.loc[rt2["partition"] == 2, "z"] += 100.0        # perturb held-out reference values
    b = mod.predict_idw(rt2, lab, 8, 2.0, 2000)
    np.testing.assert_array_equal(a["pred"].to_numpy(), b["pred"].to_numpy())


def test_window_stats_ignore_invalid_neighbours():
    a = np.ones((7, 7), np.float32)
    a[3, 3] = 100.0
    valid = np.ones_like(a, bool)
    valid[3, 3] = False
    m, s = features.window_stats(a, valid, 3)
    assert m[3, 4] == pytest.approx(1.0) and s[3, 4] == pytest.approx(0.0)


def test_metrics_counts_reconcile():
    pred = np.array([1.0, np.nan, 3.0, 4.0])
    ref = np.array([0.0, 0.0, 3.0, 2.0])
    m = metrics.summarize(pred, ref)
    assert (m["n_eval"], m["n_pred"]) == (4, 3)
    assert m["coverage"] == pytest.approx(0.75)
    assert m["bias"] == pytest.approx(1.0) and m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
