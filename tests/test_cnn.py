import numpy as np
import pytest

torch = pytest.importorskip("torch")

from island_altimetry.cnn import dataset, inference, model  # noqa: E402
from island_altimetry.cnn.sites import context_valid, densest_window, select_scene  # noqa: E402

DIL = [1, 2, 4, 8, 1]
CH = [32, 32, 64, 64, 32]


def _net(dil=DIL, seed=0):
    torch.manual_seed(seed)
    m = model.build(4, CH, dil).eval()
    return m


def test_receptive_field_and_offset():
    assert model.receptive_field(DIL) == 33 and model.output_offset(DIL) == 16
    assert model.receptive_field([1, 1, 2, 3, 1]) == 17 and model.output_offset([1, 1, 2, 3, 1]) == 8
    m = _net()
    assert m(torch.zeros(1, 4, 33, 33)).shape == (1, 1, 1)
    assert m(torch.zeros(1, 4, 100, 70)).shape == (1, 68, 38)


def test_no_normalization_or_striding_layers():
    m = _net()
    for mod in m.modules():
        assert not isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.Dropout,
                                    torch.nn.Linear, torch.nn.AdaptiveAvgPool2d))
        if isinstance(mod, torch.nn.Conv2d):
            assert mod.stride == (1, 1) and mod.padding == (0, 0)


def test_patch_equals_fcn_and_tiled():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(4, 120, 97)).astype(np.float32)
    m = _net()
    full = inference.predict_full(m, img, "cpu")
    o = m.offset
    assert np.isnan(full[:o]).all() and np.isnan(full[:, -o:]).all() and np.isfinite(full[o:-o, o:-o]).all()
    rows = rng.integers(o, 120 - o, 50)
    cols = rng.integers(o, 97 - o, 50)
    pat = inference.predict_patches(m, img, rows, cols, "cpu")
    np.testing.assert_allclose(pat, full[rows, cols], atol=1e-5)
    for tile in (7, 32, 64, 500):     # including ragged border tiles
        np.testing.assert_allclose(inference.predict_tiled(m, img, "cpu", tile=tile), full, atol=1e-5, equal_nan=True)


def test_output_index_alignment_with_coordinate_ramp():
    """Output (i, j) belongs to input pixel (i + 16, j + 16): a model that
    copies its center input pixel reproduces a coordinate ramp exactly there."""
    m = _net()
    with torch.no_grad():
        for mod in m.net:
            if isinstance(mod, torch.nn.Conv2d):
                mod.weight.zero_()
                mod.bias.zero_()
                k = mod.kernel_size[0] // 2
                n = min(mod.in_channels, mod.out_channels)
                for c in range(n):
                    mod.weight[c, c, k, k] = 1.0
    H, W = 60, 50
    rr, cc = np.mgrid[0:H, 0:W]
    img = np.stack([rr * 1000.0 + cc + 1, np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))]).astype(np.float32)
    full = inference.predict_full(m, img, "cpu")
    o = m.offset
    np.testing.assert_allclose(full[o:-o, o:-o], img[0, o:-o, o:-o])


def test_center_loss_ignores_neighbour_labels():
    """The loss sees only one target per patch: changing every other label
    of the reference raster leaves loss and gradients unchanged."""
    rng = np.random.default_rng(1)
    img = rng.normal(size=(4, 80, 80)).astype(np.float32)
    zref = rng.normal(size=(80, 80)).astype(np.float32)
    rows = np.array([20, 40, 55])
    cols = np.array([30, 41, 22])
    imgs = {"a": img}
    gp = dataset.GpuPatches(imgs, {"mean": [0] * 4, "std": [1] * 4}, 16, "cpu")

    def loss_and_grad(z):
        m = _net()
        x = gp.gather(torch.zeros(3, dtype=torch.long), torch.as_tensor(rows), torch.as_tensor(cols))
        y = m(x)[:, 0, 0]
        loss = ((y - torch.as_tensor(z[rows, cols])) ** 2).mean()
        loss.backward()
        return loss.item(), m.net[0].weight.grad.clone()

    l1, g1 = loss_and_grad(zref)
    z2 = zref + 100.0
    z2[rows, cols] = zref[rows, cols]
    l2, g2 = loss_and_grad(z2)
    assert l1 == l2 and torch.equal(g1, g2)


def test_gather_matches_slicing_and_augment_keeps_center():
    rng = np.random.default_rng(2)
    img = rng.normal(size=(4, 50, 60)).astype(np.float32)
    gp = dataset.GpuPatches({"a": img, "b": img[:, ::-1].copy()}, {"mean": [0] * 4, "std": [1] * 4}, 16, "cpu")
    rows, cols, site = torch.tensor([16, 30, 20]), torch.tensor([16, 40, 25]), torch.tensor([0, 0, 1])
    x = gp.gather(site, rows, cols)
    np.testing.assert_array_equal(x[1].numpy(), img[:, 14:47, 24:57])
    np.testing.assert_array_equal(x[2].numpy(), img[:, ::-1][:, 4:37, 9:42])
    for k in range(4):
        for f in (False, True):
            a = dataset.augment(x, k, f)
            np.testing.assert_array_equal(a[:, :, 16, 16].numpy(), x[:, :, 16, 16].numpy())


def test_tiny_subset_overfits():
    rng = np.random.default_rng(3)
    torch.manual_seed(0)
    x = torch.as_tensor(rng.normal(size=(64, 4, 33, 33)).astype(np.float32))
    z = torch.as_tensor(rng.normal(size=64).astype(np.float32))
    m = model.build(4, CH, DIL).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = ((m(x)[:, 0, 0] - z) ** 2).mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.05 * first


def test_context_valid_requires_full_window():
    v = np.ones((40, 40), bool)
    v[20, 20] = False
    c = context_valid(v, 5)
    assert not c[:2].any() and not c[:, -2:].any()        # window reaches past the raster edge
    assert not c[18:23, 18:23].any()                       # window contains the invalid pixel
    assert c[17, 20] and c[10, 10] and c[2, 2]


def test_balanced_indices_equal_share():
    site_of = np.array([0] * 1000 + [1] * 10)
    idx = dataset.balanced_indices(site_of, 2, 200, np.random.default_rng(0))
    assert (site_of[idx] == 0).sum() == 100 and (site_of[idx] == 1).sum() == 100


def test_densest_window():
    m = np.zeros((50, 50), bool)
    m[30:40, 5:15] = True
    assert densest_window(m, 10) == (30, 5)


def test_scene_rule_prefers_nearest_eligible():
    import pandas as pd
    t0, t1 = pd.Timestamp("2022-05-10", tz="UTC"), pd.Timestamp("2022-05-17", tz="UTC")
    s = pd.DataFrame({"strip_id": ["a", "b", "c", "d"],
                      "acquired": pd.to_datetime(["2022-05-12", "2022-05-25", "2022-05-01", "2022-05-18"], utc=True),
                      "aoi_footprint": [1.0, 1.0, 1.0, 0.5], "aoi_clear": [0.90, 0.99, 0.97, 1.0],
                      "bundle": ["analytic_8b_sr_udm2"] * 4})
    best, led = select_scene(s, (t0, t1), {"aoi_footprint_min": 0.99, "aoi_clear_min": 0.95,
                                           "bundles": ["analytic_8b_sr_udm2", "analytic_sr_udm2"]})
    assert best["strip_id"] == "b" and best["gap_days"] == pytest.approx(8.0)   # c is 9 d before, a cloudy, d partial
    assert set(led["status"]) >= {"selected", "eligible"}


def test_track_layout_spacing_is_exact():
    from island_altimetry.coastal import tracks
    lines = tracks.layout_by_spacing((0, 0, 5200, 3000), 0, 1000, 0.0)
    xs = [ln.coords[0][0] for ln in lines]
    assert np.allclose(np.diff(xs), 1000) and xs[0] == pytest.approx(500) and len(lines) == 5
