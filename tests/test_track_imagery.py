import numpy as np
import pandas as pd
import pytest

from island_altimetry import imagery
from scripts_loader import load_script

pti = load_script("plot_track_imagery")
T = pd.Timestamp("2021-09-21T15:00:00Z")


def orders(**kw):
    base = {"strip_id": "s", "bundle": "analytic_8b_sr_udm2", "dir": "d", "download_status": "ok"}
    return {name: {**base, "strip_id": name, "acquired_utc": t, **extra} for name, (t, extra) in kw.items()}


def test_nearest_strip_picks_smallest_abs_dt_with_sign():
    o = orders(a=("2021-09-24T00:48:27Z", {}), b=("2021-09-20T01:00:00Z", {}))
    s = pti.nearest_strip(T, o, {"b": 0.9}, 7 * 24)
    assert s["strip_id"] == "b" and s["dt_hours"] == pytest.approx(-38.0) and s["measured_clear_core"] == 0.9


def test_nearest_strip_window_and_download_status():
    o = orders(far=("2021-10-15T00:00:00Z", {}), failed=("2021-09-21T16:00:00Z", {"download_status": "retrieval_failed"}))
    assert pti.nearest_strip(T, o, {}, 7 * 24) is None
    assert pti.nearest_strip(T, o, {}, 30 * 24)["strip_id"] == "far"


def test_gedi_along_track_monotone_zero_at_closest():
    n = 21
    x = np.linspace(-1000, 1000, n) + 300           # passes 300 m east of the centre
    y = np.linspace(-1000, 1000, n)
    df = pd.DataFrame({"beam": 5, "time_utc": pd.date_range("2021-09-19", periods=n, freq="s"),
                       "x_local_m": x, "y_local_m": y})
    df["r_m"] = np.hypot(df["x_local_m"], df["y_local_m"])
    s = pti.gedi_along_track(df.sample(frac=1, random_state=0)).sort_index()
    assert (np.diff(s.to_numpy()) > 0).all()
    assert s.iloc[int(df["r_m"].idxmin())] == pytest.approx(0.0)
    assert s.iloc[-1] - s.iloc[0] == pytest.approx(np.hypot(2000, 2000))


def test_height_window_ignores_clouds_and_keeps_island():
    geoid = 47.7
    h = pd.Series(np.r_[np.full(500, 48.0), np.linspace(48, 76, 50), np.full(30, 700.0)])
    lo, hi = pti.height_window(h, geoid)
    assert lo == pytest.approx(geoid - 10) and 76 < hi < 120


def test_to_raster_xy_centre_maps_to_site_lonlat():
    from pyproj import Transformer
    f = imagery.to_raster_xy("EPSG:32654", (141.481, 24.285))
    x, y = f(0.0, 0.0)
    lon, lat = Transformer.from_crs("EPSG:32654", "EPSG:4326", always_xy=True).transform(x, y)
    assert lon == pytest.approx(141.481, abs=1e-7) and lat == pytest.approx(24.285, abs=1e-7)
