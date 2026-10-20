"""Normalized analysis views of raw chunks. Raw chunks are never modified; views
add zone labels, local metric coordinates, fill masks, and source identity while
keeping native variables and raw values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from .aoi import aeqd_crs

FLOAT_FILL_MIN = 1e38  # float32 max fills (3.4028235e38) and float64 equivalents

# Height variables per product, with their vertical reference (kept separate).
HEIGHTS = {
    "atl03": {"height": "ellipsoid_itrf2014"},
    "atl08": {"terrain/h_te_best_fit": "ellipsoid_wgs84", "h_te_median": "ellipsoid_wgs84",
              "terrain/h_te_mean": "ellipsoid_wgs84"},
    "atl24": {"ellipse_h": "ellipsoid_itrf2014", "ortho_h": "egm2008", "surface_h": "egm2008"},
    "gedi_l2a": {"elevation_lm": "ellipsoid_wgs84", "elevation_hr": "ellipsoid_wgs84"},
}
PRIMARY_HEIGHT = {"atl03": "height", "atl08": "terrain/h_te_best_fit", "atl24": "ellipse_h", "gedi_l2a": "elevation_lm"}


class Zoner:
    """Assigns core / buffer / outer_margin from geodesic-equivalent metric radius
    in the same site-centred AEQD projection used to build the AOIs."""

    def __init__(self, lon: float, lat: float, core_m: float, outer_m: float):
        self.tr = Transformer.from_crs("EPSG:4326", aeqd_crs(lon, lat), always_xy=True)
        self.core_m, self.outer_m = core_m, outer_m

    def xy(self, lon, lat):
        return self.tr.transform(np.asarray(lon, float), np.asarray(lat, float))

    def apply(self, df: pd.DataFrame, lon, lat) -> pd.DataFrame:
        x, y = self.xy(lon, lat)
        r = np.hypot(x, y)
        df["x_local_m"], df["y_local_m"], df["r_m"] = x, y, r
        df["zone"] = np.where(r <= self.core_m, "core", np.where(r <= self.outer_m, "buffer", "outer_margin"))
        return df


def mask_fills(df: pd.DataFrame) -> dict:
    """Replace float max fills with NaN in the view; return counts per column."""
    counts = {}
    for c in df.columns:
        if c != "geometry" and df[c].dtype.kind == "f":
            m = df[c].abs() >= FLOAT_FILL_MIN
            n = int(m.sum())
            if n:
                df.loc[m, c] = np.nan
                counts[c] = n
    return counts


def view(pid: str, raw, granule: str, zoner: Zoner) -> tuple[pd.DataFrame, dict]:
    """Normalized view of one raw chunk (GeoDataFrame or ATL09 profile table)."""
    if pid == "atl09":
        df = pd.DataFrame(raw).copy()
        df["time_utc"] = pd.to_datetime(df["high_rate/delta_time"], unit="s", origin=pd.Timestamp("2018-01-01"), utc=True)
        lon, lat = df["lon_mean"].to_numpy(), df["lat_mean"].to_numpy()
    else:
        df = pd.DataFrame(raw.drop(columns="geometry"))
        df["time_utc"] = pd.to_datetime(raw.index, utc=True)
        df = df.reset_index(drop=True)
        lon, lat = raw.geometry.x.to_numpy(), raw.geometry.y.to_numpy()
        df["lon"], df["lat"] = lon, lat
    df["granule"] = granule
    fills = mask_fills(df)
    zoner.apply(df, lon, lat)
    if pid in PRIMARY_HEIGHT:
        h = PRIMARY_HEIGHT[pid]
        df["height_primary"] = df[h].astype("float64")
        df["height_primary_ref"] = HEIGHTS[pid][h]
    return df, fills
