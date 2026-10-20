"""Site geometry: metric buffers built in a site-centred azimuthal equidistant
projection, then transformed to WGS84 (EPSG:4326). Never buffers degrees."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point, Polygon, mapping, shape

GEOD = Geod(ellps="WGS84")
N_VERTICES = 128


def aeqd_crs(lon: float, lat: float) -> CRS:
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")


def _to_wgs84(lon: float, lat: float):
    tr = Transformer.from_crs(aeqd_crs(lon, lat), "EPSG:4326", always_xy=True)

    def f(xy):
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])
    return f


def disk(lon: float, lat: float, radius_m: float, n: int = N_VERTICES) -> Polygon:
    """Geodesic disk approximated by n vertices (counter-clockwise, lon/lat)."""
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"center out of range (lon={lon}, lat={lat}); coordinates must be lon,lat")
    local = Point(0, 0).buffer(radius_m, quad_segs=n // 4)
    poly = shapely.transform(local, _to_wgs84(lon, lat))
    poly = shapely.geometry.polygon.orient(poly, sign=1.0)
    minx, _, maxx, _ = poly.bounds
    if maxx - minx > 180 or minx < -180 or maxx > 180:
        raise ValueError("AOI crosses the antimeridian; split geometry support is not implemented")
    if abs(lat) + radius_m / 111_000 > 85:
        raise ValueError("polar AOIs are not supported by this construction")
    return poly


def annulus(lon: float, lat: float, inner_m: float, outer_m: float) -> Polygon:
    return disk(lon, lat, outer_m).difference(disk(lon, lat, inner_m))


def max_geodesic_radius_m(poly: Polygon, lon: float, lat: float) -> tuple[float, float]:
    """Min and max geodesic distance from center to exterior vertices (verification)."""
    xs, ys = poly.exterior.coords.xy
    _, _, d = GEOD.inv([lon] * len(xs), [lat] * len(ys), list(xs), list(ys))
    return float(min(d)), float(max(d))


def to_sliderule_poly(poly: Polygon) -> list[dict]:
    """SlideRule polygon: counter-clockwise list of {lon, lat}, closed."""
    poly = shapely.geometry.polygon.orient(poly, sign=1.0)
    return [{"lon": round(x, 9), "lat": round(y, 9)} for x, y in poly.exterior.coords]


def geometry_hash(geom) -> str:
    """Stable hash of a geometry, rounded to ~1 mm."""
    gj = json.dumps(mapping(shapely.set_precision(geom, 1e-8)), sort_keys=True)
    return hashlib.sha256(gj.encode()).hexdigest()[:16]


def from_geojson(obj: dict):
    return shape(obj)
