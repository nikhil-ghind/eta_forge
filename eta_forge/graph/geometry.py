"""Spherical geometry, in pure Python.

PostGIS does all of this server-side, but the map matcher, the turn classifier and the
synthetic graph all need the same functions client-side, and they need to agree with
the server to the metre. Everything here uses the same spherical earth radius that
``ST_Distance(geography)`` approximates, and the same bearing convention
(``ST_Azimuth``: degrees clockwise from north, in ``[0, 360)``).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from ..types import EARTH_RADIUS_M, Location


def haversine_m(a: Location, b: Location) -> float:
    """Great-circle distance in metres.

    The haversine form is used rather than the spherical law of cosines because the
    latter loses precision for the short distances that dominate here: two GPS points
    a few metres apart are exactly the case where cosine differencing degrades.
    """
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a: Location, b: Location) -> float:
    """Initial bearing from ``a`` to ``b``, degrees clockwise from north, in [0, 360)."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlon = math.radians(b.lon - a.lon)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return normalize_bearing(math.degrees(math.atan2(x, y)))


def normalize_bearing(deg: float) -> float:
    """Wrap any angle into ``[0, 360)``."""
    return deg % 360.0


def bearing_delta_deg(incoming: float, outgoing: float) -> float:
    """Signed turn angle in ``(-180, 180]``; positive is a right turn.

    The wrap is the whole point of this function. Heading 350 degrees then 10 degrees
    is a 20-degree right drift, not a 340-degree left u-turn, and getting that wrong
    misclassifies every manoeuvre near due north.
    """
    delta = (outgoing - incoming) % 360.0
    if delta > 180.0:
        delta -= 360.0
    return delta


def polyline_length_m(points: Sequence[Location]) -> float:
    """Total great-circle length of a polyline."""
    if len(points) < 2:
        return 0.0
    return sum(
        haversine_m(points[i], points[i + 1]) for i in range(len(points) - 1)
    )


def interpolate(a: Location, b: Location, fraction: float) -> Location:
    """Point a given fraction along the great circle from ``a`` to ``b``.

    Falls back to linear interpolation for coincident points, where the spherical
    formula's ``sin(d)`` denominator vanishes.
    """
    fraction = min(1.0, max(0.0, fraction))
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    d = haversine_m(a, b) / EARTH_RADIUS_M
    if d < 1e-12:
        return Location(a.lat, a.lon)
    sin_d = math.sin(d)
    A = math.sin((1.0 - fraction) * d) / sin_d
    B = math.sin(fraction * d) / sin_d
    x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
    y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
    z = A * math.sin(lat1) + B * math.sin(lat2)
    return Location(
        math.degrees(math.atan2(z, math.sqrt(x * x + y * y))),
        math.degrees(math.atan2(y, x)),
    )


def project_point_on_segment(
    point: Location, start: Location, end: Location
) -> Tuple[Location, float, float]:
    """Project ``point`` onto the segment ``start``->``end``.

    Returns ``(projected_point, fraction_along, perpendicular_distance_m)``. The
    fraction is clamped to ``[0, 1]``, so a point beyond either end projects onto that
    end — which is the behaviour the map matcher wants, since a GPS fix past the end of
    a segment belongs to the next segment, not to an extrapolation of this one.

    The projection is done in a local equirectangular frame centred on ``start``. Over
    the tens of metres a road segment spans, the distortion is far below GPS noise, and
    a true spherical cross-track projection would be considerably more expensive inside
    the Viterbi inner loop.
    """
    lat0 = math.radians(start.lat)
    kx = math.cos(lat0) * math.pi * EARTH_RADIUS_M / 180.0
    ky = math.pi * EARTH_RADIUS_M / 180.0

    sx, sy = 0.0, 0.0
    ex, ey = (end.lon - start.lon) * kx, (end.lat - start.lat) * ky
    px, py = (point.lon - start.lon) * kx, (point.lat - start.lat) * ky

    dx, dy = ex - sx, ey - sy
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return Location(start.lat, start.lon), 0.0, haversine_m(point, start)

    t = ((px - sx) * dx + (py - sy) * dy) / denom
    t = min(1.0, max(0.0, t))
    projected = interpolate(start, end, t)
    return projected, t, haversine_m(point, projected)


def project_point_on_polyline(
    point: Location, polyline: Sequence[Location]
) -> Tuple[Location, float, float]:
    """Nearest point on a multi-vertex polyline.

    Returns ``(projected_point, distance_along_m, perpendicular_distance_m)``. The
    "along" value is measured in metres from the start of the polyline, which is what
    turns a matched GPS point into a segment offset.
    """
    if not polyline:
        raise ValueError("polyline must have at least one point")
    if len(polyline) == 1:
        return polyline[0], 0.0, haversine_m(point, polyline[0])

    best: Optional[Tuple[Location, float, float]] = None
    travelled = 0.0
    for i in range(len(polyline) - 1):
        start, end = polyline[i], polyline[i + 1]
        span = haversine_m(start, end)
        projected, fraction, perpendicular = project_point_on_segment(point, start, end)
        if best is None or perpendicular < best[2]:
            best = (projected, travelled + fraction * span, perpendicular)
        travelled += span
    assert best is not None
    return best


def polyline_bearings(points: Sequence[Location]) -> Tuple[float, float]:
    """Entry and exit bearing of a polyline.

    On a curved segment these differ, and collapsing them to the chord bearing is what
    makes a naive turn classifier call a highway on-ramp a sharp right.
    """
    if len(points) < 2:
        return 0.0, 0.0
    return bearing_deg(points[0], points[1]), bearing_deg(points[-2], points[-1])


def destination_point(origin: Location, bearing: float, distance_m: float) -> Location:
    """Move ``distance_m`` from ``origin`` along ``bearing``. Used to build fixtures."""
    d = distance_m / EARTH_RADIUS_M
    theta = math.radians(normalize_bearing(bearing))
    lat1, lon1 = math.radians(origin.lat), math.radians(origin.lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(theta)
    )
    lon2 = lon1 + math.atan2(
        math.sin(theta) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return Location(math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0)


def bbox_of(points: Iterable[Location], *, pad_m: float = 0.0) -> Tuple[float, float, float, float]:
    """``(min_lat, min_lon, max_lat, max_lon)``, optionally padded by metres."""
    lats: List[float] = []
    lons: List[float] = []
    for p in points:
        lats.append(p.lat)
        lons.append(p.lon)
    if not lats:
        raise ValueError("cannot compute a bounding box of no points")
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    if pad_m > 0:
        dlat = pad_m / (math.pi * EARTH_RADIUS_M / 180.0)
        mean_lat = math.radians((min_lat + max_lat) / 2.0)
        dlon = pad_m / max(1e-9, math.cos(mean_lat) * math.pi * EARTH_RADIUS_M / 180.0)
        min_lat, max_lat = min_lat - dlat, max_lat + dlat
        min_lon, max_lon = min_lon - dlon, max_lon + dlon
    return min_lat, min_lon, max_lat, max_lon


__all__ = [
    "haversine_m",
    "bearing_deg",
    "normalize_bearing",
    "bearing_delta_deg",
    "polyline_length_m",
    "interpolate",
    "project_point_on_segment",
    "project_point_on_polyline",
    "polyline_bearings",
    "destination_point",
    "bbox_of",
]
