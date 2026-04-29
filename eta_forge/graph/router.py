"""Baseline routing over pgRouting.

Three database round trips per request, in this order:

1. **snap** origin and destination to the nearest routable vertices (one query, both
   points, via a lateral join — two separate KNN queries would double the latency for
   no benefit);
2. **route** with ``pgr_dijkstra`` over the time-dependent cost expression;
3. **coverage**: what fraction of the chosen route had a real mined profile behind it.

Step 3 is not free and could be folded into step 2, but keeping it separate means the
routing query returns exactly the columns needed to build legs and nothing else, and
the coverage number stays honest when the route came out of a cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..config import Config
from ..db.runner import SqlRunner, safe_identifier
from ..types import (
    BaselineRoute,
    HighwayClass,
    Location,
    NoRouteFound,
    RouteLeg,
)
from .costs import (
    build_cost_sql,
    build_dijkstra_sql,
    build_profile_coverage_sql,
    build_snap_sql,
    time_bucket,
)
from .geometry import bbox_of, haversine_m, polyline_bearings


@dataclass(frozen=True)
class SnapResult:
    """A request coordinate tied to a graph vertex."""

    vertex_id: int
    location: Location
    distance_m: float


class RouterProtocol(Protocol):
    """What the ETA service needs from a router.

    :class:`Router` (PostGIS) and ``SyntheticGraph`` (in-memory) both satisfy it, which
    is what lets the entire pipeline above this line run with no database.
    """

    def route(
        self, origin: Location, destination: Location, departure_at: datetime
    ) -> BaselineRoute: ...


# Both endpoints are snapped in one statement. The lateral join runs the same KNN
# search twice against the same index rather than paying two network round trips.
SNAP_PAIR_SQL_TEMPLATE = """
    WITH request(role, pt) AS (
        VALUES ('origin',      ST_SetSRID(ST_MakePoint(%s, %s), {srid})),
               ('destination', ST_SetSRID(ST_MakePoint(%s, %s), {srid}))
    )
    SELECT r.role,
           v.id                                                    AS vertex_id,
           ST_Y(v.the_geom)                                        AS lat,
           ST_X(v.the_geom)                                        AS lon,
           ST_Distance(v.the_geom::geography, r.pt::geography)     AS distance_m
    FROM request r
    CROSS JOIN LATERAL (
        SELECT vv.id, vv.the_geom
        FROM {vertex_table} vv
        WHERE ST_DWithin(vv.the_geom::geography, r.pt::geography, %s)
        ORDER BY vv.the_geom <-> r.pt
        LIMIT 1
    ) v
"""


class Router:
    """pgRouting-backed baseline router."""

    def __init__(self, cfg: Config, runner: SqlRunner):
        self._cfg = cfg
        self._runner = runner

    # -- public ------------------------------------------------------------------

    def route(
        self,
        origin: Location,
        destination: Location,
        departure_at: datetime,
        *,
        use_profiles: bool = True,
    ) -> BaselineRoute:
        """Compute the baseline route, or raise :class:`NoRouteFound`."""
        crow_m = haversine_m(origin, destination)
        if crow_m > self._cfg.graph.max_route_distance_m:
            raise NoRouteFound(
                f"straight-line distance {crow_m / 1000:.1f} km exceeds the configured "
                f"maximum {self._cfg.graph.max_route_distance_m / 1000:.1f} km"
            )

        snapped = self.snap_pair(origin, destination)
        start, end = snapped["origin"], snapped["destination"]
        if start.vertex_id == end.vertex_id:
            # Both ends land on the same vertex: a walk, not a drive. An empty route is
            # the honest answer; the service turns it into a near-zero travel term.
            return BaselineRoute(
                legs=(),
                origin=origin,
                destination=destination,
                departure_at=departure_at,
                used_speed_profiles=use_profiles,
                profile_coverage=1.0,
                snapped_origin_distance_m=start.distance_m,
                snapped_destination_distance_m=end.distance_m,
            )

        dow, tod = time_bucket(
            departure_at,
            tod_buckets=self._cfg.profiles.tod_buckets,
            dow_buckets=self._cfg.profiles.dow_buckets,
            tz=self._cfg.profiles.timezone,
        )
        # Pad the search box by half the crow-flight distance (floor 2 km): enough room
        # for a real detour around a river or a park, without materialising the city.
        pad = max(2_000.0, crow_m * 0.5)
        box = bbox_of([origin, destination, start.location, end.location], pad_m=pad)

        cost = build_cost_sql(
            self._cfg.graph,
            self._cfg.profiles,
            dow_bucket=dow,
            tod_bucket=tod,
            use_profiles=use_profiles,
            bbox=box,
        )
        rows = self._runner.fetch_all(
            build_dijkstra_sql(self._cfg.graph, cost),
            (start.vertex_id, end.vertex_id),
            label="router.dijkstra",
        )
        legs = self._build_legs(rows)
        if not legs and not self._runner.dry_run:
            raise NoRouteFound(
                f"no path between vertices {start.vertex_id} and {end.vertex_id}"
            )

        coverage = self._profile_coverage(legs, dow, tod) if use_profiles else 0.0
        return BaselineRoute(
            legs=tuple(legs),
            origin=origin,
            destination=destination,
            departure_at=departure_at,
            used_speed_profiles=use_profiles,
            profile_coverage=coverage,
            snapped_origin_distance_m=start.distance_m,
            snapped_destination_distance_m=end.distance_m,
        )

    def snap_pair(
        self, origin: Location, destination: Location
    ) -> Mapping[str, SnapResult]:
        """Snap both endpoints in one query."""
        sql = SNAP_PAIR_SQL_TEMPLATE.format(
            vertex_table=safe_identifier(self._cfg.graph.vertex_table),
            srid=int(self._cfg.graph.srid),
        ).strip()
        rows = self._runner.fetch_all(
            sql,
            (
                origin.lon,
                origin.lat,
                destination.lon,
                destination.lat,
                self._cfg.graph.snap_radius_m,
            ),
            label="router.snap_pair",
        )
        if self._runner.dry_run:
            return {
                "origin": SnapResult(1, origin, 0.0),
                "destination": SnapResult(2, destination, 0.0),
            }

        found = {
            row["role"]: SnapResult(
                vertex_id=int(row["vertex_id"]),
                location=Location(float(row["lat"]), float(row["lon"])),
                distance_m=float(row["distance_m"]),
            )
            for row in rows
        }
        for role in ("origin", "destination"):
            if role not in found:
                raise NoRouteFound(
                    f"no routable vertex within {self._cfg.graph.snap_radius_m:.0f} m "
                    f"of the {role}"
                )
        return found

    def snap_one(self, point: Location) -> SnapResult:
        """Single-point snap, used by the map matcher's endpoint handling."""
        rows = self._runner.fetch_all(
            build_snap_sql(self._cfg.graph),
            (
                point.lon, point.lat,
                point.lon, point.lat,
                self._cfg.graph.snap_radius_m,
                point.lon, point.lat,
                self._cfg.graph.snap_candidates,
            ),
            label="router.snap_one",
        )
        if not rows:
            if self._runner.dry_run:
                return SnapResult(1, point, 0.0)
            raise NoRouteFound(
                f"no routable vertex within {self._cfg.graph.snap_radius_m:.0f} m"
            )
        row = rows[0]
        return SnapResult(
            vertex_id=int(row["id"]),
            location=Location(float(row["lat"]), float(row["lon"])),
            distance_m=float(row["distance_m"]),
        )

    # -- internals ---------------------------------------------------------------

    def _build_legs(self, rows: Sequence[Mapping[str, Any]]) -> List[RouteLeg]:
        legs: List[RouteLeg] = []
        for row in rows:
            # pgr_dijkstra's final row is the destination node with edge = -1; there is
            # no segment behind it and it must not become a leg.
            if row.get("edge") in (None, -1):
                continue
            geometry = _decode_geojson_line(row.get("geojson"))
            length_m = float(row["length_m"])
            cost_s = float(row["cost"])
            speed_kph = (length_m / cost_s) * 3.6 if cost_s > 0 else 1.0
            bearing_in = row.get("bearing_in_deg")
            bearing_out = row.get("bearing_out_deg")
            if bearing_in is None or bearing_out is None:
                bearing_in, bearing_out = polyline_bearings(geometry)
            legs.append(
                RouteLeg(
                    segment_id=int(row["segment_id"]),
                    source_vertex=int(row["source"]),
                    target_vertex=int(row["target"]),
                    length_m=length_m,
                    highway_class=HighwayClass.coerce(row.get("highway_class")),
                    speed_kph=max(0.1, speed_kph),
                    duration_s=cost_s,
                    bearing_in_deg=float(bearing_in),
                    bearing_out_deg=float(bearing_out),
                    has_signal_at_end=bool(row.get("has_signal_at_end", False)),
                    geometry=tuple(geometry),
                )
            )
        return legs

    def _profile_coverage(
        self, legs: Sequence[RouteLeg], dow: int, tod: int
    ) -> float:
        if not legs:
            return 0.0
        row = self._runner.fetch_one(
            build_profile_coverage_sql(self._cfg.profiles),
            (
                [leg.segment_id for leg in legs],
                dow,
                tod,
                self._cfg.profiles.min_samples_per_cell,
            ),
            label="router.profile_coverage",
        )
        if not row:
            return 0.0
        segments = float(row.get("segments") or 0.0)
        if segments <= 0:
            return 0.0
        return min(1.0, float(row.get("profiled") or 0.0) / segments)


def _decode_geojson_line(raw: Any) -> List[Location]:
    """``ST_AsGeoJSON`` output -> a list of Locations. GeoJSON is [lon, lat]."""
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        return [Location(float(c[1]), float(c[0])) for c in data.get("coordinates", [])]
    except (ValueError, KeyError, TypeError, IndexError):
        return []


__all__ = ["Router", "RouterProtocol", "SnapResult"]
