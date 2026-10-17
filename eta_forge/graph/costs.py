"""Composition of the ``pgr_dijkstra`` cost expression.

``pgr_dijkstra`` takes an *SQL string* that must yield ``(id, source, target, cost,
reverse_cost)``. That indirection is the hook this project hangs the whole first stage
on: the cost is not a stored column, it is an expression evaluated at query time
against the mined speed profiles, so routing at 08:00 on a Thursday and at 03:00 on a
Sunday traverses two different graphs over the same topology.

Cost is **seconds**, not metres. Routing on distance and multiplying by an average
speed afterwards gives the shortest path, which on a congested grid is frequently not
the fastest one — the entire point of mining profiles is lost if the router cannot see
them while choosing.

Three fallbacks are stacked, cheapest first:

1. the smoothed profile cell for this (segment, dow, tod), when it rests on enough
   evidence — observed traversals plus whatever the shrinkage borrowed from adjacent
   hours on the same segment (``effective_sample_count``);
2. the segment's OSM ``maxspeed``, damped, when it has one;
3. the highway-class default.

The damping in (2) matters: a posted 50 km/h limit on a Manhattan avenue is an
aspiration, and using it raw makes every baseline optimistic in exactly the places
where congestion is worst.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import GraphConfig, ProfilesConfig
from ..db.runner import safe_identifier
from ..types import ValidationError

# How much of the posted speed limit a vehicle actually achieves, by class. Motorways
# roughly hit their limit off-peak; a residential street's posted 30 is never met.
SPEED_LIMIT_DAMPING: Dict[str, float] = {
    "motorway": 0.95,
    "trunk": 0.90,
    "primary": 0.80,
    "secondary": 0.78,
    "tertiary": 0.75,
    "residential": 0.65,
    "service": 0.55,
    "unclassified": 0.70,
}


def time_bucket(
    when: datetime, *, tod_buckets: int = 24, dow_buckets: int = 7, tz: Optional[str] = None
) -> Tuple[int, int]:
    """Map a timestamp onto ``(dow_bucket, tod_bucket)``.

    The timestamp is converted into the profile's local timezone first. Mining rush
    hour in UTC would smear 08:00 local across two different buckets for half the
    year, once daylight saving moves.
    """
    if when.tzinfo is None:
        raise ValidationError("time_bucket needs a timezone-aware timestamp")
    if 24 % tod_buckets != 0:
        raise ValidationError("tod_buckets must divide 24")
    if dow_buckets not in (1, 7):
        raise ValidationError("dow_buckets must be 1 or 7")

    local = when
    if tz:
        try:
            from zoneinfo import ZoneInfo

            local = when.astimezone(ZoneInfo(tz))
        except Exception:  # pragma: no cover - missing tzdata on a slim image
            local = when

    hours_per_bucket = 24 // tod_buckets
    tod = (local.hour // hours_per_bucket) % tod_buckets
    dow = local.weekday() if dow_buckets == 7 else 0
    return dow, tod


@dataclass(frozen=True)
class CostExpression:
    """A ``pgr_dijkstra`` edge query plus the parameters it binds."""

    sql: str
    params: Sequence[Any]
    uses_profiles: bool

    def as_dijkstra_argument(self) -> str:
        """The inner SQL exactly as ``pgr_dijkstra`` expects it."""
        return self.sql

    def describe(self) -> str:
        return f"{self.sql}\n-- params: {list(self.params)!r}"


def _damping_case(alias: str) -> str:
    branches = "\n            ".join(
        f"WHEN {alias}.highway_class = '{cls}' THEN {factor}"
        for cls, factor in SPEED_LIMIT_DAMPING.items()
    )
    return f"CASE\n            {branches}\n            ELSE 0.7\n        END"


def _class_default_case(alias: str, defaults: Dict[str, float]) -> str:
    branches = "\n            ".join(
        f"WHEN {alias}.highway_class = '{cls}' THEN {float(speed)}"
        for cls, speed in sorted(defaults.items())
    )
    return f"CASE\n            {branches}\n            ELSE 30.0\n        END"


def build_cost_sql(
    graph: GraphConfig,
    profiles: ProfilesConfig,
    *,
    dow_bucket: int,
    tod_bucket: int,
    use_profiles: bool = True,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> CostExpression:
    """Compose the edge query handed to ``pgr_dijkstra``.

    ``bbox`` restricts the edge set to a padded box around origin and destination.
    Without it, every routing call materialises the whole city's edge list, which for
    a 2 km trip is three orders of magnitude more rows than the search will ever touch.

    ``reverse_cost`` is ``-1`` on one-way segments. pgRouting treats a negative reverse
    cost as "this edge cannot be traversed in that direction", which is the only
    correct encoding — a large positive cost would let the router drive the wrong way
    up a one-way street whenever the alternative was long enough.

    Every value in this expression is inlined as a numeric literal rather than bound as
    a parameter. That is deliberate: the whole string is nested inside a dollar-quoted
    argument to ``pgr_dijkstra``, and a driver's placeholder substitution has no reason
    to respect a dollar-quoted region. All the values are ints and floats coerced here,
    and all the identifiers pass ``safe_identifier``, so nothing request-derived ever
    reaches the SQL text.
    """
    edge = safe_identifier(graph.edge_table)
    attr = safe_identifier(graph.attr_table)
    profile_table = safe_identifier(profiles.smoothed_table)

    damping = _damping_case("a")
    class_default = _class_default_case("a", graph.default_speed_kph)

    if use_profiles:
        speed_expr = f"""COALESCE(
            p.speed_kph,
            a.speed_limit_kph * {damping},
            {class_default}
        )"""
        profile_join = f"""
        LEFT JOIN {profile_table} p
               ON p.segment_id = a.segment_id
              AND p.dow_bucket = {int(dow_bucket)}
              AND p.tod_bucket = {int(tod_bucket)}
              AND p.effective_sample_count >= {int(profiles.min_samples_per_cell)}"""
    else:
        speed_expr = f"""COALESCE(
            a.speed_limit_kph * {damping},
            {class_default}
        )"""
        profile_join = ""

    where_clauses = ["a.length_m > 0"]
    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
        where_clauses.append(
            "e.the_geom && ST_MakeEnvelope("
            f"{float(min_lon)}, {float(min_lat)}, "
            f"{float(max_lon)}, {float(max_lat)}, {int(graph.srid)})"
        )

    where = "\n          AND ".join(where_clauses)

    # Speed is clamped into the mined bounds before it becomes a divisor: a profile
    # cell holding 0.2 km/h (a vehicle that stopped for lunch mid-segment) would
    # otherwise turn one edge into an hours-long cost and distort the whole search.
    sql = f"""
        SELECT
            e.gid AS id,
            e.source,
            e.target,
            a.length_m / (LEAST(GREATEST({speed_expr}, {profiles.min_speed_kph}),
                                {profiles.max_speed_kph}) / 3.6)
              + CASE WHEN a.has_signal_at_end THEN {float(graph.signal_penalty_s)}
                     ELSE 0 END                                     AS cost,
            CASE
                WHEN a.oneway THEN -1
                ELSE a.length_m / (LEAST(GREATEST({speed_expr}, {profiles.min_speed_kph}),
                                         {profiles.max_speed_kph}) / 3.6)
                     + CASE WHEN a.has_signal_at_end
                            THEN {float(graph.signal_penalty_s)} ELSE 0 END
            END                                                     AS reverse_cost
        FROM {edge} e
        JOIN {attr} a ON a.segment_id = e.gid{profile_join}
        WHERE {where}
    """.strip()

    return CostExpression(sql=sql, params=(), uses_profiles=use_profiles)


SNAP_VERTEX_SQL_TEMPLATE = """
    SELECT v.id,
           ST_Y(v.the_geom) AS lat,
           ST_X(v.the_geom) AS lon,
           ST_Distance(v.the_geom::geography, ST_SetSRID(ST_MakePoint(%s, %s), {srid})::geography)
               AS distance_m
    FROM {vertex_table} v
    WHERE ST_DWithin(
            v.the_geom::geography,
            ST_SetSRID(ST_MakePoint(%s, %s), {srid})::geography,
            %s)
    ORDER BY v.the_geom <-> ST_SetSRID(ST_MakePoint(%s, %s), {srid})
    LIMIT %s
"""


def build_snap_sql(graph: GraphConfig) -> str:
    """KNN vertex search.

    The ``ORDER BY ... <-> point`` is what makes this an index scan rather than a
    filter-then-sort: PostGIS's GiST index answers nearest-neighbour ordering directly.
    The ``ST_DWithin`` predicate then bounds it, so a request dropped in the ocean
    returns nothing quickly instead of scanning to find the nearest road in New Jersey.
    """
    return SNAP_VERTEX_SQL_TEMPLATE.format(
        vertex_table=safe_identifier(graph.vertex_table), srid=int(graph.srid)
    ).strip()


DIJKSTRA_SQL_TEMPLATE = """
    SELECT
        d.seq,
        d.node,
        d.edge,
        d.cost,
        d.agg_cost,
        a.segment_id,
        a.highway_class,
        a.length_m,
        a.speed_limit_kph,
        a.bearing_in_deg,
        a.bearing_out_deg,
        a.has_signal_at_end,
        e.source,
        e.target,
        ST_AsGeoJSON(e.the_geom) AS geojson
    FROM pgr_dijkstra(
            $${cost_sql}$$,
            %s::bigint, %s::bigint,
            directed => true
         ) d
    JOIN {edge_table} e ON e.gid = d.edge
    JOIN {attr_table} a ON a.segment_id = d.edge
    ORDER BY d.seq
"""


def build_dijkstra_sql(graph: GraphConfig, cost: CostExpression) -> str:
    """Wrap a cost expression in the ``pgr_dijkstra`` call and the attribute join.

    Note the dollar-quoting of the inner cost SQL: it contains its own quotes, and
    escaping them into a single-quoted literal is both unreadable and easy to get
    subtly wrong. The inner SQL is composed entirely from validated identifiers and
    numeric literals, never from request data.
    """
    return DIJKSTRA_SQL_TEMPLATE.format(
        cost_sql=cost.sql,
        edge_table=safe_identifier(graph.edge_table),
        attr_table=safe_identifier(graph.attr_table),
    ).strip()


PROFILE_COVERAGE_SQL_TEMPLATE = """
    SELECT
        count(*)::double precision                                   AS segments,
        count(p.segment_id)::double precision                        AS profiled,
        COALESCE(max(p.refreshed_at), 'epoch'::timestamptz)          AS newest_profile
    FROM unnest(%s::bigint[]) AS s(segment_id)
    LEFT JOIN {profile_table} p
           ON p.segment_id = s.segment_id
          AND p.dow_bucket = %s
          AND p.tod_bucket = %s
          AND p.effective_sample_count >= %s
"""


def build_profile_coverage_sql(profiles: ProfilesConfig) -> str:
    """How much of a chosen route was actually covered by mined profiles.

    Returned on every response: a baseline built entirely from class defaults is a
    guess wearing a routing engine's clothes, and the caller deserves to know.
    """
    return PROFILE_COVERAGE_SQL_TEMPLATE.format(
        profile_table=safe_identifier(profiles.smoothed_table)
    ).strip()


__all__ = [
    "CostExpression",
    "build_cost_sql",
    "build_snap_sql",
    "build_dijkstra_sql",
    "build_profile_coverage_sql",
    "time_bucket",
    "SPEED_LIMIT_DAMPING",
]
