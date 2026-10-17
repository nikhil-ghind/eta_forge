"""Candidate generation and network distances for the HMM map matcher.

The matcher itself (``hmm.py``) is a pure algorithm. Everything it needs to know about
the road network comes through the :class:`CandidateSource` protocol:

* which segments lie near a GPS fix, and where on each segment the fix projects;
* how far a vehicle must *drive* to get from one candidate position to another, and
  which segments it crosses on the way.

Two implementations: :class:`GraphCandidateSource` over the in-memory synthetic grid,
and :class:`PostgisCandidateSource`, which asks PostGIS for candidates via a GiST
radius search and pgRouting for the network distance. Tests use a third, hand-built
one, which is the reason the protocol exists.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..db.runner import SqlRunner, safe_identifier
from ..graph.geometry import project_point_on_polyline, polyline_length_m
from ..graph.synthetic import SyntheticGraph
from ..types import HighwayClass, Location

# Two projections onto the same segment closer than this, but "backwards", are GPS
# jitter around a stationary vehicle, not a reversal.
SAME_SEGMENT_BACKTRACK_TOLERANCE_M = 8.0


@dataclass(frozen=True)
class Candidate:
    """One possible matched position for a GPS fix."""

    segment_id: int
    source_vertex: int
    target_vertex: int
    length_m: float
    offset_m: float          # distance along the segment from its source vertex
    distance_m: float        # perpendicular distance from the fix
    projected: Location
    highway_class: HighwayClass = HighwayClass.UNCLASSIFIED

    @property
    def remaining_m(self) -> float:
        return max(0.0, self.length_m - self.offset_m)


@dataclass(frozen=True)
class PathPiece:
    """A traversed stretch of one segment, from ``start_m`` to ``end_m`` along it."""

    segment_id: int
    length_m: float
    start_m: float
    end_m: float

    @property
    def covered_m(self) -> float:
        return max(0.0, self.end_m - self.start_m)


@dataclass(frozen=True)
class PathBetween:
    """The driven path between two candidate positions."""

    distance_m: float
    pieces: Tuple[PathPiece, ...]


class CandidateSource(Protocol):
    def candidates(
        self, point: Location, *, radius_m: float, limit: int
    ) -> List[Candidate]: ...

    def path_between(
        self, a: Candidate, b: Candidate, *, max_distance_m: float
    ) -> Optional[PathBetween]: ...


def same_segment_path(a: Candidate, b: Candidate) -> Optional[PathBetween]:
    """The trivial case: both positions on one segment, ``b`` at or ahead of ``a``."""
    if a.segment_id != b.segment_id:
        return None
    if b.offset_m + SAME_SEGMENT_BACKTRACK_TOLERANCE_M < a.offset_m:
        return None
    start, end = a.offset_m, max(a.offset_m, b.offset_m)
    return PathBetween(
        distance_m=end - start,
        pieces=(PathPiece(a.segment_id, a.length_m, start, end),),
    )


def stitch_path(
    a: Candidate,
    b: Candidate,
    via: Sequence[Tuple[int, float]],
    network_m: float,
) -> PathBetween:
    """Compose ``a``'s remainder + full intermediate segments + ``b``'s prefix."""
    pieces: List[PathPiece] = [PathPiece(a.segment_id, a.length_m, a.offset_m, a.length_m)]
    for segment_id, length in via:
        pieces.append(PathPiece(segment_id, length, 0.0, length))
    pieces.append(PathPiece(b.segment_id, b.length_m, 0.0, b.offset_m))
    return PathBetween(
        distance_m=a.remaining_m + network_m + b.offset_m, pieces=tuple(pieces)
    )


# --------------------------------------------------------------------------------------
# in-memory
# --------------------------------------------------------------------------------------


class GraphCandidateSource:
    """Candidates and network distances over a :class:`SyntheticGraph`.

    Candidate search is a brute-force scan with a cheap bounding pre-filter — the grid
    is a few thousand edges — and network distance is a length-weighted Dijkstra with a
    cutoff, cached per (from-vertex, to-vertex), because the Viterbi lattice asks for the
    same vertex pairs many times over consecutive fixes.
    """

    def __init__(self, graph: SyntheticGraph):
        self._graph = graph
        self._edges = {e.segment_id: e for e in graph.edges}
        self._out: Dict[int, List[int]] = {}
        for edge in self._edges.values():
            self._out.setdefault(edge.source, []).append(edge.segment_id)
        self._cache: Dict[Tuple[int, int], Optional[Tuple[float, Tuple[int, ...]]]] = {}

    def candidates(self, point: Location, *, radius_m: float, limit: int) -> List[Candidate]:
        # ~111 km per degree of latitude; the pre-filter is deliberately loose.
        pad = radius_m / 100_000.0 * 1.5
        found: List[Candidate] = []
        for edge in self._edges.values():
            if not (
                min(edge.start.lat, edge.end.lat) - pad <= point.lat
                <= max(edge.start.lat, edge.end.lat) + pad
                and min(edge.start.lon, edge.end.lon) - pad * 1.4 <= point.lon
                <= max(edge.start.lon, edge.end.lon) + pad * 1.4
            ):
                continue
            projected, along, perpendicular = project_point_on_polyline(
                point, (edge.start, edge.end)
            )
            if perpendicular > radius_m:
                continue
            found.append(
                Candidate(
                    segment_id=edge.segment_id,
                    source_vertex=edge.source,
                    target_vertex=edge.target,
                    length_m=edge.length_m,
                    offset_m=min(edge.length_m, along),
                    distance_m=perpendicular,
                    projected=projected,
                    highway_class=edge.highway_class,
                )
            )
        found.sort(key=lambda c: (c.distance_m, c.segment_id))
        return found[:limit]

    def _network(
        self, start: int, goal: int, cutoff_m: float
    ) -> Optional[Tuple[float, Tuple[int, ...]]]:
        key = (start, goal)
        if key in self._cache:
            cached = self._cache[key]
            if cached is None or cached[0] <= cutoff_m:
                return cached
        if start == goal:
            self._cache[key] = (0.0, ())
            return self._cache[key]
        best: Dict[int, float] = {start: 0.0}
        prev: Dict[int, Tuple[int, int]] = {}
        heap: List[Tuple[float, int]] = [(0.0, start)]
        result: Optional[Tuple[float, Tuple[int, ...]]] = None
        while heap:
            dist, node = heapq.heappop(heap)
            if dist > cutoff_m:
                break
            if dist > best.get(node, float("inf")):
                continue
            if node == goal:
                edges: List[int] = []
                cursor = node
                while cursor in prev:
                    parent, edge_id = prev[cursor]
                    edges.append(edge_id)
                    cursor = parent
                result = (dist, tuple(reversed(edges)))
                break
            for edge_id in self._out.get(node, ()):
                edge = self._edges[edge_id]
                nd = dist + edge.length_m
                if nd < best.get(edge.target, float("inf")):
                    best[edge.target] = nd
                    prev[edge.target] = (node, edge_id)
                    heapq.heappush(heap, (nd, edge.target))
        # Only a found path is cached unconditionally; a miss under one cutoff might be
        # a hit under a larger one.
        if result is not None:
            self._cache[key] = result
        return result

    def path_between(
        self, a: Candidate, b: Candidate, *, max_distance_m: float
    ) -> Optional[PathBetween]:
        direct = same_segment_path(a, b)
        if direct is not None:
            return direct
        budget = max_distance_m - a.remaining_m - b.offset_m
        if budget < 0:
            return None
        found = self._network(a.target_vertex, b.source_vertex, budget)
        if found is None:
            return None
        network_m, edge_ids = found
        via = [(eid, self._edges[eid].length_m) for eid in edge_ids]
        return stitch_path(a, b, via, network_m)


# --------------------------------------------------------------------------------------
# PostGIS
# --------------------------------------------------------------------------------------

CANDIDATES_SQL_TEMPLATE = """
    WITH fix AS (SELECT ST_SetSRID(ST_MakePoint(%s, %s), 4326) AS pt)
    SELECT
        e.gid                                                     AS segment_id,
        e.source,
        e.target,
        a.length_m,
        a.highway_class,
        ST_LineLocatePoint(e.the_geom, fix.pt) * a.length_m       AS offset_m,
        ST_Distance(e.the_geom::geography, fix.pt::geography)     AS distance_m,
        ST_Y(ST_ClosestPoint(e.the_geom, fix.pt))                 AS lat,
        ST_X(ST_ClosestPoint(e.the_geom, fix.pt))                 AS lon
    FROM {edge_table} e
    JOIN {attr_table} a ON a.segment_id = e.gid
    CROSS JOIN fix
    WHERE ST_DWithin(e.the_geom::geography, fix.pt::geography, %s)
    ORDER BY e.the_geom <-> fix.pt
    LIMIT %s
"""

# Distance-weighted (not time-weighted): the HMM compares *driven distance* against
# great-circle distance, so the edge cost here must be metres. The vertex ids inside
# the dollar-quoted edge query are inlined as integers, never bound: a driver's
# placeholder substitution does not respect a dollar-quoted region.
NETWORK_PATH_SQL_TEMPLATE = """
    SELECT d.seq, d.edge, d.agg_cost, a.length_m
    FROM pgr_dijkstra(
        $$SELECT e.gid AS id, e.source, e.target,
                 a.length_m AS cost,
                 CASE WHEN a.oneway THEN -1 ELSE a.length_m END AS reverse_cost
            FROM {edge_table} e JOIN {attr_table} a ON a.segment_id = e.gid
           WHERE e.the_geom && ST_Expand(
                     (SELECT ST_Collect(the_geom) FROM {vertex_table}
                       WHERE id IN ({start}, {goal})), {pad_deg})$$,
        %s::bigint, %s::bigint, directed => true) d
    JOIN {attr_table} a ON a.segment_id = d.edge
    ORDER BY d.seq
"""


class PostgisCandidateSource:
    """Candidates via a GiST radius search; paths via a bounded ``pgr_dijkstra``.

    Every network path is one round trip, so results are cached per vertex pair. A
    typical trace revisits the same few dozen vertex pairs across consecutive fixes,
    and the cache turns an O(points x candidates^2) query count into roughly
    O(distinct junctions).
    """

    def __init__(
        self,
        runner: SqlRunner,
        *,
        edge_table: str = "road_segment",
        attr_table: str = "segment_attr",
        vertex_table: str = "road_segment_vertices_pgr",
        pad_deg: float = 0.01,
    ):
        self._runner = runner
        self._edge = safe_identifier(edge_table)
        self._attr = safe_identifier(attr_table)
        self._vertex = safe_identifier(vertex_table)
        self._pad = float(pad_deg)
        self._cache: Dict[Tuple[int, int], Optional[Tuple[float, Tuple[Tuple[int, float], ...]]]] = {}

    def candidates(self, point: Location, *, radius_m: float, limit: int) -> List[Candidate]:
        sql = CANDIDATES_SQL_TEMPLATE.format(
            edge_table=self._edge, attr_table=self._attr
        ).strip()
        rows = self._runner.fetch_all(
            sql, (point.lon, point.lat, radius_m, limit), label="mapmatch.candidates"
        )
        return [_candidate_from_row(r) for r in rows]

    def _network(
        self, start: int, goal: int
    ) -> Optional[Tuple[float, Tuple[Tuple[int, float], ...]]]:
        key = (start, goal)
        if key in self._cache:
            return self._cache[key]
        if start == goal:
            self._cache[key] = (0.0, ())
            return self._cache[key]
        sql = NETWORK_PATH_SQL_TEMPLATE.format(
            edge_table=self._edge,
            attr_table=self._attr,
            vertex_table=self._vertex,
            pad_deg=self._pad,
            start=int(start),
            goal=int(goal),
        ).strip()
        rows = self._runner.fetch_all(
            sql, (int(start), int(goal)), label="mapmatch.network_path"
        )
        if not rows:
            self._cache[key] = None
            return None
        via = tuple((int(r["edge"]), float(r["length_m"])) for r in rows if r["edge"] != -1)
        total = sum(length for _, length in via)
        self._cache[key] = (total, via)
        return self._cache[key]

    def path_between(
        self, a: Candidate, b: Candidate, *, max_distance_m: float
    ) -> Optional[PathBetween]:
        direct = same_segment_path(a, b)
        if direct is not None:
            return direct
        found = self._network(a.target_vertex, b.source_vertex)
        if found is None:
            return None
        network_m, via = found
        path = stitch_path(a, b, via, network_m)
        return path if path.distance_m <= max_distance_m else None


def _candidate_from_row(row: Mapping[str, Any]) -> Candidate:
    length = float(row["length_m"])
    return Candidate(
        segment_id=int(row["segment_id"]),
        source_vertex=int(row["source"]),
        target_vertex=int(row["target"]),
        length_m=length,
        offset_m=min(length, max(0.0, float(row["offset_m"]))),
        distance_m=float(row["distance_m"]),
        projected=Location(float(row["lat"]), float(row["lon"])),
        highway_class=HighwayClass.coerce(row.get("highway_class")),
    )


def candidate_on_polyline(
    point: Location,
    *,
    segment_id: int,
    polyline: Sequence[Location],
    source_vertex: int,
    target_vertex: int,
    highway_class: HighwayClass = HighwayClass.UNCLASSIFIED,
) -> Candidate:
    """Build a candidate by projecting onto a known polyline. Used by fixtures."""
    projected, along, perpendicular = project_point_on_polyline(point, polyline)
    length = polyline_length_m(polyline)
    return Candidate(
        segment_id=segment_id,
        source_vertex=source_vertex,
        target_vertex=target_vertex,
        length_m=length,
        offset_m=min(length, along),
        distance_m=perpendicular,
        projected=projected,
        highway_class=highway_class,
    )


__all__ = [
    "Candidate",
    "CandidateSource",
    "PathBetween",
    "PathPiece",
    "GraphCandidateSource",
    "PostgisCandidateSource",
    "same_segment_path",
    "stitch_path",
    "candidate_on_polyline",
    "SAME_SEGMENT_BACKTRACK_TOLERANCE_M",
]
