"""An in-memory grid graph that satisfies ``RouterProtocol``.

This is not a toy for tests alone — it is what makes the rest of the project provable
without PostGIS. Everything downstream of routing (map matching, profile mining,
feature building, training, evaluation, the service itself) can be exercised end to
end against a graph that lives in a dictionary.

The grid is Manhattan-like on purpose: a rectangular street network with avenues
(faster, primary) and streets (slower, residential), alternating one-way directions,
and a congestion field that varies with time of day. That structure means a residual
model trained on synthetic trips learns something recognisable — rush hour is slow,
avenues are fast, left turns cost — rather than fitting noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..types import (
    BaselineRoute,
    HighwayClass,
    Location,
    NoRouteFound,
    RouteLeg,
)
from .costs import time_bucket
from .geometry import bearing_deg, haversine_m
from .turns import classify_turn, turn_penalty_s


@dataclass(frozen=True)
class SyntheticEdge:
    segment_id: int
    source: int
    target: int
    start: Location
    end: Location
    length_m: float
    highway_class: HighwayClass
    free_flow_kph: float
    has_signal_at_end: bool

    @property
    def bearing(self) -> float:
        return bearing_deg(self.start, self.end)


class SyntheticGraph:
    """A deterministic grid graph with a time-of-day congestion field."""

    def __init__(
        self,
        *,
        rows: int = 24,
        cols: int = 24,
        block_m: float = 180.0,
        origin: Location = Location(40.7000, -74.0100),
        seed: int = 7,
        avenue_every: int = 4,
    ):
        if rows < 2 or cols < 2:
            raise ValueError("grid must be at least 2x2")
        self.rows = rows
        self.cols = cols
        self.block_m = block_m
        self.origin = origin
        self.avenue_every = max(1, avenue_every)
        self._rng = random.Random(seed)

        self._nodes: Dict[int, Location] = {}
        self._edges: Dict[int, SyntheticEdge] = {}
        self._out: Dict[int, List[int]] = {}
        self._build()

    # -- construction ------------------------------------------------------------

    def _node_id(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _node_location(self, row: int, col: int) -> Location:
        # Degrees per metre, latitude-corrected so the blocks are actually square.
        dlat = self.block_m / 111_320.0
        dlon = self.block_m / (111_320.0 * math.cos(math.radians(self.origin.lat)))
        return Location(self.origin.lat + row * dlat, self.origin.lon + col * dlon)

    def _build(self) -> None:
        for row in range(self.rows):
            for col in range(self.cols):
                self._nodes[self._node_id(row, col)] = self._node_location(row, col)

        next_id = 1
        for row in range(self.rows):
            for col in range(self.cols):
                here = self._node_id(row, col)
                # Avenues run north-south (varying row) every `avenue_every` columns.
                if row + 1 < self.rows:
                    is_avenue = col % self.avenue_every == 0
                    # Alternate one-way direction, as a real grid does.
                    northbound = (col // self.avenue_every) % 2 == 0
                    if is_avenue and not northbound:
                        pair = (self._node_id(row + 1, col), here)
                    else:
                        pair = (here, self._node_id(row + 1, col))
                    next_id = self._add_edge(next_id, pair, is_avenue, row, col)
                    if not is_avenue:
                        next_id = self._add_edge(
                            next_id, (pair[1], pair[0]), is_avenue, row, col
                        )
                if col + 1 < self.cols:
                    is_avenue = row % self.avenue_every == 0
                    eastbound = (row // self.avenue_every) % 2 == 0
                    if is_avenue and not eastbound:
                        pair = (self._node_id(row, col + 1), here)
                    else:
                        pair = (here, self._node_id(row, col + 1))
                    next_id = self._add_edge(next_id, pair, is_avenue, row, col)
                    if not is_avenue:
                        next_id = self._add_edge(
                            next_id, (pair[1], pair[0]), is_avenue, row, col
                        )

    def _add_edge(
        self, next_id: int, pair: Tuple[int, int], is_avenue: bool, row: int, col: int
    ) -> int:
        source, target = pair
        start, end = self._nodes[source], self._nodes[target]
        edge = SyntheticEdge(
            segment_id=next_id,
            source=source,
            target=target,
            start=start,
            end=end,
            length_m=haversine_m(start, end),
            highway_class=(
                HighwayClass.PRIMARY if is_avenue else HighwayClass.RESIDENTIAL
            ),
            free_flow_kph=(42.0 if is_avenue else 26.0),
            # Signals at avenue crossings only, which is where they actually are.
            has_signal_at_end=is_avenue and (row + col) % 3 == 0,
        )
        self._edges[next_id] = edge
        self._out.setdefault(source, []).append(next_id)
        return next_id + 1

    # -- congestion --------------------------------------------------------------

    def congestion_factor(self, edge: SyntheticEdge, when: datetime) -> float:
        """Multiplier on free-flow speed: 1.0 is free-flowing, 0.35 is gridlock.

        Two rush peaks, a weekend discount, and a spatial gradient that makes the
        centre of the grid slower than its edges — the same shape a real city has, so
        that a model trained here has real structure to find.
        """
        dow, tod = time_bucket(when, tod_buckets=24, dow_buckets=7)
        morning = math.exp(-((tod - 8.5) ** 2) / 4.0)
        evening = math.exp(-((tod - 18.0) ** 2) / 5.0)
        peak = max(morning, evening)
        if dow >= 5:  # weekend: the peaks flatten out
            peak *= 0.35

        mid_row, mid_col = self.rows / 2.0, self.cols / 2.0
        row = edge.source // self.cols
        col = edge.source % self.cols
        centrality = 1.0 - (
            (abs(row - mid_row) / mid_row + abs(col - mid_col) / mid_col) / 2.0
        )
        centrality = max(0.0, min(1.0, centrality))

        slowdown = 0.55 * peak * (0.4 + 0.6 * centrality)
        # Avenues carry the through-traffic and so suffer more in a peak.
        if edge.highway_class is HighwayClass.PRIMARY:
            slowdown *= 1.15
        return max(0.30, 1.0 - slowdown)

    def edge_duration_s(self, edge: SyntheticEdge, when: datetime) -> float:
        speed = max(3.0, edge.free_flow_kph * self.congestion_factor(edge, when))
        return edge.length_m / (speed / 3.6)

    # -- routing -----------------------------------------------------------------

    def nearest_node(self, point: Location) -> Tuple[int, float]:
        best_id, best_d = None, float("inf")
        for node_id, loc in self._nodes.items():
            d = haversine_m(point, loc)
            if d < best_d:
                best_id, best_d = node_id, d
        assert best_id is not None
        return best_id, best_d

    def route(
        self,
        origin: Location,
        destination: Location,
        departure_at: datetime,
        *,
        use_profiles: bool = True,
        include_turn_penalties: bool = True,
    ) -> BaselineRoute:
        """Dijkstra over the grid, with the same turn penalties the real router uses.

        Turn cost makes this a search over (node, arriving-edge) states rather than
        over nodes: the cheapest way to reach an intersection depends on which way you
        are facing when you get there, and collapsing that loses the u-turn ban.
        """
        import heapq

        start, start_d = self.nearest_node(origin)
        goal, goal_d = self.nearest_node(destination)

        if start == goal:
            return BaselineRoute(
                legs=(),
                origin=origin,
                destination=destination,
                departure_at=departure_at,
                used_speed_profiles=use_profiles,
                profile_coverage=1.0,
                snapped_origin_distance_m=start_d,
                snapped_destination_distance_m=goal_d,
            )

        # state: (node, incoming_edge_id or None)
        best: Dict[Tuple[int, Optional[int]], float] = {(start, None): 0.0}
        prev: Dict[Tuple[int, Optional[int]], Tuple[Tuple[int, Optional[int]], int]] = {}
        heap: List[Tuple[float, int, Optional[int]]] = [(0.0, start, None)]
        goal_state: Optional[Tuple[int, Optional[int]]] = None

        while heap:
            cost, node, incoming = heapq.heappop(heap)
            state = (node, incoming)
            if cost > best.get(state, float("inf")) + 1e-9:
                continue
            if node == goal:
                goal_state = state
                break
            for edge_id in self._out.get(node, ()):
                edge = self._edges[edge_id]
                step = self.edge_duration_s(edge, departure_at)
                if edge.has_signal_at_end:
                    step += 8.0
                if incoming is not None and include_turn_penalties:
                    turn = classify_turn(self._edges[incoming].bearing, edge.bearing)
                    step += turn_penalty_s(turn)
                nxt = (edge.target, edge_id)
                candidate = cost + step
                if candidate < best.get(nxt, float("inf")) - 1e-9:
                    best[nxt] = candidate
                    prev[nxt] = (state, edge_id)
                    heapq.heappush(heap, (candidate, edge.target, edge_id))

        if goal_state is None:
            raise NoRouteFound("synthetic grid has no path between those points")

        edge_ids: List[int] = []
        cursor = goal_state
        while cursor in prev:
            previous, edge_id = prev[cursor]
            edge_ids.append(edge_id)
            cursor = previous
        edge_ids.reverse()

        legs = [self._to_leg(self._edges[eid], departure_at) for eid in edge_ids]
        return BaselineRoute(
            legs=tuple(legs),
            origin=origin,
            destination=destination,
            departure_at=departure_at,
            used_speed_profiles=use_profiles,
            profile_coverage=1.0 if use_profiles else 0.0,
            snapped_origin_distance_m=start_d,
            snapped_destination_distance_m=goal_d,
        )

    def _to_leg(self, edge: SyntheticEdge, when: datetime) -> RouteLeg:
        duration = self.edge_duration_s(edge, when)
        return RouteLeg(
            segment_id=edge.segment_id,
            source_vertex=edge.source,
            target_vertex=edge.target,
            length_m=edge.length_m,
            highway_class=edge.highway_class,
            speed_kph=(edge.length_m / duration) * 3.6,
            duration_s=duration,
            bearing_in_deg=edge.bearing,
            bearing_out_deg=edge.bearing,
            has_signal_at_end=edge.has_signal_at_end,
            geometry=(edge.start, edge.end),
        )

    # -- helpers for fixtures ----------------------------------------------------

    @property
    def edges(self) -> Sequence[SyntheticEdge]:
        return tuple(self._edges.values())

    def edge(self, segment_id: int) -> SyntheticEdge:
        return self._edges[segment_id]

    def random_location(self, rng: Optional[random.Random] = None) -> Location:
        """A point somewhere inside the grid, jittered off the lattice."""
        r = rng or self._rng
        row = r.uniform(0.0, self.rows - 1)
        col = r.uniform(0.0, self.cols - 1)
        dlat = self.block_m / 111_320.0
        dlon = self.block_m / (111_320.0 * math.cos(math.radians(self.origin.lat)))
        return Location(self.origin.lat + row * dlat, self.origin.lon + col * dlon)

    def bounds(self) -> Tuple[Location, Location]:
        return self._node_location(0, 0), self._node_location(self.rows - 1, self.cols - 1)


__all__ = ["SyntheticGraph", "SyntheticEdge"]
