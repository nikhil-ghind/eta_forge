"""A deterministic synthetic trip source over :class:`SyntheticGraph`.

The point is not realism for its own sake; it is that the synthetic world has *known
structure the baseline does not capture*, so that every downstream stage has
something real to find:

* the baseline prices each segment at its congested speed, but the "true" trip also
  pays for manoeuvres at a higher rate than the router's turn penalties, waits at
  signals, and crawls through a downtown hotspot at peak — the residual model should
  learn all three;
* traversal times carry multiplicative log-normal noise, so p90 is genuinely wider
  than p50;
* GPS traces are sampled every few seconds with Gaussian position noise, so the map
  matcher has to work for its answer and its output can be scored against the ground
  truth traversals this module also returns.

Everything is seeded; the same arguments give byte-identical trips on every machine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..graph.geometry import destination_point, haversine_m, interpolate
from ..graph.synthetic import SyntheticGraph
from ..graph.turns import classify_turn
from ..types import (
    BaselineRoute,
    GpsPoint,
    Location,
    MatchQuality,
    NoRouteFound,
    SegmentObservation,
    TripRecord,
    TurnType,
)

# What manoeuvres *really* cost in the synthetic city, versus the router's defaults in
# eta_forge/graph/turns.py. The gap is the turn-related residual the model must learn.
TRUE_TURN_COST_S: Dict[TurnType, float] = {
    TurnType.STRAIGHT: 0.0,
    TurnType.SLIGHT_LEFT: 3.0,
    TurnType.SLIGHT_RIGHT: 2.0,
    TurnType.LEFT: 21.0,
    TurnType.RIGHT: 7.0,
    TurnType.SHARP_LEFT: 26.0,
    TurnType.SHARP_RIGHT: 12.0,
    TurnType.UTURN: 40.0,
}


@dataclass(frozen=True)
class SyntheticConfig:
    start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    days: int = 28
    trips_per_day: int = 120
    min_trip_m: float = 600.0
    gps_interval_s: float = 5.0
    gps_noise_m: float = 6.0
    noise_sigma: float = 0.14
    signal_wait_max_s: float = 35.0
    hotspot_slowdown: float = 0.45
    seed: int = 11

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            raise ValueError("synthetic start must be timezone-aware")
        if self.days < 1 or self.trips_per_day < 1:
            raise ValueError("days and trips_per_day must be >= 1")


@dataclass
class SyntheticTrip:
    """A generated trip with the ground truth the real world never hands over."""

    record: TripRecord
    route: BaselineRoute
    true_traversals: List[SegmentObservation] = field(default_factory=list)
    hotspot_s: float = 0.0
    turn_s: float = 0.0
    signal_s: float = 0.0

    @property
    def baseline_s(self) -> float:
        return self.route.duration_s

    @property
    def residual_s(self) -> float:
        return self.record.duration_s - self.route.duration_s


class SyntheticSource:
    """Generates trips, traces and ground-truth traversals over a synthetic grid."""

    def __init__(self, graph: SyntheticGraph, cfg: Optional[SyntheticConfig] = None):
        self.graph = graph
        self.cfg = cfg or SyntheticConfig()
        self._rng = random.Random(self.cfg.seed)

    # -- the world's hidden structure --------------------------------------------

    def hotspot_factor(self, location: Location, when: datetime) -> float:
        """Slowdown multiplier on time spent inside the downtown hotspot.

        A box in the north-centre of the grid ("Midtown") that is badly congested at
        the weekday evening peak. The baseline's congestion field does not know about
        it; only the residual model can.
        """
        (lo, hi) = self.graph.bounds()
        lat_frac = (location.lat - lo.lat) / max(1e-9, hi.lat - lo.lat)
        lon_frac = (location.lon - lo.lon) / max(1e-9, hi.lon - lo.lon)
        inside = 0.55 <= lat_frac <= 0.85 and 0.35 <= lon_frac <= 0.65
        if not inside:
            return 1.0
        hour = when.hour + when.minute / 60.0
        evening = math.exp(-((hour - 18.0) ** 2) / 3.0)
        if when.weekday() >= 5:
            evening *= 0.3
        return 1.0 + self.cfg.hotspot_slowdown * evening

    # -- generation ----------------------------------------------------------------

    def departure_times(self) -> Iterator[datetime]:
        """Departures weighted toward the day's peaks, as real demand is."""
        cfg = self.cfg
        hour_weights = [
            0.2, 0.15, 0.1, 0.1, 0.15, 0.3, 0.6, 1.0, 1.3, 1.1, 0.9, 0.9,
            1.0, 0.95, 0.9, 1.0, 1.2, 1.4, 1.5, 1.3, 1.0, 0.8, 0.6, 0.4,
        ]
        for day in range(cfg.days):
            base = cfg.start + timedelta(days=day)
            for _ in range(cfg.trips_per_day):
                hour = self._rng.choices(range(24), weights=hour_weights)[0]
                seconds = self._rng.uniform(0, 3600)
                yield base + timedelta(hours=hour, seconds=seconds)

    def trips(self, *, limit: Optional[int] = None) -> Iterator[SyntheticTrip]:
        produced = 0
        for index, departure in enumerate(self.departure_times()):
            if limit is not None and produced >= limit:
                return
            trip = self._one_trip(index, departure)
            if trip is None:
                continue
            produced += 1
            yield trip

    def _pick_endpoints(self) -> Tuple[Location, Location]:
        for _ in range(20):
            a = self.graph.random_location(self._rng)
            b = self.graph.random_location(self._rng)
            if haversine_m(a, b) >= self.cfg.min_trip_m:
                return a, b
        return a, b

    def _one_trip(self, index: int, departure: datetime) -> Optional[SyntheticTrip]:
        origin, destination = self._pick_endpoints()
        try:
            route = self.graph.route(origin, destination, departure)
        except NoRouteFound:
            return None
        if not route.legs:
            return None

        clock = departure
        traversals: List[SegmentObservation] = []
        trace: List[GpsPoint] = []
        hotspot_total = turn_total = signal_total = 0.0
        next_fix = departure
        trip_id = f"syn-{self.cfg.seed}-{index:07d}"

        for i, leg in enumerate(route.legs):
            # Manoeuvre cost is paid at the start of the leg, standing at the junction.
            if i > 0:
                prev = route.legs[i - 1]
                turn = classify_turn(prev.bearing_out_deg, leg.bearing_in_deg)
                turn_cost = TRUE_TURN_COST_S[turn] * self._rng.uniform(0.6, 1.4)
                clock += timedelta(seconds=turn_cost)
                turn_total += turn_cost

            midpoint = leg.geometry[0] if leg.geometry else route.origin
            hotspot = self.hotspot_factor(midpoint, clock)
            noise = math.exp(self._rng.gauss(0.0, self.cfg.noise_sigma))
            moving_s = leg.duration_s * hotspot * noise
            hotspot_total += leg.duration_s * (hotspot - 1.0)

            entered = clock
            # GPS fixes while moving along this leg.
            start = leg.geometry[0] if leg.geometry else route.origin
            end = leg.geometry[-1] if leg.geometry else route.destination
            while next_fix <= entered + timedelta(seconds=moving_s):
                fraction = (next_fix - entered).total_seconds() / max(moving_s, 1e-6)
                trace.append(self._fix(next_fix, interpolate(start, end, fraction)))
                next_fix += timedelta(seconds=self.cfg.gps_interval_s)
            clock = entered + timedelta(seconds=moving_s)

            # Signal wait at the end of the leg, stationary at the stop line.
            wait = 0.0
            if leg.has_signal_at_end and self._rng.random() < 0.55:
                wait = self._rng.uniform(0.0, self.cfg.signal_wait_max_s)
                while next_fix <= clock + timedelta(seconds=wait):
                    trace.append(self._fix(next_fix, end))
                    next_fix += timedelta(seconds=self.cfg.gps_interval_s)
                clock += timedelta(seconds=wait)
                signal_total += wait

            traversals.append(
                SegmentObservation(
                    segment_id=leg.segment_id,
                    entered_at=entered,
                    traversal_s=moving_s + wait,
                    length_m=leg.length_m,
                    highway_class=leg.highway_class,
                    trip_id=trip_id,
                )
            )

        trace.append(self._fix(clock, route.legs[-1].geometry[-1]))
        duration = (clock - departure).total_seconds()
        record = TripRecord(
            trip_id=trip_id,
            pickup_at=departure,
            dropoff_at=clock,
            origin=origin,
            destination=destination,
            distance_m=route.distance_m,
            duration_s=duration,
            match_quality=MatchQuality.ENDPOINTS,
            vehicle_type="car",
            gps_trace=tuple(trace),
        )
        return SyntheticTrip(
            record=record,
            route=route,
            true_traversals=traversals,
            hotspot_s=hotspot_total,
            turn_s=turn_total,
            signal_s=signal_total,
        )

    def _fix(self, at: datetime, true_location: Location) -> GpsPoint:
        """A GPS sample: the true position displaced by isotropic Gaussian noise."""
        dx = self._rng.gauss(0.0, self.cfg.gps_noise_m)
        dy = self._rng.gauss(0.0, self.cfg.gps_noise_m)
        distance = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dx, dy))
        noisy = destination_point(true_location, bearing, distance) if distance > 0 else true_location
        return GpsPoint(at=at, location=noisy, accuracy_m=max(3.0, self.cfg.gps_noise_m))


def ground_truth_observations(trips: Sequence[SyntheticTrip]) -> Iterator[SegmentObservation]:
    for trip in trips:
        yield from trip.true_traversals


__all__ = [
    "SyntheticConfig",
    "SyntheticSource",
    "SyntheticTrip",
    "TRUE_TURN_COST_S",
    "ground_truth_observations",
]
