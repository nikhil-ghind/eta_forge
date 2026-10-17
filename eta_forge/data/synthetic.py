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
  truth traversals this module also returns;
* rain slows every trip in proportion to its intensity, and an hourly weather feed
  (with realistic gaps) is published for the weather join to find;
* merchants have a cuisine-dependent base prep time, a sensitivity to how many orders
  are already open, and lunch/dinner peaks; destinations inside the hotspot are
  towers with long handoff dwell.

Everything is seeded; the same arguments give byte-identical trips on every machine.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..features.h3_cells import cell_for
from ..features.weather import WeatherStation, WeatherTable
from ..graph.geometry import destination_point, haversine_m, interpolate
from ..graph.synthetic import SyntheticGraph
from ..graph.turns import classify_turn
from ..types import (
    BaselineRoute,
    DwellObservation,
    GpsPoint,
    Location,
    MatchQuality,
    NoRouteFound,
    PrepObservation,
    SegmentObservation,
    TripRecord,
    TurnType,
    WeatherObservation,
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
    rain_slowdown_per_mm: float = 0.025
    weather_gap_rate: float = 0.03
    merchants: int = 40
    orders_per_day: int = 150
    seed: int = 11

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            raise ValueError("synthetic start must be timezone-aware")
        if self.days < 1 or self.trips_per_day < 1:
            raise ValueError("days and trips_per_day must be >= 1")


CUISINE_BASE_PREP_S: Dict[str, float] = {
    "pizza": 840.0,
    "burgers": 600.0,
    "sushi": 960.0,
    "salad": 420.0,
    "chinese": 720.0,
    "indian": 900.0,
}


@dataclass(frozen=True)
class SyntheticMerchant:
    merchant_id: str
    cuisine: str
    location: Location
    base_prep_s: float
    load_sensitivity: float


@dataclass
class SyntheticTrip:
    """A generated trip with the ground truth the real world never hands over."""

    record: TripRecord
    route: BaselineRoute
    true_traversals: List[SegmentObservation] = field(default_factory=list)
    hotspot_s: float = 0.0
    turn_s: float = 0.0
    signal_s: float = 0.0
    weather_s: float = 0.0

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
        self._rain: Dict[int, List[Tuple[float, float, float]]] = {}
        self._merchants: Optional[List[SyntheticMerchant]] = None

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

    def _rain_episodes(self, day: int) -> List[Tuple[float, float, float]]:
        """``(start_hour, end_hour, mm_per_hour)`` rain episodes for a day.

        Drawn from a per-day RNG, not the trip stream, so the weather is a fixed fact
        about the synthetic world regardless of how many trips are generated.
        """
        if day not in self._rain:
            rng = random.Random(self.cfg.seed * 7_919 + day)
            episodes = []
            for _ in range(rng.choices([0, 1, 2], weights=[0.55, 0.3, 0.15])[0]):
                start = rng.uniform(0.0, 22.0)
                episodes.append((start, start + rng.uniform(1.5, 5.0), rng.uniform(0.5, 9.0)))
            self._rain[day] = episodes
        return self._rain[day]

    def weather_at(self, when: datetime) -> WeatherObservation:
        day = (when - self.cfg.start).days
        hour = when.hour + when.minute / 60.0
        precip = sum(mm for start, end, mm in self._rain_episodes(day) if start <= hour < end)
        season = math.cos(2.0 * math.pi * (when.timetuple().tm_yday - 200) / 365.0)
        temperature = 13.0 + 11.0 * season + 4.0 * math.sin(2.0 * math.pi * (hour - 9.0) / 24.0)
        if precip >= 4.0:
            condition = "snow" if temperature < 0.5 else "heavy_rain"
        elif precip > 0.0:
            condition = "snow" if temperature < 0.5 else "rain"
        else:
            condition = "cloudy" if (day + int(hour)) % 3 == 0 else "clear"
        return WeatherObservation(
            observed_at=when.replace(minute=0, second=0, microsecond=0),
            temperature_c=round(temperature, 1),
            precipitation_mm_h=round(precip, 2),
            wind_kph=round(8.0 + 6.0 * abs(math.sin(day + hour / 5.0)), 1),
            condition=condition,
        )

    def weather_factor(self, when: datetime) -> float:
        precip = self.weather_at(when).precipitation_mm_h
        return 1.0 + self.cfg.rain_slowdown_per_mm * min(precip, 12.0)

    def weather_table(self) -> WeatherTable:
        """An hourly feed from one station at the grid centre, with random gaps."""
        lo, hi = self.graph.bounds()
        centre = Location((lo.lat + hi.lat) / 2.0, (lo.lon + hi.lon) / 2.0)
        table = WeatherTable()
        table.add_station(WeatherStation("SYN0001", centre))
        gaps = random.Random(self.cfg.seed * 31)
        for hour in range(self.cfg.days * 24 + 1):
            if gaps.random() < self.cfg.weather_gap_rate:
                continue
            table.add("SYN0001", self.weather_at(self.cfg.start + timedelta(hours=hour)))
        return table

    # -- merchants, prep and dwell ---------------------------------------------

    def merchants(self) -> List[SyntheticMerchant]:
        if self._merchants is None:
            rng = random.Random(self.cfg.seed * 101)
            cuisines = sorted(CUISINE_BASE_PREP_S)
            out = []
            for i in range(self.cfg.merchants):
                cuisine = cuisines[i % len(cuisines)]
                out.append(
                    SyntheticMerchant(
                        merchant_id=f"m{i:04d}",
                        cuisine=cuisine,
                        location=self.graph.random_location(rng),
                        base_prep_s=CUISINE_BASE_PREP_S[cuisine] * rng.uniform(0.7, 1.4),
                        load_sensitivity=rng.uniform(0.02, 0.09),
                    )
                )
            self._merchants = out
        return self._merchants

    def prep_observations(self) -> List[PrepObservation]:
        """Historical orders with their true prep time."""
        rng = random.Random(self.cfg.seed * 211)
        merchants = self.merchants()
        # Long-tailed popularity: a few kitchens take most of the orders.
        weights = [1.0 / (1 + i) ** 0.8 for i in range(len(merchants))]
        meal_hours = [0.1] * 24
        for h, w in ((11, 0.8), (12, 1.6), (13, 1.2), (17, 0.9), (18, 1.7), (19, 1.9), (20, 1.3), (21, 0.7)):
            meal_hours[h] = w
        out: List[PrepObservation] = []
        for day in range(self.cfg.days):
            for n in range(self.cfg.orders_per_day):
                merchant = rng.choices(merchants, weights=weights)[0]
                hour = rng.choices(range(24), weights=meal_hours)[0]
                placed = self.cfg.start + timedelta(days=day, hours=hour, seconds=rng.uniform(0, 3600))
                items = rng.randint(1, 6)
                peak = meal_hours[hour] / 1.9
                open_orders = max(0, int(rng.gauss(2.0 + 8.0 * peak, 1.5)))
                prep = (
                    merchant.base_prep_s
                    * (1.0 + 0.06 * (items - 1))
                    * (1.0 + merchant.load_sensitivity * open_orders)
                    * math.exp(rng.gauss(0.0, 0.22))
                )
                out.append(
                    PrepObservation(
                        order_id=f"o{self.cfg.seed}-{day:03d}-{n:05d}",
                        merchant_id=merchant.merchant_id,
                        placed_at=placed,
                        prep_s=max(60.0, prep),
                        cuisine_category=merchant.cuisine,
                        item_count=items,
                        order_value_usd=round(items * rng.uniform(8.0, 18.0), 2),
                        open_orders_at_merchant=open_orders,
                    )
                )
        return out

    def dwell_seconds(self, location: Location, rng: random.Random) -> float:
        lo, hi = self.graph.bounds()
        lat_frac = (location.lat - lo.lat) / max(1e-9, hi.lat - lo.lat)
        lon_frac = (location.lon - lo.lon) / max(1e-9, hi.lon - lo.lon)
        tower = 0.55 <= lat_frac <= 0.85 and 0.35 <= lon_frac <= 0.65
        base = 210.0 if tower else 75.0
        return base * math.exp(rng.gauss(0.0, 0.45))

    def dwell_observations(
        self, *, resolution: int = 8, scheme: Optional[str] = None
    ) -> List[DwellObservation]:
        rng = random.Random(self.cfg.seed * 307)
        out: List[DwellObservation] = []
        for day in range(self.cfg.days):
            for n in range(self.cfg.orders_per_day):
                location = self.graph.random_location(rng)
                out.append(
                    DwellObservation(
                        order_id=f"d{self.cfg.seed}-{day:03d}-{n:05d}",
                        destination_cell=cell_for(location, resolution, scheme=scheme),
                        arrived_at=self.cfg.start + timedelta(days=day, hours=rng.uniform(10, 22)),
                        dwell_s=self.dwell_seconds(location, rng),
                    )
                )
        return out

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
        hotspot_total = turn_total = signal_total = weather_total = 0.0
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
            rain = self.weather_factor(clock)
            noise = math.exp(self._rng.gauss(0.0, self.cfg.noise_sigma))
            moving_s = leg.duration_s * hotspot * rain * noise
            hotspot_total += leg.duration_s * (hotspot - 1.0)
            weather_total += leg.duration_s * hotspot * (rain - 1.0)

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
            weather_s=weather_total,
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
    "SyntheticMerchant",
    "CUISINE_BASE_PREP_S",
    "SyntheticConfig",
    "SyntheticSource",
    "SyntheticTrip",
    "TRUE_TURN_COST_S",
    "ground_truth_observations",
]
