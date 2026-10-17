"""Building the residual model's training set.

Each historical trip is re-routed through the *same* router the service uses, at its
own pickup time, so the baseline in a training row is exactly the baseline the
service would have produced for that request. The target is then

    y = actual_duration - baseline_duration

Training on a baseline computed any other way — a different profile generation, a
cached route from another hour — teaches the model to correct a baseline it will never
see in production.

Rows keep their slice keys (hour, weekday, distance, cells) alongside the features so
the evaluation harness can slice without rebuilding anything.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..features.builder import FeatureBuilder, FeatureRow
from ..features.weather import WeatherTable
from ..graph.router import RouterProtocol
from ..types import (
    EtaForgeError,
    Location,
    MatchQuality,
    TripRecord,
    _parse_ts,
)


@dataclass
class TrainingRow:
    trip_id: str
    departure_at: datetime
    feature_row: FeatureRow
    baseline_s: float
    actual_s: float
    distance_m: float
    merchant_id: Optional[str] = None

    @property
    def residual_s(self) -> float:
        return self.actual_s - self.baseline_s

    @property
    def origin_cell(self) -> str:
        return self.feature_row.origin_cell

    @property
    def dest_cell(self) -> str:
        return self.feature_row.dest_cell

    def slice_keys(self) -> Dict[str, Any]:
        features = self.feature_row.features
        return {
            "hour": int(features.get("local_hour", self.departure_at.hour)),
            "dow": int(features.get("dow", self.departure_at.weekday())),
            "distance_m": self.distance_m,
            "origin_cell": self.origin_cell,
            "dest_cell": self.dest_cell,
            "merchant_id": self.merchant_id,
        }


@dataclass
class BuildReport:
    trips_read: int = 0
    rows_built: int = 0
    skipped: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trips_read": self.trips_read,
            "rows_built": self.rows_built,
            "skipped": dict(self.skipped),
        }


def build_rows(
    trips: Iterable[TripRecord],
    router: RouterProtocol,
    builder: FeatureBuilder,
    *,
    weather: Optional[WeatherTable] = None,
    max_residual_abs_s: float = 3_600.0,
) -> Tuple[List[TrainingRow], BuildReport]:
    """Route every trip at its pickup time and build its base features.

    A trip whose residual exceeds ``max_residual_abs_s`` is dropped, not clipped: a
    two-hour residual on a 12-minute baseline is a meter left running, and clipping it
    to an hour would still teach the model a 60-minute correction.
    """
    rows: List[TrainingRow] = []
    report = BuildReport()
    for trip in trips:
        report.trips_read += 1
        if trip.match_quality is MatchQuality.FAILED:
            report.skipped["match_failed"] += 1
            continue
        try:
            route = router.route(trip.origin, trip.destination, trip.pickup_at)
        except EtaForgeError:
            report.skipped["no_route"] += 1
            continue
        if not route.legs or route.duration_s <= 0:
            report.skipped["empty_route"] += 1
            continue
        if abs(trip.duration_s - route.duration_s) > max_residual_abs_s:
            report.skipped["residual_outlier"] += 1
            continue
        observation = weather.as_of(trip.origin, trip.pickup_at) if weather else None
        feature_row = builder.base_features(route, weather=observation)
        rows.append(
            TrainingRow(
                trip_id=trip.trip_id,
                departure_at=trip.pickup_at,
                feature_row=feature_row,
                baseline_s=route.duration_s,
                actual_s=trip.duration_s,
                distance_m=route.distance_m,
                merchant_id=trip.merchant_id,
            )
        )
        report.rows_built += 1
    return rows, report


WAREHOUSE_TRIPS_SQL = """
SELECT trip_id, pickup_at, dropoff_at, duration_s, distance_m,
       ST_Y(origin) AS origin_lat, ST_X(origin) AS origin_lon,
       ST_Y(destination) AS dest_lat, ST_X(destination) AS dest_lon,
       match_quality, merchant_id, vehicle_type
FROM trip
WHERE pickup_at >= %s AND pickup_at < %s
  AND match_quality <> 'failed'
  AND origin IS NOT NULL AND destination IS NOT NULL
ORDER BY pickup_at
LIMIT %s
"""


def warehouse_trips(
    runner: Any, *, since: datetime, until: datetime, limit: int = 2_000_000
) -> Iterator[TripRecord]:
    """Trips from the ``trip`` table, oldest first, as domain objects."""
    rows = runner.fetch_all(WAREHOUSE_TRIPS_SQL, (since, until, limit), label="dataset.trips")
    for row in rows:
        try:
            yield TripRecord(
                trip_id=str(row["trip_id"]),
                pickup_at=_parse_ts(row["pickup_at"]),
                dropoff_at=_parse_ts(row["dropoff_at"]),
                origin=Location(float(row["origin_lat"]), float(row["origin_lon"])),
                destination=Location(float(row["dest_lat"]), float(row["dest_lon"])),
                distance_m=float(row["distance_m"]),
                duration_s=float(row["duration_s"]),
                match_quality=MatchQuality(row["match_quality"]),
                merchant_id=row.get("merchant_id"),
                vehicle_type=str(row.get("vehicle_type") or "car"),
            )
        except (ValueError, KeyError, EtaForgeError):
            continue


def training_window(rows: Sequence[TrainingRow]) -> Dict[str, Optional[str]]:
    if not rows:
        return {"start": None, "end": None}
    times = [r.departure_at for r in rows]
    return {"start": min(times).isoformat(), "end": max(times).isoformat()}


__all__ = [
    "TrainingRow",
    "BuildReport",
    "build_rows",
    "warehouse_trips",
    "training_window",
    "WAREHOUSE_TRIPS_SQL",
]
