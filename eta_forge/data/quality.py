"""Trip quality filtering, with a tally of everything thrown away.

The TLC corpus is large and dirty in well-known ways: dropoffs before pickups, meter
left running overnight, zero-mile trips that took forty minutes, coordinates of
``(0, 0)`` in the Gulf of Guinea, and a long tail of 300 km/h "trips" from a GPS that
lost its fix. Every one of those, left in, becomes a residual target the model then
tries to learn.

The filter is deliberately loud about what it drops. A loader that silently discards
18% of a month is indistinguishable from one that loaded a different month, and the
only defence is a per-reason count that ends up in the ingest ledger and the model
manifest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional

from ..config import DataConfig
from ..graph.geometry import haversine_m
from ..types import BoundingBox, Location

# Reason codes, stable because they are stored in ingest_batch.rejections.
MISSING_TIMESTAMP = "missing_timestamp"
NON_POSITIVE_DURATION = "non_positive_duration"
DURATION_TOO_SHORT = "duration_too_short"
DURATION_TOO_LONG = "duration_too_long"
ZERO_DISTANCE = "zero_distance"
DISTANCE_TOO_SHORT = "distance_too_short"
MISSING_LOCATION = "missing_location"
UNKNOWN_ZONE = "unknown_zone"
OUT_OF_BBOX = "out_of_bbox"
IMPLAUSIBLE_SPEED = "implausible_speed"
DISTANCE_BELOW_CROW = "distance_below_crow_line"
MALFORMED = "malformed"

ALL_REASONS = (
    MISSING_TIMESTAMP,
    NON_POSITIVE_DURATION,
    DURATION_TOO_SHORT,
    DURATION_TOO_LONG,
    ZERO_DISTANCE,
    DISTANCE_TOO_SHORT,
    MISSING_LOCATION,
    UNKNOWN_ZONE,
    OUT_OF_BBOX,
    IMPLAUSIBLE_SPEED,
    DISTANCE_BELOW_CROW,
    MALFORMED,
)


@dataclass
class RejectionTally:
    """Rows read, rows kept, and a count per rejection reason."""

    read: int = 0
    kept: int = 0
    rejected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def keep(self) -> None:
        self.read += 1
        self.kept += 1

    def reject(self, reason: str) -> None:
        self.read += 1
        self.rejected[reason] += 1

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

    @property
    def kept_fraction(self) -> float:
        return self.kept / self.read if self.read else 0.0

    def merge(self, other: "RejectionTally") -> "RejectionTally":
        self.read += other.read
        self.kept += other.kept
        for reason, count in other.rejected.items():
            self.rejected[reason] += count
        return self

    def as_dict(self) -> Dict[str, object]:
        return {
            "read": self.read,
            "kept": self.kept,
            "kept_fraction": round(self.kept_fraction, 4),
            "rejected": dict(sorted(self.rejected.items())),
        }

    def summary_lines(self) -> List[str]:
        lines = [
            f"read {self.read:,}  kept {self.kept:,} ({100.0 * self.kept_fraction:.1f}%)"
        ]
        for reason, count in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            share = 100.0 * count / self.read if self.read else 0.0
            lines.append(f"  {reason:<26}{count:>12,}  {share:5.2f}%")
        return lines


@dataclass(frozen=True)
class QualityRules:
    """Thresholds for a plausible trip. Defaults mirror ``DataConfig``."""

    min_duration_s: float = 60.0
    max_duration_s: float = 14_400.0
    min_distance_m: float = 100.0
    max_speed_kph: float = 130.0
    bbox: Optional[BoundingBox] = None
    # A metered distance shorter than the straight line between the endpoints is
    # physically impossible; some slack absorbs centroid placement for zone trips.
    crow_slack: float = 0.85

    @classmethod
    def from_config(cls, cfg: DataConfig) -> "QualityRules":
        return cls(
            min_duration_s=cfg.min_trip_duration_s,
            max_duration_s=cfg.max_trip_duration_s,
            min_distance_m=cfg.min_trip_distance_m,
            max_speed_kph=cfg.max_trip_speed_kph,
            bbox=cfg.bbox(),
        )


def check_trip(
    rules: QualityRules,
    *,
    pickup_at: Optional[datetime],
    dropoff_at: Optional[datetime],
    distance_m: Optional[float],
    origin: Optional[Location],
    destination: Optional[Location],
    check_crow_line: bool = True,
) -> Optional[str]:
    """Return the first rejection reason for a trip, or ``None`` if it is plausible.

    Checks run cheapest and most common first; only the first failing reason is
    counted, so the tally is a partition of the rejected rows rather than an
    overlapping set of flags that sums to more than was read.

    ``distance_m=None`` means "not reported" (the FHV schemas carry no distance) and
    skips the distance checks rather than failing them.
    """
    if pickup_at is None or dropoff_at is None:
        return MISSING_TIMESTAMP
    duration_s = (dropoff_at - pickup_at).total_seconds()
    if duration_s <= 0:
        return NON_POSITIVE_DURATION
    if duration_s < rules.min_duration_s:
        return DURATION_TOO_SHORT
    if duration_s > rules.max_duration_s:
        return DURATION_TOO_LONG

    if distance_m is not None:
        if distance_m <= 0:
            return ZERO_DISTANCE
        if distance_m < rules.min_distance_m:
            return DISTANCE_TOO_SHORT

    if origin is None or destination is None:
        return MISSING_LOCATION
    if rules.bbox is not None and not (
        rules.bbox.contains(origin) and rules.bbox.contains(destination)
    ):
        return OUT_OF_BBOX

    if distance_m is not None:
        speed_kph = (distance_m / duration_s) * 3.6
        if speed_kph > rules.max_speed_kph:
            return IMPLAUSIBLE_SPEED
        if check_crow_line:
            crow = haversine_m(origin, destination)
            if crow > 500.0 and distance_m < crow * rules.crow_slack:
                return DISTANCE_BELOW_CROW
    return None


def is_null_island(lat: Optional[float], lon: Optional[float]) -> bool:
    """Pre-2016 TLC files encode a missing coordinate as 0.0, not as null."""
    if lat is None or lon is None:
        return True
    return abs(lat) < 1e-6 and abs(lon) < 1e-6


def rejection_rates(tally: RejectionTally) -> Mapping[str, float]:
    """Reason -> fraction of rows read, for the ingest report."""
    if not tally.read:
        return {}
    return {reason: count / tally.read for reason, count in tally.rejected.items()}


__all__ = [
    "RejectionTally",
    "QualityRules",
    "check_trip",
    "is_null_island",
    "rejection_rates",
    "ALL_REASONS",
    "MISSING_TIMESTAMP",
    "NON_POSITIVE_DURATION",
    "DURATION_TOO_SHORT",
    "DURATION_TOO_LONG",
    "ZERO_DISTANCE",
    "DISTANCE_TOO_SHORT",
    "MISSING_LOCATION",
    "UNKNOWN_ZONE",
    "OUT_OF_BBOX",
    "IMPLAUSIBLE_SPEED",
    "DISTANCE_BELOW_CROW",
    "MALFORMED",
]
