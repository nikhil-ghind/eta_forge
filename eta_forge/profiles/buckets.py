"""Time bucketing for speed profiles.

One rule, applied everywhere: a bucket is defined in the *city's* local time, not UTC.
Rush hour is a human phenomenon anchored to local clocks, and mining it in UTC smears
08:00 across two buckets for half the year as daylight saving moves.

The day-of-week convention is Python's (``0 = Monday``), matched by
``sql/006_routing_functions.sql``'s ``eta_time_bucket``. PostgreSQL's ``EXTRACT(DOW)``
is 0 = Sunday, and mixing the two shifts every profile by one day — a bug that looks
exactly like "the model learned that Sundays are busy".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional, Tuple

from ..config import ProfilesConfig
from ..types import ValidationError

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class BucketSpec:
    """How the profile grid is cut up."""

    tod_buckets: int = 24
    dow_buckets: int = 7
    timezone: Optional[str] = "America/New_York"

    def __post_init__(self) -> None:
        if self.tod_buckets < 1 or 24 % self.tod_buckets != 0:
            raise ValidationError("tod_buckets must be a positive divisor of 24")
        if self.dow_buckets not in (1, 7):
            raise ValidationError("dow_buckets must be 1 or 7")

    @property
    def hours_per_bucket(self) -> int:
        return 24 // self.tod_buckets

    @property
    def cell_count(self) -> int:
        return self.tod_buckets * self.dow_buckets

    @classmethod
    def from_config(cls, cfg: ProfilesConfig) -> "BucketSpec":
        return cls(
            tod_buckets=cfg.tod_buckets,
            dow_buckets=cfg.dow_buckets,
            timezone=cfg.timezone,
        )


def _localize(when: datetime, tz: Optional[str]) -> datetime:
    if when.tzinfo is None:
        raise ValidationError("bucketing needs a timezone-aware timestamp")
    if not tz:
        return when
    try:
        from zoneinfo import ZoneInfo

        return when.astimezone(ZoneInfo(tz))
    except Exception:  # pragma: no cover - missing tzdata on a slim image
        return when


def bucket_of(when: datetime, spec: BucketSpec) -> Tuple[int, int]:
    """``(dow_bucket, tod_bucket)`` for a timestamp."""
    local = _localize(when, spec.timezone)
    tod = (local.hour // spec.hours_per_bucket) % spec.tod_buckets
    dow = local.weekday() if spec.dow_buckets == 7 else 0
    return dow, tod


def bucket_label(dow: int, tod: int, spec: BucketSpec) -> str:
    """Human-readable cell name, e.g. ``Thu 18:00-19:00``. Used in reports."""
    start = tod * spec.hours_per_bucket
    end = start + spec.hours_per_bucket
    window = f"{start:02d}:00-{end:02d}:00"
    if spec.dow_buckets == 1:
        return f"all {window}"
    return f"{DAY_NAMES[dow % 7]} {window}"


def iter_cells(spec: BucketSpec) -> Iterator[Tuple[int, int]]:
    """Every ``(dow, tod)`` cell in the grid, in a stable order."""
    for dow in range(spec.dow_buckets):
        for tod in range(spec.tod_buckets):
            yield dow, tod


def neighbouring_cells(
    dow: int, tod: int, spec: BucketSpec, *, radius: int = 1
) -> Iterator[Tuple[int, int]]:
    """Cells adjacent in time-of-day, wrapping across midnight.

    Used when a cell is too sparse to trust on its own: 03:00 on a Tuesday borrows
    from 02:00 and 04:00 on a Tuesday before it borrows from the class prior, because
    adjacent hours are far more alike than a class average is.

    The wrap matters. Without it, the 00:00 and 23:00 cells — the two quietest, most
    similar hours of the night — are the only two that never help each other.
    """
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        yield dow, (tod + offset) % spec.tod_buckets


def is_weekend(dow: int, spec: BucketSpec) -> bool:
    return spec.dow_buckets == 7 and dow >= 5


__all__ = [
    "BucketSpec",
    "bucket_of",
    "bucket_label",
    "iter_cells",
    "neighbouring_cells",
    "is_weekend",
    "DAY_NAMES",
]
