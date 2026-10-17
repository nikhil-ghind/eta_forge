"""Calendar features: cyclical time, weekday structure, rush hour, holidays.

Hour of day is encoded as ``(sin, cos)`` rather than as an integer. As an integer, 23
and 0 are maximally far apart, and a tree has to spend two splits to learn that late
night is one regime; on the circle they are neighbours.

Holidays use the ``holidays`` package when it is installed and an exact built-in
computation of the US federal calendar (with the Saturday -> Friday / Sunday -> Monday
observance rule) when it is not, so a training run and a serving process on a slim
image never disagree about whether a date was a holiday.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Sequence

from ..types import ValidationError


def localize(when: datetime, tz: Optional[str]) -> datetime:
    if when.tzinfo is None:
        raise ValidationError("calendar features need a timezone-aware timestamp")
    if not tz:
        return when
    try:
        from zoneinfo import ZoneInfo

        return when.astimezone(ZoneInfo(tz))
    except Exception:  # pragma: no cover - missing tzdata on a slim image
        return when


def cyclical(value: float, period: float) -> tuple:
    angle = 2.0 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


# --------------------------------------------------------------------------------------
# holidays
# --------------------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (0=Mon) of a month; ``n=-1`` for the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def us_federal_holidays(year: int) -> FrozenSet[date]:
    """US federal holidays for a year, as observed."""
    fixed = [
        date(year, 1, 1),     # New Year's Day
        date(year, 7, 4),     # Independence Day
        date(year, 11, 11),   # Veterans Day
        date(year, 12, 25),   # Christmas Day
    ]
    if year >= 2021:
        fixed.append(date(year, 6, 19))  # Juneteenth
    floating = [
        _nth_weekday(year, 1, 0, 3),     # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),     # Washington's Birthday
        _nth_weekday(year, 5, 0, -1),    # Memorial Day
        _nth_weekday(year, 9, 0, 1),     # Labor Day
        _nth_weekday(year, 10, 0, 2),    # Columbus Day
        _nth_weekday(year, 11, 3, 4),    # Thanksgiving
    ]
    return frozenset([_observed(d) for d in fixed] + floating)


class HolidayCalendar:
    """``is_holiday(date)`` backed by the ``holidays`` package or the built-in table."""

    def __init__(self, country: str = "US", *, use_package: bool = True):
        self.country = country
        self._package = None
        if use_package:
            try:
                import holidays  # lazily: optional dependency

                self._package = holidays.country_holidays(country)
            except Exception:
                self._package = None

    @property
    def source(self) -> str:
        return "holidays" if self._package is not None else "builtin"

    def is_holiday(self, day: date) -> bool:
        if self._package is not None:
            return day in self._package
        if self.country.upper() != "US":
            return False
        # New Year's Day on a Saturday is observed on 31 December of the year before,
        # so the following year's table has to be consulted too.
        return day in us_federal_holidays(day.year) or day in us_federal_holidays(day.year + 1)


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------

CALENDAR_FEATURES = (
    "hour_sin",
    "hour_cos",
    "dow",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_rush_hour",
    "is_holiday",
    "local_hour",
)


def in_rush_hour(local: datetime, windows: Sequence[Sequence[int]]) -> bool:
    if local.weekday() >= 5:
        return False
    hour = local.hour + local.minute / 60.0
    return any(start <= hour < end for start, end in windows)


def calendar_features(
    when: datetime,
    *,
    tz: Optional[str] = "America/New_York",
    rush_hour_windows: Sequence[Sequence[int]] = ((7, 10), (16, 19)),
    holidays: Optional[HolidayCalendar] = None,
) -> Dict[str, float]:
    """Calendar features for a departure time, in the city's local time."""
    local = localize(when, tz)
    fractional_hour = local.hour + local.minute / 60.0 + local.second / 3600.0
    hour_sin, hour_cos = cyclical(fractional_hour, 24.0)
    dow_sin, dow_cos = cyclical(local.weekday(), 7.0)
    holiday = bool(holidays.is_holiday(local.date())) if holidays is not None else False
    return {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow": float(local.weekday()),
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": 1.0 if local.weekday() >= 5 else 0.0,
        # A holiday weekday runs on a weekend traffic pattern; no rush hour.
        "is_rush_hour": 1.0 if (in_rush_hour(local, rush_hour_windows) and not holiday) else 0.0,
        "is_holiday": 1.0 if holiday else 0.0,
        "local_hour": float(local.hour),
    }


__all__ = [
    "HolidayCalendar",
    "calendar_features",
    "cyclical",
    "in_rush_hour",
    "localize",
    "us_federal_holidays",
    "CALENDAR_FEATURES",
]
