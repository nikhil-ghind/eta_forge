"""Weather join: nearest station, as-of in time, bounded staleness.

Weather is the textbook optional feature: genuinely predictive (heavy rain adds
double-digit percentages to urban travel time), but the feed is the first thing to go
missing in production. So the join is built to fail *gracefully*, and to fail the same
way in training and in serving:

* the observation used is the latest one **at or before** the departure time — never
  after, which would leak the future into training rows;
* an observation older than ``max_staleness`` is treated as missing, not as current;
* missing weather becomes NaN numerics plus ``weather_missing = 1``, a state the model
  sees in training (the feed had gaps then too) and therefore knows how to handle.
"""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..graph.geometry import haversine_m
from ..types import Location, WeatherObservation, _parse_ts

CONDITION_CODES: Dict[str, int] = {
    "clear": 0,
    "cloudy": 1,
    "fog": 2,
    "rain": 3,
    "heavy_rain": 4,
    "snow": 5,
    "storm": 6,
}

WEATHER_FEATURES = (
    "temperature_c",
    "precipitation_mm_h",
    "wind_kph",
    "weather_condition",
    "weather_missing",
)

NAN = float("nan")


def condition_code(condition: Optional[str]) -> int:
    if condition is None:
        return -1
    return CONDITION_CODES.get(str(condition).strip().lower(), CONDITION_CODES["cloudy"])


def weather_features(observation: Optional[WeatherObservation]) -> Dict[str, float]:
    """The weather slice of the feature vector, NaN-and-flag when absent."""
    if observation is None:
        return {
            "temperature_c": NAN,
            "precipitation_mm_h": NAN,
            "wind_kph": NAN,
            # LightGBM treats a negative categorical value as missing.
            "weather_condition": -1.0,
            "weather_missing": 1.0,
        }
    return {
        "temperature_c": float(observation.temperature_c),
        "precipitation_mm_h": float(observation.precipitation_mm_h),
        "wind_kph": float(observation.wind_kph),
        "weather_condition": float(condition_code(observation.condition)),
        "weather_missing": 0.0,
    }


@dataclass(frozen=True)
class WeatherStation:
    station_id: str
    location: Location


@dataclass
class _Series:
    times: List[datetime] = field(default_factory=list)
    observations: List[WeatherObservation] = field(default_factory=list)


class WeatherTable:
    """Hourly observations by station, with nearest-station as-of lookup."""

    def __init__(self, *, max_staleness: timedelta = timedelta(minutes=90),
                 max_station_distance_m: float = 40_000.0):
        self._stations: Dict[str, WeatherStation] = {}
        self._series: Dict[str, _Series] = {}
        self._sorted = True
        self.max_staleness = max_staleness
        self.max_station_distance_m = max_station_distance_m

    def __len__(self) -> int:
        return sum(len(s.observations) for s in self._series.values())

    @property
    def stations(self) -> List[WeatherStation]:
        return list(self._stations.values())

    def add_station(self, station: WeatherStation) -> None:
        self._stations[station.station_id] = station
        self._series.setdefault(station.station_id, _Series())

    def add(self, station_id: str, observation: WeatherObservation) -> None:
        if station_id not in self._stations:
            raise KeyError(f"unknown weather station {station_id!r}")
        series = self._series[station_id]
        if series.times and observation.observed_at < series.times[-1]:
            self._sorted = False
        series.times.append(observation.observed_at)
        series.observations.append(observation)

    def _ensure_sorted(self) -> None:
        if self._sorted:
            return
        for series in self._series.values():
            pairs = sorted(zip(series.times, series.observations), key=lambda p: p[0])
            series.times = [t for t, _ in pairs]
            series.observations = [o for _, o in pairs]
        self._sorted = True

    def nearest_station(self, location: Location) -> Optional[WeatherStation]:
        best: Optional[Tuple[float, WeatherStation]] = None
        for station in self._stations.values():
            d = haversine_m(location, station.location)
            if best is None or d < best[0]:
                best = (d, station)
        if best is None or best[0] > self.max_station_distance_m:
            return None
        return best[1]

    def as_of(self, location: Location, when: datetime) -> Optional[WeatherObservation]:
        """Latest observation at or before ``when`` from the nearest station."""
        station = self.nearest_station(location)
        if station is None:
            return None
        self._ensure_sorted()
        series = self._series[station.station_id]
        index = bisect.bisect_right(series.times, when) - 1
        if index < 0:
            return None
        observation = series.observations[index]
        if when - observation.observed_at > self.max_staleness:
            return None
        return observation

    def features(self, location: Location, when: datetime) -> Dict[str, float]:
        return weather_features(self.as_of(location, when))

    # -- loading -----------------------------------------------------------------

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> "WeatherTable":
        """Rows with station_id, lat, lon, observed_at, temperature_c, ... columns."""
        table = cls(**kwargs)
        for row in rows:
            station_id = str(row["station_id"])
            if station_id not in table._stations:
                table.add_station(
                    WeatherStation(station_id, Location(float(row["lat"]), float(row["lon"])))
                )
            table.add(
                station_id,
                WeatherObservation(
                    observed_at=_parse_ts(row["observed_at"]),
                    temperature_c=_num(row.get("temperature_c"), 15.0),
                    precipitation_mm_h=_num(row.get("precipitation_mm_h"), 0.0),
                    wind_kph=_num(row.get("wind_kph"), 0.0),
                    condition=str(row.get("condition") or "clear"),
                ),
            )
        return table

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "WeatherTable":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return cls.from_rows(csv.DictReader(handle), **kwargs)

    @classmethod
    def from_runner(cls, runner: Any, since: datetime, until: datetime, **kwargs: Any) -> "WeatherTable":
        rows = runner.fetch_all(WEATHER_WINDOW_SQL, (since, until), label="weather.window")
        return cls.from_rows(rows, **kwargs)


def _num(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


WEATHER_WINDOW_SQL = """
SELECT s.station_id, ST_Y(s.location) AS lat, ST_X(s.location) AS lon,
       o.observed_at, o.temperature_c, o.precipitation_mm_h, o.wind_kph, o.condition
FROM weather_observation o
JOIN weather_station s USING (station_id)
WHERE o.observed_at >= %s - interval '3 hours' AND o.observed_at < %s
ORDER BY s.station_id, o.observed_at
"""


__all__ = [
    "WeatherTable",
    "WeatherStation",
    "weather_features",
    "condition_code",
    "CONDITION_CODES",
    "WEATHER_FEATURES",
    "WEATHER_WINDOW_SQL",
]
