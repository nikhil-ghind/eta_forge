"""Domain types shared by the routing, training, evaluation and serving layers.

Everything here is a plain dataclass with validation in ``__post_init__`` and a
``to_dict`` / ``from_dict`` pair, so any object can cross a process boundary (JSON
request body, cached feature row, evaluation fixture) without a serialisation library.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

# --------------------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------------------


class EtaForgeError(Exception):
    """Base class for every error this package raises deliberately."""


class ValidationError(EtaForgeError, ValueError):
    """A domain object was constructed with values that cannot describe reality."""


class NoRouteFound(EtaForgeError):
    """The router could not connect origin to destination on the routable graph."""


class ModelNotFound(EtaForgeError):
    """A model version was requested that the registry does not hold."""


class FeatureSpecMismatch(EtaForgeError):
    """A model was trained against a different feature layout than the one serving."""


# --------------------------------------------------------------------------------------
# geography
# --------------------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class Location:
    """A WGS84 point. ``lat``/``lon`` in degrees."""

    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not isinstance(self.lat, (int, float)) or not isinstance(self.lon, (int, float)):
            raise ValidationError("lat and lon must be numbers")
        if math.isnan(self.lat) or math.isnan(self.lon):
            raise ValidationError("lat and lon must not be NaN")
        if not -90.0 <= self.lat <= 90.0:
            raise ValidationError(f"lat {self.lat} outside [-90, 90]")
        if not -180.0 <= self.lon <= 180.0:
            raise ValidationError(f"lon {self.lon} outside [-180, 180]")

    def to_dict(self) -> Dict[str, float]:
        return {"lat": self.lat, "lon": self.lon}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Location":
        try:
            return cls(lat=float(data["lat"]), lon=float(data["lon"]))
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValidationError(f"location missing key {exc}") from exc

    @classmethod
    def from_pair(cls, pair: Sequence[float]) -> "Location":
        if len(pair) != 2:
            raise ValidationError("location pair must have exactly two elements")
        return cls(lat=float(pair[0]), lon=float(pair[1]))


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned lat/lon box used to reject out-of-region coordinates."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def __post_init__(self) -> None:
        if self.min_lat >= self.max_lat:
            raise ValidationError("min_lat must be < max_lat")
        if self.min_lon >= self.max_lon:
            raise ValidationError("min_lon must be < max_lon")

    def contains(self, loc: Location) -> bool:
        return (
            self.min_lat <= loc.lat <= self.max_lat
            and self.min_lon <= loc.lon <= self.max_lon
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundingBox":
        return cls(
            min_lat=float(data["min_lat"]),
            min_lon=float(data["min_lon"]),
            max_lat=float(data["max_lat"]),
            max_lon=float(data["max_lon"]),
        )


# --------------------------------------------------------------------------------------
# road network
# --------------------------------------------------------------------------------------


class HighwayClass(str, Enum):
    """The OSM ``highway`` values kept in the routable graph, coarsest first.

    The ordering matters: class rank is used as a feature and as the fallback key for
    speed priors, so it must be stable.
    """

    MOTORWAY = "motorway"
    TRUNK = "trunk"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    RESIDENTIAL = "residential"
    SERVICE = "service"
    UNCLASSIFIED = "unclassified"

    @classmethod
    def coerce(cls, value: Optional[str]) -> "HighwayClass":
        """Map a raw OSM tag (including ``*_link`` variants) onto a kept class."""
        if not value:
            return cls.UNCLASSIFIED
        raw = str(value).strip().lower()
        if raw.endswith("_link"):
            raw = raw[: -len("_link")]
        aliases = {
            "living_street": "residential",
            "road": "unclassified",
            "motorway_junction": "motorway",
            "busway": "secondary",
        }
        raw = aliases.get(raw, raw)
        try:
            return cls(raw)
        except ValueError:
            return cls.UNCLASSIFIED

    @property
    def rank(self) -> int:
        return list(HighwayClass).index(self)


class TurnType(str, Enum):
    """Manoeuvre classification derived from the bearing delta between two legs."""

    STRAIGHT = "straight"
    SLIGHT_LEFT = "slight_left"
    SLIGHT_RIGHT = "slight_right"
    LEFT = "left"
    RIGHT = "right"
    SHARP_LEFT = "sharp_left"
    SHARP_RIGHT = "sharp_right"
    UTURN = "uturn"

    @property
    def is_left(self) -> bool:
        return self in (TurnType.SLIGHT_LEFT, TurnType.LEFT, TurnType.SHARP_LEFT)

    @property
    def is_right(self) -> bool:
        return self in (TurnType.SLIGHT_RIGHT, TurnType.RIGHT, TurnType.SHARP_RIGHT)

    @property
    def is_major(self) -> bool:
        """A manoeuvre that costs real time: a full turn, sharp turn or u-turn."""
        return self not in (
            TurnType.STRAIGHT,
            TurnType.SLIGHT_LEFT,
            TurnType.SLIGHT_RIGHT,
        )


@dataclass(frozen=True)
class RouteLeg:
    """One traversed road segment within a baseline route."""

    segment_id: int
    source_vertex: int
    target_vertex: int
    length_m: float
    highway_class: HighwayClass
    speed_kph: float
    duration_s: float
    bearing_in_deg: float
    bearing_out_deg: float
    has_signal_at_end: bool = False
    geometry: Sequence[Location] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.length_m < 0:
            raise ValidationError("length_m must be >= 0")
        if self.speed_kph <= 0:
            raise ValidationError("speed_kph must be > 0")
        if self.duration_s < 0:
            raise ValidationError("duration_s must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source_vertex": self.source_vertex,
            "target_vertex": self.target_vertex,
            "length_m": self.length_m,
            "highway_class": self.highway_class.value,
            "speed_kph": self.speed_kph,
            "duration_s": self.duration_s,
            "bearing_in_deg": self.bearing_in_deg,
            "bearing_out_deg": self.bearing_out_deg,
            "has_signal_at_end": self.has_signal_at_end,
            "geometry": [p.to_dict() for p in self.geometry],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouteLeg":
        return cls(
            segment_id=int(data["segment_id"]),
            source_vertex=int(data["source_vertex"]),
            target_vertex=int(data["target_vertex"]),
            length_m=float(data["length_m"]),
            highway_class=HighwayClass.coerce(data["highway_class"]),
            speed_kph=float(data["speed_kph"]),
            duration_s=float(data["duration_s"]),
            bearing_in_deg=float(data.get("bearing_in_deg", 0.0)),
            bearing_out_deg=float(data.get("bearing_out_deg", 0.0)),
            has_signal_at_end=bool(data.get("has_signal_at_end", False)),
            geometry=tuple(Location.from_dict(p) for p in data.get("geometry", ())),
        )


@dataclass(frozen=True)
class BaselineRoute:
    """The stage-one output: a physical path with a free-flow-ish duration."""

    legs: Sequence[RouteLeg]
    origin: Location
    destination: Location
    departure_at: datetime
    used_speed_profiles: bool = False
    profile_coverage: float = 0.0
    snapped_origin_distance_m: float = 0.0
    snapped_destination_distance_m: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.profile_coverage <= 1.0:
            raise ValidationError("profile_coverage must be in [0, 1]")
        if self.departure_at.tzinfo is None:
            raise ValidationError("departure_at must be timezone-aware")

    @property
    def distance_m(self) -> float:
        return sum(leg.length_m for leg in self.legs)

    @property
    def duration_s(self) -> float:
        return sum(leg.duration_s for leg in self.legs)

    @property
    def mean_speed_kph(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return (self.distance_m / self.duration_s) * 3.6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "legs": [leg.to_dict() for leg in self.legs],
            "origin": self.origin.to_dict(),
            "destination": self.destination.to_dict(),
            "departure_at": self.departure_at.isoformat(),
            "used_speed_profiles": self.used_speed_profiles,
            "profile_coverage": self.profile_coverage,
            "snapped_origin_distance_m": self.snapped_origin_distance_m,
            "snapped_destination_distance_m": self.snapped_destination_distance_m,
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
        }


# --------------------------------------------------------------------------------------
# trips and observations
# --------------------------------------------------------------------------------------


class MatchQuality(str, Enum):
    """How confidently a historical trip was tied to the road graph."""

    MATCHED = "matched"      # full GPS trace, HMM map-matched onto segments
    ZONE = "zone"            # only pickup/dropoff zone ids, centroid-to-centroid
    ENDPOINTS = "endpoints"  # coordinates for both ends, no intermediate trace
    FAILED = "failed"


@dataclass
class TripRecord:
    """One historical trip: the supervised example the residual model learns from."""

    trip_id: str
    pickup_at: datetime
    dropoff_at: datetime
    origin: Location
    destination: Location
    distance_m: float
    duration_s: float
    match_quality: MatchQuality = MatchQuality.ENDPOINTS
    merchant_id: Optional[str] = None
    vehicle_type: str = "car"
    gps_trace: Sequence["GpsPoint"] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.pickup_at.tzinfo is None or self.dropoff_at.tzinfo is None:
            raise ValidationError("trip timestamps must be timezone-aware")
        if self.dropoff_at < self.pickup_at:
            raise ValidationError("dropoff_at precedes pickup_at")
        if self.duration_s <= 0:
            raise ValidationError("duration_s must be > 0")
        if self.distance_m < 0:
            raise ValidationError("distance_m must be >= 0")

    @property
    def mean_speed_kph(self) -> float:
        return (self.distance_m / self.duration_s) * 3.6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "pickup_at": self.pickup_at.isoformat(),
            "dropoff_at": self.dropoff_at.isoformat(),
            "origin": self.origin.to_dict(),
            "destination": self.destination.to_dict(),
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "match_quality": self.match_quality.value,
            "merchant_id": self.merchant_id,
            "vehicle_type": self.vehicle_type,
            "gps_trace": [p.to_dict() for p in self.gps_trace],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TripRecord":
        return cls(
            trip_id=str(data["trip_id"]),
            pickup_at=_parse_ts(data["pickup_at"]),
            dropoff_at=_parse_ts(data["dropoff_at"]),
            origin=Location.from_dict(data["origin"]),
            destination=Location.from_dict(data["destination"]),
            distance_m=float(data["distance_m"]),
            duration_s=float(data["duration_s"]),
            match_quality=MatchQuality(data.get("match_quality", "endpoints")),
            merchant_id=data.get("merchant_id"),
            vehicle_type=data.get("vehicle_type", "car"),
            gps_trace=tuple(GpsPoint.from_dict(p) for p in data.get("gps_trace", ())),
        )


@dataclass(frozen=True)
class GpsPoint:
    """One sample of a vehicle trace, fed to the map matcher."""

    at: datetime
    location: Location
    speed_kph: Optional[float] = None
    accuracy_m: float = 15.0

    def __post_init__(self) -> None:
        if self.at.tzinfo is None:
            raise ValidationError("gps timestamp must be timezone-aware")
        if self.accuracy_m <= 0:
            raise ValidationError("accuracy_m must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "location": self.location.to_dict(),
            "speed_kph": self.speed_kph,
            "accuracy_m": self.accuracy_m,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GpsPoint":
        return cls(
            at=_parse_ts(data["at"]),
            location=Location.from_dict(data["location"]),
            speed_kph=(None if data.get("speed_kph") is None else float(data["speed_kph"])),
            accuracy_m=float(data.get("accuracy_m", 15.0)),
        )


@dataclass(frozen=True)
class SegmentObservation:
    """A traversal of one road segment attributed out of a map-matched trip.

    This is the atom the speed profiles are mined from.
    """

    segment_id: int
    entered_at: datetime
    traversal_s: float
    length_m: float
    highway_class: HighwayClass = HighwayClass.UNCLASSIFIED
    trip_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.traversal_s <= 0:
            raise ValidationError("traversal_s must be > 0")
        if self.length_m <= 0:
            raise ValidationError("length_m must be > 0")
        if self.entered_at.tzinfo is None:
            raise ValidationError("entered_at must be timezone-aware")

    @property
    def speed_kph(self) -> float:
        return (self.length_m / self.traversal_s) * 3.6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "entered_at": self.entered_at.isoformat(),
            "traversal_s": self.traversal_s,
            "length_m": self.length_m,
            "highway_class": self.highway_class.value,
            "trip_id": self.trip_id,
        }


@dataclass(frozen=True)
class PrepObservation:
    """One historical order's merchant preparation time.

    ``prep_is_upper_bound`` marks orders where the merchant never confirmed readiness
    and prep was inferred from courier pickup, which includes courier wait.
    """

    order_id: str
    merchant_id: str
    placed_at: datetime
    prep_s: float
    cuisine_category: str = "other"
    item_count: int = 1
    order_value_usd: Optional[float] = None
    open_orders_at_merchant: Optional[int] = None
    prep_is_upper_bound: bool = False

    def __post_init__(self) -> None:
        if self.placed_at.tzinfo is None:
            raise ValidationError("placed_at must be timezone-aware")
        if self.prep_s <= 0:
            raise ValidationError("prep_s must be > 0")
        if self.item_count < 0:
            raise ValidationError("item_count must be >= 0")


@dataclass(frozen=True)
class DwellObservation:
    """Handoff dwell at a destination: arrival at the kerb to order handed over."""

    order_id: str
    destination_cell: str
    arrived_at: datetime
    dwell_s: float

    def __post_init__(self) -> None:
        if self.arrived_at.tzinfo is None:
            raise ValidationError("arrived_at must be timezone-aware")
        if self.dwell_s < 0:
            raise ValidationError("dwell_s must be >= 0")


# --------------------------------------------------------------------------------------
# service request / response
# --------------------------------------------------------------------------------------


@dataclass
class WeatherObservation:
    """Hourly weather joined onto a request or a training row."""

    observed_at: datetime
    temperature_c: float
    precipitation_mm_h: float = 0.0
    wind_kph: float = 0.0
    condition: str = "clear"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_at": self.observed_at.isoformat(),
            "temperature_c": self.temperature_c,
            "precipitation_mm_h": self.precipitation_mm_h,
            "wind_kph": self.wind_kph,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WeatherObservation":
        return cls(
            observed_at=_parse_ts(data["observed_at"]),
            temperature_c=float(data["temperature_c"]),
            precipitation_mm_h=float(data.get("precipitation_mm_h", 0.0)),
            wind_kph=float(data.get("wind_kph", 0.0)),
            condition=str(data.get("condition", "clear")),
        )


@dataclass
class EtaRequest:
    """What a caller asks for.

    ``merchant_id`` switches on the delivery decomposition: without it the answer is
    ``travel`` only; with it the answer is ``prep + travel + handoff``.
    """

    origin: Location
    destination: Location
    departure_at: Optional[datetime] = None
    merchant_id: Optional[str] = None
    order_item_count: Optional[int] = None
    order_value_usd: Optional[float] = None
    open_orders_at_merchant: Optional[int] = None
    vehicle_type: str = "car"
    model_version: Optional[str] = None
    include_route: bool = False

    def __post_init__(self) -> None:
        if self.departure_at is None:
            self.departure_at = datetime.now(timezone.utc)
        elif self.departure_at.tzinfo is None:
            raise ValidationError("departure_at must be timezone-aware")
        if self.order_item_count is not None and self.order_item_count < 0:
            raise ValidationError("order_item_count must be >= 0")
        if self.order_value_usd is not None and self.order_value_usd < 0:
            raise ValidationError("order_value_usd must be >= 0")
        if self.open_orders_at_merchant is not None and self.open_orders_at_merchant < 0:
            raise ValidationError("open_orders_at_merchant must be >= 0")

    @property
    def is_delivery(self) -> bool:
        return self.merchant_id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin.to_dict(),
            "destination": self.destination.to_dict(),
            "departure_at": self.departure_at.isoformat() if self.departure_at else None,
            "merchant_id": self.merchant_id,
            "order_item_count": self.order_item_count,
            "order_value_usd": self.order_value_usd,
            "open_orders_at_merchant": self.open_orders_at_merchant,
            "vehicle_type": self.vehicle_type,
            "model_version": self.model_version,
            "include_route": self.include_route,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EtaRequest":
        departure = data.get("departure_at")
        return cls(
            origin=Location.from_dict(data["origin"]),
            destination=Location.from_dict(data["destination"]),
            departure_at=_parse_ts(departure) if departure else None,
            merchant_id=data.get("merchant_id"),
            order_item_count=data.get("order_item_count"),
            order_value_usd=data.get("order_value_usd"),
            open_orders_at_merchant=data.get("open_orders_at_merchant"),
            vehicle_type=data.get("vehicle_type", "car"),
            model_version=data.get("model_version"),
            include_route=bool(data.get("include_route", False)),
        )


@dataclass(frozen=True)
class QuantilePair:
    """A p50/p90 pair in seconds. p90 is clamped to never fall below p50."""

    p50: float
    p90: float

    def __post_init__(self) -> None:
        if self.p50 < 0:
            raise ValidationError("p50 must be >= 0")
        if self.p90 < self.p50:
            # Independently fitted quantile models can cross; the domain object refuses
            # to represent a crossed pair rather than leaking it into a response.
            object.__setattr__(self, "p90", self.p50)

    @property
    def spread(self) -> float:
        return self.p90 - self.p50

    def scaled(self, factor: float) -> "QuantilePair":
        return QuantilePair(self.p50 * factor, self.p90 * factor)

    def to_dict(self) -> Dict[str, float]:
        return {"p50": self.p50, "p90": self.p90}

    @classmethod
    def zero(cls) -> "QuantilePair":
        return cls(0.0, 0.0)


@dataclass
class EtaBreakdown:
    """The decomposition, kept in the response so an ETA can be argued with."""

    prep: QuantilePair
    travel: QuantilePair
    handoff: QuantilePair
    baseline_travel_s: float
    residual_correction_p50_s: float
    residual_correction_p90_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prep": self.prep.to_dict(),
            "travel": self.travel.to_dict(),
            "handoff": self.handoff.to_dict(),
            "baseline_travel_s": self.baseline_travel_s,
            "residual_correction_p50_s": self.residual_correction_p50_s,
            "residual_correction_p90_s": self.residual_correction_p90_s,
        }


@dataclass
class EtaResponse:
    """What the service answers with."""

    eta: QuantilePair
    breakdown: EtaBreakdown
    distance_m: float
    model_versions: Mapping[str, str]
    degraded: bool = False
    degraded_reason: Optional[str] = None
    profile_coverage: float = 0.0
    route: Optional[BaselineRoute] = None
    computed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "eta_seconds": self.eta.to_dict(),
            "breakdown": self.breakdown.to_dict(),
            "distance_m": self.distance_m,
            "model_versions": dict(self.model_versions),
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "profile_coverage": self.profile_coverage,
            "computed_at": (self.computed_at or datetime.now(timezone.utc)).isoformat(),
        }
        if self.route is not None:
            payload["route"] = self.route.to_dict()
        return payload


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _parse_ts(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp, insisting on timezone awareness.

    A naive timestamp is the single most common way a time-bucketed model silently
    trains on the wrong hour, so it is an error rather than an assumed UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValidationError("timestamp must be timezone-aware")
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(f"unparseable timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"timestamp {value!r} has no timezone")
    return parsed


__all__ = [
    "EtaForgeError",
    "ValidationError",
    "NoRouteFound",
    "ModelNotFound",
    "FeatureSpecMismatch",
    "EARTH_RADIUS_M",
    "Location",
    "BoundingBox",
    "HighwayClass",
    "TurnType",
    "RouteLeg",
    "BaselineRoute",
    "MatchQuality",
    "TripRecord",
    "GpsPoint",
    "SegmentObservation",
    "PrepObservation",
    "DwellObservation",
    "WeatherObservation",
    "EtaRequest",
    "QuantilePair",
    "EtaBreakdown",
    "EtaResponse",
]
