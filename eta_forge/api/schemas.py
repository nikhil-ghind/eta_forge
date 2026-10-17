"""Request parsing and response shaping for the HTTP API, in pure Python.

The API validates bodies here rather than with Pydantic models. That keeps the whole
contract — which keys are accepted, what each must look like, what an error body says
— importable and unit-testable without FastAPI or Pydantic installed, and it means
every validation failure, wherever it is caught, produces the same error shape:

    {"error": "validation_error", "detail": [{"loc": ["origin", "lat"], "msg": "..."}]}

Unknown keys are rejected, not ignored. A client sending ``departure_time`` instead of
``departure_at`` would otherwise silently get an ETA for *now*.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..types import (
    EtaForgeError,
    EtaRequest,
    EtaResponse,
    FeatureSpecMismatch,
    Location,
    ModelNotFound,
    NoRouteFound,
    ValidationError,
    _parse_ts,
)

RIDE_KEYS = frozenset({
    "origin", "destination", "departure_at", "vehicle_type", "model_version", "include_route",
})
DELIVERY_KEYS = RIDE_KEYS | frozenset({
    "merchant_id", "order_item_count", "order_value_usd", "open_orders_at_merchant",
})
VEHICLE_TYPES = frozenset({"car", "taxi", "fhv", "bike", "scooter"})


class SchemaError(ValidationError):
    """A request body or query string failed validation; carries every problem found."""

    def __init__(self, errors: List[Dict[str, Any]]):
        self.errors = errors
        super().__init__("; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors))


class _Collector:
    def __init__(self) -> None:
        self.errors: List[Dict[str, Any]] = []

    def add(self, loc: Sequence[Any], msg: str) -> None:
        self.errors.append({"loc": list(loc), "msg": msg})

    def raise_if_any(self) -> None:
        if self.errors:
            raise SchemaError(self.errors)


def _location(value: Any, field: str, errors: _Collector) -> Optional[Location]:
    """``{"lat": .., "lon": ..}`` or ``[lat, lon]``."""
    if value is None:
        errors.add([field], "field required")
        return None
    try:
        if isinstance(value, Mapping):
            extra = set(value) - {"lat", "lon"}
            if extra:
                errors.add([field], f"unknown keys {sorted(extra)}")
                return None
            if "lat" not in value or "lon" not in value:
                errors.add([field], "lat and lon are required")
                return None
            lat, lon = value["lat"], value["lon"]
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            lat, lon = value
        else:
            errors.add([field], "must be {lat, lon} or [lat, lon]")
            return None
        if isinstance(lat, bool) or isinstance(lon, bool):
            raise ValidationError("lat and lon must be numbers")
        return Location(float(lat), float(lon))
    except (TypeError, ValueError) as exc:
        errors.add([field], str(exc) or "invalid coordinate")
        return None


def _optional_int(payload: Mapping[str, Any], key: str, errors: _Collector, *, minimum: int = 0) -> Optional[int]:
    if payload.get(key) is None:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        errors.add([key], "must be an integer")
        return None
    if value < minimum:
        errors.add([key], f"must be >= {minimum}")
        return None
    return int(value)


def _optional_float(payload: Mapping[str, Any], key: str, errors: _Collector) -> Optional[float]:
    if payload.get(key) is None:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.add([key], "must be a number")
        return None
    if value < 0:
        errors.add([key], "must be >= 0")
        return None
    return float(value)


def _timestamp(value: Any, field: str, errors: _Collector) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return _parse_ts(value)
    except ValidationError as exc:
        errors.add([field], str(exc))
        return None


def parse_eta_request(payload: Any, *, delivery: bool) -> EtaRequest:
    """Validate a ``POST /v1/eta`` or ``/v1/eta/delivery`` body into an :class:`EtaRequest`."""
    errors = _Collector()
    if not isinstance(payload, Mapping):
        errors.add([], "request body must be a JSON object")
        errors.raise_if_any()
    allowed = DELIVERY_KEYS if delivery else RIDE_KEYS
    for key in sorted(set(payload) - allowed):
        hint = " (delivery fields go to /v1/eta/delivery)" if key in DELIVERY_KEYS else ""
        errors.add([key], f"unknown field{hint}")

    origin = _location(payload.get("origin"), "origin", errors)
    destination = _location(payload.get("destination"), "destination", errors)
    departure = _timestamp(payload.get("departure_at"), "departure_at", errors)

    vehicle = payload.get("vehicle_type", "car")
    if not isinstance(vehicle, str) or vehicle not in VEHICLE_TYPES:
        errors.add(["vehicle_type"], f"must be one of {sorted(VEHICLE_TYPES)}")
    model_version = payload.get("model_version")
    if model_version is not None and (not isinstance(model_version, str) or not model_version):
        errors.add(["model_version"], "must be a non-empty string")
    include_route = payload.get("include_route", False)
    if not isinstance(include_route, bool):
        errors.add(["include_route"], "must be a boolean")

    merchant_id = items = value = open_orders = None
    if delivery:
        merchant_id = payload.get("merchant_id")
        if not isinstance(merchant_id, str) or not merchant_id:
            errors.add(["merchant_id"], "field required for a delivery ETA")
        items = _optional_int(payload, "order_item_count", errors)
        value = _optional_float(payload, "order_value_usd", errors)
        open_orders = _optional_int(payload, "open_orders_at_merchant", errors)

    errors.raise_if_any()
    assert origin is not None and destination is not None
    try:
        return EtaRequest(
            origin=origin,
            destination=destination,
            departure_at=departure,
            merchant_id=merchant_id,
            order_item_count=items,
            order_value_usd=value,
            open_orders_at_merchant=open_orders,
            vehicle_type=vehicle,
            model_version=model_version,
            include_route=include_route,
        )
    except ValidationError as exc:
        raise SchemaError([{"loc": [], "msg": str(exc)}]) from exc


def parse_point(text: Optional[str], field: str) -> Location:
    """``"40.75,-73.98"`` from a query string."""
    errors = _Collector()
    if not text:
        errors.add([field], "field required, as lat,lon")
        errors.raise_if_any()
    parts = str(text).split(",")
    if len(parts) != 2:
        errors.add([field], "must be lat,lon")
        errors.raise_if_any()
    loc = _location([p.strip() for p in parts], field, errors)
    errors.raise_if_any()
    assert loc is not None
    return loc


def parse_route_query(params: Mapping[str, Any]) -> Tuple[Location, Location, Optional[datetime]]:
    origin = parse_point(params.get("origin"), "origin")
    destination = parse_point(params.get("destination"), "destination")
    errors = _Collector()
    departure = _timestamp(params.get("departure_at"), "departure_at", errors)
    errors.raise_if_any()
    return origin, destination, departure


# --------------------------------------------------------------------------------------
# responses
# --------------------------------------------------------------------------------------


def eta_response_body(response: EtaResponse) -> Dict[str, Any]:
    body = response.to_dict()
    body["eta_minutes"] = {
        "p50": round(response.eta.p50 / 60.0, 1),
        "p90": round(response.eta.p90 / 60.0, 1),
    }
    return body


ERROR_STATUS = (
    (SchemaError, 422, "validation_error"),
    (FeatureSpecMismatch, 409, "feature_spec_mismatch"),
    (ModelNotFound, 404, "model_not_found"),
    (NoRouteFound, 404, "no_route"),
    (ValidationError, 422, "validation_error"),
    (EtaForgeError, 500, "eta_forge_error"),
)


def error_body(exc: BaseException) -> Tuple[int, Dict[str, Any]]:
    """``(status, body)`` for any exception, most specific type first."""
    for kind, status, code in ERROR_STATUS:
        if isinstance(exc, kind):
            detail = exc.errors if isinstance(exc, SchemaError) else [{"loc": [], "msg": str(exc)}]
            return status, {"error": code, "detail": detail}
    return 500, {"error": "internal_error", "detail": [{"loc": [], "msg": "internal error"}]}


__all__ = [
    "SchemaError",
    "parse_eta_request",
    "parse_route_query",
    "parse_point",
    "eta_response_body",
    "error_body",
    "RIDE_KEYS",
    "DELIVERY_KEYS",
]
