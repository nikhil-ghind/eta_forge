"""Layered configuration: YAML file < environment < CLI overrides.

The three layers exist because the same code runs in three places with three different
sources of truth: a checked-in YAML for defaults, ``ETA_FORGE_*`` environment variables
for a container, and ``--set section.key=value`` for a one-off experiment.

Every section is a dataclass with typed fields, so a typo in a key is an error at load
time rather than a silently-ignored setting that makes an offline experiment
irreproducible.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .types import BoundingBox, ValidationError

ENV_PREFIX = "ETA_FORGE_"


class ConfigError(ValidationError):
    """A configuration value is missing, unknown, or of the wrong type."""


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "eta_forge"
    user: str = "eta"
    password: str = ""
    sslmode: str = "prefer"
    min_pool_size: int = 1
    max_pool_size: int = 10
    connect_timeout_s: float = 5.0
    statement_timeout_ms: int = 10_000
    application_name: str = "eta_forge"

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"database.port {self.port} outside 1..65535")
        if self.min_pool_size < 0:
            raise ConfigError("database.min_pool_size must be >= 0")
        if self.max_pool_size < max(1, self.min_pool_size):
            raise ConfigError("database.max_pool_size must be >= min_pool_size and >= 1")
        if self.statement_timeout_ms <= 0:
            raise ConfigError("database.statement_timeout_ms must be > 0")


@dataclass
class GraphConfig:
    """Routable-graph and routing-query behaviour."""

    edge_table: str = "road_segment"
    vertex_table: str = "road_segment_vertices_pgr"
    attr_table: str = "segment_attr"
    srid: int = 4326
    snap_radius_m: float = 300.0
    snap_candidates: int = 10
    max_route_distance_m: float = 120_000.0
    # Fallback speeds by highway class, km/h, used when no mined profile covers a cell.
    default_speed_kph: Dict[str, float] = field(
        default_factory=lambda: {
            "motorway": 90.0,
            "trunk": 70.0,
            "primary": 50.0,
            "secondary": 45.0,
            "tertiary": 40.0,
            "residential": 30.0,
            "service": 20.0,
            "unclassified": 30.0,
        }
    )
    # Seconds added per manoeuvre; a left across traffic is not free.
    turn_penalty_s: Dict[str, float] = field(
        default_factory=lambda: {
            "straight": 0.0,
            "slight_left": 1.5,
            "slight_right": 1.0,
            "left": 12.0,
            "right": 5.0,
            "sharp_left": 16.0,
            "sharp_right": 9.0,
            "uturn": 25.0,
        }
    )
    signal_penalty_s: float = 8.0
    # Straight-line fallback when the routing query times out or the graph is missing.
    fallback_detour_factor: float = 1.32
    fallback_speed_kph: float = 24.0

    def validate(self) -> None:
        if self.snap_radius_m <= 0:
            raise ConfigError("graph.snap_radius_m must be > 0")
        if self.snap_candidates < 1:
            raise ConfigError("graph.snap_candidates must be >= 1")
        if self.fallback_detour_factor < 1.0:
            raise ConfigError("graph.fallback_detour_factor must be >= 1.0")
        if self.fallback_speed_kph <= 0:
            raise ConfigError("graph.fallback_speed_kph must be > 0")
        for cls_name, speed in self.default_speed_kph.items():
            if speed <= 0:
                raise ConfigError(f"graph.default_speed_kph[{cls_name}] must be > 0")


@dataclass
class ProfilesConfig:
    """Speed-profile mining: bucketing, filtering, and hierarchical shrinkage."""

    tod_buckets: int = 24
    dow_buckets: int = 7
    timezone: str = "America/New_York"
    min_samples_per_cell: int = 5
    # Shrinkage strength at each level of the hierarchy: cell -> segment -> class -> global.
    shrink_k_segment: float = 20.0
    shrink_k_class_tod: float = 40.0
    shrink_k_global: float = 80.0
    min_speed_kph: float = 1.0
    max_speed_kph: float = 130.0
    profile_table: str = "segment_speed_profile"
    smoothed_table: str = "segment_speed_profile_smoothed"
    max_profile_age_days: int = 90

    def validate(self) -> None:
        if 24 % self.tod_buckets != 0:
            raise ConfigError("profiles.tod_buckets must divide 24")
        if self.dow_buckets not in (1, 7):
            raise ConfigError("profiles.dow_buckets must be 1 (all days) or 7")
        if self.min_samples_per_cell < 1:
            raise ConfigError("profiles.min_samples_per_cell must be >= 1")
        if self.min_speed_kph <= 0 or self.max_speed_kph <= self.min_speed_kph:
            raise ConfigError("profiles speed bounds must satisfy 0 < min < max")
        for name in ("shrink_k_segment", "shrink_k_class_tod", "shrink_k_global"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"profiles.{name} must be > 0")


@dataclass
class FeaturesConfig:
    h3_resolution_coarse: int = 8
    h3_resolution_fine: int = 9
    use_weather: bool = True
    weather_max_staleness_min: int = 90
    target_encoding_smoothing: float = 30.0
    holiday_country: str = "US"
    rush_hour_windows: List[List[int]] = field(
        default_factory=lambda: [[7, 10], [16, 19]]
    )

    def validate(self) -> None:
        for name in ("h3_resolution_coarse", "h3_resolution_fine"):
            value = getattr(self, name)
            if not 0 <= value <= 15:
                raise ConfigError(f"features.{name} must be in 0..15")
        if self.h3_resolution_coarse >= self.h3_resolution_fine:
            raise ConfigError("features.h3_resolution_coarse must be < h3_resolution_fine")
        if self.target_encoding_smoothing <= 0:
            raise ConfigError("features.target_encoding_smoothing must be > 0")
        for window in self.rush_hour_windows:
            if len(window) != 2 or not 0 <= window[0] < window[1] <= 24:
                raise ConfigError(f"features.rush_hour_windows entry {window} is invalid")


@dataclass
class ModelConfig:
    """LightGBM hyper-parameters and the registry contract."""

    registry_dir: str = "artifacts/models"
    quantiles: List[float] = field(default_factory=lambda: [0.5, 0.9])
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 600
    min_child_samples: int = 40
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 17
    # Fractions of the *time-ordered* corpus, not random rows.
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    # A candidate whose held-out p90 coverage falls outside this band is not promoted.
    promote_coverage_band: List[float] = field(default_factory=lambda: [0.86, 0.94])
    max_residual_abs_s: float = 3_600.0

    def validate(self) -> None:
        if not self.quantiles:
            raise ConfigError("model.quantiles must not be empty")
        for q in self.quantiles:
            if not 0.0 < q < 1.0:
                raise ConfigError(f"model.quantiles entry {q} must be in (0, 1)")
        if sorted(self.quantiles) != list(self.quantiles):
            raise ConfigError("model.quantiles must be given in ascending order")
        if self.train_fraction <= 0 or self.validation_fraction <= 0:
            raise ConfigError("model split fractions must be > 0")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ConfigError("model.train_fraction + validation_fraction must be < 1.0")
        if len(self.promote_coverage_band) != 2:
            raise ConfigError("model.promote_coverage_band must be [low, high]")
        low, high = self.promote_coverage_band
        if not 0.0 < low < high < 1.0:
            raise ConfigError("model.promote_coverage_band must satisfy 0 < low < high < 1")


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    route_timeout_ms: int = 400
    model_alias: str = "production"
    reload_interval_s: int = 300
    default_handoff_p50_s: float = 90.0
    default_handoff_p90_s: float = 240.0
    default_prep_p50_s: float = 600.0
    default_prep_p90_s: float = 1_020.0
    # Correlation assumed between the prep / travel / handoff tails when composing p90.
    # 1.0 reproduces the naive sum; 0.0 adds them in quadrature.
    component_tail_correlation: float = 0.35
    log_feature_hash: bool = True

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"service.port {self.port} outside 1..65535")
        if self.route_timeout_ms <= 0:
            raise ConfigError("service.route_timeout_ms must be > 0")
        if not 0.0 <= self.component_tail_correlation <= 1.0:
            raise ConfigError("service.component_tail_correlation must be in [0, 1]")
        if self.default_handoff_p90_s < self.default_handoff_p50_s:
            raise ConfigError("service.default_handoff_p90_s must be >= p50")
        if self.default_prep_p90_s < self.default_prep_p50_s:
            raise ConfigError("service.default_prep_p90_s must be >= p50")


@dataclass
class EvalConfig:
    report_dir: str = "artifacts/eval"
    mape_floor_s: float = 60.0
    distance_buckets_m: List[float] = field(
        default_factory=lambda: [1_000.0, 3_000.0, 8_000.0, 20_000.0]
    )
    calibration_quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )
    min_slice_size: int = 30

    def validate(self) -> None:
        if self.mape_floor_s <= 0:
            raise ConfigError("eval.mape_floor_s must be > 0 (MAPE explodes near zero)")
        if sorted(self.distance_buckets_m) != list(self.distance_buckets_m):
            raise ConfigError("eval.distance_buckets_m must be ascending")
        if self.min_slice_size < 1:
            raise ConfigError("eval.min_slice_size must be >= 1")


@dataclass
class DataConfig:
    """Where source data lives and which region is in scope."""

    tlc_dir: str = "data/tlc"
    osm_extract: str = "data/osm/new-york.osm.pbf"
    weather_csv: str = "data/weather/nyc_hourly.csv"
    region_bbox: List[float] = field(
        default_factory=lambda: [40.4774, -74.2591, 40.9176, -73.7004]
    )
    min_trip_duration_s: float = 60.0
    max_trip_duration_s: float = 14_400.0
    min_trip_distance_m: float = 100.0
    max_trip_speed_kph: float = 130.0

    def validate(self) -> None:
        if len(self.region_bbox) != 4:
            raise ConfigError("data.region_bbox must be [min_lat, min_lon, max_lat, max_lon]")
        self.bbox()  # raises if the box is degenerate
        if self.min_trip_duration_s >= self.max_trip_duration_s:
            raise ConfigError("data.min_trip_duration_s must be < max_trip_duration_s")

    def bbox(self) -> BoundingBox:
        min_lat, min_lon, max_lat, max_lon = self.region_bbox
        return BoundingBox(min_lat, min_lon, max_lat, max_lon)


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    profiles: ProfilesConfig = field(default_factory=ProfilesConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> "Config":
        for f in fields(self):
            section = getattr(self, f.name)
            validate = getattr(section, "validate", None)
            if callable(validate):
                validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _dataclass_to_dict(getattr(self, f.name)) for f in fields(self)}


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def load_config(
    path: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    overrides: Optional[Sequence[str]] = None,
) -> Config:
    """Build a validated :class:`Config` from the three layers, lowest priority first."""
    data: Dict[str, Any] = {}
    if path:
        data = _read_yaml(path)
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a top-level mapping")
    _merge(data, _env_overrides(os.environ if env is None else env))
    for item in overrides or ():
        _merge(data, _cli_override(item))
    return _build(Config, data).validate()


def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # imported lazily so the package imports without PyYAML
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigError(
            "PyYAML is required to read a config file; pip install pyyaml"
        ) from exc
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc


def _env_overrides(env: Mapping[str, str]) -> Dict[str, Any]:
    """``ETA_FORGE_DATABASE__HOST=db`` -> ``{"database": {"host": "db"}}``.

    A double underscore separates section from key so that keys containing single
    underscores (``max_pool_size``) survive intact.
    """
    out: Dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX) :].lower()
        parts = path.split("__") if "__" in path else _split_known_section(path)
        if parts is None:
            continue
        _assign(out, parts, _parse_scalar(raw_value))
    return out


_SECTION_NAMES = tuple(f.name for f in fields(Config))


def _split_known_section(path: str) -> Optional[List[str]]:
    for section in _SECTION_NAMES:
        prefix = section + "_"
        if path.startswith(prefix):
            return [section, path[len(prefix) :]]
    return None


def _cli_override(item: str) -> Dict[str, Any]:
    if "=" not in item:
        raise ConfigError(f"override {item!r} must be of the form section.key=value")
    key, value = item.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if len(parts) < 2:
        raise ConfigError(f"override {item!r} must name a section and a key")
    out: Dict[str, Any] = {}
    _assign(out, parts, _parse_scalar(value))
    return out


def _assign(target: Dict[str, Any], parts: Sequence[str], value: Any) -> None:
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"cannot descend into scalar at {part!r}")
        cursor = nxt
    cursor[parts[-1]] = value


def _merge(base: Dict[str, Any], extra: Mapping[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value


def _parse_scalar(raw: str) -> Any:
    """Best-effort typing of an environment/CLI string.

    JSON is tried first so lists and nested maps can be passed on one line; a bare
    word that is not JSON falls back to the string itself.
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text[:1] in "[{" or text[:1] == '"':
        import json

        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def _resolve_hints(cls: Any) -> Dict[str, Any]:
    """Resolve string annotations to real types.

    ``from __future__ import annotations`` turns every annotation into a string, so
    ``dataclasses.Field.type`` cannot be compared against ``int`` or tested with
    ``is_dataclass``. ``get_type_hints`` gives back the objects.
    """
    import typing

    return typing.get_type_hints(cls)


def _build(cls: Any, data: Mapping[str, Any]) -> Any:
    """Instantiate a nested dataclass tree, rejecting unknown and mistyped keys."""
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    hints = _resolve_hints(cls)
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    for name in known:
        if name not in data:
            continue
        value = data[name]
        annotation = hints.get(name)
        if isinstance(annotation, type) and is_dataclass(annotation):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{cls.__name__}.{name} must be a mapping")
            kwargs[name] = _build(annotation, value)
        else:
            kwargs[name] = _coerce(cls.__name__, name, annotation, value)
    return cls(**kwargs)


def _coerce(owner: str, name: str, annotation: Any, value: Any) -> Any:
    """Coerce a scalar to the annotated primitive; containers pass through."""
    if value is None:
        return value
    target = annotation if isinstance(annotation, type) else None
    if target in (int, float, str, bool):
        if target is bool:
            if isinstance(value, bool):
                return value
            raise ConfigError(f"{owner}.{name} must be a boolean, got {value!r}")
        if target is str:
            return str(value)
        if isinstance(value, bool):
            raise ConfigError(f"{owner}.{name} must be numeric, got a boolean")
        try:
            return target(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{owner}.{name} must be {target.__name__}, got {value!r}"
            ) from exc
    return value


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


__all__ = [
    "Config",
    "ConfigError",
    "DatabaseConfig",
    "GraphConfig",
    "ProfilesConfig",
    "FeaturesConfig",
    "ModelConfig",
    "ServiceConfig",
    "EvalConfig",
    "DataConfig",
    "load_config",
]
