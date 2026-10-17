"""The feature contract between training and serving.

A :class:`FeatureSpec` is an ordered tuple of feature names plus the subset that is
categorical. Its hash is stored in every model manifest and checked when a model is
loaded for serving. If the code that builds the vector at serve time would produce a
different layout from the one the model was trained on — a feature added, two
swapped, a categorical turned numeric — the load fails loudly with
``FeatureSpecMismatch`` instead of serving predictions computed from the wrong columns,
which is the failure mode that otherwise produces a plausible-looking, badly wrong ETA.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..types import FeatureSpecMismatch, HighwayClass, TurnType, ValidationError
from .calendar import CALENDAR_FEATURES
from .weather import WEATHER_FEATURES

NAN = float("nan")


@dataclass(frozen=True)
class FeatureSpec:
    names: Tuple[str, ...]
    categorical: Tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            dupes = sorted({n for n in self.names if self.names.count(n) > 1})
            raise ValidationError(f"duplicate feature names: {dupes}")
        missing = [c for c in self.categorical if c not in self.names]
        if missing:
            raise ValidationError(f"categorical features not in spec: {missing}")

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def categorical_indices(self) -> List[int]:
        return [self.names.index(c) for c in self.categorical]

    @property
    def hash(self) -> str:
        payload = json.dumps(
            {"version": self.version, "names": list(self.names), "categorical": list(self.categorical)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def vectorize(
        self, features: Mapping[str, float], *, strict: bool = False
    ) -> List[float]:
        """Order a feature mapping into a vector. Missing features become NaN.

        ``strict`` turns a missing feature into an error; training uses it so that a
        builder bug cannot silently train a model on a column of NaNs.
        """
        out: List[float] = []
        for name in self.names:
            if name not in features:
                if strict:
                    raise ValidationError(f"feature {name!r} missing from row")
                out.append(NAN)
                continue
            value = features[name]
            out.append(NAN if value is None else float(value))
        return out

    def check(self, expected_hash: str) -> None:
        if expected_hash != self.hash:
            raise FeatureSpecMismatch(
                f"model was trained on feature spec {expected_hash}, "
                f"serving code builds {self.hash}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "names": list(self.names),
            "categorical": list(self.categorical),
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSpec":
        spec = cls(
            names=tuple(data["names"]),
            categorical=tuple(data.get("categorical", ())),
            version=int(data.get("version", 1)),
        )
        recorded = data.get("hash")
        if recorded and recorded != spec.hash:
            raise FeatureSpecMismatch(
                f"feature spec file hash {recorded} does not match its contents ({spec.hash})"
            )
        return spec


def route_feature_names() -> List[str]:
    """Exactly the keys ``RouteSummary.to_features`` produces, in a fixed order."""
    names = [
        "distance_m",
        "baseline_duration_s",
        "baseline_mean_speed_kph",
        "leg_count",
        "distinct_classes",
        "crow_distance_m",
        "detour_ratio",
        "od_bearing_deg",
        "profile_coverage",
        "snap_distance_m",
    ]
    names += [f"class_frac_{cls.value}" for cls in HighwayClass]
    names += [f"turn_{t.value}" for t in TurnType]
    names += [
        "turn_left_total",
        "turn_right_total",
        "turn_major_total",
        "turn_uturn_total",
        "signal_count",
        "class_changes",
        "turn_penalty_s",
        "max_turn_deg",
    ]
    return names


SPATIAL_FEATURES = (
    "origin_cell_te",
    "dest_cell_te",
    "origin_cell_n",
    "dest_cell_n",
    "origin_cell_id",
    "dest_cell_id",
    "od_bearing_sin",
    "od_bearing_cos",
)

RESIDUAL_CATEGORICAL = ("dow", "origin_cell_id", "dest_cell_id", "weather_condition")


def residual_spec(*, use_weather: bool = True) -> FeatureSpec:
    """The residual model's feature layout. Weather can be switched off as a whole."""
    names = list(CALENDAR_FEATURES) + route_feature_names() + list(SPATIAL_FEATURES)
    categorical = [c for c in RESIDUAL_CATEGORICAL if c != "weather_condition"]
    if use_weather:
        names += list(WEATHER_FEATURES)
        categorical.append("weather_condition")
    return FeatureSpec(names=tuple(names), categorical=tuple(categorical), version=1)


def finite_or_nan(value: float) -> float:
    return value if (value is not None and math.isfinite(value)) else NAN


__all__ = [
    "FeatureSpec",
    "residual_spec",
    "route_feature_names",
    "SPATIAL_FEATURES",
    "RESIDUAL_CATEGORICAL",
    "finite_or_nan",
]
