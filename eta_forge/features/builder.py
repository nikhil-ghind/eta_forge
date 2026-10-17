"""Assembling the residual model's feature vector.

Building a vector happens in two steps, because the spatial target encodings cannot
be known when a training row is first built:

1. :meth:`FeatureBuilder.base_features` — everything that depends only on the request
   and the baseline route: calendar, route structure, O-D geometry, weather. Done once
   per trip, at dataset build time.
2. :meth:`FeatureBuilder.finalize` — adds the cell encodings and orders the mapping by
   the :class:`FeatureSpec`. At training time the trainer passes out-of-fold encodings
   in explicitly; at serving time the builder uses the encoders it was fitted with.

The same class runs in both places, which is the point: there is exactly one
definition of every feature.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..graph.summary import RouteSummary, summarize_route
from ..types import BaselineRoute, Location, WeatherObservation
from .calendar import HolidayCalendar, calendar_features
from .h3_cells import CellVocabulary, cell_for, cell_scheme
from .spec import FeatureSpec, residual_spec
from .target_encoding import TargetEncoder
from .weather import weather_features


@dataclass
class FeatureRow:
    """Base features for one trip, plus the keys needed to finish it later."""

    features: Dict[str, float]
    origin_cell: str
    dest_cell: str
    summary: RouteSummary


class FeatureBuilder:
    def __init__(
        self,
        cfg: Config,
        *,
        spec: Optional[FeatureSpec] = None,
        origin_encoder: Optional[TargetEncoder] = None,
        dest_encoder: Optional[TargetEncoder] = None,
        vocabulary: Optional[CellVocabulary] = None,
        holidays: Optional[HolidayCalendar] = None,
        scheme: Optional[str] = None,
    ):
        self._cfg = cfg
        self.spec = spec or residual_spec(use_weather=cfg.features.use_weather)
        smoothing = cfg.features.target_encoding_smoothing
        self.origin_encoder = origin_encoder or TargetEncoder(smoothing=smoothing)
        self.dest_encoder = dest_encoder or TargetEncoder(smoothing=smoothing)
        self.vocabulary = vocabulary or CellVocabulary()
        self.holidays = holidays or HolidayCalendar(cfg.features.holiday_country)
        self.scheme = scheme or cell_scheme()

    # -- cells -------------------------------------------------------------------

    def cell(self, location: Location) -> str:
        return cell_for(location, self._cfg.features.h3_resolution_coarse, scheme=self.scheme)

    # -- step 1 ------------------------------------------------------------------

    def base_features(
        self,
        route: BaselineRoute,
        *,
        weather: Optional[WeatherObservation] = None,
        departure_at: Optional[datetime] = None,
    ) -> FeatureRow:
        when = departure_at or route.departure_at
        summary = summarize_route(
            route,
            turn_penalties=self._cfg.graph.turn_penalty_s,
            signal_penalty_s=self._cfg.graph.signal_penalty_s,
        )
        features: Dict[str, float] = {}
        features.update(
            calendar_features(
                when,
                tz=self._cfg.profiles.timezone,
                rush_hour_windows=self._cfg.features.rush_hour_windows,
                holidays=self.holidays,
            )
        )
        features.update(summary.to_features())
        bearing = math.radians(summary.od_bearing_deg)
        features["od_bearing_sin"] = math.sin(bearing)
        features["od_bearing_cos"] = math.cos(bearing)
        if self._cfg.features.use_weather:
            features.update(weather_features(weather))
        return FeatureRow(
            features=features,
            origin_cell=self.cell(route.origin),
            dest_cell=self.cell(route.destination),
            summary=summary,
        )

    # -- step 2 ------------------------------------------------------------------

    def finalize(
        self,
        row: FeatureRow,
        *,
        origin_te: Optional[float] = None,
        dest_te: Optional[float] = None,
        strict: bool = False,
    ) -> List[float]:
        features = dict(row.features)
        features["origin_cell_te"] = (
            origin_te if origin_te is not None else self.origin_encoder.transform_one(row.origin_cell)
        )
        features["dest_cell_te"] = (
            dest_te if dest_te is not None else self.dest_encoder.transform_one(row.dest_cell)
        )
        features["origin_cell_n"] = math.log1p(self.origin_encoder.count(row.origin_cell))
        features["dest_cell_n"] = math.log1p(self.dest_encoder.count(row.dest_cell))
        features["origin_cell_id"] = float(self.vocabulary.encode(row.origin_cell))
        features["dest_cell_id"] = float(self.vocabulary.encode(row.dest_cell))
        return self.spec.vectorize(features, strict=strict)

    def vector(
        self,
        route: BaselineRoute,
        *,
        weather: Optional[WeatherObservation] = None,
    ) -> Tuple[List[float], FeatureRow]:
        """Serving path: both steps with the fitted encoders."""
        row = self.base_features(route, weather=weather)
        return self.finalize(row), row

    # -- persistence -------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "origin_encoder": self.origin_encoder.to_dict(),
            "dest_encoder": self.dest_encoder.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "cell_scheme": self.scheme,
            "cell_resolution": self._cfg.features.h3_resolution_coarse,
        }

    @classmethod
    def from_state(cls, cfg: Config, state: Mapping[str, Any]) -> "FeatureBuilder":
        spec = FeatureSpec.from_dict(state["spec"])
        # The serving code's own idea of the layout must match what was trained.
        spec.check(residual_spec(use_weather=cfg.features.use_weather).hash)
        return cls(
            cfg,
            spec=spec,
            origin_encoder=TargetEncoder.from_dict(state["origin_encoder"]),
            dest_encoder=TargetEncoder.from_dict(state["dest_encoder"]),
            vocabulary=CellVocabulary.from_dict(state["vocabulary"]),
            scheme=str(state.get("cell_scheme", "grid")),
        )


def feature_hash(vector: Sequence[float]) -> str:
    """Stable short hash of a feature vector, logged with every request for replay.

    NaN is canonicalised before packing: there are many NaN bit patterns and two
    identical vectors must hash identically.
    """
    digest = hashlib.sha1()
    for value in vector:
        canonical = float("nan") if value != value else round(float(value), 6)
        digest.update(struct.pack("<d", canonical) if canonical == canonical else b"NaN")
    return digest.hexdigest()[:16]


__all__ = ["FeatureBuilder", "FeatureRow", "feature_hash"]
