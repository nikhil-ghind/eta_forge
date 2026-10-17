"""The stage-two residual model.

The learner never predicts a duration. It predicts the *correction* to the physical
baseline, at each configured quantile:

    travel_q = baseline + residual_q

Two consequences, both deliberate:

* The model's job is small. Most of a 25-minute trip is explained by its route; the
  residual is what the route got wrong (turns, signals, the hotspot, the rain), and
  a tree model learns that far more sample-efficiently than it learns "duration".
* When the model is unavailable or untrusted, the baseline alone is still a valid
  ETA. A model that predicts durations directly has no such fallback.

Quantiles are fitted independently, so they can cross. Serving repairs that after
the fact (``p90 = max(p90, p50)``) and the training report records how often it was
needed: a high crossing rate is a sign the upper model is under-fitted.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..features.builder import FeatureBuilder
from ..features.h3_cells import CellVocabulary
from ..features.target_encoding import TargetEncoder
from ..types import QuantilePair
from .dataset import TrainingRow, training_window
from .quantile import (
    alpha_key,
    coverage,
    crossing_rate,
    load_quantile_model,
    make_quantile_model,
    mean_pinball_loss,
    resolve_backend,
)
from .splits import TimeSplit, assert_no_leakage, time_blocks, time_split

MODEL_FILE = "residual.json"
BUILDER_FILE = "feature_state.json"


def upper_alpha(quantiles: Sequence[float]) -> float:
    """The quantile served as "p90": 0.9 when configured, else the highest one."""
    return 0.9 if 0.9 in quantiles else max(quantiles)


class ResidualModel:
    def __init__(
        self,
        builder: FeatureBuilder,
        models: Mapping[float, Any],
        *,
        max_residual_abs_s: float = 3_600.0,
        min_travel_fraction: float = 0.3,
        backend: str = "stumps",
    ):
        if 0.5 not in models:
            raise ValueError("a residual model needs a 0.5 quantile")
        self.builder = builder
        self.models = dict(models)
        self.max_residual_abs_s = max_residual_abs_s
        self.min_travel_fraction = min_travel_fraction
        self.backend = backend

    @property
    def quantiles(self) -> List[float]:
        return sorted(self.models)

    @property
    def spec_hash(self) -> str:
        return self.builder.spec.hash

    def _clip(self, value: float) -> float:
        return max(-self.max_residual_abs_s, min(self.max_residual_abs_s, value))

    def predict_residuals(self, vectors: Sequence[Sequence[float]]) -> Dict[float, List[float]]:
        return {
            alpha: [self._clip(v) for v in model.predict(vectors)]
            for alpha, model in self.models.items()
        }

    def compose(self, baseline_s: float, r50: float, r90: float) -> QuantilePair:
        floor = self.min_travel_fraction * baseline_s
        p50 = max(floor, baseline_s + r50)
        p90 = max(p50, baseline_s + r90)  # monotonicity repair
        return QuantilePair(p50, p90)

    def predict_travel_batch(
        self, baselines: Sequence[float], vectors: Sequence[Sequence[float]]
    ) -> List[Tuple[QuantilePair, float, float]]:
        residuals = self.predict_residuals(vectors)
        hi = upper_alpha(self.quantiles)
        out = []
        for i, baseline in enumerate(baselines):
            r50, r90 = residuals[0.5][i], residuals[hi][i]
            out.append((self.compose(baseline, r50, r90), r50, r90))
        return out

    def predict_travel(
        self, baseline_s: float, vector: Sequence[float]
    ) -> Tuple[QuantilePair, float, float]:
        return self.predict_travel_batch([baseline_s], [vector])[0]

    # -- persistence -------------------------------------------------------------

    def save(self, directory: str) -> Dict[str, Any]:
        os.makedirs(directory, exist_ok=True)
        entries = [model.save(directory, f"residual_{alpha_key(a)}") for a, model in sorted(self.models.items())]
        meta = {
            "kind": "residual",
            "backend": self.backend,
            "quantiles": entries,
            "max_residual_abs_s": self.max_residual_abs_s,
            "min_travel_fraction": self.min_travel_fraction,
            "feature_spec_hash": self.spec_hash,
        }
        with open(os.path.join(directory, MODEL_FILE), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        with open(os.path.join(directory, BUILDER_FILE), "w", encoding="utf-8") as handle:
            json.dump(self.builder.state(), handle, separators=(",", ":"), sort_keys=True)
        return meta

    @classmethod
    def load(cls, cfg: Config, directory: str) -> "ResidualModel":
        with open(os.path.join(directory, MODEL_FILE), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        with open(os.path.join(directory, BUILDER_FILE), "r", encoding="utf-8") as handle:
            builder = FeatureBuilder.from_state(cfg, json.load(handle))
        models = {
            float(entry["alpha"]): load_quantile_model(entry, directory)
            for entry in meta["quantiles"]
        }
        return cls(
            builder,
            models,
            max_residual_abs_s=float(meta.get("max_residual_abs_s", 3_600.0)),
            min_travel_fraction=float(meta.get("min_travel_fraction", 0.3)),
            backend=str(meta.get("backend", "stumps")),
        )


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------


@dataclass
class TrainingReport:
    sizes: Dict[str, int] = field(default_factory=dict)
    window: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    backend: str = ""
    feature_spec_hash: str = ""
    feature_count: int = 0
    vocabulary_size: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sizes": self.sizes,
            "window": self.window,
            "metrics": self.metrics,
            "backend": self.backend,
            "feature_spec_hash": self.feature_spec_hash,
            "feature_count": self.feature_count,
            "vocabulary_size": self.vocabulary_size,
        }

    def coverage(self, split: str = "test") -> Optional[float]:
        return self.metrics.get(split, {}).get("coverage_p90")


def split_metrics(
    model: ResidualModel,
    rows: Sequence[TrainingRow],
    vectors: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """Headline numbers for one split: what the promotion gate and the manifest read."""
    if not rows:
        return {}
    predictions = model.predict_travel_batch([r.baseline_s for r in rows], vectors)
    actual = [r.actual_s for r in rows]
    p50 = [p.p50 for p, _, _ in predictions]
    p90 = [p.p90 for p, _, _ in predictions]
    hi = upper_alpha(model.quantiles)
    raw = model.predict_residuals(vectors)
    return {
        "rows": float(len(rows)),
        "mae_p50": fmean(abs(a - p) for a, p in zip(actual, p50)),
        "baseline_mae": fmean(abs(a - r.baseline_s) for a, r in zip(actual, rows)),
        "pinball_p50": mean_pinball_loss(actual, p50, 0.5),
        "pinball_p90": mean_pinball_loss(actual, p90, hi),
        "coverage_p90": coverage(actual, p90),
        "coverage_p50": coverage(actual, p50),
        "sharpness_s": fmean(b - a for a, b in zip(p50, p90)),
        "crossing_rate": crossing_rate(raw[0.5], raw[hi]),
    }


class ResidualTrainer:
    """Fit the residual quantile models on a time-split dataset."""

    def __init__(self, cfg: Config, *, backend: Optional[str] = None, te_blocks: int = 5):
        self._cfg = cfg
        self._backend = resolve_backend(backend or cfg.model.backend)
        self._te_blocks = te_blocks

    def split(self, rows: Sequence[TrainingRow]) -> TimeSplit:
        split = time_split(
            [r.departure_at for r in rows],
            train_fraction=self._cfg.model.train_fraction,
            validation_fraction=self._cfg.model.validation_fraction,
        )
        assert_no_leakage(split, [r.departure_at for r in rows])
        return split

    def fit(
        self, rows: Sequence[TrainingRow]
    ) -> Tuple[ResidualModel, TrainingReport, TimeSplit]:
        cfg = self._cfg
        if len(rows) < 30:
            raise ValueError(f"need at least 30 training rows, got {len(rows)}")
        split = self.split(rows)
        train = [rows[i] for i in split.train]
        validation = [rows[i] for i in split.validation]
        test = [rows[i] for i in split.test]

        clip = cfg.model.max_residual_abs_s
        y_train = [max(-clip, min(clip, r.residual_s)) for r in train]

        vocabulary = CellVocabulary(min_count=max(5, cfg.model.min_child_samples // 2)).fit(
            [r.origin_cell for r in train] + [r.dest_cell for r in train]
        )
        smoothing = cfg.features.target_encoding_smoothing
        folds = time_blocks(list(range(len(train))), self._te_blocks)
        origin_encoder = TargetEncoder(smoothing=smoothing)
        dest_encoder = TargetEncoder(smoothing=smoothing)
        origin_oof = origin_encoder.fit_transform_oof(
            [r.origin_cell for r in train], y_train, folds, mode="past"
        )
        dest_oof = dest_encoder.fit_transform_oof(
            [r.dest_cell for r in train], y_train, folds, mode="past"
        )

        builder = FeatureBuilder(
            cfg,
            origin_encoder=origin_encoder,
            dest_encoder=dest_encoder,
            vocabulary=vocabulary,
        )
        if train:
            builder.scheme = _scheme_of(train[0].origin_cell)

        X_train = [
            builder.finalize(r.feature_row, origin_te=o, dest_te=d, strict=True)
            for r, o, d in zip(train, origin_oof, dest_oof)
        ]
        X_val = [builder.finalize(r.feature_row, strict=True) for r in validation]
        X_test = [builder.finalize(r.feature_row, strict=True) for r in test]
        y_val = [max(-clip, min(clip, r.residual_s)) for r in validation]

        categorical = builder.spec.categorical_indices
        models: Dict[float, Any] = {}
        for alpha in cfg.model.quantiles:
            model = make_quantile_model(
                alpha, cfg.model, categorical=categorical, backend=self._backend
            )
            models[alpha] = model.fit(
                X_train, y_train, X_val=X_val or None, y_val=y_val or None
            )

        residual = ResidualModel(
            builder,
            models,
            max_residual_abs_s=clip,
            min_travel_fraction=cfg.model.min_travel_fraction,
            backend=self._backend,
        )
        report = TrainingReport(
            sizes=split.sizes(),
            window={
                "train": training_window(train),
                "validation": training_window(validation),
                "test": training_window(test),
            },
            metrics={
                "train": split_metrics(residual, train, X_train),
                "validation": split_metrics(residual, validation, X_val),
                "test": split_metrics(residual, test, X_test),
            },
            backend=self._backend,
            feature_spec_hash=builder.spec.hash,
            feature_count=len(builder.spec),
            vocabulary_size=len(vocabulary),
        )
        return residual, report, split


def _scheme_of(cell: str) -> str:
    return "grid" if cell.startswith("g") and ":" in cell else "h3"


def finite_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return fmean(finite) if finite else float("nan")


__all__ = [
    "ResidualModel",
    "ResidualTrainer",
    "TrainingReport",
    "split_metrics",
    "upper_alpha",
    "MODEL_FILE",
    "BUILDER_FILE",
]
