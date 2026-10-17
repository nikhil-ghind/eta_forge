"""The offline evaluation harness.

Input is a list of :class:`EvalRecord` — held-out trips with what actually happened,
what the routing-only baseline said, and what the full service said. Output is an
:class:`EvalResult` that answers four questions:

1. **How accurate is the p50?** MAE / MAPE / RMSE / bias overall.
2. **Is the p90 a p90?** Coverage, pinball loss and sharpness, and a calibration curve
   across the implied quantiles.
3. **What did the learned stage add?** The same point metrics for routing-only and
   routing + residual side by side, with the relative improvement — so the value of
   stage two is a number, not a claim.
4. **Where is it wrong?** Every metric per hour, weekday, distance bucket, cell and
   merchant.

Records come either from training rows scored by a residual model directly (fast; what
``scripts/evaluate.py`` uses) or from the full :class:`EtaService` request path (slower,
but it exercises routing fallbacks and composition too).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import EvalConfig
from ..features.calendar import localize
from ..models.dataset import TrainingRow
from ..models.residual import ResidualModel
from ..types import EtaRequest, TripRecord
from .calibration import CalibrationPoint, calibration_curve, calibration_error
from .metrics import point_metrics, quantile_metrics, relative_improvement
from .slices import EvalRecord, SliceMetrics, slice_metrics, standard_slicers, worst_slices


def records_from_rows(rows: Sequence[TrainingRow], model: ResidualModel) -> List[EvalRecord]:
    """Score training-format rows with a residual model, using its fitted encoders."""
    if not rows:
        return []
    vectors = [model.builder.finalize(r.feature_row) for r in rows]
    predictions = model.predict_travel_batch([r.baseline_s for r in rows], vectors)
    out: List[EvalRecord] = []
    for row, (pair, _, _) in zip(rows, predictions):
        keys = row.slice_keys()
        out.append(
            EvalRecord(
                trip_id=row.trip_id,
                departure_at=row.departure_at,
                actual_s=row.actual_s,
                baseline_s=row.baseline_s,
                p50_s=pair.p50,
                p90_s=pair.p90,
                distance_m=row.distance_m,
                hour=int(keys["hour"]),
                dow=int(keys["dow"]),
                origin_cell=row.origin_cell,
                dest_cell=row.dest_cell,
                merchant_id=row.merchant_id,
            )
        )
    return out


def records_from_service(
    trips: Iterable[TripRecord], service: Any, *, tz: Optional[str] = "America/New_York"
) -> List[EvalRecord]:
    """Score trips through the full service path as ride requests."""
    out: List[EvalRecord] = []
    for trip in trips:
        response = service.estimate(
            EtaRequest(origin=trip.origin, destination=trip.destination, departure_at=trip.pickup_at)
        )
        local = localize(trip.pickup_at, tz)
        bundle = service.bundle
        builder = bundle.residual.builder if bundle.residual else None
        out.append(
            EvalRecord(
                trip_id=trip.trip_id,
                departure_at=trip.pickup_at,
                actual_s=trip.duration_s,
                baseline_s=response.breakdown.baseline_travel_s,
                p50_s=response.eta.p50,
                p90_s=response.eta.p90,
                distance_m=response.distance_m,
                hour=local.hour,
                dow=local.weekday(),
                origin_cell=builder.cell(trip.origin) if builder else None,
                dest_cell=builder.cell(trip.destination) if builder else None,
                merchant_id=trip.merchant_id,
                degraded=response.degraded,
            )
        )
    return out


@dataclass
class EvalResult:
    overall: Dict[str, float]
    baseline: Dict[str, float]
    comparison: List[Dict[str, Any]]
    calibration: List[CalibrationPoint]
    calibration_error: Dict[str, float]
    slices: Dict[str, List[SliceMetrics]] = field(default_factory=dict)
    dropped_slices: Dict[str, int] = field(default_factory=dict)
    worst_late: List[SliceMetrics] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "baseline": self.baseline,
            "comparison": self.comparison,
            "calibration": [p.to_dict() for p in self.calibration],
            "calibration_error": self.calibration_error,
            "dropped_slices": self.dropped_slices,
            "worst_late": [s.to_row() for s in self.worst_late],
            "meta": self.meta,
        }


COMPARED = ("mae_s", "mape", "rmse_s", "bias_s", "pinball_p50", "late_rate")


def evaluate(
    records: Sequence[EvalRecord],
    cfg: EvalConfig,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    actual = [r.actual_s for r in records]
    p50 = [r.p50_s for r in records]
    p90 = [r.p90_s for r in records]
    baseline_pred = [r.baseline_s for r in records]

    overall = quantile_metrics(actual, p50, p90, mape_floor_s=cfg.mape_floor_s)
    overall["degraded_rate"] = (
        sum(1 for r in records if r.degraded) / len(records) if records else 0.0
    )
    baseline = point_metrics(actual, baseline_pred, mape_floor_s=cfg.mape_floor_s)

    comparison = [
        dict({"predictor": "routing only"}, **{k: baseline[k] for k in COMPARED}),
        dict({"predictor": "routing + residual p50"}, **{k: overall[k] for k in COMPARED}),
        dict(
            {"predictor": "relative improvement"},
            **{
                k: relative_improvement(abs(overall[k]), abs(baseline[k]))
                for k in COMPARED
            },
        ),
    ]

    curve = calibration_curve(actual, p50, p90, cfg.calibration_quantiles)

    slices: Dict[str, List[SliceMetrics]] = {}
    dropped: Dict[str, int] = {}
    for dimension, fn in standard_slicers(cfg).items():
        found, n_dropped = slice_metrics(
            records, dimension, fn, min_size=cfg.min_slice_size, mape_floor_s=cfg.mape_floor_s
        )
        if found or n_dropped:
            slices[dimension] = found
            dropped[dimension] = n_dropped

    pooled = [s for dim in ("hour", "distance", "origin_cell") for s in slices.get(dim, [])]
    info = dict(meta or {})
    if records:
        info.setdefault("window_start", min(r.departure_at for r in records).isoformat())
        info.setdefault("window_end", max(r.departure_at for r in records).isoformat())
    info.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

    return EvalResult(
        overall=overall,
        baseline=baseline,
        comparison=comparison,
        calibration=curve,
        calibration_error=calibration_error(curve),
        slices=slices,
        dropped_slices=dropped,
        worst_late=worst_slices(pooled, "bias_s", n=5, lowest=True),
        meta=info,
    )


__all__ = ["EvalResult", "evaluate", "records_from_rows", "records_from_service", "COMPARED"]
