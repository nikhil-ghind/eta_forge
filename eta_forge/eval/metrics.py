"""Point and quantile accuracy metrics for ETAs.

Two that need care:

**MAPE** is ``mean(|actual - predicted| / actual)``, and it explodes for short trips: a
90-second trip predicted at 3 minutes is a 100% error that dominates a corpus of
25-minute trips predicted within 10%. The denominator is floored at
``eval.mape_floor_s`` (60 s by default), which keeps the metric about the trips people
actually take.

**Coverage** is the fraction of actuals at or below the predicted p90. It is the single
most important number in this file: it is what makes a p90 a p90. Coverage is reported
together with **sharpness** (mean p90 - p50), because 100% coverage is trivially
achievable by predicting an hour for everything.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

from ..models.quantile import coverage, mean_pinball_loss


def _check(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _check(actual, predicted)
    if not actual:
        return float("nan")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _check(actual, predicted)
    if not actual:
        return float("nan")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mape(actual: Sequence[float], predicted: Sequence[float], *, floor_s: float = 60.0) -> float:
    """Mean absolute percentage error with a floored denominator. Returns a fraction."""
    _check(actual, predicted)
    if floor_s <= 0:
        raise ValueError("floor_s must be > 0")
    if not actual:
        return float("nan")
    return sum(abs(a - p) / max(a, floor_s) for a, p in zip(actual, predicted)) / len(actual)


def bias(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean (predicted - actual): negative means the service is late on average."""
    _check(actual, predicted)
    if not actual:
        return float("nan")
    return sum(p - a for a, p in zip(actual, predicted)) / len(actual)


def late_rate(actual: Sequence[float], predicted: Sequence[float], *, tolerance_s: float = 0.0) -> float:
    """Fraction of trips that arrived later than predicted (by more than the tolerance)."""
    _check(actual, predicted)
    if not actual:
        return float("nan")
    return sum(1 for a, p in zip(actual, predicted) if a > p + tolerance_s) / len(actual)


def sharpness(p50: Sequence[float], p90: Sequence[float]) -> float:
    _check(p50, p90)
    if not p50:
        return float("nan")
    return sum(hi - lo for lo, hi in zip(p50, p90)) / len(p50)


def point_metrics(
    actual: Sequence[float], predicted: Sequence[float], *, mape_floor_s: float = 60.0
) -> Dict[str, float]:
    return {
        "n": float(len(actual)),
        "mae_s": mae(actual, predicted),
        "rmse_s": rmse(actual, predicted),
        "mape": mape(actual, predicted, floor_s=mape_floor_s),
        "bias_s": bias(actual, predicted),
        "pinball_p50": mean_pinball_loss(actual, predicted, 0.5) if actual else float("nan"),
        "late_rate": late_rate(actual, predicted),
    }


def quantile_metrics(
    actual: Sequence[float],
    p50: Sequence[float],
    p90: Sequence[float],
    *,
    mape_floor_s: float = 60.0,
    upper_alpha: float = 0.9,
) -> Dict[str, float]:
    out = point_metrics(actual, p50, mape_floor_s=mape_floor_s)
    out.update({
        "pinball_p90": mean_pinball_loss(actual, p90, upper_alpha) if actual else float("nan"),
        "coverage_p50": coverage(actual, p50) if actual else float("nan"),
        "coverage_p90": coverage(actual, p90) if actual else float("nan"),
        "sharpness_s": sharpness(p50, p90),
    })
    return out


def relative_improvement(model: float, baseline: float) -> Optional[float]:
    """``1 - model / baseline``: 0.3 means the model removed 30% of the baseline's error."""
    if baseline is None or baseline == 0 or math.isnan(baseline) or math.isnan(model):
        return None
    return 1.0 - model / baseline


__all__ = [
    "mae",
    "rmse",
    "mape",
    "bias",
    "late_rate",
    "sharpness",
    "point_metrics",
    "quantile_metrics",
    "relative_improvement",
]
