"""Calibration across nominal quantiles.

The service answers with two quantiles, p50 and p90. A caller asking "what's the
chance I'm there within 30 minutes?" is implicitly asking about every quantile in
between, so the evaluation checks the *implied* distribution too: a quantile at level
tau is interpolated through the served pair under a Gaussian shape,

    q_tau = p50 + (p90 - p50) * z(tau) / z(0.9)

and its observed coverage is compared with tau. A calibrated service lies on the
diagonal; one whose curve bows below it at high tau has a tail that is too short, and
one whose p25 covers 40% of trips has a median that is right but a spread that is
wrong, which coverage at p90 alone would never reveal.

The inverse normal CDF is Acklam's rational approximation (relative error < 1.2e-9),
so none of this needs SciPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

# Acklam's coefficients.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)
_P_LOW = 0.02425


def inverse_normal_cdf(p: float) -> float:
    """``z`` such that ``Phi(z) = p``."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p > 1.0 - _P_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    )


def implied_quantile(p50: float, p90: float, tau: float, *, upper_alpha: float = 0.9) -> float:
    """The tau-quantile implied by a served (p50, p90) pair, floored at zero."""
    scale = (p90 - p50) / inverse_normal_cdf(upper_alpha)
    return max(0.0, p50 + scale * inverse_normal_cdf(tau))


@dataclass(frozen=True)
class CalibrationPoint:
    nominal: float
    observed: float
    n: int

    @property
    def gap(self) -> float:
        return self.observed - self.nominal

    def to_dict(self) -> Dict[str, float]:
        return {"nominal": self.nominal, "observed": self.observed, "gap": self.gap, "n": self.n}


def calibration_curve(
    actual: Sequence[float],
    p50: Sequence[float],
    p90: Sequence[float],
    quantiles: Sequence[float],
    *,
    upper_alpha: float = 0.9,
) -> List[CalibrationPoint]:
    if not (len(actual) == len(p50) == len(p90)):
        raise ValueError("actual, p50 and p90 must have the same length")
    points: List[CalibrationPoint] = []
    n = len(actual)
    for tau in quantiles:
        if n == 0:
            points.append(CalibrationPoint(tau, float("nan"), 0))
            continue
        covered = sum(
            1 for a, lo, hi in zip(actual, p50, p90)
            if a <= implied_quantile(lo, hi, tau, upper_alpha=upper_alpha)
        )
        points.append(CalibrationPoint(tau, covered / n, n))
    return points


def calibration_error(points: Sequence[CalibrationPoint]) -> Dict[str, float]:
    """Mean and worst absolute gap between observed and nominal coverage."""
    gaps = [abs(p.gap) for p in points if p.n and not math.isnan(p.observed)]
    if not gaps:
        return {"mean_abs_gap": float("nan"), "max_abs_gap": float("nan")}
    return {"mean_abs_gap": sum(gaps) / len(gaps), "max_abs_gap": max(gaps)}


__all__ = [
    "inverse_normal_cdf",
    "implied_quantile",
    "CalibrationPoint",
    "calibration_curve",
    "calibration_error",
]
