"""Composing ``ETA = prep + travel + handoff`` at two quantiles.

Medians add (approximately — exactly for symmetric components, and well enough for
these). **Tails do not.** Adding three p90s claims that the kitchen is slow, the bridge
is jammed, *and* the lift is out, all at once, on one in ten orders. If the three were
perfectly correlated that would be right; they are not, and the naive sum produces a
p90 that holds on 97-98% of orders — an ETA so padded that customers learn to ignore it.

Each component's tail is summarised by its spread ``s_i = p90_i - p50_i``. Under a
Gaussian approximation ``s_i = z_0.9 * sigma_i``, and for a sum of components with
pairwise correlation ``rho``

    sigma_total^2 = sum_i sigma_i^2 + 2 * rho * sum_{i<j} sigma_i * sigma_j

so, since ``z_0.9`` scales every term equally,

    spread_total = sqrt( sum_i s_i^2 + 2 * rho * sum_{i<j} s_i * s_j )

At ``rho = 1`` this is exactly the naive sum of spreads; at ``rho = 0`` it is the
quadrature sum. The default (``service.component_tail_correlation = 0.35``) reflects
that prep and travel tails share a cause — a dinner rush is also a traffic rush — but
are far from the same event.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .types import EtaBreakdown, QuantilePair, ValidationError


def compose_quantiles(components: Sequence[QuantilePair], rho: float) -> QuantilePair:
    """Combine independent-ish components into one p50/p90 under correlation ``rho``."""
    if not 0.0 <= rho <= 1.0:
        raise ValidationError("tail correlation must be in [0, 1]")
    if not components:
        return QuantilePair.zero()
    p50 = sum(c.p50 for c in components)
    spreads = [c.spread for c in components]
    variance = sum(s * s for s in spreads)
    for i in range(len(spreads)):
        for j in range(i + 1, len(spreads)):
            variance += 2.0 * rho * spreads[i] * spreads[j]
    return QuantilePair(p50, p50 + math.sqrt(max(0.0, variance)))


def naive_sum(components: Sequence[QuantilePair]) -> QuantilePair:
    """The p90-of-a-sum-is-the-sum-of-p90s mistake, for comparison in reports."""
    return QuantilePair(sum(c.p50 for c in components), sum(c.p90 for c in components))


@dataclass(frozen=True)
class Decomposition:
    total: QuantilePair
    breakdown: EtaBreakdown
    naive_total: QuantilePair

    @property
    def tail_saving_s(self) -> float:
        """How much padding the correlation-aware composition removed."""
        return self.naive_total.p90 - self.total.p90


def decompose(
    *,
    prep: QuantilePair,
    travel: QuantilePair,
    handoff: QuantilePair,
    baseline_travel_s: float,
    residual_p50_s: float,
    residual_p90_s: float,
    rho: float,
) -> Decomposition:
    components = [c for c in (prep, travel, handoff)]
    total = compose_quantiles(components, rho)
    breakdown = EtaBreakdown(
        prep=prep,
        travel=travel,
        handoff=handoff,
        baseline_travel_s=baseline_travel_s,
        residual_correction_p50_s=residual_p50_s,
        residual_correction_p90_s=residual_p90_s,
    )
    return Decomposition(total=total, breakdown=breakdown, naive_total=naive_sum(components))


def share_of_total(decomposition: Decomposition) -> Dict[str, float]:
    """Fraction of the median ETA attributable to each component."""
    total = decomposition.total.p50
    if total <= 0:
        return {"prep": 0.0, "travel": 0.0, "handoff": 0.0}
    b = decomposition.breakdown
    return {
        "prep": b.prep.p50 / total,
        "travel": b.travel.p50 / total,
        "handoff": b.handoff.p50 / total,
    }


__all__ = [
    "compose_quantiles",
    "naive_sum",
    "decompose",
    "Decomposition",
    "share_of_total",
]
