"""Manoeuvre classification and turn cost.

A router that only sums edge traversal times systematically under-predicts on
dense urban routes, because the time is not on the edges — it is at the nodes. A left
turn across two lanes of oncoming traffic at an unprotected intersection costs more
than the 30 metres of road it covers, and a route with fourteen of them is not the
same trip as a route of the same length with two.

The classification is deliberately coarse and rule-based rather than learned: it feeds
*features* to the residual model, which then learns what those manoeuvres actually cost
in this city at this hour. Learning the classification too would just move the same
information around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from ..types import HighwayClass, RouteLeg, TurnType
from .geometry import bearing_delta_deg

# Degree thresholds on the absolute bearing delta. The straight band is generous
# because urban blocks are not perfectly gridded and a 15-degree drift down a curving
# avenue is not a manoeuvre.
STRAIGHT_MAX_DEG = 20.0
SLIGHT_MAX_DEG = 45.0
TURN_MAX_DEG = 120.0
SHARP_MAX_DEG = 160.0
# Beyond SHARP_MAX_DEG it is a u-turn.

DEFAULT_TURN_PENALTY_S: Dict[TurnType, float] = {
    TurnType.STRAIGHT: 0.0,
    TurnType.SLIGHT_LEFT: 1.5,
    TurnType.SLIGHT_RIGHT: 1.0,
    TurnType.LEFT: 12.0,
    TurnType.RIGHT: 5.0,
    TurnType.SHARP_LEFT: 16.0,
    TurnType.SHARP_RIGHT: 9.0,
    TurnType.UTURN: 25.0,
}


def classify_turn(incoming_bearing: float, outgoing_bearing: float) -> TurnType:
    """Classify the manoeuvre between two headings.

    Right-hand traffic is assumed (this is New York): a positive delta is a right
    turn, which is the cheap one because it does not cross oncoming traffic.
    """
    delta = bearing_delta_deg(incoming_bearing, outgoing_bearing)
    magnitude = abs(delta)
    if magnitude <= STRAIGHT_MAX_DEG:
        return TurnType.STRAIGHT
    right = delta > 0
    if magnitude <= SLIGHT_MAX_DEG:
        return TurnType.SLIGHT_RIGHT if right else TurnType.SLIGHT_LEFT
    if magnitude <= TURN_MAX_DEG:
        return TurnType.RIGHT if right else TurnType.LEFT
    if magnitude <= SHARP_MAX_DEG:
        return TurnType.SHARP_RIGHT if right else TurnType.SHARP_LEFT
    return TurnType.UTURN


def turn_penalty_s(
    turn: TurnType, penalties: Optional[Mapping[str, float]] = None
) -> float:
    """Seconds charged for a manoeuvre, from config or the built-in defaults."""
    if penalties is None:
        return DEFAULT_TURN_PENALTY_S[turn]
    return float(penalties.get(turn.value, DEFAULT_TURN_PENALTY_S[turn]))


@dataclass(frozen=True)
class TurnProfile:
    """Everything about the manoeuvres on a route, as model features."""

    counts: Mapping[str, int]
    left_turns: int
    right_turns: int
    major_turns: int
    uturns: int
    signal_count: int
    class_changes: int
    total_penalty_s: float
    max_turn_deg: float

    @property
    def total_turns(self) -> int:
        return sum(v for k, v in self.counts.items() if k != TurnType.STRAIGHT.value)

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {f"turn_{k}": float(v) for k, v in self.counts.items()}
        out.update(
            {
                "turn_left_total": float(self.left_turns),
                "turn_right_total": float(self.right_turns),
                "turn_major_total": float(self.major_turns),
                "turn_uturn_total": float(self.uturns),
                "signal_count": float(self.signal_count),
                "class_changes": float(self.class_changes),
                "turn_penalty_s": self.total_penalty_s,
                "max_turn_deg": self.max_turn_deg,
            }
        )
        return out


def route_turn_profile(
    legs: Sequence[RouteLeg],
    *,
    penalties: Optional[Mapping[str, float]] = None,
    signal_penalty_s: float = 8.0,
) -> TurnProfile:
    """Walk a route's legs and summarise every manoeuvre between them.

    The delta compared is leg *i*'s exit bearing against leg *i+1*'s entry bearing —
    not chord to chord. On a curved segment those are materially different, and using
    the chord makes a gentle merge look like a turn.
    """
    counts: Dict[str, int] = {t.value: 0 for t in TurnType}
    left = right = major = uturns = 0
    penalty = 0.0
    max_delta = 0.0
    class_changes = 0

    for i in range(len(legs) - 1):
        current, following = legs[i], legs[i + 1]
        delta = bearing_delta_deg(current.bearing_out_deg, following.bearing_in_deg)
        max_delta = max(max_delta, abs(delta))
        turn = classify_turn(current.bearing_out_deg, following.bearing_in_deg)
        counts[turn.value] += 1
        penalty += turn_penalty_s(turn, penalties)
        if turn.is_left:
            left += 1
        if turn.is_right:
            right += 1
        if turn.is_major:
            major += 1
        if turn is TurnType.UTURN:
            uturns += 1
        if current.highway_class is not following.highway_class:
            class_changes += 1

    signals = sum(1 for leg in legs if leg.has_signal_at_end)
    penalty += signals * signal_penalty_s

    return TurnProfile(
        counts=counts,
        left_turns=left,
        right_turns=right,
        major_turns=major,
        uturns=uturns,
        signal_count=signals,
        class_changes=class_changes,
        total_penalty_s=penalty,
        max_turn_deg=max_delta,
    )


def class_mix(legs: Sequence[RouteLeg]) -> Dict[str, float]:
    """Fraction of route *distance* on each highway class.

    Distance-weighted, not leg-count-weighted: fifty 20-metre service-road segments
    should not outweigh one 3-kilometre stretch of motorway when describing what kind
    of trip this is.
    """
    total = sum(leg.length_m for leg in legs)
    mix = {cls.value: 0.0 for cls in HighwayClass}
    if total <= 0:
        return mix
    for leg in legs:
        mix[leg.highway_class.value] += leg.length_m
    return {k: v / total for k, v in mix.items()}


__all__ = [
    "classify_turn",
    "turn_penalty_s",
    "route_turn_profile",
    "class_mix",
    "TurnProfile",
    "DEFAULT_TURN_PENALTY_S",
    "STRAIGHT_MAX_DEG",
    "SLIGHT_MAX_DEG",
    "TURN_MAX_DEG",
    "SHARP_MAX_DEG",
]
