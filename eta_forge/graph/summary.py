"""Route summarisation: turning a path into the numbers a learner can use.

The residual model never sees a geometry. It sees this: how far, how long the baseline
thinks it takes, what kind of roads, how many manoeuvres, how many signals, and how
much of the baseline was built on real observations rather than a class default.

That last one is the interesting feature. A baseline with 5% profile coverage is a
guess and its residual is large and noisy; a baseline with 90% coverage is close and
its residual is a small correction. Handing the model the coverage lets it learn to
distrust the baseline exactly where the baseline deserves it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from ..types import BaselineRoute, HighwayClass, RouteLeg
from .geometry import bearing_deg, haversine_m
from .turns import TurnProfile, class_mix, route_turn_profile


@dataclass(frozen=True)
class RouteSummary:
    """Structural description of a baseline route."""

    distance_m: float
    baseline_duration_s: float
    mean_speed_kph: float
    leg_count: int
    class_mix: Mapping[str, float]
    distinct_classes: int
    turns: TurnProfile
    crow_distance_m: float
    detour_ratio: float
    od_bearing_deg: float
    profile_coverage: float
    snap_distance_m: float

    def to_features(self) -> Dict[str, float]:
        """Flat name -> value map, consumed by the feature builder.

        Names are stable and are frozen into the FeatureSpec; renaming one here
        without bumping the spec hash is what makes a served model silently read the
        wrong column.
        """
        out: Dict[str, float] = {
            "distance_m": self.distance_m,
            "baseline_duration_s": self.baseline_duration_s,
            "baseline_mean_speed_kph": self.mean_speed_kph,
            "leg_count": float(self.leg_count),
            "distinct_classes": float(self.distinct_classes),
            "crow_distance_m": self.crow_distance_m,
            "detour_ratio": self.detour_ratio,
            "od_bearing_deg": self.od_bearing_deg,
            "profile_coverage": self.profile_coverage,
            "snap_distance_m": self.snap_distance_m,
        }
        for cls in HighwayClass:
            out[f"class_frac_{cls.value}"] = float(self.class_mix.get(cls.value, 0.0))
        out.update(self.turns.to_dict())
        return out


def summarize_route(
    route: BaselineRoute,
    *,
    turn_penalties: Optional[Mapping[str, float]] = None,
    signal_penalty_s: float = 8.0,
) -> RouteSummary:
    """Summarise a route into features."""
    legs: Sequence[RouteLeg] = route.legs
    distance = route.distance_m
    duration = route.duration_s
    crow = haversine_m(route.origin, route.destination)

    mix = class_mix(legs)
    turns = route_turn_profile(
        legs, penalties=turn_penalties, signal_penalty_s=signal_penalty_s
    )

    # detour_ratio < 1 is geometrically impossible for a real path and signals that
    # snapping moved an endpoint further than the trip itself. Clamping at 1.0 keeps
    # a nonsense value out of the feature vector.
    detour = max(1.0, distance / crow) if crow > 1.0 else 1.0

    return RouteSummary(
        distance_m=distance,
        baseline_duration_s=duration,
        mean_speed_kph=route.mean_speed_kph,
        leg_count=len(legs),
        class_mix=mix,
        distinct_classes=sum(1 for v in mix.values() if v > 0.0),
        turns=turns,
        crow_distance_m=crow,
        detour_ratio=detour,
        od_bearing_deg=bearing_deg(route.origin, route.destination),
        profile_coverage=route.profile_coverage,
        snap_distance_m=(
            route.snapped_origin_distance_m + route.snapped_destination_distance_m
        ),
    )


__all__ = ["RouteSummary", "summarize_route"]
