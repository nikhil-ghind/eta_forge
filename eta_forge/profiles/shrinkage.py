"""Hierarchical shrinkage of sparse profile cells.

A ``(segment, Tuesday, 03:00)`` cell with three observations is not evidence. Used raw
it will confidently tell the router that this block moves at 71 km/h because one
vehicle had a clear run at 3am, and the router will happily send every request through
it forever.

The fix is empirical-Bayes shrinkage toward progressively broader priors:

    cell  ->  same segment, all times
          ->  same highway class, same time cell
          ->  same highway class, all times
          ->  the configured class default

The three ``k`` values in ``ProfilesConfig`` map onto the levels as: ``shrink_k_segment``
for cell -> segment (and adjacent hours), ``shrink_k_class_tod`` for segment -> class,
and ``shrink_k_global`` for the class levels -> the configured default.

Each level uses the standard credibility weight ``w = n / (n + k)``. With ``n = 0`` the
estimate is exactly the prior; with ``n >> k`` it is exactly the observation; ``k`` is
the number of observations at which the two are weighted equally, which makes it a
directly interpretable knob rather than an abstract regularisation strength.

The shrinkage is applied in *speed* space rather than in duration space. Duration is
1/speed, and shrinking a reciprocal introduces a bias (Jensen's inequality) that
systematically over-predicts on slow segments — the ones that matter most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from ..config import ProfilesConfig
from ..types import HighwayClass
from .buckets import BucketSpec, iter_cells, neighbouring_cells
from .mining import CellStats, SegmentProfile, class_stats, class_tod_stats


def credibility_weight(n: float, k: float) -> float:
    """``n / (n + k)``, clamped into [0, 1]."""
    if k <= 0:
        raise ValueError("shrinkage k must be > 0")
    if n <= 0:
        return 0.0
    return n / (n + k)


def shrink_cell(observation: float, prior: float, n: float, k: float) -> float:
    """One shrinkage step: blend an observation with a prior by credibility."""
    w = credibility_weight(n, k)
    return w * observation + (1.0 - w) * prior


@dataclass
class PriorHierarchy:
    """The class-level priors a cell falls back through.

    Each level carries its sample count, not just its mean, because the class levels
    are themselves shrunk: ``service`` roads at 04:00 on a Sunday may be backed by
    eleven traversals city-wide, and that mean deserves no more trust than a sparse
    segment cell does.
    """

    class_tod: Mapping[Tuple[str, int, int], Tuple[float, int]] = field(default_factory=dict)
    class_all: Mapping[str, Tuple[float, int]] = field(default_factory=dict)
    class_default: Mapping[str, float] = field(default_factory=dict)
    global_default_kph: float = 30.0
    k_global: float = 80.0

    @classmethod
    def build(
        cls,
        profiles: Mapping[int, SegmentProfile],
        spec: BucketSpec,
        class_default: Mapping[str, float],
        *,
        k_global: float = 80.0,
    ) -> "PriorHierarchy":
        return cls(
            class_tod=class_tod_stats(profiles, spec),
            class_all=class_stats(profiles),
            class_default=dict(class_default),
            global_default_kph=float(class_default.get("unclassified", 30.0)),
            k_global=k_global,
        )

    def default_for(self, highway_class: HighwayClass) -> float:
        value = self.class_default.get(highway_class.value)
        if value is not None and value > 0:
            return float(value)
        return self.global_default_kph

    def class_prior(self, highway_class: HighwayClass) -> float:
        """The class's all-time observed mean, shrunk toward the configured default."""
        default = self.default_for(highway_class)
        observed = self.class_all.get(highway_class.value)
        if observed is None:
            return default
        mean, n = observed
        return shrink_cell(mean, default, n, self.k_global)

    def base_prior(self, highway_class: HighwayClass, dow: int, tod: int) -> float:
        """The broadest defensible estimate for a cell with no segment data at all.

        ``(class, dow, tod)`` shrunk toward ``class`` shrunk toward the default. With no
        observations anywhere in the class this returns the configured default exactly.
        """
        class_level = self.class_prior(highway_class)
        observed = self.class_tod.get((highway_class.value, dow, tod))
        if observed is None:
            return class_level
        mean, n = observed
        return shrink_cell(mean, class_level, n, self.k_global)


@dataclass
class SmoothedCell:
    """A cell after shrinkage, with the provenance kept for auditing."""

    segment_id: int
    dow_bucket: int
    tod_bucket: int
    speed_kph: float
    raw_speed_kph: Optional[float]
    sample_count: int
    effective_sample_count: float
    prior_kph: float
    shrink_weight: float
    spread_kph: float
    source: str  # observed | neighbour | segment | class | default

    def to_row(self) -> Tuple:
        return (
            self.segment_id,
            self.dow_bucket,
            self.tod_bucket,
            self.speed_kph,
            self.sample_count,
            self.effective_sample_count,
            self.prior_kph,
            self.shrink_weight,
            self.spread_kph,
            self.source,
        )


def _neighbour_estimate(
    profile: SegmentProfile, dow: int, tod: int, spec: BucketSpec
) -> Tuple[Optional[float], float]:
    """Harmonic-mean speed of the adjacent time cells on the same segment.

    Returns ``(speed_kph, sample_count)``. Adjacent hours on the same segment are the
    single most informative prior available — far better than a class average — so they
    are consulted before the hierarchy proper.
    """
    total_m = 0.0
    total_s = 0.0
    count = 0
    for ndow, ntod in neighbouring_cells(dow, tod, spec):
        cell = profile.cell(ndow, ntod)
        if cell is None:
            continue
        total_m += cell.total_length_m
        total_s += cell.total_traversal_s
        count += cell.sample_count
    if total_m <= 0 or total_s <= 0:
        return None, 0.0
    return (total_m / total_s) * 3.6, float(count)


def shrink_profiles(
    profiles: Mapping[int, SegmentProfile],
    cfg: ProfilesConfig,
    *,
    spec: Optional[BucketSpec] = None,
    class_default: Optional[Mapping[str, float]] = None,
    fill_empty_cells: bool = True,
    all_segments: Optional[Mapping[int, HighwayClass]] = None,
) -> Dict[int, Dict[Tuple[int, int], SmoothedCell]]:
    """Smooth every cell of every segment through the prior hierarchy.

    With ``fill_empty_cells`` the output is dense: every segment gets a value in every
    cell of the grid, sourced from the priors where nothing was observed. A dense table
    is what lets the routing cost expression be a single indexed lookup with no
    conditional fallback logic in the hot path.

    ``all_segments`` (segment id -> class, typically every row of ``segment_attr``)
    extends the density to segments no trip ever traversed: they get the class prior
    for each cell, tagged ``class`` when the class had observations and ``default``
    when even the class did not.
    """
    spec = spec or BucketSpec.from_config(cfg)
    defaults = dict(class_default or {})
    hierarchy = PriorHierarchy.build(
        profiles, spec, defaults, k_global=cfg.shrink_k_global
    )

    out: Dict[int, Dict[Tuple[int, int], SmoothedCell]] = {}

    for segment_id, profile in profiles.items():
        smoothed: Dict[Tuple[int, int], SmoothedCell] = {}
        cells = iter_cells(spec) if fill_empty_cells else list(profile.cells)

        for dow, tod in cells:
            cell: Optional[CellStats] = profile.cell(dow, tod)
            base_prior = hierarchy.base_prior(profile.highway_class, dow, tod)

            # Level 3 -> 2: the segment's own all-time mean, shrunk toward the class
            # priors. A segment that is reliably slow all week should stay slow in a
            # cell where nothing was seen.
            segment_prior = base_prior
            if profile.segment_mean_kph is not None:
                segment_prior = shrink_cell(
                    profile.segment_mean_kph,
                    base_prior,
                    profile.segment_sample_count,
                    cfg.shrink_k_class_tod,
                )

            # Level 2 -> 1: adjacent time cells on this segment.
            neighbour_speed, neighbour_n = _neighbour_estimate(profile, dow, tod, spec)
            prior = segment_prior
            if neighbour_speed is not None:
                prior = shrink_cell(
                    neighbour_speed, segment_prior, neighbour_n, cfg.shrink_k_segment
                )

            if cell is None:
                # The effective sample count of a filled cell is the evidence it
                # borrowed: the adjacent hours' traversals, else the segment's own
                # all-time traversals, else nothing. The router trusts a cell by this
                # number, so a cell filled from a well-observed neighbour hour is
                # routable while one filled from a class average is not.
                if neighbour_speed is not None:
                    source, borrowed = "neighbour", neighbour_n
                elif profile.segment_mean_kph is not None:
                    source = "segment"
                    borrowed = float(profile.segment_sample_count)
                else:
                    source, borrowed = "class", 0.0
                smoothed[(dow, tod)] = SmoothedCell(
                    segment_id=segment_id,
                    dow_bucket=dow,
                    tod_bucket=tod,
                    speed_kph=_clamp(prior, cfg),
                    raw_speed_kph=None,
                    sample_count=0,
                    effective_sample_count=borrowed,
                    prior_kph=prior,
                    shrink_weight=0.0,
                    spread_kph=0.0,
                    source=source,
                )
                continue

            # Level 1: the observation itself.
            weight = credibility_weight(cell.sample_count, cfg.shrink_k_segment)
            speed = shrink_cell(
                cell.speed_kph, prior, cell.sample_count, cfg.shrink_k_segment
            )
            smoothed[(dow, tod)] = SmoothedCell(
                segment_id=segment_id,
                dow_bucket=dow,
                tod_bucket=tod,
                speed_kph=_clamp(speed, cfg),
                raw_speed_kph=cell.speed_kph,
                sample_count=cell.sample_count,
                effective_sample_count=float(cell.sample_count) + neighbour_n,
                prior_kph=prior,
                shrink_weight=weight,
                spread_kph=cell.spread_kph,
                source="observed",
            )
        out[segment_id] = smoothed

    if all_segments and fill_empty_cells:
        for segment_id, highway_class in all_segments.items():
            if segment_id in out:
                continue
            observed_class = highway_class.value in hierarchy.class_all
            cells_out: Dict[Tuple[int, int], SmoothedCell] = {}
            for dow, tod in iter_cells(spec):
                prior = hierarchy.base_prior(highway_class, dow, tod)
                cells_out[(dow, tod)] = SmoothedCell(
                    segment_id=segment_id,
                    dow_bucket=dow,
                    tod_bucket=tod,
                    speed_kph=_clamp(prior, cfg),
                    raw_speed_kph=None,
                    sample_count=0,
                    effective_sample_count=0.0,
                    prior_kph=prior,
                    shrink_weight=0.0,
                    spread_kph=0.0,
                    source="class" if observed_class else "default",
                )
            out[segment_id] = cells_out
    return out


def _clamp(speed: float, cfg: ProfilesConfig) -> float:
    return min(cfg.max_speed_kph, max(cfg.min_speed_kph, speed))


__all__ = [
    "PriorHierarchy",
    "SmoothedCell",
    "credibility_weight",
    "shrink_cell",
    "shrink_profiles",
]
