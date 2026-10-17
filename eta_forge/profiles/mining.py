"""Mining segment speed profiles from map-matched traversals.

The one piece of arithmetic that has to be right: **speeds over a fixed distance
average harmonically, not arithmetically.**

Drive 1 km at 60 km/h and 1 km at 20 km/h. The arithmetic mean says 40 km/h, which
implies 3 minutes for the 2 km. The trip actually took 1 + 3 = 4 minutes, an average of
30 km/h — the harmonic mean. Since the whole purpose of a profile is to be divided into
a length to produce a duration, using the arithmetic mean builds a systematic optimism
into every baseline, worst exactly where the spread is widest, which is rush hour.

Equivalently and more robustly, the harmonic mean of speeds over a fixed distance is
the distance divided by the *total time*, which is how it is computed here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import ProfilesConfig
from ..types import HighwayClass, SegmentObservation
from .buckets import BucketSpec, bucket_of


def harmonic_mean_speed(
    observations: Sequence[Tuple[float, float]]
) -> Optional[float]:
    """Harmonic-mean speed in km/h from ``(length_m, traversal_s)`` pairs.

    Returns ``None`` for an empty or degenerate input rather than a fabricated zero.
    """
    total_m = 0.0
    total_s = 0.0
    for length_m, traversal_s in observations:
        if length_m <= 0 or traversal_s <= 0:
            continue
        total_m += length_m
        total_s += traversal_s
    if total_m <= 0 or total_s <= 0:
        return None
    return (total_m / total_s) * 3.6


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in [0, 1]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class CellStats:
    """One ``(segment, dow, tod)`` cell of the profile grid."""

    segment_id: int
    dow_bucket: int
    tod_bucket: int
    speed_kph: float
    sample_count: int
    p15_kph: float
    p50_kph: float
    p85_kph: float
    total_length_m: float = 0.0
    total_traversal_s: float = 0.0
    highway_class: HighwayClass = HighwayClass.UNCLASSIFIED

    @property
    def spread_kph(self) -> float:
        """p85 - p15: how unreliable this segment is in this hour.

        Fed to the residual model as a feature, because a segment with a 30 km/h
        spread deserves a wider p90 than one that is reliably slow.
        """
        return self.p85_kph - self.p15_kph

    def to_row(self) -> Tuple:
        return (
            self.segment_id,
            self.dow_bucket,
            self.tod_bucket,
            self.speed_kph,
            self.sample_count,
            self.p15_kph,
            self.p50_kph,
            self.p85_kph,
            self.total_length_m,
            self.total_traversal_s,
        )


@dataclass
class SegmentProfile:
    """Every mined cell for one segment, plus the segment-level aggregate."""

    segment_id: int
    highway_class: HighwayClass
    cells: Dict[Tuple[int, int], CellStats] = field(default_factory=dict)
    segment_mean_kph: Optional[float] = None
    segment_sample_count: int = 0

    def cell(self, dow: int, tod: int) -> Optional[CellStats]:
        return self.cells.get((dow, tod))


@dataclass
class MiningReport:
    """What mining saw and what it threw away."""

    observations_read: int = 0
    observations_kept: int = 0
    segments: int = 0
    cells: int = 0
    cells_below_threshold: int = 0
    rejected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> Dict[str, object]:
        return {
            "observations_read": self.observations_read,
            "observations_kept": self.observations_kept,
            "segments": self.segments,
            "cells": self.cells,
            "cells_below_threshold": self.cells_below_threshold,
            "rejected": dict(self.rejected),
        }


def mine_profiles(
    observations: Iterable[SegmentObservation],
    cfg: ProfilesConfig,
    *,
    spec: Optional[BucketSpec] = None,
    segment_classes: Optional[Mapping[int, HighwayClass]] = None,
) -> Tuple[Dict[int, SegmentProfile], MiningReport]:
    """Aggregate traversals into per-segment, per-cell speed statistics.

    Observations outside ``[min_speed_kph, max_speed_kph]`` are dropped before
    aggregation. That filter is doing real work: a map-matched traversal at 0.4 km/h is
    a vehicle that parked mid-segment, and one at 180 km/h is a matching error. Either
    one, harmonically averaged, moves a cell a long way.
    """
    spec = spec or BucketSpec.from_config(cfg)
    report = MiningReport()

    # (segment, dow, tod) -> list of (length_m, traversal_s, speed_kph)
    buckets: Dict[Tuple[int, int, int], List[Tuple[float, float, float]]] = defaultdict(list)
    classes: Dict[int, HighwayClass] = dict(segment_classes or {})

    for obs in observations:
        report.observations_read += 1
        speed = obs.speed_kph
        if speed < cfg.min_speed_kph:
            report.rejected["too_slow"] += 1
            continue
        if speed > cfg.max_speed_kph:
            report.rejected["too_fast"] += 1
            continue
        dow, tod = bucket_of(obs.entered_at, spec)
        buckets[(obs.segment_id, dow, tod)].append(
            (obs.length_m, obs.traversal_s, speed)
        )
        classes.setdefault(obs.segment_id, obs.highway_class)
        report.observations_kept += 1

    profiles: Dict[int, SegmentProfile] = {}
    segment_totals: Dict[int, Tuple[float, float, int]] = defaultdict(
        lambda: (0.0, 0.0, 0)
    )

    for (segment_id, dow, tod), samples in buckets.items():
        speeds = [s for _, _, s in samples]
        pairs = [(length, traversal) for length, traversal, _ in samples]
        mean = harmonic_mean_speed(pairs)
        if mean is None:  # pragma: no cover - filtered above
            continue
        total_m = sum(length for length, _ in pairs)
        total_s = sum(traversal for _, traversal in pairs)

        stats = CellStats(
            segment_id=segment_id,
            dow_bucket=dow,
            tod_bucket=tod,
            speed_kph=mean,
            sample_count=len(samples),
            p15_kph=percentile(speeds, 0.15),
            p50_kph=percentile(speeds, 0.50),
            p85_kph=percentile(speeds, 0.85),
            total_length_m=total_m,
            total_traversal_s=total_s,
            highway_class=classes.get(segment_id, HighwayClass.UNCLASSIFIED),
        )
        profile = profiles.setdefault(
            segment_id,
            SegmentProfile(
                segment_id=segment_id,
                highway_class=classes.get(segment_id, HighwayClass.UNCLASSIFIED),
            ),
        )
        profile.cells[(dow, tod)] = stats
        report.cells += 1
        if stats.sample_count < cfg.min_samples_per_cell:
            report.cells_below_threshold += 1

        prev_m, prev_s, prev_n = segment_totals[segment_id]
        segment_totals[segment_id] = (
            prev_m + total_m,
            prev_s + total_s,
            prev_n + len(samples),
        )

    for segment_id, profile in profiles.items():
        total_m, total_s, count = segment_totals[segment_id]
        profile.segment_mean_kph = (
            (total_m / total_s) * 3.6 if total_m > 0 and total_s > 0 else None
        )
        profile.segment_sample_count = count

    report.segments = len(profiles)
    return profiles, report


def _class_totals(
    profiles: Mapping[int, SegmentProfile], *, by_cell: bool
) -> Dict[Tuple, Tuple[float, float, int]]:
    """``key -> (total_m, total_s, samples)`` per class, optionally per time cell.

    Totals rather than means are accumulated so that the harmonic mean comes out as
    distance over time across *every* traversal in the group, not as a mean of
    per-segment means that would weight a 3-sample segment like a 3000-sample one.
    """
    totals: Dict[Tuple, Tuple[float, float, int]] = defaultdict(lambda: (0.0, 0.0, 0))
    for profile in profiles.values():
        key_class = profile.highway_class.value
        for (dow, tod), cell in profile.cells.items():
            key = (key_class, dow, tod) if by_cell else (key_class,)
            m, s, n = totals[key]
            totals[key] = (
                m + cell.total_length_m,
                s + cell.total_traversal_s,
                n + cell.sample_count,
            )
    return totals


def class_tod_stats(
    profiles: Mapping[int, SegmentProfile], spec: BucketSpec
) -> Dict[Tuple[str, int, int], Tuple[float, int]]:
    """``(highway_class, dow, tod) -> (harmonic mean kph, sample count)``."""
    return {
        key: ((m / s) * 3.6, n)
        for key, (m, s, n) in _class_totals(profiles, by_cell=True).items()
        if m > 0 and s > 0
    }


def class_stats(profiles: Mapping[int, SegmentProfile]) -> Dict[str, Tuple[float, int]]:
    """``highway_class -> (harmonic mean kph, sample count)`` across all time cells."""
    return {
        key[0]: ((m / s) * 3.6, n)
        for key, (m, s, n) in _class_totals(profiles, by_cell=False).items()
        if m > 0 and s > 0
    }


def class_tod_means(
    profiles: Mapping[int, SegmentProfile], spec: BucketSpec
) -> Dict[Tuple[str, int, int], float]:
    """Harmonic mean speed per ``(highway_class, dow, tod)``.

    The second level of the shrinkage hierarchy: a sparse cell on a residential street
    at 08:00 should first borrow from *residential streets at 08:00*, not from a single
    global number that averages a motorway at 03:00 into it.
    """
    return {key: mean for key, (mean, _) in class_tod_stats(profiles, spec).items()}


def class_means(profiles: Mapping[int, SegmentProfile]) -> Dict[str, float]:
    """Harmonic mean speed per highway class across all time cells."""
    return {key: mean for key, (mean, _) in class_stats(profiles).items()}


__all__ = [
    "CellStats",
    "SegmentProfile",
    "MiningReport",
    "mine_profiles",
    "harmonic_mean_speed",
    "percentile",
    "class_tod_means",
    "class_means",
    "class_tod_stats",
    "class_stats",
]
