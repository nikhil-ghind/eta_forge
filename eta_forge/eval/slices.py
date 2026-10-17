"""Slice metrics: accuracy broken down by when, where, how far and who.

A service can post a fine overall MAE while being systematically four minutes late in
Midtown at 18:00 and two minutes early on every airport run — errors that cancel in
the average and compound in customer trust. Every headline number is therefore also
reported per slice: hour of day, day of week, distance bucket, origin cell and
merchant. Slices smaller than ``eval.min_slice_size`` are dropped (and counted), since
a 6-trip slice's coverage is noise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import EvalConfig
from ..profiles.buckets import DAY_NAMES
from .metrics import quantile_metrics


@dataclass
class EvalRecord:
    """One held-out trip with the service's prediction and the baseline's."""

    trip_id: str
    departure_at: datetime
    actual_s: float
    baseline_s: float
    p50_s: float
    p90_s: float
    distance_m: float
    hour: int
    dow: int
    origin_cell: Optional[str] = None
    dest_cell: Optional[str] = None
    merchant_id: Optional[str] = None
    degraded: bool = False


SliceFn = Callable[[EvalRecord], Optional[str]]


def by_hour(record: EvalRecord) -> str:
    return f"{record.hour:02d}:00"


def by_dow(record: EvalRecord) -> str:
    return f"{record.dow}-{DAY_NAMES[record.dow % 7]}"


def by_origin_cell(record: EvalRecord) -> Optional[str]:
    return record.origin_cell


def by_dest_cell(record: EvalRecord) -> Optional[str]:
    return record.dest_cell


def by_merchant(record: EvalRecord) -> Optional[str]:
    return record.merchant_id


def distance_label(distance_m: float, buckets_m: Sequence[float]) -> str:
    edges = list(buckets_m)
    for i, edge in enumerate(edges):
        if distance_m < edge:
            low = 0.0 if i == 0 else edges[i - 1]
            if i == 0:
                return f"<{edge / 1000:g}km"
            return f"{low / 1000:g}-{edge / 1000:g}km"
    return f">={edges[-1] / 1000:g}km" if edges else "all"


def distance_slicer(buckets_m: Sequence[float]) -> SliceFn:
    def slice_fn(record: EvalRecord) -> str:
        return distance_label(record.distance_m, buckets_m)
    return slice_fn


def standard_slicers(cfg: EvalConfig) -> Dict[str, SliceFn]:
    return {
        "hour": by_hour,
        "dow": by_dow,
        "distance": distance_slicer(cfg.distance_buckets_m),
        "origin_cell": by_origin_cell,
        "merchant": by_merchant,
    }


@dataclass
class SliceMetrics:
    dimension: str
    key: str
    n: int
    metrics: Dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.metrics.get(name, float("nan"))

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"dimension": self.dimension, "key": self.key, "n": self.n}
        row.update(self.metrics)
        return row


def slice_metrics(
    records: Sequence[EvalRecord],
    dimension: str,
    slice_fn: SliceFn,
    *,
    min_size: int = 30,
    mape_floor_s: float = 60.0,
) -> Tuple[List[SliceMetrics], int]:
    """Metrics per slice key, sorted by key; returns ``(slices, dropped_slice_count)``."""
    groups: Dict[str, List[EvalRecord]] = defaultdict(list)
    for record in records:
        key = slice_fn(record)
        if key is not None:
            groups[key].append(record)
    out: List[SliceMetrics] = []
    dropped = 0
    for key in sorted(groups):
        members = groups[key]
        if len(members) < min_size:
            dropped += 1
            continue
        out.append(
            SliceMetrics(
                dimension=dimension,
                key=key,
                n=len(members),
                metrics=quantile_metrics(
                    [r.actual_s for r in members],
                    [r.p50_s for r in members],
                    [r.p90_s for r in members],
                    mape_floor_s=mape_floor_s,
                ),
            )
        )
    return out, dropped


def worst_slices(
    slices: Sequence[SliceMetrics], metric: str = "bias_s", *, n: int = 5, lowest: bool = True
) -> List[SliceMetrics]:
    """The ``n`` slices with the lowest (or highest) value of ``metric``.

    With the default ``bias_s`` and ``lowest=True`` these are where the service is most
    systematically *late* — the list to read first.
    """
    ranked = sorted(slices, key=lambda s: s.get(metric), reverse=not lowest)
    return list(ranked[:n])


__all__ = [
    "EvalRecord",
    "SliceMetrics",
    "SliceFn",
    "by_hour",
    "by_dow",
    "by_origin_cell",
    "by_dest_cell",
    "by_merchant",
    "distance_label",
    "distance_slicer",
    "standard_slicers",
    "slice_metrics",
    "worst_slices",
]
