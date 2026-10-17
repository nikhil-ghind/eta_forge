"""Turning a decoded match into per-segment traversal times.

Between two consecutive matched fixes the vehicle drove a known path (the one the
transition probability was computed over) in a known elapsed time. That time is spread
across the path's pieces in proportion to distance — the only defensible assumption
with nothing observed in between — and accumulated per segment *visit*.

Only **complete** visits become observations: the vehicle must be seen entering the
segment at its source vertex and leaving it at its target. The first and last segments
of every trip are therefore dropped, and that is deliberate: a pickup segment's
traversal includes the time spent pulling over and loading, and folding it into the
segment's profile makes every pickup street look gridlocked.

Time spent stationary (two fixes projecting onto the same position) is charged to the
segment the vehicle is standing on. A signal wait at the end of a block is part of
the cost of traversing that block, which is exactly how the router charges it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, List, Mapping, Optional, Sequence

from ..types import HighwayClass, SegmentObservation, ValidationError
from .hmm import MatchResult

# A piece that starts within this many metres of the segment's source vertex counts as
# entering it; likewise for leaving at the target.
ENDPOINT_TOLERANCE_M = 1.5
# Distance below which two consecutive fixes are treated as stationary.
STATIONARY_M = 0.5


@dataclass
class AttributedTraversal:
    observation: SegmentObservation
    confidence: float


@dataclass
class _Visit:
    segment_id: int
    length_m: float
    entered_at: Optional[datetime]
    saw_start: bool
    saw_end: bool = False
    covered_m: float = 0.0
    time_s: float = 0.0
    last_end_m: float = 0.0
    confidences: List[float] = field(default_factory=list)


def attribute_traversals(
    result: MatchResult,
    *,
    trip_id: Optional[str] = None,
    segment_classes: Optional[Mapping[int, HighwayClass]] = None,
    min_coverage: float = 0.9,
) -> List[AttributedTraversal]:
    """Complete segment traversals from a :class:`MatchResult`, in driving order."""
    classes = segment_classes or {}
    out: List[AttributedTraversal] = []
    current: Optional[_Visit] = None

    def finish(visit: Optional[_Visit]) -> None:
        if visit is None or not (visit.saw_start and visit.saw_end):
            return
        if visit.entered_at is None or visit.time_s <= 0:
            return
        if visit.covered_m < min_coverage * visit.length_m:
            return
        try:
            observation = SegmentObservation(
                segment_id=visit.segment_id,
                entered_at=visit.entered_at,
                traversal_s=visit.time_s,
                length_m=visit.length_m,
                highway_class=classes.get(visit.segment_id, HighwayClass.UNCLASSIFIED),
                trip_id=trip_id,
            )
        except ValidationError:
            return
        confidence = min(visit.confidences) if visit.confidences else 0.0
        out.append(AttributedTraversal(observation, confidence))

    for k in range(len(result.points) - 1):
        a, b = result.points[k], result.points[k + 1]
        path = result.paths[k] if k < len(result.paths) else None
        if path is None:
            # A lattice break: nothing is known about what happened in the gap, so the
            # visit in progress can never be proven complete.
            current = None
            continue
        elapsed = (b.point.at - a.point.at).total_seconds()
        if elapsed <= 0:
            continue
        pair_confidence = min(a.confidence, b.confidence)

        if path.distance_m < STATIONARY_M:
            seg = a.candidate
            if current is None or current.segment_id != seg.segment_id:
                finish(current)
                current = _Visit(
                    segment_id=seg.segment_id,
                    length_m=seg.length_m,
                    entered_at=None,
                    saw_start=False,
                    last_end_m=seg.offset_m,
                )
            current.time_s += elapsed
            current.confidences.append(pair_confidence)
            continue

        travelled = 0.0
        for piece in path.pieces:
            share = piece.covered_m / path.distance_m
            starts_at = a.point.at + timedelta(seconds=elapsed * travelled / path.distance_m)
            re_entry = (
                current is not None
                and current.segment_id == piece.segment_id
                and piece.start_m + ENDPOINT_TOLERANCE_M < current.last_end_m
            )
            if current is None or current.segment_id != piece.segment_id or re_entry:
                finish(current)
                entering = piece.start_m <= ENDPOINT_TOLERANCE_M
                current = _Visit(
                    segment_id=piece.segment_id,
                    length_m=piece.length_m,
                    entered_at=starts_at if entering else None,
                    saw_start=entering,
                    last_end_m=piece.start_m,
                )
            current.time_s += elapsed * share
            current.covered_m += piece.covered_m
            current.last_end_m = max(current.last_end_m, piece.end_m)
            current.confidences.append(pair_confidence)
            if piece.end_m >= piece.length_m - ENDPOINT_TOLERANCE_M:
                current.saw_end = True
            travelled += piece.covered_m

    # The trailing visit is the dropoff segment; it is only complete if the trace
    # happened to end exactly at a vertex, which `finish` checks.
    finish(current)
    return out


TRIP_SEGMENT_INSERT_SQL = """
INSERT INTO trip_segment (
    trip_id, seq, segment_id, entered_at, traversal_s, length_m, match_confidence
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (trip_id, seq) DO UPDATE SET
    segment_id       = EXCLUDED.segment_id,
    entered_at       = EXCLUDED.entered_at,
    traversal_s      = EXCLUDED.traversal_s,
    length_m         = EXCLUDED.length_m,
    match_confidence = EXCLUDED.match_confidence
"""


def trip_segment_rows(
    trip_id: str, traversals: Sequence[AttributedTraversal]
) -> List[Sequence[Any]]:
    return [
        (
            trip_id,
            seq,
            t.observation.segment_id,
            t.observation.entered_at,
            t.observation.traversal_s,
            t.observation.length_m,
            round(t.confidence, 6),
        )
        for seq, t in enumerate(traversals)
    ]


__all__ = [
    "AttributedTraversal",
    "attribute_traversals",
    "trip_segment_rows",
    "TRIP_SEGMENT_INSERT_SQL",
    "ENDPOINT_TOLERANCE_M",
]
