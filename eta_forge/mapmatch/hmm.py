"""Hidden-Markov-model map matching (Newson & Krumm, 2009).

Hidden states are candidate positions on road segments; observations are GPS fixes.

**Emission** — how plausible is it that a vehicle *at* this candidate produced this
fix? GPS error is roughly Gaussian in the plane, so the log-likelihood falls off with
the square of the perpendicular distance:

    log p(z | c) = -0.5 * (d / sigma)^2 - log(sqrt(2 pi) * sigma)

**Transition** — how plausible is it to drive from candidate ``a`` to candidate ``b``
between two fixes? Newson & Krumm's key empirical observation: for a correct pair, the
*driven* network distance is very close to the great-circle distance between the
fixes, and the difference is exponentially distributed:

    log p(b | a) = -|route_m - crow_m| / beta - log(beta)

That one term is what stops a matcher from hopping onto a parallel street or an
overpass that happens to be closer to a noisy fix: getting there and back would
require a long detour, and the transition probability punishes it.

**Decoding** is Viterbi in log space. When every transition into a step is impossible
(a gap in the trace, a tunnel, a fix teleported 2 km), the lattice is *broken* there
and decoding restarts, rather than forcing a nonsense path through the gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from ..graph.geometry import haversine_m
from ..types import GpsPoint
from .candidates import Candidate, CandidateSource, PathBetween

NEG_INF = float("-inf")


@dataclass(frozen=True)
class HmmParams:
    # GPS noise standard deviation. Newson & Krumm measured 4.07 m for their data;
    # urban-canyon phone GPS in Manhattan is considerably worse.
    sigma_m: float = 8.0
    # Transition scale: how much driven-minus-crow distance is "normal".
    beta_m: float = 6.0
    search_radius_m: float = 45.0
    max_candidates: int = 8
    # A transition whose driven distance exceeds this multiple of the crow distance
    # (plus a floor) is not searched at all — it is not a plausible step.
    max_route_factor: float = 4.0
    max_route_floor_m: float = 300.0
    # Fixes closer than 2 sigma to the previous kept fix carry no new information and
    # only add lattice width (Newson & Krumm, section 4).
    min_spacing_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.sigma_m <= 0 or self.beta_m <= 0:
            raise ValueError("sigma_m and beta_m must be > 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")


def emission_logp(distance_m: float, sigma_m: float) -> float:
    """Log-density of a fix at ``distance_m`` from its candidate."""
    return -0.5 * (distance_m / sigma_m) ** 2 - math.log(math.sqrt(2.0 * math.pi) * sigma_m)


def transition_logp(route_m: float, crow_m: float, beta_m: float) -> float:
    """Log-density of driving ``route_m`` between fixes ``crow_m`` apart."""
    return -abs(route_m - crow_m) / beta_m - math.log(beta_m)


def logsumexp(values: Sequence[float]) -> float:
    finite = [v for v in values if v != NEG_INF]
    if not finite:
        return NEG_INF
    top = max(finite)
    return top + math.log(sum(math.exp(v - top) for v in finite))


def viterbi(
    emissions: Sequence[Sequence[float]],
    transition: Callable[[int, int, int], float],
) -> Tuple[List[int], List[List[float]]]:
    """Generic log-space Viterbi.

    ``emissions[t][j]`` is the log-emission of state ``j`` at step ``t``;
    ``transition(t, i, j)`` is the log-probability of moving from state ``i`` at
    ``t - 1`` to state ``j`` at ``t``. Returns the best state index per step and the
    per-step forward scores (useful for a confidence measure).

    The caller guarantees the lattice is unbroken (every step has at least one state
    reachable from the previous step); :class:`HmmMapMatcher` splits at breaks before
    calling this.
    """
    if not emissions:
        return [], []
    scores: List[List[float]] = [list(emissions[0])]
    back: List[List[int]] = [[-1] * len(emissions[0])]
    for t in range(1, len(emissions)):
        row: List[float] = []
        ptr: List[int] = []
        prev = scores[t - 1]
        for j, emit in enumerate(emissions[t]):
            best, arg = NEG_INF, -1
            for i, score in enumerate(prev):
                if score == NEG_INF:
                    continue
                trans = transition(t, i, j)
                if trans == NEG_INF:
                    continue
                candidate = score + trans
                if candidate > best:
                    best, arg = candidate, i
            row.append(best + emit if arg >= 0 else NEG_INF)
            ptr.append(arg)
        if all(v == NEG_INF for v in row):
            # Every reachable predecessor was itself a dead state. Restart from the
            # emissions, offset by the best previous score so later rows stay
            # comparable; the backtrack below bridges the gap with an argmax.
            floor = max((v for v in prev if v != NEG_INF), default=0.0)
            row = [floor + emit for emit in emissions[t]]
            ptr = [-1] * len(row)
        scores.append(row)
        back.append(ptr)

    def argmax(values: Sequence[float]) -> int:
        return max(range(len(values)), key=lambda j: values[j])

    state = argmax(scores[-1])
    path = [state]
    for t in range(len(emissions) - 1, 0, -1):
        state = back[t][state]
        if state < 0:
            state = argmax(scores[t - 1])
        path.append(state)
    path.reverse()
    return path, scores


@dataclass
class MatchedPoint:
    """A GPS fix tied to a road position."""

    index: int               # index into the original trace
    point: GpsPoint
    candidate: Candidate
    confidence: float        # in (0, 1]: share of forward probability on this state


@dataclass
class MatchResult:
    points: List[MatchedPoint] = field(default_factory=list)
    # paths[k] connects points[k] -> points[k + 1]; None marks a lattice break.
    paths: List[Optional[PathBetween]] = field(default_factory=list)
    breaks: int = 0
    dropped_no_candidates: int = 0
    dropped_too_close: int = 0
    trace_length: int = 0

    @property
    def matched_fraction(self) -> float:
        if not self.trace_length:
            return 0.0
        return len(self.points) / self.trace_length

    @property
    def mean_confidence(self) -> float:
        if not self.points:
            return 0.0
        return sum(p.confidence for p in self.points) / len(self.points)

    @property
    def matched_segments(self) -> List[int]:
        """Distinct segment ids along the decoded path, in driving order."""
        out: List[int] = []
        for k, point in enumerate(self.points):
            if not out or out[-1] != point.candidate.segment_id:
                out.append(point.candidate.segment_id)
            if k < len(self.paths) and self.paths[k] is not None:
                for piece in self.paths[k].pieces:  # type: ignore[union-attr]
                    if not out or out[-1] != piece.segment_id:
                        out.append(piece.segment_id)
        return out


class HmmMapMatcher:
    """Matches a GPS trace onto road segments via an injected :class:`CandidateSource`."""

    def __init__(self, source: CandidateSource, params: Optional[HmmParams] = None):
        self._source = source
        self._params = params or HmmParams()

    @property
    def params(self) -> HmmParams:
        return self._params

    def match(self, trace: Sequence[GpsPoint]) -> MatchResult:
        p = self._params
        result = MatchResult(trace_length=len(trace))

        # 1. thin the trace and collect candidates per kept fix
        kept: List[Tuple[int, GpsPoint, List[Candidate]]] = []
        last_kept: Optional[GpsPoint] = None
        for index, fix in enumerate(trace):
            is_last = index == len(trace) - 1
            if (
                last_kept is not None
                and not is_last
                and haversine_m(last_kept.location, fix.location)
                < p.min_spacing_factor * p.sigma_m
            ):
                result.dropped_too_close += 1
                continue
            cands = self._source.candidates(
                fix.location, radius_m=p.search_radius_m, limit=p.max_candidates
            )
            if not cands:
                result.dropped_no_candidates += 1
                continue
            kept.append((index, fix, cands))
            last_kept = fix

        if not kept:
            return result

        # 2. build the lattice piecewise, splitting wherever it breaks
        pieces: List[List[Tuple[int, GpsPoint, List[Candidate]]]] = []
        current: List[Tuple[int, GpsPoint, List[Candidate]]] = [kept[0]]
        path_cache: dict = {}

        def path(t_prev: Tuple[int, GpsPoint, List[Candidate]], i: int,
                 t_next: Tuple[int, GpsPoint, List[Candidate]], j: int) -> Optional[PathBetween]:
            key = (t_prev[0], i, t_next[0], j)
            if key not in path_cache:
                crow = haversine_m(t_prev[1].location, t_next[1].location)
                budget = max(p.max_route_floor_m, crow * p.max_route_factor)
                path_cache[key] = self._source.path_between(
                    t_prev[2][i], t_next[2][j], max_distance_m=budget
                )
            return path_cache[key]

        for step in kept[1:]:
            prev = current[-1]
            reachable = any(
                path(prev, i, step, j) is not None
                for i in range(len(prev[2]))
                for j in range(len(step[2]))
            )
            if reachable:
                current.append(step)
            else:
                pieces.append(current)
                current = [step]
                result.breaks += 1
        pieces.append(current)

        # 3. decode each unbroken piece
        for piece_index, piece in enumerate(pieces):
            emissions = [
                [emission_logp(c.distance_m, p.sigma_m) for c in step[2]] for step in piece
            ]

            def transition(t: int, i: int, j: int, _piece=piece) -> float:
                found = path(_piece[t - 1], i, _piece[t], j)
                if found is None:
                    return NEG_INF
                crow = haversine_m(_piece[t - 1][1].location, _piece[t][1].location)
                return transition_logp(found.distance_m, crow, p.beta_m)

            states, scores = viterbi(emissions, transition)
            for t, (step, state) in enumerate(zip(piece, states)):
                norm = logsumexp(scores[t])
                confidence = (
                    math.exp(scores[t][state] - norm) if norm != NEG_INF else 0.0
                )
                if result.points:
                    if t == 0:
                        result.paths.append(None)  # the break between pieces
                    else:
                        result.paths.append(path(piece[t - 1], states[t - 1], step, state))
                result.points.append(
                    MatchedPoint(
                        index=step[0],
                        point=step[1],
                        candidate=step[2][state],
                        confidence=max(1e-6, min(1.0, confidence)),
                    )
                )
        return result


__all__ = [
    "HmmParams",
    "HmmMapMatcher",
    "MatchResult",
    "MatchedPoint",
    "emission_logp",
    "transition_logp",
    "viterbi",
    "logsumexp",
    "NEG_INF",
]
