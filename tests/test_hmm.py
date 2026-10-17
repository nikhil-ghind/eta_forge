import math
import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.graph.geometry import destination_point
from eta_forge.mapmatch.attribution import attribute_traversals, trip_segment_rows
from eta_forge.mapmatch.candidates import (
    candidate_on_polyline,
    same_segment_path,
    stitch_path,
)
from eta_forge.mapmatch.hmm import (
    NEG_INF,
    HmmMapMatcher,
    HmmParams,
    emission_logp,
    logsumexp,
    transition_logp,
    viterbi,
)
from eta_forge.types import GpsPoint, Location

UTC = timezone.utc
T0 = datetime(2024, 3, 5, 8, 0, tzinfo=UTC)
P0 = Location(40.7500, -73.9900)
P1 = destination_point(P0, 90.0, 200.0)
P2 = destination_point(P1, 90.0, 200.0)
P3 = destination_point(P2, 90.0, 200.0)
# A parallel street 25 m north of segment 1, connected to nothing the trip uses.
Q0 = destination_point(P0, 0.0, 25.0)
Q1 = destination_point(P1, 0.0, 25.0)


class FixtureSource:
    """Three collinear eastbound segments 1 -> 2 -> 3 plus a disconnected parallel 9."""

    SEGMENTS = {
        1: ((P0, P1), 1, 2),
        2: ((P1, P2), 2, 3),
        3: ((P2, P3), 3, 4),
        9: ((Q0, Q1), 10, 11),
    }

    def candidates(self, point, *, radius_m, limit):
        found = []
        for segment_id, (polyline, source, target) in self.SEGMENTS.items():
            cand = candidate_on_polyline(
                point,
                segment_id=segment_id,
                polyline=polyline,
                source_vertex=source,
                target_vertex=target,
            )
            if cand.distance_m <= radius_m:
                found.append(cand)
        found.sort(key=lambda c: c.distance_m)
        return found[:limit]

    def path_between(self, a, b, *, max_distance_m):
        direct = same_segment_path(a, b)
        if direct is not None:
            return direct
        via = []
        vertex = a.target_vertex
        for segment_id in (1, 2, 3):
            if vertex == b.source_vertex:
                break
            _, source, target = self.SEGMENTS[segment_id]
            if source == vertex and segment_id not in (a.segment_id, b.segment_id):
                via.append((segment_id, 200.0))
                vertex = target
        if vertex != b.source_vertex:
            return None
        path = stitch_path(a, b, via, 200.0 * len(via))
        return path if path.distance_m <= max_distance_m else None


def fix(seconds, along_m, north_m=0.0):
    point = destination_point(P0, 90.0, along_m)
    if north_m:
        point = destination_point(point, 0.0, north_m)
    return GpsPoint(at=T0 + timedelta(seconds=seconds), location=point)


class TestProbabilityModel(unittest.TestCase):
    def test_emission_falls_with_distance(self):
        self.assertGreater(emission_logp(0.0, 8.0), emission_logp(5.0, 8.0))
        self.assertGreater(emission_logp(5.0, 8.0), emission_logp(30.0, 8.0))

    def test_emission_is_a_gaussian_density(self):
        self.assertAlmostEqual(
            emission_logp(0.0, 1.0), -math.log(math.sqrt(2.0 * math.pi))
        )

    def test_transition_peaks_when_route_matches_crow_line(self):
        self.assertGreater(transition_logp(100.0, 100.0, 6.0), transition_logp(130.0, 100.0, 6.0))
        self.assertAlmostEqual(
            transition_logp(110.0, 100.0, 6.0), transition_logp(90.0, 100.0, 6.0)
        )

    def test_logsumexp(self):
        self.assertAlmostEqual(logsumexp([math.log(1.0), math.log(3.0)]), math.log(4.0))
        self.assertEqual(logsumexp([NEG_INF, NEG_INF]), NEG_INF)

    def test_params_validate(self):
        with self.assertRaises(ValueError):
            HmmParams(sigma_m=0.0)
        with self.assertRaises(ValueError):
            HmmParams(max_candidates=0)


class TestViterbi(unittest.TestCase):
    EMISSIONS = [[-1.0, -2.0], [-1.5, -1.0], [-1.0, -2.0]]

    def test_switching_penalty_keeps_the_state(self):
        path, _ = viterbi(self.EMISSIONS, lambda t, i, j: 0.0 if i == j else -5.0)
        self.assertEqual(path, [0, 0, 0])

    def test_free_transitions_follow_the_emissions(self):
        path, _ = viterbi(self.EMISSIONS, lambda t, i, j: 0.0)
        self.assertEqual(path, [0, 1, 0])

    def test_impossible_transitions_are_never_taken(self):
        def transition(t, i, j):
            return NEG_INF if (t == 1 and j == 0) else 0.0

        path, _ = viterbi(self.EMISSIONS, transition)
        self.assertEqual(path[1], 1)

    def test_dead_row_does_not_crash(self):
        path, _ = viterbi(self.EMISSIONS, lambda t, i, j: NEG_INF if t == 1 else 0.0)
        self.assertEqual(len(path), 3)
        self.assertTrue(all(state >= 0 for state in path))

    def test_empty(self):
        self.assertEqual(viterbi([], lambda t, i, j: 0.0), ([], []))


class TestMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = HmmMapMatcher(FixtureSource(), HmmParams(sigma_m=6.0, search_radius_m=45.0))
        # 10 m/s eastbound from 20 m into segment 1 to 70 m into segment 3.
        self.trace = [fix(t, 20.0 + 10.0 * t) for t in range(0, 56, 5)]

    def test_decodes_the_driven_segments_in_order(self):
        result = self.matcher.match(self.trace)
        self.assertEqual(result.breaks, 0)
        self.assertEqual(result.matched_segments, [1, 2, 3])
        self.assertEqual(len(result.paths), len(result.points) - 1)
        self.assertAlmostEqual(result.matched_fraction, 1.0)

    def test_network_structure_beats_a_closer_parallel_street(self):
        # Fixes on segment 1 drift 14 m north: 11 m from the parallel street, closer
        # than the true road — but the parallel street goes nowhere the trip goes.
        trace = [
            fix(t, 20.0 + 10.0 * t, north_m=14.0 if t < 15 else 0.0)
            for t in range(0, 56, 5)
        ]
        result = self.matcher.match(trace)
        self.assertNotIn(9, result.matched_segments)
        self.assertEqual(result.points[0].candidate.segment_id, 1)

    def test_a_teleport_breaks_the_lattice(self):
        trace = [fix(0, 20.0), fix(5, 70.0), GpsPoint(T0 + timedelta(seconds=10), Q0),
                 fix(15, 170.0)]
        result = self.matcher.match(trace)
        self.assertGreaterEqual(result.breaks, 1)
        self.assertIn(None, result.paths)

    def test_fixes_far_from_any_road_are_dropped(self):
        trace = list(self.trace)
        trace.insert(3, GpsPoint(T0 + timedelta(seconds=11), destination_point(P1, 180.0, 500.0)))
        result = self.matcher.match(trace)
        self.assertEqual(result.dropped_no_candidates, 1)

    def test_stationary_jitter_is_thinned(self):
        trace = [fix(0, 20.0), fix(1, 21.0), fix(2, 22.0), fix(10, 120.0)]
        result = self.matcher.match(trace)
        self.assertEqual(result.dropped_too_close, 2)


class TestAttribution(unittest.TestCase):
    def setUp(self):
        matcher = HmmMapMatcher(FixtureSource(), HmmParams(sigma_m=6.0))
        trace = [fix(t, 20.0 + 10.0 * t) for t in range(0, 56, 5)]
        self.result = matcher.match(trace)

    def test_only_complete_traversals_are_kept(self):
        traversals = attribute_traversals(self.result, trip_id="t1")
        # Segment 1 was entered mid-way and segment 3 left mid-way: only 2 is complete.
        self.assertEqual([t.observation.segment_id for t in traversals], [2])

    def test_constant_speed_traversal_time(self):
        observation = attribute_traversals(self.result, trip_id="t1")[0].observation
        self.assertAlmostEqual(observation.traversal_s, 20.0, delta=0.5)
        self.assertAlmostEqual(observation.length_m, 200.0, delta=0.5)
        self.assertAlmostEqual(observation.speed_kph, 36.0, delta=1.0)
        # Entered segment 2 after driving 180 m at 10 m/s.
        self.assertAlmostEqual(
            (observation.entered_at - T0).total_seconds(), 18.0, delta=0.5
        )
        self.assertEqual(observation.trip_id, "t1")

    def test_a_stop_is_charged_to_the_segment_it_happens_on(self):
        matcher = HmmMapMatcher(FixtureSource(), HmmParams(sigma_m=6.0))
        trace = (
            [fix(t, 20.0 + 10.0 * t) for t in range(0, 26, 5)]           # to 270 m
            + [fix(25 + 30, 270.0)]                                        # 30 s stopped
            + [fix(55 + t, 270.0 + 10.0 * t) for t in range(5, 31, 5)]     # to 570 m
        )
        observation = attribute_traversals(matcher.match(trace))[0].observation
        self.assertEqual(observation.segment_id, 2)
        self.assertAlmostEqual(observation.traversal_s, 50.0, delta=1.0)

    def test_rows_for_trip_segment(self):
        rows = trip_segment_rows("t1", attribute_traversals(self.result, trip_id="t1"))
        self.assertEqual(len(rows), 1)
        trip_id, seq, segment_id, _, traversal_s, length_m, confidence = rows[0]
        self.assertEqual((trip_id, seq, segment_id), ("t1", 0, 2))
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
