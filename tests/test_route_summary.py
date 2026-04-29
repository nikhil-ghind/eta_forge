import unittest
from datetime import datetime, timezone

from eta_forge.graph.geometry import destination_point, haversine_m
from eta_forge.graph.summary import summarize_route
from eta_forge.graph.synthetic import SyntheticGraph
from eta_forge.types import BaselineRoute, HighwayClass, Location, RouteLeg

UTC = timezone.utc
ORIGIN = Location(40.7000, -74.0100)


def build_route(legs, *, origin=None, destination=None, coverage=0.0):
    origin = origin or ORIGIN
    destination = destination or destination_point(ORIGIN, 90.0, 1000.0)
    return BaselineRoute(
        legs=tuple(legs),
        origin=origin,
        destination=destination,
        departure_at=datetime(2024, 3, 5, 8, 0, tzinfo=UTC),
        used_speed_profiles=True,
        profile_coverage=coverage,
        snapped_origin_distance_m=12.0,
        snapped_destination_distance_m=8.0,
    )


def leg(segment_id, length_m, highway_class, bearing=90.0, duration_s=None):
    duration = duration_s if duration_s is not None else length_m / (30.0 / 3.6)
    return RouteLeg(
        segment_id=segment_id,
        source_vertex=segment_id,
        target_vertex=segment_id + 1,
        length_m=length_m,
        highway_class=highway_class,
        speed_kph=(length_m / duration) * 3.6,
        duration_s=duration,
        bearing_in_deg=bearing,
        bearing_out_deg=bearing,
    )


class TestSummarizeRoute(unittest.TestCase):
    def test_distance_and_duration_sum_the_legs(self):
        route = build_route([
            leg(1, 400.0, HighwayClass.PRIMARY),
            leg(2, 600.0, HighwayClass.RESIDENTIAL),
        ])
        summary = summarize_route(route)
        self.assertAlmostEqual(summary.distance_m, 1000.0)
        self.assertAlmostEqual(summary.baseline_duration_s, 120.0)
        self.assertAlmostEqual(summary.mean_speed_kph, 30.0, places=6)

    def test_class_mix_is_distance_weighted(self):
        summary = summarize_route(build_route([
            leg(1, 800.0, HighwayClass.PRIMARY),
            leg(2, 200.0, HighwayClass.RESIDENTIAL),
        ]))
        self.assertAlmostEqual(summary.class_mix["primary"], 0.8)
        self.assertAlmostEqual(summary.class_mix["residential"], 0.2)
        self.assertEqual(summary.distinct_classes, 2)

    def test_detour_ratio_against_the_crow_line(self):
        # A dog-leg route: 1000 m north then 1000 m east for a ~1414 m crow line.
        destination = destination_point(
            destination_point(ORIGIN, 0.0, 1000.0), 90.0, 1000.0
        )
        route = build_route(
            [leg(1, 1000.0, HighwayClass.PRIMARY, bearing=0.0),
             leg(2, 1000.0, HighwayClass.PRIMARY, bearing=90.0)],
            destination=destination,
        )
        summary = summarize_route(route)
        self.assertAlmostEqual(summary.crow_distance_m, 1414.0, delta=5.0)
        self.assertAlmostEqual(summary.detour_ratio, 2000.0 / 1414.0, delta=0.01)

    def test_detour_ratio_never_below_one(self):
        far = destination_point(ORIGIN, 90.0, 10_000.0)
        summary = summarize_route(
            build_route([leg(1, 100.0, HighwayClass.PRIMARY)], destination=far)
        )
        self.assertGreaterEqual(summary.detour_ratio, 1.0)

    def test_turn_profile_is_carried_through(self):
        summary = summarize_route(build_route([
            leg(1, 500.0, HighwayClass.PRIMARY, bearing=0.0),
            leg(2, 500.0, HighwayClass.PRIMARY, bearing=90.0),
        ]))
        self.assertEqual(summary.turns.right_turns, 1)

    def test_snap_distance_is_the_sum_of_both_ends(self):
        summary = summarize_route(build_route([leg(1, 100.0, HighwayClass.PRIMARY)]))
        self.assertAlmostEqual(summary.snap_distance_m, 20.0)

    def test_profile_coverage_is_carried_through(self):
        summary = summarize_route(
            build_route([leg(1, 100.0, HighwayClass.PRIMARY)], coverage=0.62)
        )
        self.assertAlmostEqual(summary.profile_coverage, 0.62)

    def test_empty_route(self):
        summary = summarize_route(build_route([]))
        self.assertEqual(summary.distance_m, 0.0)
        self.assertEqual(summary.leg_count, 0)
        self.assertEqual(summary.detour_ratio, 1.0)


class TestFeatureNames(unittest.TestCase):
    def test_every_class_gets_a_fraction_column(self):
        features = summarize_route(
            build_route([leg(1, 100.0, HighwayClass.PRIMARY)])
        ).to_features()
        for cls in HighwayClass:
            self.assertIn(f"class_frac_{cls.value}", features)

    def test_core_names_present_and_numeric(self):
        features = summarize_route(
            build_route([leg(1, 100.0, HighwayClass.PRIMARY)])
        ).to_features()
        for name in (
            "distance_m",
            "baseline_duration_s",
            "baseline_mean_speed_kph",
            "detour_ratio",
            "od_bearing_deg",
            "profile_coverage",
            "turn_penalty_s",
            "signal_count",
        ):
            self.assertIn(name, features)
            self.assertIsInstance(features[name], float)


class TestSyntheticGraphRouting(unittest.TestCase):
    def setUp(self):
        self.graph = SyntheticGraph(rows=10, cols=10, block_m=180.0)
        self.when = datetime(2024, 3, 5, 11, 0, tzinfo=UTC)

    def test_routes_between_two_corners(self):
        south_west, north_east = self.graph.bounds()
        route = self.graph.route(south_west, north_east, self.when)
        self.assertGreater(len(route.legs), 10)
        self.assertGreater(route.distance_m, haversine_m(south_west, north_east))

    def test_path_is_connected(self):
        south_west, north_east = self.graph.bounds()
        route = self.graph.route(south_west, north_east, self.when)
        for a, b in zip(route.legs, route.legs[1:]):
            self.assertEqual(a.target_vertex, b.source_vertex)

    def test_identical_endpoints_give_an_empty_route(self):
        point = self.graph.random_location()
        route = self.graph.route(point, point, self.when)
        self.assertEqual(len(route.legs), 0)
        self.assertEqual(route.duration_s, 0.0)

    def test_rush_hour_is_slower_than_the_small_hours(self):
        south_west, north_east = self.graph.bounds()
        rush = self.graph.route(
            south_west, north_east, datetime(2024, 3, 5, 8, 30, tzinfo=UTC)
        )
        quiet = self.graph.route(
            south_west, north_east, datetime(2024, 3, 5, 3, 0, tzinfo=UTC)
        )
        self.assertGreater(rush.duration_s, quiet.duration_s * 1.15)

    def test_weekend_peak_is_flatter_than_a_weekday_peak(self):
        south_west, north_east = self.graph.bounds()
        weekday = self.graph.route(
            south_west, north_east, datetime(2024, 3, 5, 8, 30, tzinfo=UTC)
        )
        weekend = self.graph.route(
            south_west, north_east, datetime(2024, 3, 9, 8, 30, tzinfo=UTC)
        )
        self.assertLess(weekend.duration_s, weekday.duration_s)

    def test_summary_of_a_synthetic_route(self):
        south_west, north_east = self.graph.bounds()
        summary = summarize_route(self.graph.route(south_west, north_east, self.when))
        self.assertGreater(summary.distance_m, 0.0)
        self.assertGreaterEqual(summary.detour_ratio, 1.0)
        self.assertAlmostEqual(sum(summary.class_mix.values()), 1.0, places=6)

    def test_avenues_are_one_way(self):
        # Every avenue edge should exist in exactly one direction.
        pairs = {(e.source, e.target) for e in self.graph.edges
                 if e.highway_class.value == "primary"}
        reversed_pairs = {(t, s) for s, t in pairs}
        self.assertEqual(len(pairs & reversed_pairs), 0)

    def test_deterministic_across_instances(self):
        other = SyntheticGraph(rows=10, cols=10, block_m=180.0)
        south_west, north_east = self.graph.bounds()
        a = self.graph.route(south_west, north_east, self.when)
        b = other.route(south_west, north_east, self.when)
        self.assertAlmostEqual(a.duration_s, b.duration_s, places=9)
        self.assertEqual(
            [l.segment_id for l in a.legs], [l.segment_id for l in b.legs]
        )


if __name__ == "__main__":
    unittest.main()
