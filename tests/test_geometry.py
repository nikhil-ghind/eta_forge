import math
import unittest

from eta_forge.graph.geometry import (
    bbox_of,
    bearing_deg,
    bearing_delta_deg,
    destination_point,
    haversine_m,
    interpolate,
    normalize_bearing,
    polyline_bearings,
    polyline_length_m,
    project_point_on_polyline,
    project_point_on_segment,
)
from eta_forge.types import Location

# Two well-known points, ~5.57 km apart.
TIMES_SQUARE = Location(40.7580, -73.9855)
UNION_SQUARE = Location(40.7359, -73.9911)


class TestHaversine(unittest.TestCase):
    def test_known_distance(self):
        d = haversine_m(TIMES_SQUARE, UNION_SQUARE)
        self.assertAlmostEqual(d, 2504.0, delta=25.0)

    def test_identity_is_zero(self):
        self.assertEqual(haversine_m(TIMES_SQUARE, TIMES_SQUARE), 0.0)

    def test_symmetry(self):
        self.assertAlmostEqual(
            haversine_m(TIMES_SQUARE, UNION_SQUARE),
            haversine_m(UNION_SQUARE, TIMES_SQUARE),
            places=6,
        )

    def test_one_degree_of_latitude(self):
        d = haversine_m(Location(0.0, 0.0), Location(1.0, 0.0))
        self.assertAlmostEqual(d, 111_195.0, delta=50.0)

    def test_antimeridian(self):
        d = haversine_m(Location(0.0, 179.999), Location(0.0, -179.999))
        self.assertLess(d, 300.0)


class TestBearing(unittest.TestCase):
    def test_due_north(self):
        self.assertAlmostEqual(
            bearing_deg(Location(0.0, 0.0), Location(1.0, 0.0)), 0.0, places=6
        )

    def test_due_east(self):
        self.assertAlmostEqual(
            bearing_deg(Location(0.0, 0.0), Location(0.0, 1.0)), 90.0, places=6
        )

    def test_due_south_and_west(self):
        self.assertAlmostEqual(
            bearing_deg(Location(1.0, 0.0), Location(0.0, 0.0)), 180.0, places=6
        )
        self.assertAlmostEqual(
            bearing_deg(Location(0.0, 1.0), Location(0.0, 0.0)), 270.0, places=6
        )

    def test_always_in_range(self):
        for lat, lon in ((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)):
            b = bearing_deg(Location(0.0, 0.0), Location(lat, lon))
            self.assertTrue(0.0 <= b < 360.0)

    def test_normalize(self):
        self.assertAlmostEqual(normalize_bearing(-10.0), 350.0)
        self.assertAlmostEqual(normalize_bearing(370.0), 10.0)
        self.assertAlmostEqual(normalize_bearing(720.0), 0.0)


class TestBearingDelta(unittest.TestCase):
    def test_no_turn(self):
        self.assertAlmostEqual(bearing_delta_deg(90.0, 90.0), 0.0)

    def test_right_is_positive(self):
        self.assertAlmostEqual(bearing_delta_deg(0.0, 90.0), 90.0)

    def test_left_is_negative(self):
        self.assertAlmostEqual(bearing_delta_deg(90.0, 0.0), -90.0)

    def test_wrap_across_north(self):
        # 350 -> 10 is a 20-degree right drift, not a 340-degree left u-turn.
        self.assertAlmostEqual(bearing_delta_deg(350.0, 10.0), 20.0)
        self.assertAlmostEqual(bearing_delta_deg(10.0, 350.0), -20.0)

    def test_exact_reversal_is_positive_180(self):
        self.assertAlmostEqual(bearing_delta_deg(0.0, 180.0), 180.0)

    def test_always_in_range(self):
        for incoming in range(0, 360, 17):
            for outgoing in range(0, 360, 23):
                delta = bearing_delta_deg(float(incoming), float(outgoing))
                self.assertTrue(-180.0 < delta <= 180.0, (incoming, outgoing, delta))


class TestPolyline(unittest.TestCase):
    def test_length_of_two_hops(self):
        mid = Location(40.7470, -73.9880)
        total = polyline_length_m([TIMES_SQUARE, mid, UNION_SQUARE])
        direct = haversine_m(TIMES_SQUARE, UNION_SQUARE)
        self.assertGreaterEqual(total, direct - 1.0)

    def test_degenerate(self):
        self.assertEqual(polyline_length_m([]), 0.0)
        self.assertEqual(polyline_length_m([TIMES_SQUARE]), 0.0)

    def test_entry_and_exit_bearings_differ_on_a_curve(self):
        a = Location(40.70, -74.00)
        b = destination_point(a, 0.0, 100.0)
        c = destination_point(b, 90.0, 100.0)
        entry, exit_ = polyline_bearings([a, b, c])
        self.assertAlmostEqual(entry, 0.0, delta=0.5)
        self.assertAlmostEqual(exit_, 90.0, delta=0.5)


class TestInterpolate(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(interpolate(TIMES_SQUARE, UNION_SQUARE, 0.0).lat,
                               TIMES_SQUARE.lat, places=9)
        self.assertAlmostEqual(interpolate(TIMES_SQUARE, UNION_SQUARE, 1.0).lat,
                               UNION_SQUARE.lat, places=9)

    def test_midpoint_is_halfway(self):
        mid = interpolate(TIMES_SQUARE, UNION_SQUARE, 0.5)
        total = haversine_m(TIMES_SQUARE, UNION_SQUARE)
        self.assertAlmostEqual(haversine_m(TIMES_SQUARE, mid), total / 2.0, delta=1.0)

    def test_coincident_points(self):
        same = interpolate(TIMES_SQUARE, TIMES_SQUARE, 0.5)
        self.assertAlmostEqual(same.lat, TIMES_SQUARE.lat)

    def test_fraction_is_clamped(self):
        self.assertAlmostEqual(
            interpolate(TIMES_SQUARE, UNION_SQUARE, 5.0).lat, UNION_SQUARE.lat, places=9
        )


class TestProjection(unittest.TestCase):
    def setUp(self):
        self.start = Location(40.70, -74.00)
        self.end = destination_point(self.start, 90.0, 200.0)   # 200 m due east

    def test_point_on_the_line_projects_onto_itself(self):
        on = destination_point(self.start, 90.0, 100.0)
        projected, fraction, perpendicular = project_point_on_segment(
            on, self.start, self.end
        )
        self.assertAlmostEqual(fraction, 0.5, delta=0.02)
        self.assertLess(perpendicular, 1.0)

    def test_perpendicular_offset(self):
        offset = destination_point(
            destination_point(self.start, 90.0, 100.0), 0.0, 30.0
        )
        _, fraction, perpendicular = project_point_on_segment(offset, self.start, self.end)
        self.assertAlmostEqual(fraction, 0.5, delta=0.02)
        self.assertAlmostEqual(perpendicular, 30.0, delta=1.0)

    def test_beyond_the_end_clamps(self):
        beyond = destination_point(self.start, 90.0, 400.0)
        _, fraction, perpendicular = project_point_on_segment(beyond, self.start, self.end)
        self.assertAlmostEqual(fraction, 1.0)
        self.assertAlmostEqual(perpendicular, 200.0, delta=2.0)

    def test_before_the_start_clamps(self):
        before = destination_point(self.start, 270.0, 50.0)
        _, fraction, _ = project_point_on_segment(before, self.start, self.end)
        self.assertAlmostEqual(fraction, 0.0)

    def test_degenerate_segment(self):
        _, fraction, perpendicular = project_point_on_segment(
            Location(40.7001, -74.0), self.start, self.start
        )
        self.assertEqual(fraction, 0.0)
        self.assertGreater(perpendicular, 0.0)

    def test_polyline_projection_reports_distance_along(self):
        corner = destination_point(self.start, 90.0, 200.0)
        far = destination_point(corner, 0.0, 200.0)
        target = destination_point(corner, 0.0, 50.0)
        _, along, perpendicular = project_point_on_polyline(
            target, [self.start, corner, far]
        )
        self.assertAlmostEqual(along, 250.0, delta=3.0)
        self.assertLess(perpendicular, 1.0)

    def test_polyline_requires_points(self):
        with self.assertRaises(ValueError):
            project_point_on_polyline(self.start, [])


class TestDestinationPoint(unittest.TestCase):
    def test_round_trip_distance(self):
        moved = destination_point(TIMES_SQUARE, 47.0, 1234.0)
        self.assertAlmostEqual(haversine_m(TIMES_SQUARE, moved), 1234.0, delta=0.5)

    def test_round_trip_bearing(self):
        moved = destination_point(TIMES_SQUARE, 47.0, 1234.0)
        self.assertAlmostEqual(bearing_deg(TIMES_SQUARE, moved), 47.0, delta=0.05)


class TestBoundingBox(unittest.TestCase):
    def test_encloses_points(self):
        min_lat, min_lon, max_lat, max_lon = bbox_of([TIMES_SQUARE, UNION_SQUARE])
        self.assertLessEqual(min_lat, UNION_SQUARE.lat)
        self.assertGreaterEqual(max_lat, TIMES_SQUARE.lat)
        self.assertLessEqual(min_lon, UNION_SQUARE.lon)

    def test_padding_widens(self):
        tight = bbox_of([TIMES_SQUARE, UNION_SQUARE])
        padded = bbox_of([TIMES_SQUARE, UNION_SQUARE], pad_m=1000.0)
        self.assertLess(padded[0], tight[0])
        self.assertGreater(padded[2], tight[2])
        # 1 km of padding is roughly 0.009 degrees of latitude.
        self.assertAlmostEqual(tight[0] - padded[0], 0.00899, delta=0.0005)

    def test_empty_is_an_error(self):
        with self.assertRaises(ValueError):
            bbox_of([])


if __name__ == "__main__":
    unittest.main()
