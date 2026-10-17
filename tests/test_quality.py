import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.config import DataConfig
from eta_forge.data import quality as q
from eta_forge.data.quality import QualityRules, RejectionTally, check_trip, is_null_island
from eta_forge.types import Location

UTC = timezone.utc
PICKUP = datetime(2024, 3, 5, 8, 0, tzinfo=UTC)
MIDTOWN = Location(40.7580, -73.9855)
UPTOWN = Location(40.7760, -73.9800)   # ~2.0 km north-north-east


def check(**overrides):
    rules = overrides.pop("rules", QualityRules.from_config(DataConfig()))
    trip = dict(
        pickup_at=PICKUP,
        dropoff_at=PICKUP + timedelta(minutes=15),
        distance_m=3_000.0,
        origin=MIDTOWN,
        destination=UPTOWN,
    )
    trip.update(overrides)
    return check_trip(rules, **trip)


class TestCheckTrip(unittest.TestCase):
    def test_plausible_trip_passes(self):
        self.assertIsNone(check())

    def test_timestamps(self):
        self.assertEqual(check(pickup_at=None), q.MISSING_TIMESTAMP)
        self.assertEqual(check(dropoff_at=PICKUP), q.NON_POSITIVE_DURATION)
        self.assertEqual(check(dropoff_at=PICKUP - timedelta(minutes=1)), q.NON_POSITIVE_DURATION)

    def test_duration_window(self):
        self.assertEqual(check(dropoff_at=PICKUP + timedelta(seconds=30)), q.DURATION_TOO_SHORT)
        self.assertEqual(check(dropoff_at=PICKUP + timedelta(hours=5)), q.DURATION_TOO_LONG)

    def test_distance(self):
        self.assertEqual(check(distance_m=0.0), q.ZERO_DISTANCE)
        self.assertEqual(check(distance_m=50.0), q.DISTANCE_TOO_SHORT)

    def test_missing_distance_skips_the_distance_checks(self):
        self.assertIsNone(check(distance_m=None))

    def test_locations(self):
        self.assertEqual(check(origin=None), q.MISSING_LOCATION)
        self.assertEqual(check(destination=Location(42.0, -73.98)), q.OUT_OF_BBOX)

    def test_no_bbox_means_no_region_check(self):
        rules = QualityRules(bbox=None)
        self.assertIsNone(check(rules=rules, destination=Location(40.80, -73.95), distance_m=6_000.0))

    def test_implausible_speed(self):
        # 100 km in 15 minutes.
        self.assertEqual(check(distance_m=100_000.0), q.IMPLAUSIBLE_SPEED)

    def test_metered_distance_below_the_crow_line(self):
        self.assertEqual(check(distance_m=1_000.0), q.DISTANCE_BELOW_CROW)
        self.assertIsNone(check(distance_m=1_000.0, check_crow_line=False))

    def test_first_failing_reason_only(self):
        # Both too short in time and zero distance: only the first check counts.
        self.assertEqual(
            check(dropoff_at=PICKUP + timedelta(seconds=10), distance_m=0.0),
            q.DURATION_TOO_SHORT,
        )


class TestRejectionTally(unittest.TestCase):
    def test_counts_partition_the_rows(self):
        tally = RejectionTally()
        tally.keep()
        tally.keep()
        tally.reject(q.ZERO_DISTANCE)
        tally.reject(q.ZERO_DISTANCE)
        tally.reject(q.OUT_OF_BBOX)
        self.assertEqual(tally.read, 5)
        self.assertEqual(tally.kept, 2)
        self.assertEqual(tally.rejected_total, 3)
        self.assertEqual(tally.read, tally.kept + tally.rejected_total)
        self.assertAlmostEqual(tally.kept_fraction, 0.4)

    def test_merge(self):
        a, b = RejectionTally(), RejectionTally()
        a.keep()
        a.reject(q.MALFORMED)
        b.reject(q.MALFORMED)
        a.merge(b)
        self.assertEqual(a.read, 3)
        self.assertEqual(a.rejected[q.MALFORMED], 2)

    def test_serialises_and_summarises(self):
        tally = RejectionTally()
        tally.keep()
        tally.reject(q.IMPLAUSIBLE_SPEED)
        payload = tally.as_dict()
        self.assertEqual(payload["rejected"], {q.IMPLAUSIBLE_SPEED: 1})
        self.assertEqual(payload["kept_fraction"], 0.5)
        lines = tally.summary_lines()
        self.assertIn("kept 1", lines[0])
        self.assertTrue(any(q.IMPLAUSIBLE_SPEED in line for line in lines[1:]))

    def test_empty_tally(self):
        self.assertEqual(RejectionTally().kept_fraction, 0.0)
        self.assertEqual(q.rejection_rates(RejectionTally()), {})


class TestNullIsland(unittest.TestCase):
    def test_zero_coordinates_are_missing(self):
        self.assertTrue(is_null_island(0.0, 0.0))
        self.assertTrue(is_null_island(None, -73.9))
        self.assertFalse(is_null_island(40.75, -73.98))


if __name__ == "__main__":
    unittest.main()
