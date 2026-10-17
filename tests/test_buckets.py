import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.config import ProfilesConfig
from eta_forge.graph.costs import time_bucket
from eta_forge.profiles.buckets import (
    BucketSpec,
    bucket_label,
    bucket_of,
    is_weekend,
    iter_cells,
    neighbouring_cells,
)
from eta_forge.types import ValidationError

UTC = timezone.utc


class TestBucketSpec(unittest.TestCase):
    def test_defaults_are_hourly_by_weekday(self):
        spec = BucketSpec()
        self.assertEqual(spec.hours_per_bucket, 1)
        self.assertEqual(spec.cell_count, 168)

    def test_coarse_bucketing(self):
        spec = BucketSpec(tod_buckets=6, dow_buckets=1)
        self.assertEqual(spec.hours_per_bucket, 4)
        self.assertEqual(spec.cell_count, 6)

    def test_rejects_non_divisor(self):
        with self.assertRaises(ValidationError):
            BucketSpec(tod_buckets=5)
        with self.assertRaises(ValidationError):
            BucketSpec(dow_buckets=2)

    def test_from_config(self):
        spec = BucketSpec.from_config(ProfilesConfig(tod_buckets=12, timezone="UTC"))
        self.assertEqual(spec.tod_buckets, 12)
        self.assertEqual(spec.timezone, "UTC")


class TestBucketOf(unittest.TestCase):
    def test_utc_when_no_timezone(self):
        spec = BucketSpec(timezone=None)
        self.assertEqual(bucket_of(datetime(2024, 3, 7, 18, 30, tzinfo=UTC), spec), (3, 18))

    def test_local_time_moves_hour_and_day(self):
        # 03:00 UTC on a Friday is 22:00 Thursday in New York (EST, before DST).
        spec = BucketSpec()
        self.assertEqual(bucket_of(datetime(2024, 3, 8, 3, 0, tzinfo=UTC), spec), (3, 22))

    def test_daylight_saving_keeps_local_rush_hour_in_one_bucket(self):
        # 08:30 local on either side of the March 2024 DST change is a different UTC
        # hour but must be the same profile cell.
        spec = BucketSpec()
        before = datetime(2024, 3, 8, 13, 30, tzinfo=UTC)   # 08:30 EST, Friday
        after = datetime(2024, 3, 15, 12, 30, tzinfo=UTC)   # 08:30 EDT, Friday
        self.assertEqual(bucket_of(before, spec), bucket_of(after, spec))

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValidationError):
            bucket_of(datetime(2024, 3, 7, 18, 30), BucketSpec())

    def test_agrees_with_the_router_bucketing(self):
        # The miner and the router must put every timestamp in the same cell, or the
        # router reads a profile mined for a different hour.
        spec = BucketSpec(tod_buckets=24, dow_buckets=7, timezone="America/New_York")
        when = datetime(2024, 1, 1, tzinfo=UTC)
        for step in range(0, 24 * 14, 5):
            ts = when + timedelta(hours=step, minutes=17)
            self.assertEqual(
                bucket_of(ts, spec),
                time_bucket(ts, tod_buckets=24, dow_buckets=7, tz="America/New_York"),
            )


class TestCellHelpers(unittest.TestCase):
    def test_label(self):
        self.assertEqual(bucket_label(3, 18, BucketSpec()), "Thu 18:00-19:00")
        self.assertEqual(
            bucket_label(0, 2, BucketSpec(tod_buckets=8, dow_buckets=1)), "all 06:00-09:00"
        )

    def test_iter_cells_is_complete_and_ordered(self):
        cells = list(iter_cells(BucketSpec()))
        self.assertEqual(len(cells), 168)
        self.assertEqual(len(set(cells)), 168)
        self.assertEqual(cells[0], (0, 0))
        self.assertEqual(cells[-1], (6, 23))

    def test_neighbours_wrap_across_midnight(self):
        spec = BucketSpec()
        self.assertEqual(set(neighbouring_cells(2, 0, spec)), {(2, 23), (2, 1)})
        self.assertEqual(set(neighbouring_cells(2, 23, spec)), {(2, 22), (2, 0)})

    def test_neighbour_radius(self):
        self.assertEqual(len(list(neighbouring_cells(0, 10, BucketSpec(), radius=2))), 4)

    def test_weekend(self):
        spec = BucketSpec()
        self.assertTrue(is_weekend(5, spec))
        self.assertTrue(is_weekend(6, spec))
        self.assertFalse(is_weekend(4, spec))
        self.assertFalse(is_weekend(0, BucketSpec(dow_buckets=1)))


if __name__ == "__main__":
    unittest.main()
