import unittest
from datetime import datetime, timezone

from eta_forge.config import ProfilesConfig
from eta_forge.profiles.buckets import BucketSpec
from eta_forge.profiles.mining import (
    class_means,
    class_stats,
    class_tod_means,
    harmonic_mean_speed,
    mine_profiles,
    percentile,
)
from eta_forge.types import HighwayClass, SegmentObservation

UTC = timezone.utc
SPEC = BucketSpec(timezone=None)


def obs(segment_id, hour, length_m, traversal_s, cls=HighwayClass.RESIDENTIAL, day=5):
    # 2024-03-05 is a Tuesday -> dow 1.
    return SegmentObservation(
        segment_id=segment_id,
        entered_at=datetime(2024, 3, day, hour, 10, tzinfo=UTC),
        traversal_s=traversal_s,
        length_m=length_m,
        highway_class=cls,
    )


class TestHarmonicMean(unittest.TestCase):
    def test_harmonic_not_arithmetic(self):
        # 1 km at 60 km/h + 1 km at 20 km/h: 4 minutes for 2 km is 30 km/h, not 40.
        speed = harmonic_mean_speed([(1000.0, 60.0), (1000.0, 180.0)])
        self.assertAlmostEqual(speed, 30.0)

    def test_empty_and_degenerate(self):
        self.assertIsNone(harmonic_mean_speed([]))
        self.assertIsNone(harmonic_mean_speed([(0.0, 10.0), (100.0, 0.0)]))

    def test_degenerate_pairs_are_skipped(self):
        self.assertAlmostEqual(harmonic_mean_speed([(0.0, 5.0), (100.0, 10.0)]), 36.0)


class TestPercentile(unittest.TestCase):
    def test_interpolates(self):
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([10.0, 20.0], 0.15), 11.5)

    def test_single_and_bounds(self):
        self.assertEqual(percentile([7.0], 0.9), 7.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.0), 1.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 1.0), 3.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            percentile([], 0.5)


class TestMineProfiles(unittest.TestCase):
    def setUp(self):
        self.cfg = ProfilesConfig(min_samples_per_cell=3)

    def test_cells_are_keyed_by_local_dow_and_tod(self):
        profiles, report = mine_profiles(
            [obs(1, 8, 100.0, 12.0), obs(1, 8, 100.0, 18.0), obs(1, 17, 100.0, 30.0)],
            self.cfg,
            spec=SPEC,
        )
        self.assertEqual(set(profiles[1].cells), {(1, 8), (1, 17)})
        self.assertEqual(report.cells, 2)
        self.assertEqual(report.segments, 1)

    def test_cell_statistics(self):
        profiles, _ = mine_profiles(
            [obs(1, 8, 100.0, 10.0), obs(1, 8, 100.0, 30.0)], self.cfg, spec=SPEC
        )
        cell = profiles[1].cell(1, 8)
        self.assertEqual(cell.sample_count, 2)
        self.assertAlmostEqual(cell.speed_kph, 18.0)  # 200 m in 40 s
        self.assertAlmostEqual(cell.p50_kph, 24.0)    # median of 36 and 12
        self.assertAlmostEqual(cell.total_length_m, 200.0)
        self.assertAlmostEqual(cell.total_traversal_s, 40.0)
        self.assertGreater(cell.spread_kph, 0.0)

    def test_implausible_speeds_are_rejected_and_counted(self):
        profiles, report = mine_profiles(
            [
                obs(1, 8, 100.0, 10.0),      # 36 km/h, kept
                obs(1, 8, 100.0, 1000.0),    # 0.36 km/h, parked
                obs(1, 8, 100.0, 1.0),       # 360 km/h, matching error
            ],
            self.cfg,
            spec=SPEC,
        )
        self.assertEqual(report.observations_read, 3)
        self.assertEqual(report.observations_kept, 1)
        self.assertEqual(report.rejected["too_slow"], 1)
        self.assertEqual(report.rejected["too_fast"], 1)
        self.assertEqual(profiles[1].cell(1, 8).sample_count, 1)

    def test_sparse_cells_are_counted(self):
        _, report = mine_profiles([obs(1, 8, 100.0, 10.0)], self.cfg, spec=SPEC)
        self.assertEqual(report.cells_below_threshold, 1)

    def test_segment_mean_spans_all_cells(self):
        profiles, _ = mine_profiles(
            [obs(1, 8, 100.0, 20.0), obs(1, 3, 100.0, 5.0)], self.cfg, spec=SPEC
        )
        self.assertAlmostEqual(profiles[1].segment_mean_kph, 200.0 / 25.0 * 3.6)
        self.assertEqual(profiles[1].segment_sample_count, 2)

    def test_supplied_segment_classes_take_precedence(self):
        profiles, _ = mine_profiles(
            [obs(1, 8, 100.0, 10.0, cls=HighwayClass.UNCLASSIFIED)],
            self.cfg,
            spec=SPEC,
            segment_classes={1: HighwayClass.PRIMARY},
        )
        self.assertIs(profiles[1].highway_class, HighwayClass.PRIMARY)

    def test_report_serialises(self):
        _, report = mine_profiles([obs(1, 8, 100.0, 10.0)], self.cfg, spec=SPEC)
        payload = report.as_dict()
        self.assertEqual(payload["observations_kept"], 1)
        self.assertIsInstance(payload["rejected"], dict)


class TestClassAggregates(unittest.TestCase):
    def test_class_means_pool_traversals_not_segment_means(self):
        cfg = ProfilesConfig()
        profiles, _ = mine_profiles(
            [
                obs(1, 8, 100.0, 10.0, HighwayClass.PRIMARY),
                obs(2, 8, 300.0, 90.0, HighwayClass.PRIMARY),
            ],
            cfg,
            spec=SPEC,
        )
        # 400 m in 100 s = 14.4 km/h; the mean of the two segment speeds would be 24.
        self.assertAlmostEqual(class_means(profiles)["primary"], 14.4)
        self.assertAlmostEqual(class_tod_means(profiles, SPEC)[("primary", 1, 8)], 14.4)
        mean, n = class_stats(profiles)["primary"]
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
