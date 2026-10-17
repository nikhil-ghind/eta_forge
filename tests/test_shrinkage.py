import unittest
from datetime import datetime, timezone

from eta_forge.config import GraphConfig, ProfilesConfig
from eta_forge.profiles.buckets import BucketSpec
from eta_forge.profiles.mining import CellStats, SegmentProfile, mine_profiles
from eta_forge.profiles.shrinkage import (
    PriorHierarchy,
    credibility_weight,
    shrink_cell,
    shrink_profiles,
)
from eta_forge.types import HighwayClass, SegmentObservation

UTC = timezone.utc
SPEC = BucketSpec(timezone=None)
DEFAULTS = GraphConfig().default_speed_kph


def heavy_profile(segment_id, speed_kph, n, dow=1, tod=8, cls=HighwayClass.RESIDENTIAL):
    """A profile with one cell backed by ``n`` traversals of 100 m at ``speed_kph``."""
    length = 100.0 * n
    seconds = length / (speed_kph / 3.6)
    cell = CellStats(
        segment_id=segment_id,
        dow_bucket=dow,
        tod_bucket=tod,
        speed_kph=speed_kph,
        sample_count=n,
        p15_kph=speed_kph,
        p50_kph=speed_kph,
        p85_kph=speed_kph,
        total_length_m=length,
        total_traversal_s=seconds,
        highway_class=cls,
    )
    return SegmentProfile(
        segment_id=segment_id,
        highway_class=cls,
        cells={(dow, tod): cell},
        segment_mean_kph=speed_kph,
        segment_sample_count=n,
    )


class TestCredibility(unittest.TestCase):
    def test_weight_endpoints(self):
        self.assertEqual(credibility_weight(0, 20.0), 0.0)
        self.assertAlmostEqual(credibility_weight(20, 20.0), 0.5)
        self.assertGreater(credibility_weight(1e9, 20.0), 0.999999)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            credibility_weight(5, 0.0)

    def test_zero_samples_returns_the_prior_exactly(self):
        self.assertEqual(shrink_cell(71.0, 30.0, 0, 20.0), 30.0)

    def test_huge_n_returns_the_observation(self):
        self.assertAlmostEqual(shrink_cell(71.0, 30.0, 1e12, 20.0), 71.0, places=6)

    def test_k_is_the_equal_weight_point(self):
        self.assertAlmostEqual(shrink_cell(70.0, 30.0, 20, 20.0), 50.0)


class TestPriorHierarchy(unittest.TestCase):
    def test_no_data_gives_the_configured_default(self):
        hierarchy = PriorHierarchy(class_default=DEFAULTS)
        self.assertEqual(hierarchy.base_prior(HighwayClass.PRIMARY, 0, 8), 50.0)
        self.assertEqual(hierarchy.base_prior(HighwayClass.SERVICE, 3, 2), 20.0)

    def test_unknown_class_falls_back_to_global(self):
        hierarchy = PriorHierarchy(class_default={}, global_default_kph=27.0)
        self.assertEqual(hierarchy.base_prior(HighwayClass.TRUNK, 0, 0), 27.0)

    def test_class_levels_are_shrunk_not_taken_raw(self):
        hierarchy = PriorHierarchy(
            class_tod={("primary", 1, 8): (12.0, 4)},
            class_all={"primary": (40.0, 4)},
            class_default=DEFAULTS,
            k_global=80.0,
        )
        prior = hierarchy.base_prior(HighwayClass.PRIMARY, 1, 8)
        # Four traversals city-wide barely move the default.
        self.assertGreater(prior, 45.0)
        self.assertLess(prior, 50.0)


class TestShrinkProfiles(unittest.TestCase):
    def setUp(self):
        self.cfg = ProfilesConfig()

    def test_sparse_cell_is_pulled_toward_the_prior(self):
        observations = [
            SegmentObservation(
                segment_id=7,
                entered_at=datetime(2024, 3, 5, 3, 5, tzinfo=UTC),
                traversal_s=100.0 / (70.0 / 3.6),
                length_m=100.0,
                highway_class=HighwayClass.RESIDENTIAL,
            )
            for _ in range(3)
        ]
        profiles, _ = mine_profiles(observations, self.cfg, spec=SPEC)
        smoothed = shrink_profiles(profiles, self.cfg, spec=SPEC, class_default=DEFAULTS)
        cell = smoothed[7][(1, 3)]
        self.assertEqual(cell.source, "observed")
        self.assertAlmostEqual(cell.raw_speed_kph, 70.0, places=6)
        self.assertAlmostEqual(cell.shrink_weight, 3.0 / 23.0)
        # Three lucky 3am runs must not make a residential block a 70 km/h road.
        self.assertGreater(cell.speed_kph, 30.0)
        self.assertLess(cell.speed_kph, 50.0)

    def test_well_observed_cell_keeps_its_observation(self):
        profiles = {1: heavy_profile(1, 70.0, 1_000_000)}
        smoothed = shrink_profiles(profiles, self.cfg, spec=SPEC, class_default=DEFAULTS)
        self.assertAlmostEqual(smoothed[1][(1, 8)].speed_kph, 70.0, delta=0.05)

    def test_output_is_dense(self):
        profiles = {1: heavy_profile(1, 25.0, 50)}
        smoothed = shrink_profiles(profiles, self.cfg, spec=SPEC, class_default=DEFAULTS)
        self.assertEqual(len(smoothed[1]), SPEC.cell_count)

    def test_sparse_output_when_not_filling(self):
        profiles = {1: heavy_profile(1, 25.0, 50)}
        smoothed = shrink_profiles(
            profiles, self.cfg, spec=SPEC, class_default=DEFAULTS, fill_empty_cells=False
        )
        self.assertEqual(list(smoothed[1]), [(1, 8)])

    def test_adjacent_hours_borrow_from_their_neighbour(self):
        profiles = {1: heavy_profile(1, 12.0, 200)}
        smoothed = shrink_profiles(profiles, self.cfg, spec=SPEC, class_default=DEFAULTS)
        neighbour = smoothed[1][(1, 9)]
        far = smoothed[1][(4, 2)]
        self.assertEqual(neighbour.source, "neighbour")
        self.assertEqual(neighbour.effective_sample_count, 200.0)
        self.assertEqual(far.source, "segment")
        self.assertEqual(neighbour.sample_count, 0)
        self.assertAlmostEqual(neighbour.speed_kph, 12.0, delta=1.0)

    def test_untraversed_segments_get_the_default(self):
        profiles = {1: heavy_profile(1, 25.0, 50, cls=HighwayClass.RESIDENTIAL)}
        smoothed = shrink_profiles(
            profiles,
            self.cfg,
            spec=SPEC,
            class_default=DEFAULTS,
            all_segments={1: HighwayClass.RESIDENTIAL, 99: HighwayClass.SERVICE},
        )
        cell = smoothed[99][(0, 0)]
        self.assertEqual(cell.source, "default")
        self.assertEqual(cell.speed_kph, 20.0)
        self.assertEqual(cell.effective_sample_count, 0.0)
        self.assertEqual(len(smoothed[99]), SPEC.cell_count)

    def test_untraversed_segment_of_an_observed_class(self):
        profiles = {1: heavy_profile(1, 25.0, 50, cls=HighwayClass.RESIDENTIAL)}
        smoothed = shrink_profiles(
            profiles,
            self.cfg,
            spec=SPEC,
            class_default=DEFAULTS,
            all_segments={2: HighwayClass.RESIDENTIAL},
        )
        self.assertEqual(smoothed[2][(1, 8)].source, "class")

    def test_speeds_are_clamped(self):
        profiles = {1: heavy_profile(1, 129.0, 10_000_000, cls=HighwayClass.MOTORWAY)}
        cfg = ProfilesConfig(max_speed_kph=100.0)
        smoothed = shrink_profiles(profiles, cfg, spec=SPEC, class_default=DEFAULTS)
        self.assertTrue(all(c.speed_kph <= 100.0 for c in smoothed[1].values()))

    def test_rows_match_the_table_layout(self):
        profiles = {1: heavy_profile(1, 25.0, 50)}
        smoothed = shrink_profiles(profiles, self.cfg, spec=SPEC, class_default=DEFAULTS)
        row = smoothed[1][(1, 8)].to_row()
        self.assertEqual(len(row), 10)
        self.assertEqual(row[:3], (1, 1, 8))
        self.assertEqual(row[-1], "observed")


if __name__ == "__main__":
    unittest.main()
