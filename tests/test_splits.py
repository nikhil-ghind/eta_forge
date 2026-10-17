import random
import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.models.splits import assert_no_leakage, time_blocks, time_split, TimeSplit
from eta_forge.types import ValidationError

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def shuffled_times(n, seed=3):
    times = [T0 + timedelta(hours=i) for i in range(n)]
    random.Random(seed).shuffle(times)
    return times


class TestTimeSplit(unittest.TestCase):
    def test_sizes_follow_the_fractions(self):
        split = time_split(shuffled_times(100), train_fraction=0.7, validation_fraction=0.15)
        self.assertEqual(len(split.train), 70)
        self.assertEqual(len(split.validation), 15)
        self.assertEqual(len(split.test), 15)

    def test_split_is_by_time_not_by_position(self):
        times = shuffled_times(200)
        split = time_split(times, train_fraction=0.6, validation_fraction=0.2)
        assert_no_leakage(split, times)
        self.assertLess(max(times[i] for i in split.train), min(times[i] for i in split.validation))
        self.assertLess(max(times[i] for i in split.validation), min(times[i] for i in split.test))

    def test_each_part_is_time_ordered(self):
        times = shuffled_times(50)
        split = time_split(times)
        for part in (split.train, split.validation, split.test):
            self.assertEqual([times[i] for i in part], sorted(times[i] for i in part))

    def test_every_row_is_used_exactly_once(self):
        times = shuffled_times(77)
        split = time_split(times)
        used = split.train + split.validation + split.test
        self.assertEqual(sorted(used), list(range(77)))

    def test_embargo_drops_rows_near_the_boundaries(self):
        times = [T0 + timedelta(hours=i) for i in range(100)]
        split = time_split(times, embargo=timedelta(hours=3, minutes=30))
        self.assertEqual(split.embargoed, 6)
        self.assertGreater(min(times[i] for i in split.validation), split.train_end + timedelta(hours=3))

    def test_invalid_fractions(self):
        with self.assertRaises(ValidationError):
            time_split(shuffled_times(10), train_fraction=0.9, validation_fraction=0.2)
        with self.assertRaises(ValidationError):
            time_split(shuffled_times(10), train_fraction=0.0)

    def test_tiny_inputs(self):
        split = time_split([T0, T0 + timedelta(hours=1)])
        self.assertEqual(split.train, [0, 1])
        self.assertEqual(split.test, [])


class TestBlocksAndLeakage(unittest.TestCase):
    def test_blocks_are_contiguous_and_cover(self):
        blocks = time_blocks(list(range(10)), 3)
        self.assertEqual(sum(blocks, []), list(range(10)))
        self.assertEqual([len(b) for b in blocks], [3, 3, 4])

    def test_more_blocks_than_rows(self):
        self.assertEqual(time_blocks([5, 6], 5), [[5], [6]])

    def test_leakage_is_detected(self):
        times = [T0 + timedelta(hours=i) for i in range(6)]
        bad = TimeSplit(train=[0, 1, 5], validation=[2, 3], test=[4])
        with self.assertRaises(ValidationError):
            assert_no_leakage(bad, times)


if __name__ == "__main__":
    unittest.main()
