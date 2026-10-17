import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.config import EvalConfig
from eta_forge.eval.slices import (
    EvalRecord,
    by_dow,
    by_hour,
    distance_label,
    slice_metrics,
    standard_slicers,
    worst_slices,
)

UTC = timezone.utc
T0 = datetime(2024, 3, 4, tzinfo=UTC)


def record(i, *, hour=8, dow=0, distance=2_500.0, actual=600.0, p50=600.0, p90=800.0, cell="c1",
           merchant=None):
    return EvalRecord(
        trip_id=f"t{i}", departure_at=T0 + timedelta(minutes=i), actual_s=actual,
        baseline_s=p50 - 30.0, p50_s=p50, p90_s=p90, distance_m=distance, hour=hour, dow=dow,
        origin_cell=cell, dest_cell=cell, merchant_id=merchant,
    )


class TestSliceKeys(unittest.TestCase):
    def test_distance_labels(self):
        buckets = [1_000.0, 3_000.0, 8_000.0]
        self.assertEqual(distance_label(500.0, buckets), "<1km")
        self.assertEqual(distance_label(1_000.0, buckets), "1-3km")
        self.assertEqual(distance_label(7_999.0, buckets), "3-8km")
        self.assertEqual(distance_label(20_000.0, buckets), ">=8km")

    def test_hour_and_dow(self):
        self.assertEqual(by_hour(record(0, hour=7)), "07:00")
        self.assertEqual(by_dow(record(0, dow=4)), "4-Fri")

    def test_standard_slicers(self):
        self.assertEqual(
            set(standard_slicers(EvalConfig())), {"hour", "dow", "distance", "origin_cell", "merchant"}
        )


class TestSliceMetrics(unittest.TestCase):
    def test_small_slices_are_dropped_and_counted(self):
        records = [record(i, hour=8) for i in range(40)] + [record(100 + i, hour=3) for i in range(5)]
        slices, dropped = slice_metrics(records, "hour", by_hour, min_size=30)
        self.assertEqual([s.key for s in slices], ["08:00"])
        self.assertEqual(dropped, 1)
        self.assertEqual(slices[0].n, 40)

    def test_none_keys_are_skipped(self):
        slices, dropped = slice_metrics(
            [record(i) for i in range(40)], "merchant", lambda r: r.merchant_id, min_size=1
        )
        self.assertEqual(slices, [])
        self.assertEqual(dropped, 0)

    def test_a_late_slice_shows_up_first(self):
        # Midtown at 18:00 is four minutes late; everywhere else is on time.
        on_time = [record(i, hour=10, cell="uptown") for i in range(50)]
        late = [record(100 + i, hour=18, cell="midtown", actual=840.0) for i in range(50)]
        slices, _ = slice_metrics(on_time + late, "origin_cell", lambda r: r.origin_cell, min_size=30)
        worst = worst_slices(slices, "bias_s", n=1)
        self.assertEqual(worst[0].key, "midtown")
        self.assertAlmostEqual(worst[0].get("bias_s"), -240.0)
        self.assertAlmostEqual(worst[0].get("coverage_p90"), 0.0)

    def test_overall_average_can_hide_a_bad_slice(self):
        early = [record(i, cell="a", actual=360.0) for i in range(50)]     # 4 min early
        late = [record(100 + i, cell="b", actual=840.0) for i in range(50)]  # 4 min late
        slices, _ = slice_metrics(early + late, "origin_cell", lambda r: r.origin_cell, min_size=30)
        biases = sorted(s.get("bias_s") for s in slices)
        self.assertAlmostEqual(biases[0], -240.0)
        self.assertAlmostEqual(biases[1], 240.0)

    def test_rows(self):
        slices, _ = slice_metrics([record(i) for i in range(30)], "hour", by_hour, min_size=30)
        row = slices[0].to_row()
        self.assertEqual(row["dimension"], "hour")
        self.assertIn("coverage_p90", row)


if __name__ == "__main__":
    unittest.main()
