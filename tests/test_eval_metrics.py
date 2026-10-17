import math
import unittest

from eta_forge.eval.metrics import (
    bias,
    late_rate,
    mae,
    mape,
    point_metrics,
    quantile_metrics,
    relative_improvement,
    rmse,
    sharpness,
)


class TestPointMetrics(unittest.TestCase):
    def test_mae_rmse_bias(self):
        actual = [100.0, 200.0, 300.0]
        predicted = [110.0, 190.0, 330.0]
        self.assertAlmostEqual(mae(actual, predicted), 50.0 / 3)
        self.assertAlmostEqual(rmse(actual, predicted), math.sqrt((100 + 100 + 900) / 3))
        self.assertAlmostEqual(bias(actual, predicted), 10.0)

    def test_mape_floor_tames_short_trips(self):
        # A 10-second trip predicted at 70 s is a 600% error unfloored.
        self.assertAlmostEqual(mape([10.0], [70.0], floor_s=60.0), 1.0)
        self.assertAlmostEqual(mape([600.0], [660.0], floor_s=60.0), 0.1)
        with self.assertRaises(ValueError):
            mape([1.0], [1.0], floor_s=0.0)

    def test_late_rate(self):
        self.assertAlmostEqual(late_rate([100.0, 200.0, 300.0, 400.0], [150.0, 150.0, 350.0, 350.0]), 0.5)

    def test_empty_inputs_are_nan(self):
        self.assertTrue(math.isnan(mae([], [])))
        self.assertTrue(math.isnan(sharpness([], [])))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            mae([1.0], [1.0, 2.0])

    def test_point_metric_bundle(self):
        m = point_metrics([100.0, 200.0], [100.0, 220.0])
        self.assertEqual(m["n"], 2.0)
        self.assertAlmostEqual(m["mae_s"], 10.0)
        self.assertAlmostEqual(m["bias_s"], 10.0)


class TestQuantileMetrics(unittest.TestCase):
    def test_coverage_and_sharpness(self):
        actual = [100.0, 200.0, 300.0, 400.0]
        p50 = [90.0, 210.0, 280.0, 380.0]
        p90 = [150.0, 260.0, 290.0, 500.0]
        m = quantile_metrics(actual, p50, p90)
        self.assertAlmostEqual(m["coverage_p90"], 0.75)
        self.assertAlmostEqual(m["coverage_p50"], 0.25)
        self.assertAlmostEqual(m["sharpness_s"], (60 + 50 + 10 + 120) / 4)
        self.assertGreater(m["pinball_p90"], 0.0)

    def test_relative_improvement(self):
        self.assertAlmostEqual(relative_improvement(70.0, 100.0), 0.3)
        self.assertIsNone(relative_improvement(1.0, 0.0))
        self.assertIsNone(relative_improvement(1.0, float("nan")))


if __name__ == "__main__":
    unittest.main()
