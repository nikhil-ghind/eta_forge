import math
import random
import unittest

from eta_forge.eval.calibration import (
    calibration_curve,
    calibration_error,
    implied_quantile,
    inverse_normal_cdf,
)


class TestInverseNormal(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(inverse_normal_cdf(0.5), 0.0, places=9)
        self.assertAlmostEqual(inverse_normal_cdf(0.9), 1.2815515655, places=7)
        self.assertAlmostEqual(inverse_normal_cdf(0.975), 1.9599639845, places=7)
        self.assertAlmostEqual(inverse_normal_cdf(0.01), -2.3263478740, places=7)

    def test_symmetry(self):
        for p in (0.001, 0.1, 0.3):
            self.assertAlmostEqual(inverse_normal_cdf(p), -inverse_normal_cdf(1.0 - p), places=8)

    def test_domain(self):
        with self.assertRaises(ValueError):
            inverse_normal_cdf(0.0)
        with self.assertRaises(ValueError):
            inverse_normal_cdf(1.0)


class TestImpliedQuantile(unittest.TestCase):
    def test_passes_through_the_served_pair(self):
        self.assertAlmostEqual(implied_quantile(600.0, 900.0, 0.5), 600.0)
        self.assertAlmostEqual(implied_quantile(600.0, 900.0, 0.9), 900.0)

    def test_monotone_and_floored(self):
        values = [implied_quantile(100.0, 400.0, tau) for tau in (0.05, 0.25, 0.5, 0.75, 0.95)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(implied_quantile(100.0, 400.0, 0.001), 0.0)


class TestCalibrationCurve(unittest.TestCase):
    def test_a_correct_gaussian_forecast_is_calibrated(self):
        rng = random.Random(5)
        mu, sigma = 1000.0, 120.0
        actual = [rng.gauss(mu, sigma) for _ in range(20_000)]
        p50 = [mu] * len(actual)
        p90 = [mu + 1.2815515655 * sigma] * len(actual)
        curve = calibration_curve(actual, p50, p90, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        for point in curve:
            self.assertAlmostEqual(point.observed, point.nominal, delta=0.015)
        self.assertLess(calibration_error(curve)["max_abs_gap"], 0.015)

    def test_a_too_narrow_forecast_undercovers_the_tail(self):
        rng = random.Random(6)
        actual = [rng.gauss(1000.0, 200.0) for _ in range(5_000)]
        curve = calibration_curve(actual, [1000.0] * 5_000, [1050.0] * 5_000, [0.9])
        self.assertLess(curve[0].observed, 0.75)
        self.assertLess(curve[0].gap, 0.0)

    def test_empty(self):
        curve = calibration_curve([], [], [], [0.5])
        self.assertEqual(curve[0].n, 0)
        self.assertTrue(math.isnan(calibration_error(curve)["mean_abs_gap"]))


if __name__ == "__main__":
    unittest.main()
