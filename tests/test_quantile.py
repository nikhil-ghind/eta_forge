import math
import random
import shutil
import tempfile
import unittest

from eta_forge.config import ModelConfig
from eta_forge.models.quantile import (
    QuantilePrior,
    StumpQuantileBooster,
    alpha_key,
    coverage,
    crossing_rate,
    empirical_prior,
    enforce_monotone,
    load_quantile_model,
    make_quantile_model,
    mean_pinball_loss,
    pinball_loss,
    resolve_backend,
    shrink_prior,
)


def step_data(n, seed):
    """y = 100 * [x0 > 0.5] + U(0, 20); x1 is pure noise."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x0, x1 = rng.random(), rng.random()
        X.append([x0, x1])
        y.append((100.0 if x0 > 0.5 else 0.0) + rng.uniform(0.0, 20.0))
    return X, y


class TestPinball(unittest.TestCase):
    def test_asymmetry(self):
        # Under-predicting by 10 at alpha=0.9 costs 9; over-predicting by 10 costs 1.
        self.assertAlmostEqual(pinball_loss(110.0, 100.0, 0.9), 9.0)
        self.assertAlmostEqual(pinball_loss(90.0, 100.0, 0.9), 1.0)
        self.assertAlmostEqual(pinball_loss(110.0, 100.0, 0.5), 5.0)

    def test_minimised_at_the_quantile(self):
        ys = [float(v) for v in range(1, 101)]
        losses = {q: mean_pinball_loss(ys, [q] * len(ys), 0.9) for q in (50.0, 80.0, 90.0, 99.0)}
        self.assertEqual(min(losses, key=losses.get), 90.0)

    def test_coverage_and_crossing(self):
        self.assertAlmostEqual(coverage([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 5.0]), 0.75)
        self.assertAlmostEqual(crossing_rate([5.0, 5.0], [4.0, 6.0]), 0.5)
        self.assertEqual(enforce_monotone([5.0, 5.0], [4.0, 6.0]), [(5.0, 5.0), (5.0, 6.0)])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            mean_pinball_loss([1.0], [1.0, 2.0], 0.5)

    def test_alpha_key(self):
        self.assertEqual(alpha_key(0.5), "q50")
        self.assertEqual(alpha_key(0.9), "q90")
        self.assertEqual(alpha_key(0.05), "q05")


class TestStumpBooster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = step_data(800, seed=1)
        cls.X_test, cls.y_test = step_data(800, seed=2)
        cls.p90 = StumpQuantileBooster(alpha=0.9, n_rounds=120, early_stopping_rounds=None).fit(cls.X, cls.y)
        cls.p50 = StumpQuantileBooster(alpha=0.5, n_rounds=120, early_stopping_rounds=None).fit(cls.X, cls.y)

    def test_learns_the_step(self):
        low = self.p50.predict_one([0.2, 0.5])
        high = self.p50.predict_one([0.8, 0.5])
        self.assertAlmostEqual(low, 10.0, delta=3.0)
        self.assertAlmostEqual(high, 110.0, delta=3.0)

    def test_splits_on_the_informative_feature(self):
        self.assertEqual(self.p50.stumps[0].feature, 0)

    def test_held_out_coverage_matches_the_quantile(self):
        self.assertAlmostEqual(coverage(self.y_test, self.p90.predict(self.X_test)), 0.9, delta=0.05)
        self.assertAlmostEqual(coverage(self.y_test, self.p50.predict(self.X_test)), 0.5, delta=0.06)

    def test_nan_features_do_not_crash(self):
        value = self.p90.predict_one([float("nan"), 0.5])
        self.assertTrue(math.isfinite(value))

    def test_round_trip(self):
        directory = tempfile.mkdtemp()
        try:
            meta = self.p90.save(directory, "residual_q90")
            restored = load_quantile_model(meta, directory)
            self.assertEqual(restored.predict(self.X_test[:20]), self.p90.predict(self.X_test[:20]))
        finally:
            shutil.rmtree(directory)

    def test_early_stopping_truncates(self):
        model = StumpQuantileBooster(alpha=0.5, n_rounds=400, early_stopping_rounds=5).fit(
            self.X, self.y, X_val=self.X_test, y_val=self.y_test
        )
        self.assertLess(len(model.stumps), 400)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            StumpQuantileBooster(alpha=0.5).fit([], [])


class TestFactory(unittest.TestCase):
    def test_stumps_backend(self):
        self.assertEqual(resolve_backend("stumps"), "stumps")
        cfg = ModelConfig(stump_rounds=7)
        model = make_quantile_model(0.9, cfg, backend="stumps")
        self.assertIsInstance(model, StumpQuantileBooster)
        self.assertEqual(model.n_rounds, 7)
        self.assertIn(resolve_backend("auto"), ("lightgbm", "stumps"))


class TestPriors(unittest.TestCase):
    def test_empirical(self):
        prior = empirical_prior([float(v) for v in range(1, 11)])
        self.assertAlmostEqual(prior.p50, 5.5)
        self.assertAlmostEqual(prior.p90, 9.1)
        self.assertEqual(prior.n, 10)
        self.assertIsNone(empirical_prior([]))

    def test_shrinkage_endpoints(self):
        parent = QuantilePrior(600.0, 1000.0, 5000)
        self.assertEqual(shrink_prior(None, parent, 25.0).p50, 600.0)
        big = shrink_prior(QuantilePrior(300.0, 400.0, 10_000_000), parent, 25.0)
        self.assertAlmostEqual(big.p50, 300.0, delta=0.01)
        half = shrink_prior(QuantilePrior(300.0, 400.0, 25), parent, 25.0)
        self.assertAlmostEqual(half.p50, 450.0)
        self.assertGreaterEqual(half.p90, half.p50)


if __name__ == "__main__":
    unittest.main()
