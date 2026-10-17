import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.config import Config
from eta_forge.data.synthetic import SyntheticConfig, SyntheticSource
from eta_forge.graph.synthetic import SyntheticGraph
from eta_forge.models.dwell import DwellEstimator
from eta_forge.models.prep_time import PrepPriors, PrepTimeModel
from eta_forge.types import DwellObservation, PrepObservation

UTC = timezone.utc
T0 = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)


def order(i, merchant, cuisine, prep_s, minutes=0):
    return PrepObservation(
        order_id=f"o{i}",
        merchant_id=merchant,
        placed_at=T0 + timedelta(minutes=minutes + i),
        prep_s=prep_s,
        cuisine_category=cuisine,
        item_count=2,
    )


def small_config():
    cfg = Config()
    cfg.model.backend = "stumps"
    cfg.model.stump_rounds = 40
    return cfg


class TestPrepPriors(unittest.TestCase):
    def setUp(self):
        obs = [order(i, "busy", "pizza", 900.0 + (i % 10) * 10) for i in range(400)]
        obs += [order(400 + i, "other", "pizza", 600.0) for i in range(40)]
        obs += [order(500 + i, "sushi1", "sushi", 1200.0) for i in range(40)]
        obs += [order(600, "new", "pizza", 300.0)]
        self.priors = PrepPriors.fit(obs, k=25.0)

    def test_well_observed_merchant_keeps_its_own_history(self):
        prior, cold = self.priors.lookup("busy", "pizza")
        self.assertFalse(cold)
        self.assertAlmostEqual(prior.p50, 945.0, delta=25.0)

    def test_sparse_merchant_is_pulled_to_its_cuisine(self):
        prior, _ = self.priors.lookup("new", "pizza")
        cuisine = self.priors.cuisine["pizza"]
        # One 5-minute order must not make this a 5-minute kitchen.
        self.assertGreater(prior.p50, 300.0 + 0.8 * (cuisine.p50 - 300.0))

    def test_cold_start_uses_the_cuisine_then_global(self):
        prior, cold = self.priors.lookup("never-seen", "sushi")
        self.assertTrue(cold)
        self.assertEqual(prior, self.priors.cuisine["sushi"])
        prior, cold = self.priors.lookup("never-seen", "tacos")
        self.assertEqual(prior, self.priors.global_prior)

    def test_round_trip(self):
        restored = PrepPriors.from_dict(self.priors.to_dict())
        self.assertEqual(restored.lookup("busy", None), self.priors.lookup("busy", None))


class TestPrepTimeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_config()
        source = SyntheticSource(
            SyntheticGraph(rows=6, cols=6),
            SyntheticConfig(days=10, trips_per_day=1, orders_per_day=120, merchants=12),
        )
        cls.observations = source.prep_observations()
        cls.model, cls.report = PrepTimeModel.fit(cls.observations, cls.cfg)

    def test_trains_corrections_with_enough_rows(self):
        self.assertTrue(self.report["corrections"])
        self.assertEqual(self.report["merchants"], 12)

    def test_predictions_are_ordered_and_positive(self):
        for o in self.observations[:50]:
            pair = self.model.predict(
                merchant_id=o.merchant_id, placed_at=o.placed_at, cuisine=o.cuisine_category,
                item_count=o.item_count, open_orders=o.open_orders_at_merchant,
            )
            self.assertGreater(pair.p50, 0.0)
            self.assertGreaterEqual(pair.p90, pair.p50)

    def test_held_out_coverage_is_plausible(self):
        coverage = self.report["test"]["coverage_p90"]
        self.assertGreater(coverage, 0.7)
        self.assertLess(coverage, 0.995)

    def test_load_signal_raises_the_estimate(self):
        o = self.observations[0]
        quiet = self.model.predict(merchant_id=o.merchant_id, placed_at=o.placed_at, open_orders=0)
        slammed = self.model.predict(merchant_id=o.merchant_id, placed_at=o.placed_at, open_orders=14)
        self.assertGreaterEqual(slammed.p50, quiet.p50)

    def test_save_and_load(self):
        directory = tempfile.mkdtemp()
        try:
            self.model.save(directory)
            restored = PrepTimeModel.load(directory)
            o = self.observations[5]
            kwargs = dict(merchant_id=o.merchant_id, placed_at=o.placed_at, item_count=3, open_orders=4)
            self.assertEqual(restored.predict(**kwargs), self.model.predict(**kwargs))
        finally:
            shutil.rmtree(directory)

    def test_small_corpus_falls_back_to_priors(self):
        model, report = PrepTimeModel.fit(self.observations[:150], self.cfg)
        self.assertFalse(report["corrections"])
        pair = model.predict(merchant_id="unknown", placed_at=T0)
        self.assertGreater(pair.p50, 0.0)


class TestDwell(unittest.TestCase):
    def test_tower_cells_dwell_longer(self):
        obs = [DwellObservation(f"a{i}", "tower", T0, 300.0 + i) for i in range(100)]
        obs += [DwellObservation(f"b{i}", "kerb", T0, 60.0 + i % 5) for i in range(100)]
        estimator = DwellEstimator.fit(obs, k=40.0)
        self.assertGreater(estimator.predict("tower").p50, estimator.predict("kerb").p50)
        self.assertGreater(estimator.parking_difficulty("tower"), 1.0)
        self.assertLess(estimator.parking_difficulty("kerb"), 1.0)
        self.assertEqual(estimator.predict("unknown").p50, estimator.global_prior.p50)

    def test_default_and_round_trip(self):
        estimator = DwellEstimator.default(90.0, 240.0)
        self.assertEqual(estimator.predict("anything").p90, 240.0)
        directory = tempfile.mkdtemp()
        try:
            fitted = DwellEstimator.fit(
                [DwellObservation("x", "c1", T0, 120.0)] * 5, k=10.0
            )
            fitted.save(directory)
            self.assertEqual(DwellEstimator.load(directory).predict("c1"), fitted.predict("c1"))
        finally:
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
