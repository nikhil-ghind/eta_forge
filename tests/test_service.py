import math
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone

from eta_forge.config import Config
from eta_forge.data.synthetic import SyntheticConfig, SyntheticSource
from eta_forge.decompose import naive_sum
from eta_forge.features.builder import FeatureBuilder
from eta_forge.graph.geometry import haversine_m
from eta_forge.graph.synthetic import SyntheticGraph
from eta_forge.models.dataset import build_rows
from eta_forge.models.dwell import DwellEstimator
from eta_forge.models.prep_time import PrepTimeModel
from eta_forge.models.registry import ModelRegistry
from eta_forge.models.residual import ResidualModel, ResidualTrainer
from eta_forge.service import EtaService, ModelBundle
from eta_forge.types import EtaRequest, ModelNotFound, NoRouteFound

UTC = timezone.utc


def small_config():
    cfg = Config()
    cfg.model.backend = "stumps"
    cfg.model.stump_rounds = 15
    cfg.model.min_child_samples = 10
    return cfg


class SlowRouter:
    def route(self, origin, destination, departure_at):
        time.sleep(0.3)
        raise AssertionError("should have timed out first")


class NoRouteRouter:
    def route(self, origin, destination, departure_at):
        raise NoRouteFound("no routable vertex within 300 m")


class BrokenRouter:
    def route(self, origin, destination, departure_at):
        raise RuntimeError("server closed the connection unexpectedly")


class TestServiceWithModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_config()
        cls.graph = SyntheticGraph(rows=8, cols=8)
        source = SyntheticSource(
            cls.graph,
            SyntheticConfig(days=4, trips_per_day=40, orders_per_day=80, merchants=6),
        )
        trips = [t.record for t in source.trips()]
        weather = source.weather_table()
        rows, _ = build_rows(trips, cls.graph, FeatureBuilder(cls.cfg, scheme="grid"), weather=weather)
        residual, cls.report, _ = ResidualTrainer(cls.cfg).fit(rows)
        prep, _ = PrepTimeModel.fit(source.prep_observations(), cls.cfg)
        dwell = DwellEstimator.fit(source.dwell_observations(scheme="grid"), k=40.0)

        cls.root = tempfile.mkdtemp()
        cls.registry = ModelRegistry(cls.root)
        cls.versions = {
            "residual": cls.registry.register("residual", residual.save, {}),
            "prep": cls.registry.register("prep", prep.save, {}),
            "dwell": cls.registry.register("dwell", dwell.save, {}),
        }
        for version in cls.versions.values():
            cls.registry.promote(version.version)
        cls.residual = residual
        cls.merchant = source.merchants()[0]
        cls.service = EtaService(
            cls.cfg, cls.graph, bundle=ModelBundle.load(cls.registry, cls.cfg),
            registry=cls.registry, weather=weather,
        )
        cls.sw, cls.ne = cls.graph.bounds()
        cls.when = datetime(2024, 1, 3, 17, 30, tzinfo=UTC)

    @classmethod
    def tearDownClass(cls):
        cls.service.close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def ride(self, **kwargs):
        return self.service.estimate(
            EtaRequest(origin=self.sw, destination=self.ne, departure_at=self.when, **kwargs)
        )

    def test_ride_is_travel_only(self):
        response = self.ride()
        self.assertFalse(response.degraded, response.degraded_reason)
        self.assertEqual(response.breakdown.prep.p50, 0.0)
        self.assertEqual(response.breakdown.handoff.p90, 0.0)
        self.assertAlmostEqual(response.eta.p50, response.breakdown.travel.p50)
        self.assertGreaterEqual(response.eta.p90, response.eta.p50)
        self.assertGreater(response.distance_m, haversine_m(self.sw, self.ne))

    def test_travel_is_baseline_plus_residual(self):
        b = self.ride().breakdown
        floor = self.cfg.model.min_travel_fraction * b.baseline_travel_s
        self.assertAlmostEqual(
            b.travel.p50, max(floor, b.baseline_travel_s + b.residual_correction_p50_s)
        )

    def test_route_is_returned_on_request(self):
        self.assertIsNone(self.ride().route)
        self.assertGreater(len(self.ride(include_route=True).route.legs), 0)

    def test_delivery_decomposition(self):
        response = self.service.estimate(EtaRequest(
            origin=self.merchant.location, destination=self.ne, departure_at=self.when,
            merchant_id=self.merchant.merchant_id, order_item_count=3, open_orders_at_merchant=5,
        ))
        b = response.breakdown
        self.assertFalse(response.degraded, response.degraded_reason)
        self.assertGreater(b.prep.p50, 0.0)
        self.assertGreater(b.handoff.p50, 0.0)
        self.assertAlmostEqual(response.eta.p50, b.prep.p50 + b.travel.p50 + b.handoff.p50)
        self.assertLessEqual(response.eta.p90, naive_sum([b.prep, b.travel, b.handoff]).p90 + 1e-6)
        self.assertGreater(response.eta.p90 - response.eta.p50, b.travel.spread)

    def test_versions_are_reported(self):
        versions = self.ride().model_versions
        self.assertEqual(versions["residual"], self.versions["residual"].version)
        self.assertEqual(versions["dwell"], self.versions["dwell"].version)

    def test_pinning(self):
        pinned = self.ride(model_version=self.versions["residual"].version)
        self.assertEqual(pinned.model_versions["residual"], self.versions["residual"].version)
        with self.assertRaises(ModelNotFound):
            self.ride(model_version="residual/20200101T0000Z-00000000")

    def test_reload_and_ready(self):
        self.assertTrue(self.service.ready())
        versions = self.service.reload()
        self.assertEqual(versions["prep"], self.versions["prep"].version)

    def test_saved_model_predicts_identically(self):
        loaded = ResidualModel.load(self.cfg, self.versions["residual"].path)
        route = self.graph.route(self.sw, self.ne, self.when)
        a, _ = self.residual.builder.vector(route)
        b, _ = loaded.builder.vector(route)
        self.assertEqual(
            [None if math.isnan(v) else v for v in a],
            [None if math.isnan(v) else v for v in b],
        )
        self.assertEqual(
            self.residual.predict_travel(route.duration_s, a)[0],
            loaded.predict_travel(route.duration_s, b)[0],
        )

    def test_training_report(self):
        self.assertEqual(self.report.backend, "stumps")
        self.assertGreater(self.report.sizes["train"], 0)
        self.assertIn("coverage_p90", self.report.metrics["test"])


class TestDegradedPaths(unittest.TestCase):
    def setUp(self):
        self.cfg = small_config()
        self.origin = SyntheticGraph(rows=4, cols=4).bounds()[0]
        _, self.destination = SyntheticGraph(rows=10, cols=10).bounds()
        self.when = datetime(2024, 1, 3, 8, 0, tzinfo=UTC)

    def estimate(self, router, **kwargs):
        service = EtaService(self.cfg, router)
        try:
            return service.estimate(EtaRequest(
                origin=self.origin, destination=self.destination, departure_at=self.when, **kwargs
            ))
        finally:
            service.close()

    def test_routing_timeout_falls_back_to_the_crow_line(self):
        self.cfg.service.route_timeout_ms = 50
        response = self.estimate(SlowRouter())
        self.assertTrue(response.degraded)
        self.assertIn("timed out", response.degraded_reason)
        crow = haversine_m(self.origin, self.destination)
        expected = crow * self.cfg.graph.fallback_detour_factor / (self.cfg.graph.fallback_speed_kph / 3.6)
        self.assertAlmostEqual(response.eta.p50, expected, places=3)
        self.assertGreater(response.eta.p90, response.eta.p50)

    def test_no_route_is_degraded_not_raised(self):
        response = self.estimate(NoRouteRouter())
        self.assertTrue(response.degraded)
        self.assertIn("routing failed", response.degraded_reason)

    def test_unexpected_router_errors_are_degraded(self):
        response = self.estimate(BrokenRouter())
        self.assertTrue(response.degraded)
        self.assertIn("RuntimeError", response.degraded_reason)

    def test_no_models_serves_the_baseline_with_defaults(self):
        graph = SyntheticGraph(rows=10, cols=10)
        response = self.estimate(graph, merchant_id="m1")
        self.assertTrue(response.degraded)
        self.assertIn("baseline only", response.degraded_reason)
        self.assertEqual(response.breakdown.prep.p50, self.cfg.service.default_prep_p50_s)
        self.assertEqual(response.breakdown.handoff.p90, self.cfg.service.default_handoff_p90_s)
        self.assertEqual(response.model_versions, {})
        self.assertFalse(EtaService(self.cfg, graph).ready())


if __name__ == "__main__":
    unittest.main()
