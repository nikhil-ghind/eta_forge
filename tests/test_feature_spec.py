import math
import unittest
from datetime import datetime, timezone

from eta_forge.config import Config
from eta_forge.features.builder import FeatureBuilder, feature_hash
from eta_forge.features.spec import FeatureSpec, residual_spec, route_feature_names
from eta_forge.graph.summary import summarize_route
from eta_forge.graph.synthetic import SyntheticGraph
from eta_forge.types import FeatureSpecMismatch, ValidationError, WeatherObservation

UTC = timezone.utc


class TestFeatureSpec(unittest.TestCase):
    def test_rejects_duplicates_and_stray_categoricals(self):
        with self.assertRaises(ValidationError):
            FeatureSpec(names=("a", "b", "a"))
        with self.assertRaises(ValidationError):
            FeatureSpec(names=("a", "b"), categorical=("c",))

    def test_hash_is_stable_and_order_sensitive(self):
        a = FeatureSpec(names=("x", "y"), categorical=("y",))
        self.assertEqual(a.hash, FeatureSpec(names=("x", "y"), categorical=("y",)).hash)
        self.assertNotEqual(a.hash, FeatureSpec(names=("y", "x"), categorical=("y",)).hash)
        self.assertNotEqual(a.hash, FeatureSpec(names=("x", "y")).hash)
        self.assertNotEqual(a.hash, FeatureSpec(names=("x", "y"), categorical=("y",), version=2).hash)

    def test_vectorize_orders_and_fills_nan(self):
        spec = FeatureSpec(names=("a", "b", "c"))
        vector = spec.vectorize({"c": 3.0, "a": 1.0, "unused": 9.0})
        self.assertEqual(vector[0], 1.0)
        self.assertTrue(math.isnan(vector[1]))
        self.assertEqual(vector[2], 3.0)

    def test_strict_vectorize_refuses_missing(self):
        with self.assertRaises(ValidationError):
            FeatureSpec(names=("a", "b")).vectorize({"a": 1.0}, strict=True)

    def test_check_detects_drift(self):
        spec = FeatureSpec(names=("a", "b"))
        spec.check(spec.hash)
        with self.assertRaises(FeatureSpecMismatch):
            spec.check(FeatureSpec(names=("b", "a")).hash)

    def test_round_trip_and_tamper_detection(self):
        spec = FeatureSpec(names=("a", "b"), categorical=("b",))
        self.assertEqual(FeatureSpec.from_dict(spec.to_dict()), spec)
        tampered = spec.to_dict()
        tampered["names"] = ["a", "b", "c"]
        with self.assertRaises(FeatureSpecMismatch):
            FeatureSpec.from_dict(tampered)

    def test_categorical_indices(self):
        spec = FeatureSpec(names=("a", "b", "c"), categorical=("c", "a"))
        self.assertEqual(spec.categorical_indices, [2, 0])


class TestResidualSpec(unittest.TestCase):
    def setUp(self):
        graph = SyntheticGraph(rows=6, cols=6)
        sw, ne = graph.bounds()
        self.route = graph.route(sw, ne, datetime(2024, 3, 5, 13, 0, tzinfo=UTC))

    def test_covers_every_route_summary_feature(self):
        produced = set(summarize_route(self.route).to_features())
        self.assertEqual(produced, set(route_feature_names()))
        self.assertTrue(produced <= set(residual_spec().names))

    def test_weather_switch(self):
        with_weather = residual_spec(use_weather=True)
        without = residual_spec(use_weather=False)
        self.assertIn("precipitation_mm_h", with_weather.names)
        self.assertNotIn("precipitation_mm_h", without.names)
        self.assertNotIn("weather_condition", without.categorical)
        self.assertNotEqual(with_weather.hash, without.hash)


class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        graph = SyntheticGraph(rows=6, cols=6)
        sw, ne = graph.bounds()
        self.route = graph.route(sw, ne, datetime(2024, 3, 5, 13, 0, tzinfo=UTC))
        self.builder = FeatureBuilder(self.cfg, scheme="grid")

    def test_vector_matches_the_spec_strictly(self):
        row = self.builder.base_features(self.route)
        vector = self.builder.finalize(row, strict=True)
        self.assertEqual(len(vector), len(self.builder.spec))

    def test_missing_weather_is_flagged_not_zeroed(self):
        row = self.builder.base_features(self.route, weather=None)
        self.assertEqual(row.features["weather_missing"], 1.0)
        self.assertTrue(math.isnan(row.features["precipitation_mm_h"]))
        rainy = WeatherObservation(datetime(2024, 3, 5, 13, 0, tzinfo=UTC), 7.0, 3.5, 12.0, "rain")
        row = self.builder.base_features(self.route, weather=rainy)
        self.assertEqual(row.features["weather_missing"], 0.0)
        self.assertEqual(row.features["precipitation_mm_h"], 3.5)

    def test_explicit_encodings_override_the_fitted_encoders(self):
        row = self.builder.base_features(self.route)
        vector = self.builder.finalize(row, origin_te=42.0, dest_te=-7.0)
        spec = self.builder.spec
        self.assertEqual(vector[spec.index("origin_cell_te")], 42.0)
        self.assertEqual(vector[spec.index("dest_cell_te")], -7.0)

    def test_state_round_trip(self):
        self.builder.origin_encoder.fit(["c1", "c1", "c2"], [10.0, 20.0, 30.0])
        restored = FeatureBuilder.from_state(self.cfg, self.builder.state())
        row = self.builder.base_features(self.route)
        a = self.builder.finalize(row)
        b = restored.finalize(row)
        self.assertEqual(feature_hash(a), feature_hash(b))

    def test_state_from_a_different_layout_is_refused(self):
        state = self.builder.state()
        other = Config()
        other.features.use_weather = False
        with self.assertRaises(FeatureSpecMismatch):
            FeatureBuilder.from_state(other, state)


class TestFeatureHash(unittest.TestCase):
    def test_stable_and_nan_canonical(self):
        self.assertEqual(feature_hash([1.0, float("nan")]), feature_hash([1.0, float("nan")]))
        self.assertNotEqual(feature_hash([1.0, 2.0]), feature_hash([2.0, 1.0]))
        self.assertEqual(len(feature_hash([0.0])), 16)


if __name__ == "__main__":
    unittest.main()
