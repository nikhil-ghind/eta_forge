import unittest
from datetime import datetime, timezone

from eta_forge.api.metrics import Counter, Histogram, ServiceMetrics
from eta_forge.api.schemas import (
    SchemaError,
    error_body,
    eta_response_body,
    parse_eta_request,
    parse_route_query,
)
from eta_forge.types import (
    EtaBreakdown,
    EtaResponse,
    FeatureSpecMismatch,
    Location,
    ModelNotFound,
    NoRouteFound,
    QuantilePair,
)

UTC = timezone.utc
RIDE = {
    "origin": {"lat": 40.7580, "lon": -73.9855},
    "destination": {"lat": 40.7061, "lon": -74.0087},
    "departure_at": "2024-03-05T13:00:00Z",
}


class TestParseEtaRequest(unittest.TestCase):
    def errors(self, payload, delivery=False):
        with self.assertRaises(SchemaError) as ctx:
            parse_eta_request(payload, delivery=delivery)
        return ctx.exception.errors

    def test_valid_ride(self):
        request = parse_eta_request(RIDE, delivery=False)
        self.assertEqual(request.origin, Location(40.7580, -73.9855))
        self.assertEqual(request.departure_at, datetime(2024, 3, 5, 13, 0, tzinfo=UTC))
        self.assertFalse(request.is_delivery)
        self.assertFalse(request.include_route)

    def test_pair_form_and_default_departure(self):
        request = parse_eta_request(
            {"origin": [40.758, -73.9855], "destination": [40.7061, -74.0087]}, delivery=False
        )
        self.assertEqual(request.destination.lat, 40.7061)
        self.assertIsNotNone(request.departure_at.tzinfo)

    def test_valid_delivery(self):
        body = dict(RIDE, merchant_id="m0001", order_item_count=3, order_value_usd=42.5,
                    open_orders_at_merchant=6)
        request = parse_eta_request(body, delivery=True)
        self.assertTrue(request.is_delivery)
        self.assertEqual(request.order_item_count, 3)

    def test_delivery_requires_a_merchant(self):
        errors = self.errors(RIDE, delivery=True)
        self.assertEqual(errors[0]["loc"], ["merchant_id"])

    def test_unknown_fields_are_rejected(self):
        errors = self.errors(dict(RIDE, departure_time="2024-03-05T13:00:00Z"))
        self.assertEqual(errors[0]["loc"], ["departure_time"])

    def test_delivery_fields_on_the_ride_endpoint_get_a_hint(self):
        errors = self.errors(dict(RIDE, merchant_id="m1"))
        self.assertIn("/v1/eta/delivery", errors[0]["msg"])

    def test_coordinates(self):
        self.assertEqual(self.errors(dict(RIDE, origin={"lat": 91.0, "lon": 0.0}))[0]["loc"], ["origin"])
        self.assertEqual(self.errors(dict(RIDE, origin={"lat": True, "lon": 0.0}))[0]["loc"], ["origin"])
        self.assertEqual(self.errors(dict(RIDE, origin={"lat": 40.0}))[0]["loc"], ["origin"])
        self.assertEqual(self.errors(dict(RIDE, origin="40.7,-73.9"))[0]["loc"], ["origin"])
        missing = dict(RIDE)
        del missing["destination"]
        self.assertEqual(self.errors(missing)[0]["loc"], ["destination"])

    def test_naive_departure_is_rejected(self):
        errors = self.errors(dict(RIDE, departure_at="2024-03-05T13:00:00"))
        self.assertEqual(errors[0]["loc"], ["departure_at"])

    def test_field_types(self):
        body = dict(RIDE, merchant_id="m1", order_item_count=2.5, open_orders_at_merchant=-1)
        locs = {tuple(e["loc"]) for e in self.errors(body, delivery=True)}
        self.assertEqual(locs, {("order_item_count",), ("open_orders_at_merchant",)})
        self.assertEqual(self.errors(dict(RIDE, include_route="yes"))[0]["loc"], ["include_route"])
        self.assertEqual(self.errors(dict(RIDE, vehicle_type="hovercraft"))[0]["loc"], ["vehicle_type"])

    def test_every_problem_is_reported_at_once(self):
        errors = self.errors({"origin": {"lat": 200, "lon": 0}, "bogus": 1})
        self.assertGreaterEqual(len(errors), 3)

    def test_body_must_be_an_object(self):
        self.assertEqual(self.errors([1, 2, 3])[0]["loc"], [])


class TestRouteQuery(unittest.TestCase):
    def test_parses_lat_lon_pairs(self):
        origin, destination, departure = parse_route_query(
            {"origin": "40.758,-73.9855", "destination": "40.7061, -74.0087"}
        )
        self.assertEqual(origin, Location(40.758, -73.9855))
        self.assertEqual(destination.lon, -74.0087)
        self.assertIsNone(departure)

    def test_errors(self):
        with self.assertRaises(SchemaError):
            parse_route_query({"origin": "40.7", "destination": "40.7,-74.0"})
        with self.assertRaises(SchemaError):
            parse_route_query({"destination": "40.7,-74.0"})


class TestResponses(unittest.TestCase):
    def test_eta_body_carries_minutes_and_breakdown(self):
        response = EtaResponse(
            eta=QuantilePair(900.0, 1260.0),
            breakdown=EtaBreakdown(
                prep=QuantilePair.zero(), travel=QuantilePair(900.0, 1260.0),
                handoff=QuantilePair.zero(), baseline_travel_s=820.0,
                residual_correction_p50_s=80.0, residual_correction_p90_s=440.0,
            ),
            distance_m=4200.0,
            model_versions={"residual": "residual/20260824T1200Z-a1b2c3d4"},
            computed_at=datetime(2024, 3, 5, 13, 0, tzinfo=UTC),
        )
        body = eta_response_body(response)
        self.assertEqual(body["eta_minutes"], {"p50": 15.0, "p90": 21.0})
        self.assertEqual(body["eta_seconds"]["p90"], 1260.0)
        self.assertEqual(body["breakdown"]["baseline_travel_s"], 820.0)
        self.assertFalse(body["degraded"])

    def test_error_statuses(self):
        self.assertEqual(error_body(SchemaError([{"loc": ["x"], "msg": "bad"}]))[0], 422)
        self.assertEqual(error_body(NoRouteFound("no path"))[0], 404)
        self.assertEqual(error_body(ModelNotFound("nope"))[0], 404)
        self.assertEqual(error_body(FeatureSpecMismatch("drift"))[0], 409)
        status, body = error_body(RuntimeError("secret connection string"))
        self.assertEqual(status, 500)
        self.assertNotIn("secret", str(body))


class TestMetrics(unittest.TestCase):
    def test_counter_and_histogram_render(self):
        counter = Counter("eta_test_total", "Test.")
        counter.inc(endpoint="/v1/eta", status="200")
        counter.inc(2, endpoint="/v1/eta", status="200")
        self.assertEqual(counter.value(endpoint="/v1/eta", status="200"), 3.0)
        histogram = Histogram("eta_test_seconds", "Test.", [0.1, 1.0])
        for value in (0.05, 0.1, 0.5, 5.0):
            histogram.observe(value)
        text = "\n".join(histogram.render())
        self.assertIn('eta_test_seconds_bucket{le="0.1"} 2', text)
        self.assertIn('eta_test_seconds_bucket{le="1"} 3', text)
        self.assertIn('eta_test_seconds_bucket{le="+Inf"} 4', text)
        self.assertIn("eta_test_seconds_count 4", text)
        with self.assertRaises(ValueError):
            counter.inc(-1)

    def test_service_metrics_bound_degraded_labels(self):
        metrics = ServiceMetrics()
        metrics.record_eta("ride", 600.0, 900.0, "routing failed: no path between 1 and 2; no dwell model")
        self.assertEqual(metrics.degraded.value(reason="routing failed"), 1.0)
        self.assertEqual(metrics.degraded.value(reason="no dwell model"), 1.0)
        metrics.set_models({"residual": "residual/x"})
        rendered = metrics.render()
        self.assertIn("# TYPE eta_forge_eta_seconds histogram", rendered)
        self.assertIn('eta_forge_model_info{model="residual",version="residual/x"} 1', rendered)


if __name__ == "__main__":
    unittest.main()
