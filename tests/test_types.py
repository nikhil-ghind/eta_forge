import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.types import (
    BoundingBox,
    EtaRequest,
    GpsPoint,
    HighwayClass,
    Location,
    MatchQuality,
    QuantilePair,
    SegmentObservation,
    TripRecord,
    TurnType,
    ValidationError,
)

UTC = timezone.utc


class TestLocation(unittest.TestCase):
    def test_round_trip(self):
        loc = Location(40.7128, -74.0060)
        self.assertEqual(Location.from_dict(loc.to_dict()), loc)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValidationError):
            Location(91.0, 0.0)
        with self.assertRaises(ValidationError):
            Location(0.0, -181.0)

    def test_rejects_nan(self):
        with self.assertRaises(ValidationError):
            Location(float("nan"), 0.0)

    def test_from_pair_requires_two(self):
        self.assertEqual(Location.from_pair([1.0, 2.0]), Location(1.0, 2.0))
        with self.assertRaises(ValidationError):
            Location.from_pair([1.0, 2.0, 3.0])


class TestBoundingBox(unittest.TestCase):
    def test_contains(self):
        box = BoundingBox(40.4774, -74.2591, 40.9176, -73.7004)
        self.assertTrue(box.contains(Location(40.75, -73.98)))
        self.assertFalse(box.contains(Location(42.0, -73.98)))

    def test_rejects_degenerate(self):
        with self.assertRaises(ValidationError):
            BoundingBox(41.0, 0.0, 40.0, 1.0)


class TestHighwayClass(unittest.TestCase):
    def test_link_variants_collapse(self):
        self.assertIs(HighwayClass.coerce("motorway_link"), HighwayClass.MOTORWAY)
        self.assertIs(HighwayClass.coerce("PRIMARY_LINK"), HighwayClass.PRIMARY)

    def test_aliases_and_unknown(self):
        self.assertIs(HighwayClass.coerce("living_street"), HighwayClass.RESIDENTIAL)
        self.assertIs(HighwayClass.coerce("pedestrian"), HighwayClass.UNCLASSIFIED)
        self.assertIs(HighwayClass.coerce(None), HighwayClass.UNCLASSIFIED)

    def test_rank_is_stable_and_ordered(self):
        self.assertEqual(HighwayClass.MOTORWAY.rank, 0)
        self.assertLess(HighwayClass.PRIMARY.rank, HighwayClass.RESIDENTIAL.rank)


class TestTurnType(unittest.TestCase):
    def test_handedness(self):
        self.assertTrue(TurnType.SHARP_LEFT.is_left)
        self.assertFalse(TurnType.SHARP_LEFT.is_right)
        self.assertFalse(TurnType.STRAIGHT.is_left)

    def test_major(self):
        self.assertTrue(TurnType.LEFT.is_major)
        self.assertTrue(TurnType.UTURN.is_major)
        self.assertFalse(TurnType.SLIGHT_RIGHT.is_major)


class TestQuantilePair(unittest.TestCase):
    def test_crossed_quantiles_are_clamped(self):
        # A p90 below p50 is a modelling artefact, never a fact about the world.
        pair = QuantilePair(p50=600.0, p90=540.0)
        self.assertEqual(pair.p90, 600.0)
        self.assertEqual(pair.spread, 0.0)

    def test_spread_and_scaling(self):
        pair = QuantilePair(300.0, 480.0)
        self.assertEqual(pair.spread, 180.0)
        self.assertEqual(pair.scaled(2.0).p90, 960.0)

    def test_negative_p50_rejected(self):
        with self.assertRaises(ValidationError):
            QuantilePair(-1.0, 10.0)


class TestTripRecord(unittest.TestCase):
    def _trip(self, **kwargs):
        base = dict(
            trip_id="t1",
            pickup_at=datetime(2024, 3, 5, 8, 30, tzinfo=UTC),
            dropoff_at=datetime(2024, 3, 5, 8, 55, tzinfo=UTC),
            origin=Location(40.75, -73.99),
            destination=Location(40.72, -73.95),
            distance_m=6000.0,
            duration_s=1500.0,
        )
        base.update(kwargs)
        return TripRecord(**base)

    def test_round_trip(self):
        trip = self._trip(match_quality=MatchQuality.MATCHED, merchant_id="m9")
        restored = TripRecord.from_dict(trip.to_dict())
        self.assertEqual(restored.trip_id, trip.trip_id)
        self.assertEqual(restored.match_quality, MatchQuality.MATCHED)
        self.assertEqual(restored.merchant_id, "m9")
        self.assertEqual(restored.pickup_at, trip.pickup_at)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValidationError):
            self._trip(pickup_at=datetime(2024, 3, 5, 8, 30))

    def test_reversed_timestamps_rejected(self):
        with self.assertRaises(ValidationError):
            self._trip(dropoff_at=datetime(2024, 3, 5, 8, 0, tzinfo=UTC))

    def test_mean_speed(self):
        self.assertAlmostEqual(self._trip().mean_speed_kph, 14.4, places=6)

    def test_gps_trace_round_trip(self):
        trace = [
            GpsPoint(datetime(2024, 3, 5, 8, 30, tzinfo=UTC), Location(40.75, -73.99)),
            GpsPoint(
                datetime(2024, 3, 5, 8, 31, tzinfo=UTC),
                Location(40.751, -73.988),
                speed_kph=22.0,
                accuracy_m=8.0,
            ),
        ]
        trip = self._trip(gps_trace=trace)
        restored = TripRecord.from_dict(trip.to_dict())
        self.assertEqual(len(restored.gps_trace), 2)
        self.assertEqual(restored.gps_trace[1].speed_kph, 22.0)
        self.assertEqual(restored.gps_trace[1].accuracy_m, 8.0)


class TestSegmentObservation(unittest.TestCase):
    def test_speed(self):
        obs = SegmentObservation(
            segment_id=7,
            entered_at=datetime(2024, 3, 5, 8, 30, tzinfo=UTC),
            traversal_s=60.0,
            length_m=500.0,
        )
        self.assertAlmostEqual(obs.speed_kph, 30.0)

    def test_zero_traversal_rejected(self):
        with self.assertRaises(ValidationError):
            SegmentObservation(
                segment_id=7,
                entered_at=datetime(2024, 3, 5, 8, 30, tzinfo=UTC),
                traversal_s=0.0,
                length_m=500.0,
            )


class TestEtaRequest(unittest.TestCase):
    def test_defaults_departure_to_now(self):
        req = EtaRequest(Location(40.75, -73.99), Location(40.72, -73.95))
        self.assertIsNotNone(req.departure_at)
        self.assertIsNotNone(req.departure_at.tzinfo)
        self.assertLess(
            abs(datetime.now(UTC) - req.departure_at), timedelta(seconds=5)
        )

    def test_is_delivery(self):
        ride = EtaRequest(Location(40.75, -73.99), Location(40.72, -73.95))
        self.assertFalse(ride.is_delivery)
        delivery = EtaRequest(
            Location(40.75, -73.99), Location(40.72, -73.95), merchant_id="m1"
        )
        self.assertTrue(delivery.is_delivery)

    def test_round_trip(self):
        req = EtaRequest(
            Location(40.75, -73.99),
            Location(40.72, -73.95),
            departure_at=datetime(2024, 3, 5, 18, 0, tzinfo=UTC),
            merchant_id="m1",
            order_item_count=4,
            order_value_usd=38.5,
        )
        restored = EtaRequest.from_dict(req.to_dict())
        self.assertEqual(restored.merchant_id, "m1")
        self.assertEqual(restored.order_item_count, 4)
        self.assertEqual(restored.departure_at, req.departure_at)

    def test_negative_order_fields_rejected(self):
        with self.assertRaises(ValidationError):
            EtaRequest(
                Location(40.75, -73.99),
                Location(40.72, -73.95),
                merchant_id="m1",
                order_item_count=-1,
            )


if __name__ == "__main__":
    unittest.main()
