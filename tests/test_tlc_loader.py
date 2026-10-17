import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.config import DataConfig
from eta_forge.data import quality as q
from eta_forge.data.tlc import (
    METRES_PER_MILE,
    TlcFormatError,
    TlcLoader,
    ZoneCentroids,
    detect_dataset,
    normalise_row,
    parse_tlc_timestamp,
    resolve_columns,
    trip_id_for,
    trip_insert_row,
)
from eta_forge.types import Location, MatchQuality

YELLOW_2015_COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "pickup_longitude", "pickup_latitude", "dropoff_longitude",
    "dropoff_latitude", "fare_amount",
]

ZONES = ZoneCentroids({
    161: Location(40.7580, -73.9776),   # Midtown Center
    236: Location(40.7809, -73.9576),   # Upper East Side North
})


def yellow_2015(**overrides):
    row = {
        "VendorID": "2",
        "tpep_pickup_datetime": "2015-01-15 08:30:00",
        "tpep_dropoff_datetime": "2015-01-15 08:45:00",
        "passenger_count": "1",
        "trip_distance": "2.10",
        "pickup_longitude": "-73.985",
        "pickup_latitude": "40.758",
        "dropoff_longitude": "-73.975",
        "dropoff_latitude": "40.775",
        "fare_amount": "11.5",
    }
    row.update(overrides)
    return row


def yellow_2023(**overrides):
    row = {
        "VendorID": 2,
        "tpep_pickup_datetime": datetime(2023, 1, 15, 8, 30),
        "tpep_dropoff_datetime": datetime(2023, 1, 15, 8, 52),
        "trip_distance": 2.4,
        "PULocationID": 161,
        "DOLocationID": 236,
    }
    row.update(overrides)
    return row


class TestColumnResolution(unittest.TestCase):
    def test_yellow_2015_has_coordinates(self):
        cmap = resolve_columns(YELLOW_2015_COLUMNS)
        self.assertTrue(cmap.has_coordinates)
        self.assertFalse(cmap.has_zones)
        self.assertEqual(cmap.columns["pickup_at"], "tpep_pickup_datetime")

    def test_zone_columns_match_case_insensitively(self):
        cmap = resolve_columns(["pickup_datetime", "dropOff_datetime", "PUlocationID", "DOlocationID"])
        self.assertTrue(cmap.has_zones)
        self.assertEqual(cmap.columns["pu_location_id"], "PUlocationID")
        self.assertEqual(cmap.columns["dropoff_at"], "dropOff_datetime")

    def test_fhvhv_distance_column(self):
        cmap = resolve_columns(
            ["pickup_datetime", "dropoff_datetime", "trip_miles", "trip_time",
             "PULocationID", "DOLocationID"]
        )
        self.assertEqual(cmap.columns["distance_miles"], "trip_miles")

    def test_unknown_schema_is_an_error(self):
        with self.assertRaises(TlcFormatError):
            resolve_columns(["fare_amount", "tip_amount"])
        with self.assertRaises(TlcFormatError):
            resolve_columns(["tpep_pickup_datetime", "tpep_dropoff_datetime"])


class TestDetectDataset(unittest.TestCase):
    def test_from_file_name(self):
        self.assertEqual(detect_dataset("data/yellow_tripdata_2015-03.parquet"), "yellow")
        self.assertEqual(detect_dataset("green_tripdata_2015-03.csv"), "green")
        self.assertEqual(detect_dataset("fhv_tripdata_2019-02.parquet"), "fhv")
        self.assertEqual(detect_dataset("fhvhv_tripdata_2023-01.parquet"), "fhvhv")

    def test_from_columns(self):
        self.assertEqual(detect_dataset("month.parquet", ["lpep_pickup_datetime"]), "green")
        self.assertEqual(detect_dataset("month.parquet", ["trip_miles"]), "fhvhv")


class TestTimestamps(unittest.TestCase):
    def test_naive_wall_clock_is_new_york_time(self):
        parsed = parse_tlc_timestamp("2015-01-15 08:30:00")
        self.assertEqual(parsed.utcoffset(), timedelta(hours=-5))
        self.assertEqual(parsed.hour, 8)

    def test_summer_offset(self):
        parsed = parse_tlc_timestamp(datetime(2015, 7, 1, 8, 30))
        self.assertEqual(parsed.utcoffset(), timedelta(hours=-4))

    def test_aware_values_are_respected(self):
        aware = datetime(2015, 1, 15, 13, 30, tzinfo=timezone.utc)
        self.assertEqual(parse_tlc_timestamp(aware), aware)

    def test_blank_and_garbage(self):
        self.assertIsNone(parse_tlc_timestamp(""))
        self.assertIsNone(parse_tlc_timestamp(None))
        self.assertIsNone(parse_tlc_timestamp("not a time"))


class TestNormaliseRow(unittest.TestCase):
    def test_miles_become_metres(self):
        cmap = resolve_columns(YELLOW_2015_COLUMNS)
        trip = normalise_row(yellow_2015(), cmap, dataset="yellow", row_index=0)
        self.assertAlmostEqual(trip.distance_m, 2.10 * METRES_PER_MILE)
        self.assertEqual(trip.origin, Location(40.758, -73.985))
        self.assertIs(trip.match_quality, MatchQuality.ENDPOINTS)

    def test_null_island_coordinates_are_missing(self):
        cmap = resolve_columns(YELLOW_2015_COLUMNS)
        trip = normalise_row(
            yellow_2015(pickup_longitude="0", pickup_latitude="0"),
            cmap, dataset="yellow", row_index=0,
        )
        self.assertIsNone(trip.origin)

    def test_zone_only_rows_use_centroids(self):
        row = yellow_2023()
        cmap = resolve_columns(row.keys())
        trip = normalise_row(row, cmap, dataset="yellow", row_index=3, zones=ZONES)
        self.assertIs(trip.match_quality, MatchQuality.ZONE)
        self.assertEqual(trip.origin, ZONES.get(161))
        self.assertEqual(trip.destination, ZONES.get(236))

    def test_unknown_zones_have_no_centroid(self):
        self.assertIsNone(ZONES.get(264))
        self.assertIsNone(ZONES.get(265))

    def test_missing_fhv_distance_is_estimated_from_centroids(self):
        row = {
            "pickup_datetime": "2019-02-01 08:00:00",
            "dropOff_datetime": "2019-02-01 08:20:00",
            "PUlocationID": "161",
            "DOlocationID": "236",
        }
        trip = normalise_row(row, resolve_columns(row), dataset="fhv", row_index=0, zones=ZONES)
        self.assertTrue(trip.distance_estimated)
        self.assertGreater(trip.distance_m, 2_000.0)


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.loader = TlcLoader(DataConfig(), zones=ZONES)

    def test_filters_and_tallies(self):
        rows = [
            yellow_2015(),
            yellow_2015(tpep_dropoff_datetime="2015-01-15 08:00:00"),       # reversed
            yellow_2015(pickup_longitude="0", pickup_latitude="0"),         # null island
            yellow_2015(trip_distance="0"),                                 # zero distance
            yellow_2015(dropoff_latitude="42.5"),                           # out of bbox
        ]
        records, result = self.loader.convert(rows, source_file="yellow_tripdata_2015-01.csv")
        kept = list(records)
        self.assertEqual(len(kept), 1)
        self.assertEqual(result.tally.read, 5)
        self.assertEqual(result.tally.rejected[q.NON_POSITIVE_DURATION], 1)
        self.assertEqual(result.tally.rejected[q.MISSING_LOCATION], 1)
        self.assertEqual(result.tally.rejected[q.ZERO_DISTANCE], 1)
        self.assertEqual(result.tally.rejected[q.OUT_OF_BBOX], 1)
        record = kept[0]
        self.assertEqual(record.duration_s, 900.0)
        self.assertEqual(record.vehicle_type, "taxi")
        self.assertIs(record.match_quality, MatchQuality.ENDPOINTS)

    def test_zone_trips_are_kept_and_flagged(self):
        rows = [yellow_2023(), yellow_2023(PULocationID=264)]
        records, result = self.loader.convert(rows, source_file="yellow_tripdata_2023-01.parquet")
        kept = list(records)
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0].match_quality, MatchQuality.ZONE)
        self.assertEqual(result.zone_trips, 1)
        self.assertEqual(result.tally.rejected[q.UNKNOWN_ZONE], 1)

    def test_limit(self):
        records, result = self.loader.convert(
            [yellow_2015()] * 10, source_file="yellow.csv", limit=3
        )
        self.assertEqual(len(list(records)), 3)
        self.assertEqual(result.tally.read, 3)

    def test_empty_input(self):
        records, result = self.loader.convert([], source_file="yellow.csv")
        self.assertEqual(list(records), [])
        self.assertEqual(result.tally.read, 0)

    def test_streams_a_csv_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "yellow_tripdata_2015-01.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=YELLOW_2015_COLUMNS)
                writer.writeheader()
                for minute in range(4):
                    writer.writerow(yellow_2015(
                        tpep_pickup_datetime=f"2015-01-15 08:{minute:02d}:00",
                        tpep_dropoff_datetime=f"2015-01-15 08:{minute + 20:02d}:00",
                    ))
            records, result = self.loader.load_file(path)
            kept = list(records)
        self.assertEqual(result.dataset, "yellow")
        self.assertEqual(len(kept), 4)
        self.assertEqual(len({r.trip_id for r in kept}), 4)

    def test_trip_ids_are_deterministic(self):
        self.assertEqual(
            trip_id_for("/a/yellow_2015-01.csv", "yellow", 7),
            trip_id_for("/b/yellow_2015-01.csv", "yellow", 7),
        )
        self.assertNotEqual(
            trip_id_for("yellow_2015-01.csv", "yellow", 7),
            trip_id_for("yellow_2015-01.csv", "yellow", 8),
        )

    def test_insert_row_is_lon_lat(self):
        records, _ = self.loader.convert([yellow_2015()], source_file="yellow.csv")
        row = trip_insert_row(next(iter(records)), "yellow")
        self.assertEqual(row[7], -73.985)   # ST_MakePoint(lon, lat)
        self.assertEqual(row[8], 40.758)
        self.assertEqual(row[-1], "endpoints")


if __name__ == "__main__":
    unittest.main()
