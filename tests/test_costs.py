import unittest
from datetime import datetime, timezone

from eta_forge.config import Config
from eta_forge.db.runner import SqlError
from eta_forge.graph.costs import (
    SPEED_LIMIT_DAMPING,
    build_cost_sql,
    build_dijkstra_sql,
    build_profile_coverage_sql,
    build_snap_sql,
    time_bucket,
)
from eta_forge.types import ValidationError

UTC = timezone.utc


class TestTimeBucket(unittest.TestCase):
    def test_hourly_buckets_utc(self):
        dow, tod = time_bucket(datetime(2024, 3, 5, 14, 30, tzinfo=UTC))
        self.assertEqual(tod, 14)
        self.assertEqual(dow, 1)  # Tuesday, Python convention

    def test_coarse_buckets(self):
        _, tod = time_bucket(datetime(2024, 3, 5, 14, 30, tzinfo=UTC), tod_buckets=8)
        self.assertEqual(tod, 4)  # 3-hour buckets, 14:30 is in [12, 15)

    def test_dow_collapsed(self):
        dow, _ = time_bucket(datetime(2024, 3, 9, 14, 0, tzinfo=UTC), dow_buckets=1)
        self.assertEqual(dow, 0)

    def test_local_timezone_shifts_the_bucket(self):
        # 02:00 UTC is 21:00 the previous day in New York — a different hour *and* a
        # different day of week. Mining in UTC would put this in the wrong cell.
        utc_dow, utc_tod = time_bucket(datetime(2024, 3, 5, 2, 0, tzinfo=UTC))
        ny_dow, ny_tod = time_bucket(
            datetime(2024, 3, 5, 2, 0, tzinfo=UTC), tz="America/New_York"
        )
        self.assertEqual((utc_dow, utc_tod), (1, 2))
        self.assertEqual((ny_dow, ny_tod), (0, 21))

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValidationError):
            time_bucket(datetime(2024, 3, 5, 2, 0))

    def test_invalid_bucketing_rejected(self):
        with self.assertRaises(ValidationError):
            time_bucket(datetime(2024, 3, 5, 2, 0, tzinfo=UTC), tod_buckets=7)
        with self.assertRaises(ValidationError):
            time_bucket(datetime(2024, 3, 5, 2, 0, tzinfo=UTC), dow_buckets=3)

    def test_unknown_timezone_falls_back_rather_than_raising(self):
        dow, tod = time_bucket(
            datetime(2024, 3, 5, 2, 0, tzinfo=UTC), tz="Mars/Olympus_Mons"
        )
        self.assertEqual((dow, tod), (1, 2))


class TestCostSql(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def build(self, **kwargs):
        params = dict(dow_bucket=1, tod_bucket=8)
        params.update(kwargs)
        return build_cost_sql(self.cfg.graph, self.cfg.profiles, **params)

    def test_yields_the_columns_pgrouting_requires(self):
        sql = self.build().sql
        for column in ("AS id", "e.source", "e.target", "AS cost", "AS reverse_cost"):
            self.assertIn(column, sql)

    def test_cost_is_seconds_not_metres(self):
        # length / (kph / 3.6) is metres per (metres/second) = seconds.
        self.assertIn("/ 3.6", self.build().sql)

    def test_profile_join_binds_the_requested_cell(self):
        cost = self.build(dow_bucket=4, tod_bucket=17)
        self.assertTrue(cost.uses_profiles)
        self.assertIn("p.dow_bucket = 4", cost.sql)
        self.assertIn("p.tod_bucket = 17", cost.sql)
        self.assertIn("segment_speed_profile_smoothed", cost.sql)

    def test_different_hours_produce_different_graphs(self):
        morning = self.build(tod_bucket=8).sql
        night = self.build(tod_bucket=3).sql
        self.assertNotEqual(morning, night)

    def test_profiles_can_be_disabled(self):
        cost = self.build(use_profiles=False)
        self.assertFalse(cost.uses_profiles)
        self.assertNotIn("segment_speed_profile_smoothed", cost.sql)
        self.assertNotIn("p.speed_kph", cost.sql)

    def test_fallback_chain_order(self):
        sql = self.build().sql
        profile_at = sql.index("p.speed_kph")
        limit_at = sql.index("a.speed_limit_kph")
        self.assertLess(profile_at, limit_at)

    def test_speed_limits_are_damped_per_class(self):
        sql = self.build().sql
        self.assertIn("0.95", sql)   # motorway damping
        self.assertIn("0.55", sql)   # service damping
        self.assertLess(SPEED_LIMIT_DAMPING["residential"], SPEED_LIMIT_DAMPING["motorway"])

    def test_oneway_is_encoded_as_negative_reverse_cost(self):
        # pgRouting reads a negative reverse cost as "not traversable"; a large
        # positive one would let the router drive the wrong way when desperate.
        self.assertIn("WHEN a.oneway THEN -1", self.build().sql)

    def test_speed_is_clamped_into_the_mined_bounds(self):
        sql = self.build().sql
        self.assertIn("LEAST(GREATEST(", sql)
        self.assertIn(str(self.cfg.profiles.max_speed_kph), sql)

    def test_bbox_restricts_the_edge_set(self):
        cost = self.build(bbox=(40.70, -74.02, 40.78, -73.94))
        self.assertIn("ST_MakeEnvelope(-74.02, 40.7, -73.94, 40.78, 4326)", cost.sql)

    def test_no_bbox_means_no_envelope(self):
        self.assertNotIn("ST_MakeEnvelope", self.build().sql)

    def test_values_are_inlined_not_bound(self):
        # The cost SQL is nested inside a dollar-quoted argument to pgr_dijkstra, so
        # it must carry no driver placeholders of its own.
        cost = self.build(bbox=(40.70, -74.02, 40.78, -73.94))
        self.assertEqual(cost.params, ())
        self.assertNotIn("%s", cost.sql)

    def test_signal_penalty_is_charged_on_both_directions(self):
        sql = self.build().sql
        self.assertEqual(sql.count("has_signal_at_end"), 2)

    def test_rejects_an_unsafe_table_name(self):
        self.cfg.graph.edge_table = "road_segment; DROP TABLE trip"
        with self.assertRaises(SqlError):
            self.build()


class TestDijkstraSql(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cost = build_cost_sql(
            self.cfg.graph, self.cfg.profiles, dow_bucket=1, tod_bucket=8
        )

    def test_wraps_the_cost_expression_dollar_quoted(self):
        sql = build_dijkstra_sql(self.cfg.graph, self.cost)
        self.assertIn("pgr_dijkstra(", sql)
        self.assertIn("$$", sql)
        self.assertIn("directed => true", sql)

    def test_joins_attributes_for_leg_assembly(self):
        sql = build_dijkstra_sql(self.cfg.graph, self.cost)
        for column in ("a.highway_class", "a.length_m", "a.bearing_in_deg",
                       "ST_AsGeoJSON"):
            self.assertIn(column, sql)

    def test_orders_by_sequence(self):
        self.assertIn("ORDER BY d.seq", build_dijkstra_sql(self.cfg.graph, self.cost))


class TestSnapAndCoverageSql(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_snap_uses_knn_operator(self):
        sql = build_snap_sql(self.cfg.graph)
        self.assertIn("<->", sql)
        self.assertIn("ST_DWithin", sql)
        self.assertIn("LIMIT", sql)

    def test_snap_measures_distance_on_the_geography(self):
        self.assertIn("::geography", build_snap_sql(self.cfg.graph))

    def test_coverage_query_counts_profiled_segments(self):
        sql = build_profile_coverage_sql(self.cfg.profiles)
        self.assertIn("unnest(%s::bigint[])", sql)
        self.assertIn("LEFT JOIN segment_speed_profile_smoothed", sql)
        self.assertIn("count(p.segment_id)", sql)


if __name__ == "__main__":
    unittest.main()
