-- 005: indexes.
--
-- Three access patterns dominate and each needs its own index:
--   1. Snap a request coordinate to the nearest routable vertex — KNN on a GiST
--      index over the vertex geometry (the `<->` operator).
--   2. pgr_dijkstra walking the edge list — btree on source/target.
--   3. Speed-profile lookup during routing — covered by the profile migration.
--
-- The map matcher adds a fourth: radius search for candidate segments near a GPS
-- point, which is again GiST but over the *line* geometry.

-- 1 & 4: geometry.
CREATE INDEX IF NOT EXISTS road_segment_geom_gix
    ON road_segment USING GIST (the_geom);
CREATE INDEX IF NOT EXISTS road_segment_vertices_geom_gix
    ON road_segment_vertices_pgr USING GIST (the_geom);
CREATE INDEX IF NOT EXISTS traffic_signal_geom_gix
    ON traffic_signal USING GIST (the_geom);

-- 2: topology walk.
CREATE INDEX IF NOT EXISTS road_segment_source_ix ON road_segment (source);
CREATE INDEX IF NOT EXISTS road_segment_target_ix ON road_segment (target);
CREATE INDEX IF NOT EXISTS road_segment_osm_id_ix ON road_segment (osm_id);

-- Attribute joins during route assembly. Including length and class in the index lets
-- the cost query answer from the index alone on the hot path.
CREATE INDEX IF NOT EXISTS segment_attr_class_ix
    ON segment_attr (highway_class) INCLUDE (length_m, speed_limit_kph);

-- Trips: mining and training both scan by time; slicing scans by H3 cell.
CREATE INDEX IF NOT EXISTS trip_pickup_at_ix       ON trip (pickup_at);
CREATE INDEX IF NOT EXISTS trip_match_quality_ix   ON trip (match_quality, pickup_at);
CREATE INDEX IF NOT EXISTS trip_merchant_ix        ON trip (merchant_id, pickup_at)
    WHERE merchant_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS trip_origin_h3_ix       ON trip (origin_h3_r8);
CREATE INDEX IF NOT EXISTS trip_destination_h3_ix  ON trip (destination_h3_r8);
CREATE INDEX IF NOT EXISTS trip_origin_geom_gix    ON trip USING GIST (origin);
CREATE INDEX IF NOT EXISTS trip_destination_geom_gix ON trip USING GIST (destination);

-- Map-matched traversals: mining groups by (segment, time), so lead with segment_id.
CREATE INDEX IF NOT EXISTS trip_segment_segment_time_ix
    ON trip_segment (segment_id, entered_at)
    INCLUDE (traversal_s, length_m);
CREATE INDEX IF NOT EXISTS trip_segment_trip_ix ON trip_segment (trip_id);

CREATE INDEX IF NOT EXISTS trip_gps_point_trip_seq_ix
    ON trip_gps_point (trip_id, recorded_at);

-- Merchants and orders.
CREATE INDEX IF NOT EXISTS merchant_order_merchant_time_ix
    ON merchant_order (merchant_id, placed_at DESC);
CREATE INDEX IF NOT EXISTS merchant_order_trip_ix
    ON merchant_order (trip_id) WHERE trip_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS merchant_h3_ix ON merchant (h3_r8);
CREATE INDEX IF NOT EXISTS handoff_dwell_h3_ix
    ON handoff_dwell_observation (destination_h3_r8);

-- Taxi zones: point-in-polygon for the coordinate-free TLC fallback.
CREATE INDEX IF NOT EXISTS taxi_zone_geom_gix ON taxi_zone USING GIST (the_geom);
