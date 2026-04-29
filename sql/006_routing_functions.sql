-- 006: server-side routing helpers.
--
-- The cost expression itself is composed in Python (eta_forge/graph/costs.py) because
-- it changes shape with the request — profiles on or off, bounding box present or
-- absent. What lives here is the stable part: the profile speed lookup the cost
-- expression calls, and a convenience function for exploring routes from psql.

-- Speed for a segment in a time cell, with the same three-level fallback the cost
-- expression uses. Marked STABLE PARALLEL SAFE so the planner can call it inside a
-- parallel scan of the edge list.
CREATE OR REPLACE FUNCTION eta_segment_speed_kph(
    p_segment_id  bigint,
    p_dow_bucket  integer,
    p_tod_bucket  integer,
    p_min_samples integer DEFAULT 5
) RETURNS double precision
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT COALESCE(
        (SELECT s.speed_kph
           FROM segment_speed_profile_smoothed s
          WHERE s.segment_id = p_segment_id
            AND s.dow_bucket = p_dow_bucket
            AND s.tod_bucket = p_tod_bucket
            AND s.sample_count >= p_min_samples),
        (SELECT a.speed_limit_kph * 0.75
           FROM segment_attr a
          WHERE a.segment_id = p_segment_id
            AND a.speed_limit_kph IS NOT NULL),
        (SELECT d.speed_kph
           FROM segment_attr a
           JOIN highway_class_default d USING (highway_class)
          WHERE a.segment_id = p_segment_id),
        30.0
    );
$$;

-- Bucket a timestamp exactly the way eta_forge/graph/costs.py:time_bucket does, so a
-- profile written by the Python miner and one written by a SQL backfill land in the
-- same cell. The timezone conversion is the part that is easy to get wrong.
CREATE OR REPLACE FUNCTION eta_time_bucket(
    p_at timestamptz,
    p_tz text DEFAULT 'America/New_York',
    p_tod_buckets integer DEFAULT 24
) RETURNS TABLE (dow_bucket integer, tod_bucket integer)
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT
        -- PostgreSQL's dow is 0=Sunday; Python's weekday() is 0=Monday. Align on the
        -- Python convention because that is what the miner and the model use.
        ((EXTRACT(DOW FROM p_at AT TIME ZONE p_tz)::integer + 6) % 7),
        (EXTRACT(HOUR FROM p_at AT TIME ZONE p_tz)::integer / (24 / p_tod_buckets))
$$;

-- Route between two coordinates, for interactive inspection from psql. Not on the
-- serving path: the service composes its own cost expression so it can restrict the
-- edge set to a bounding box.
CREATE OR REPLACE FUNCTION eta_route(
    p_from_lat double precision,
    p_from_lon double precision,
    p_to_lat   double precision,
    p_to_lon   double precision,
    p_at       timestamptz DEFAULT now()
) RETURNS TABLE (
    seq integer,
    segment_id bigint,
    highway_class text,
    length_m double precision,
    speed_kph double precision,
    duration_s double precision,
    agg_duration_s double precision
)
LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_source bigint;
    v_target bigint;
    v_dow integer;
    v_tod integer;
BEGIN
    SELECT dow_bucket, tod_bucket INTO v_dow, v_tod FROM eta_time_bucket(p_at);

    SELECT id INTO v_source FROM road_segment_vertices_pgr
     ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(p_from_lon, p_from_lat), 4326)
     LIMIT 1;
    SELECT id INTO v_target FROM road_segment_vertices_pgr
     ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(p_to_lon, p_to_lat), 4326)
     LIMIT 1;

    IF v_source IS NULL OR v_target IS NULL THEN
        RAISE EXCEPTION 'no routable vertex near one of the endpoints';
    END IF;

    RETURN QUERY
    SELECT
        d.seq::integer,
        a.segment_id,
        a.highway_class,
        a.length_m,
        eta_segment_speed_kph(a.segment_id, v_dow, v_tod),
        d.cost,
        d.agg_cost
    FROM pgr_dijkstra(
        format($q$
            SELECT e.gid AS id, e.source, e.target,
                   a.length_m / (eta_segment_speed_kph(a.segment_id, %s, %s) / 3.6)
                     + CASE WHEN a.has_signal_at_end THEN 8 ELSE 0 END AS cost,
                   CASE WHEN a.oneway THEN -1
                        ELSE a.length_m / (eta_segment_speed_kph(a.segment_id, %s, %s) / 3.6)
                             + CASE WHEN a.has_signal_at_end THEN 8 ELSE 0 END
                   END AS reverse_cost
            FROM road_segment e
            JOIN segment_attr a ON a.segment_id = e.gid
        $q$, v_dow, v_tod, v_dow, v_tod),
        v_source, v_target, directed => true) d
    JOIN segment_attr a ON a.segment_id = d.edge
    ORDER BY d.seq;
END;
$$;
