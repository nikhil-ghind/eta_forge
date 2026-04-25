-- 003: the historical trip corpus and its map-matched output.
--
-- `trip` holds one row per supervised example: NYC TLC records, plus any synthetic
-- or production trips loaded later. `trip_gps_point` holds traces where they exist —
-- TLC does not publish them, so in practice traces come from the synthetic source or
-- a production feed, and TLC rows carry match_quality='endpoints' or 'zone'.
--
-- `trip_segment` is the map matcher's output and the *only* input to speed-profile
-- mining: one row per (trip, segment) traversal with the time it took.

CREATE TABLE IF NOT EXISTS trip (
    trip_id        text PRIMARY KEY,
    source_dataset text        NOT NULL DEFAULT 'tlc',
    vehicle_type   text        NOT NULL DEFAULT 'car',
    pickup_at      timestamptz NOT NULL,
    dropoff_at     timestamptz NOT NULL,
    duration_s     double precision NOT NULL,
    distance_m     double precision NOT NULL,
    origin         geometry(Point, 4326),
    destination    geometry(Point, 4326),
    pu_location_id integer,          -- TLC taxi zone, when coordinates are absent
    do_location_id integer,
    merchant_id    text,
    match_quality  text NOT NULL DEFAULT 'endpoints',
    ingested_at    timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT trip_duration_ck    CHECK (duration_s > 0),
    CONSTRAINT trip_distance_ck    CHECK (distance_m >= 0),
    CONSTRAINT trip_ordering_ck    CHECK (dropoff_at >= pickup_at),
    CONSTRAINT trip_match_quality_ck CHECK (match_quality IN
        ('matched','zone','endpoints','failed'))
);

-- Origin/destination H3 cells are generated columns rather than application-computed
-- ones: a cell that disagrees with the geometry it was derived from is a silent
-- feature bug that only shows up as unexplained residual variance months later.
ALTER TABLE trip
    ADD COLUMN IF NOT EXISTS origin_h3_r8 h3index
        GENERATED ALWAYS AS (h3_lat_lng_to_cell(origin, 8)) STORED;
ALTER TABLE trip
    ADD COLUMN IF NOT EXISTS destination_h3_r8 h3index
        GENERATED ALWAYS AS (h3_lat_lng_to_cell(destination, 8)) STORED;
ALTER TABLE trip
    ADD COLUMN IF NOT EXISTS origin_h3_r9 h3index
        GENERATED ALWAYS AS (h3_lat_lng_to_cell(origin, 9)) STORED;
ALTER TABLE trip
    ADD COLUMN IF NOT EXISTS destination_h3_r9 h3index
        GENERATED ALWAYS AS (h3_lat_lng_to_cell(destination, 9)) STORED;

CREATE TABLE IF NOT EXISTS trip_gps_point (
    trip_id   text        NOT NULL REFERENCES trip(trip_id) ON DELETE CASCADE,
    seq       integer     NOT NULL,
    recorded_at timestamptz NOT NULL,
    the_geom  geometry(Point, 4326) NOT NULL,
    speed_kph double precision,
    accuracy_m double precision NOT NULL DEFAULT 15.0,
    PRIMARY KEY (trip_id, seq)
);

CREATE TABLE IF NOT EXISTS trip_segment (
    trip_id     text   NOT NULL REFERENCES trip(trip_id) ON DELETE CASCADE,
    seq         integer NOT NULL,
    segment_id  bigint NOT NULL,
    entered_at  timestamptz NOT NULL,
    traversal_s double precision NOT NULL,
    length_m    double precision NOT NULL,
    -- Confidence carried through from the Viterbi decode; low-confidence traversals
    -- are excluded from mining but kept for auditing a bad match.
    match_confidence double precision NOT NULL DEFAULT 1.0,
    PRIMARY KEY (trip_id, seq),
    CONSTRAINT trip_segment_traversal_ck CHECK (traversal_s > 0),
    CONSTRAINT trip_segment_length_ck    CHECK (length_m > 0)
);

-- NYC TLC taxi zones, used for the centroid fallback on coordinate-free records.
CREATE TABLE IF NOT EXISTS taxi_zone (
    location_id integer PRIMARY KEY,
    borough     text,
    zone_name   text,
    the_geom    geometry(MultiPolygon, 4326),
    centroid    geometry(Point, 4326)
);

-- Ledger of what has been loaded, so re-running the loader is safe and the training
-- window in a model manifest can be tied back to concrete input files.
CREATE TABLE IF NOT EXISTS ingest_batch (
    batch_id      bigserial PRIMARY KEY,
    source_file   text NOT NULL UNIQUE,
    file_checksum text,
    rows_read     bigint NOT NULL DEFAULT 0,
    rows_kept     bigint NOT NULL DEFAULT 0,
    rejections    jsonb  NOT NULL DEFAULT '{}'::jsonb,
    loaded_at     timestamptz NOT NULL DEFAULT now()
);
