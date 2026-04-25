-- 002: the routable road graph.
--
-- `road_segment` is written by osm2pgrouting (see scripts/import_osm.py). Its column
-- names are fixed by that tool — gid, source, target, cost, reverse_cost, the_geom —
-- so this migration declares the table only if the import has not run yet, and then
-- layers our own derived attributes beside it in `segment_attr`.
--
-- Splitting derived attributes into their own table means a re-import of the OSM
-- extract does not destroy the mined speed profiles that reference segment ids, and
-- the attribute derivation can be re-run independently.

CREATE TABLE IF NOT EXISTS road_segment (
    gid          bigserial PRIMARY KEY,
    osm_id       bigint,
    source       bigint,
    target       bigint,
    length       double precision,           -- osm2pgrouting: degrees, unhelpfully
    cost         double precision,
    reverse_cost double precision,
    x1 double precision, y1 double precision,
    x2 double precision, y2 double precision,
    the_geom     geometry(LineString, 4326)
);

CREATE TABLE IF NOT EXISTS road_segment_vertices_pgr (
    id       bigserial PRIMARY KEY,
    cnt      integer,
    chk      integer,
    ein      integer,
    eout     integer,
    the_geom geometry(Point, 4326)
);

-- Our own view of a segment: metric length, class, and the bearings the turn
-- classifier needs. `bearing_in` is the heading leaving the source vertex,
-- `bearing_out` the heading arriving at the target vertex; on a curved segment those
-- differ, and using a single straight-line bearing misclassifies turns on ramps.
CREATE TABLE IF NOT EXISTS segment_attr (
    segment_id      bigint PRIMARY KEY REFERENCES road_segment(gid) ON DELETE CASCADE,
    osm_id          bigint,
    highway_class   text        NOT NULL DEFAULT 'unclassified',
    name            text,
    lanes           smallint,
    oneway          boolean     NOT NULL DEFAULT false,
    speed_limit_kph double precision,
    length_m        double precision NOT NULL,
    bearing_in_deg  double precision NOT NULL,
    bearing_out_deg double precision NOT NULL,
    has_signal_at_end boolean   NOT NULL DEFAULT false,
    is_bridge       boolean     NOT NULL DEFAULT false,
    is_tunnel       boolean     NOT NULL DEFAULT false,
    CONSTRAINT segment_attr_class_ck CHECK (highway_class IN (
        'motorway','trunk','primary','secondary','tertiary',
        'residential','service','unclassified'
    )),
    CONSTRAINT segment_attr_length_ck CHECK (length_m > 0)
);

-- Class defaults, used as the free-flow prior wherever mining produced nothing.
-- Kept in the database (not only in YAML) because the dynamic cost expression runs
-- server-side and must be able to fall back without a round trip.
CREATE TABLE IF NOT EXISTS highway_class_default (
    highway_class text PRIMARY KEY,
    speed_kph     double precision NOT NULL CHECK (speed_kph > 0)
);

INSERT INTO highway_class_default (highway_class, speed_kph) VALUES
    ('motorway', 90.0),
    ('trunk', 70.0),
    ('primary', 50.0),
    ('secondary', 45.0),
    ('tertiary', 40.0),
    ('residential', 30.0),
    ('service', 20.0),
    ('unclassified', 30.0)
ON CONFLICT (highway_class) DO NOTHING;

-- Traffic signals imported from OSM nodes; a segment ending at one carries a fixed
-- penalty in the cost expression.
CREATE TABLE IF NOT EXISTS traffic_signal (
    osm_node_id bigint PRIMARY KEY,
    the_geom    geometry(Point, 4326) NOT NULL
);
