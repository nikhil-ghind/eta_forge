#!/usr/bin/env python3
"""Build the routable graph from an OSM extract.

Three steps, each individually re-runnable:

1. ``osm2pgrouting`` reads the ``.osm`` / ``.osm.pbf`` extract and writes
   ``road_segment`` plus ``road_segment_vertices_pgr`` with a valid topology.
2. Derived attributes are computed into ``segment_attr``: metric length (osm2pgrouting
   records length in *degrees*, which is useless for a duration), highway class
   collapsed onto our eight kept classes, and the entry/exit bearings the turn
   classifier needs.
3. Traffic signals are attached to the segments that end at them.

``--dry-run`` prints the exact shell command and SQL without touching anything, which
is how this is exercised in an environment with no PostGIS and no extract.

    python scripts/import_osm.py --config configs/eta_forge.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eta_forge.config import Config, load_config  # noqa: E402
from eta_forge.db.dsn import redacted_dsn  # noqa: E402
from eta_forge.db.pool import ConnectionPool  # noqa: E402
from eta_forge.db.runner import SqlRunner  # noqa: E402

# ----------------------------------------------------------------------------------
# step 1: osm2pgrouting
# ----------------------------------------------------------------------------------


def osm2pgrouting_command(cfg: Config, extract: str, mapping: str, *, clean: bool) -> List[str]:
    """The exact argv for osm2pgrouting. Password is passed via PGPASSWORD, not argv."""
    cmd = [
        "osm2pgrouting",
        "--f", extract,
        "--conf", mapping,
        "--dbname", cfg.database.database,
        "--username", cfg.database.user,
        "--host", cfg.database.host,
        "--port", str(cfg.database.port),
        "--prefix", "road_",
        "--suffix", "segment",
        "--addnodes",
        "--attributes",
        "--tags",
    ]
    if clean:
        cmd.append("--clean")
    return cmd


# ----------------------------------------------------------------------------------
# step 2: derived attributes
# ----------------------------------------------------------------------------------

# osm2pgrouting's `length` is in degrees. ST_Length over a geography cast gives metres
# on the spheroid, which is what every duration in this project is derived from.
#
# ST_Azimuth returns radians clockwise from north; degrees(...) then normalising into
# [0, 360) matches the convention used by eta_forge/graph/geometry.py so the bearings
# stored here and the bearings computed in Python are directly comparable.
DERIVE_ATTR_SQL = """
INSERT INTO segment_attr (
    segment_id, osm_id, highway_class, name, lanes, oneway,
    speed_limit_kph, length_m, bearing_in_deg, bearing_out_deg,
    is_bridge, is_tunnel
)
SELECT
    s.gid,
    s.osm_id,
    CASE
        WHEN c.name IS NULL THEN 'unclassified'
        WHEN c.name IN ('motorway','motorway_link')   THEN 'motorway'
        WHEN c.name IN ('trunk','trunk_link')         THEN 'trunk'
        WHEN c.name IN ('primary','primary_link')     THEN 'primary'
        WHEN c.name IN ('secondary','secondary_link') THEN 'secondary'
        WHEN c.name IN ('tertiary','tertiary_link')   THEN 'tertiary'
        WHEN c.name IN ('residential','living_street') THEN 'residential'
        WHEN c.name = 'service'                       THEN 'service'
        ELSE 'unclassified'
    END                                               AS highway_class,
    w.name,
    NULLIF(w.lanes, 0)::smallint,
    (s.reverse_cost IS NULL OR s.reverse_cost < 0)    AS oneway,
    NULLIF(w.maxspeed_forward, 0)                     AS speed_limit_kph,
    ST_Length(s.the_geom::geography)                  AS length_m,
    -- Entry bearing: the heading over the first 2% of the line, so a curved segment
    -- is not summarised by its straight-line chord.
    MOD(DEGREES(ST_Azimuth(
        ST_LineInterpolatePoint(s.the_geom, 0.0),
        ST_LineInterpolatePoint(s.the_geom, 0.02)
    ))::numeric + 360, 360)::double precision         AS bearing_in_deg,
    MOD(DEGREES(ST_Azimuth(
        ST_LineInterpolatePoint(s.the_geom, 0.98),
        ST_LineInterpolatePoint(s.the_geom, 1.0)
    ))::numeric + 360, 360)::double precision         AS bearing_out_deg,
    COALESCE(w.tag_value LIKE '%bridge%', false),
    COALESCE(w.tag_value LIKE '%tunnel%', false)
FROM road_segment s
LEFT JOIN ways        w ON w.gid = s.gid
LEFT JOIN configuration c ON c.id = w.tag_id
WHERE ST_Length(s.the_geom::geography) > 0
ON CONFLICT (segment_id) DO UPDATE SET
    highway_class   = EXCLUDED.highway_class,
    length_m        = EXCLUDED.length_m,
    bearing_in_deg  = EXCLUDED.bearing_in_deg,
    bearing_out_deg = EXCLUDED.bearing_out_deg,
    speed_limit_kph = EXCLUDED.speed_limit_kph,
    oneway          = EXCLUDED.oneway;
"""

# A segment "has a signal at the end" if a traffic-signal node sits within 12 m of its
# terminal vertex. 12 m absorbs the offset between an OSM signal node and the way's
# endpoint without picking up the signal on the far side of a narrow intersection.
ATTACH_SIGNALS_SQL = """
UPDATE segment_attr a
SET has_signal_at_end = EXISTS (
    SELECT 1
    FROM traffic_signal t
    JOIN road_segment s ON s.gid = a.segment_id
    WHERE ST_DWithin(
        t.the_geom::geography,
        ST_EndPoint(s.the_geom)::geography,
        12.0
    )
);
"""

ANALYZE_SQL = """
ANALYZE road_segment;
ANALYZE road_segment_vertices_pgr;
ANALYZE segment_attr;
"""

SUMMARY_SQL = """
SELECT
    highway_class,
    count(*)                    AS segments,
    round(sum(length_m)::numeric / 1000.0, 1) AS km,
    round(avg(length_m)::numeric, 1)          AS mean_length_m
FROM segment_attr
GROUP BY highway_class
ORDER BY km DESC;
"""


def derive_attributes(runner: SqlRunner) -> None:
    runner.execute_script(DERIVE_ATTR_SQL, label="osm.derive_attributes")
    runner.execute_script(ATTACH_SIGNALS_SQL, label="osm.attach_signals")
    runner.execute_script(ANALYZE_SQL, label="osm.analyze")


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to eta_forge.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--extract", default=None,
                        help="OSM extract path (default: data.osm_extract)")
    parser.add_argument("--mapping", default="configs/osm2pgrouting.xml")
    parser.add_argument("--clean", action="store_true",
                        help="drop and recreate the graph tables before importing")
    parser.add_argument("--skip-import", action="store_true",
                        help="only re-derive segment_attr from an existing graph")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command and SQL; touch nothing")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    extract = args.extract or cfg.data.osm_extract

    command = osm2pgrouting_command(cfg, extract, args.mapping, clean=args.clean)

    if args.dry_run:
        print("# target:", redacted_dsn(cfg.database))
        if not args.skip_import:
            print("# step 1 — import the extract")
            print("PGPASSWORD=***", shlex.join(command))
            print()
        print("# step 2 — derive segment_attr, attach signals, analyze")
        runner = SqlRunner()
        derive_attributes(runner)
        print(runner.recorder.script())
        return 0

    if not args.skip_import:
        if not os.path.exists(extract):
            print(f"error: OSM extract not found: {extract}", file=sys.stderr)
            return 2
        env = dict(os.environ)
        if cfg.database.password:
            env["PGPASSWORD"] = cfg.database.password
        print("running:", shlex.join(command))
        result = subprocess.run(command, env=env, check=False)
        if result.returncode != 0:
            print(f"error: osm2pgrouting exited {result.returncode}", file=sys.stderr)
            return result.returncode

    with ConnectionPool(cfg.database) as pool:
        runner = SqlRunner(pool)
        derive_attributes(runner)
        print(f"{'class':<14}{'segments':>10}{'km':>12}{'mean_m':>10}")
        for row in runner.fetch_all(SUMMARY_SQL):
            print(
                f"{row['highway_class']:<14}{row['segments']:>10}"
                f"{float(row['km']):>12.1f}{float(row['mean_length_m']):>10.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
