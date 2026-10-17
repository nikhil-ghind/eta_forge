#!/usr/bin/env python3
"""Map-match GPS traces onto road segments and write per-segment traversals.

    python scripts/map_match.py --synthetic --trips 200        # no database needed
    python scripts/map_match.py --since 2024-01-01 --limit 5000
    python scripts/map_match.py --since 2024-01-01 --dry-run

``--synthetic`` generates trips over the synthetic grid, matches their noisy traces,
and scores the result against the ground truth the generator knows: segment recall
and the error in attributed traversal time. That is the only place a map matcher can
be graded honestly — real traces come with no answer key.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eta_forge.config import Config, load_config  # noqa: E402
from eta_forge.data.synthetic import SyntheticConfig, SyntheticSource  # noqa: E402
from eta_forge.db.pool import ConnectionPool  # noqa: E402
from eta_forge.db.runner import SqlRunner  # noqa: E402
from eta_forge.graph.synthetic import SyntheticGraph  # noqa: E402
from eta_forge.mapmatch import (  # noqa: E402
    GraphCandidateSource,
    HmmMapMatcher,
    HmmParams,
    PostgisCandidateSource,
    attribute_traversals,
    trip_segment_rows,
)
from eta_forge.mapmatch.attribution import TRIP_SEGMENT_INSERT_SQL  # noqa: E402
from eta_forge.types import GpsPoint, Location, _parse_ts  # noqa: E402

TRACES_SQL = """
SELECT g.trip_id, g.seq, g.recorded_at, ST_Y(g.the_geom) AS lat, ST_X(g.the_geom) AS lon,
       g.speed_kph, g.accuracy_m
FROM trip_gps_point g
JOIN trip t USING (trip_id)
WHERE t.pickup_at >= %s AND t.pickup_at < %s
  AND t.match_quality IN ('endpoints', 'failed')
ORDER BY g.trip_id, g.seq
LIMIT %s
"""

MARK_TRIP_SQL = "UPDATE trip SET match_quality = %s WHERE trip_id = %s"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--trips", type=int, default=200, help="synthetic trip count")
    parser.add_argument("--since", default="2024-01-01T00:00:00+00:00")
    parser.add_argument("--until", default=None)
    parser.add_argument("--limit", type=int, default=2_000_000, help="max GPS points read")
    parser.add_argument("--sigma", type=float, default=HmmParams.sigma_m)
    parser.add_argument("--beta", type=float, default=HmmParams.beta_m)
    parser.add_argument("--radius", type=float, default=HmmParams.search_radius_m)
    parser.add_argument("--min-matched", type=float, default=0.6,
                        help="matched-fraction below which a trip is marked failed")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_synthetic(args: argparse.Namespace) -> int:
    graph = SyntheticGraph()
    source = SyntheticSource(graph, SyntheticConfig(days=2, trips_per_day=max(1, args.trips)))
    matcher = HmmMapMatcher(
        GraphCandidateSource(graph),
        HmmParams(sigma_m=args.sigma, beta_m=args.beta, search_radius_m=args.radius),
    )
    classes = {e.segment_id: e.highway_class for e in graph.edges}

    recall_hits = recall_total = 0
    abs_errors: List[float] = []
    breaks = 0
    observations = 0
    for trip in source.trips(limit=args.trips):
        result = matcher.match(trip.record.gps_trace)
        breaks += result.breaks
        decoded = set(result.matched_segments)
        truth = {t.segment_id: t for t in trip.true_traversals}
        recall_total += len(truth)
        recall_hits += len(decoded & set(truth))
        for attributed in attribute_traversals(
            result, trip_id=trip.record.trip_id, segment_classes=classes
        ):
            observations += 1
            true = truth.get(attributed.observation.segment_id)
            if true is not None:
                abs_errors.append(abs(attributed.observation.traversal_s - true.traversal_s))

    recall = recall_hits / recall_total if recall_total else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    print(f"trips matched          {args.trips}")
    print(f"segment recall         {recall:.3f}")
    print(f"lattice breaks         {breaks}")
    print(f"traversals attributed  {observations}")
    print(f"traversal-time MAE     {mae:.2f} s  (over {len(abs_errors)} matched to truth)")
    return 0


def _group_traces(rows) -> Dict[str, List[GpsPoint]]:
    traces: Dict[str, List[GpsPoint]] = defaultdict(list)
    for row in rows:
        traces[str(row["trip_id"])].append(
            GpsPoint(
                at=_parse_ts(row["recorded_at"]),
                location=Location(float(row["lat"]), float(row["lon"])),
                speed_kph=row.get("speed_kph"),
                accuracy_m=float(row.get("accuracy_m") or 15.0),
            )
        )
    return traces


def run_database(cfg: Config, args: argparse.Namespace) -> int:
    since = _parse_ts(args.since)
    until = _parse_ts(args.until) if args.until else datetime.now(timezone.utc)
    pool = None if args.dry_run else ConnectionPool(cfg.database).open()
    runner = SqlRunner(pool)
    try:
        matcher = HmmMapMatcher(
            PostgisCandidateSource(
                runner,
                edge_table=cfg.graph.edge_table,
                attr_table=cfg.graph.attr_table,
                vertex_table=cfg.graph.vertex_table,
            ),
            HmmParams(sigma_m=args.sigma, beta_m=args.beta, search_radius_m=args.radius),
        )
        traces = _group_traces(
            runner.fetch_all(TRACES_SQL, (since, until, args.limit), label="mapmatch.traces")
        )
        matched = failed = written = 0
        for trip_id, trace in traces.items():
            result = matcher.match(trace)
            if result.matched_fraction < args.min_matched:
                runner.execute(MARK_TRIP_SQL, ("failed", trip_id), label="mapmatch.mark")
                failed += 1
                continue
            rows = trip_segment_rows(trip_id, attribute_traversals(result, trip_id=trip_id))
            written += runner.execute_many(
                TRIP_SEGMENT_INSERT_SQL, rows, label="mapmatch.insert"
            )
            runner.execute(MARK_TRIP_SQL, ("matched", trip_id), label="mapmatch.mark")
            matched += 1
        print(f"traces {len(traces):,}  matched {matched:,}  failed {failed:,}  "
              f"traversals written {written:,}")
        if args.dry_run and runner.recorder is not None:
            print(runner.recorder.script())
    finally:
        if pool is not None:
            pool.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.synthetic:
        return run_synthetic(args)
    cfg = load_config(args.config, overrides=args.overrides)
    return run_database(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
