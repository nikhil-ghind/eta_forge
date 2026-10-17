#!/usr/bin/env python3
"""Mine, smooth and publish the segment speed profiles.

    python scripts/build_profiles.py --synthetic                 # in memory, no database
    python scripts/build_profiles.py --days 90                   # last 90 days of traversals
    python scripts/build_profiles.py --days 90 --dry-run         # print the write SQL
    python scripts/build_profiles.py --days 90 --no-swap         # stage only, swap later

The publish is a staging-table swap: routing keeps reading the old profiles until the
new ones are complete, and the previous generation survives as ``*_previous`` for a
one-rename rollback.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eta_forge.config import load_config  # noqa: E402
from eta_forge.data.synthetic import SyntheticConfig, SyntheticSource  # noqa: E402
from eta_forge.db.pool import ConnectionPool  # noqa: E402
from eta_forge.db.runner import SqlRunner  # noqa: E402
from eta_forge.graph.synthetic import SyntheticGraph  # noqa: E402
from eta_forge.mapmatch import (  # noqa: E402
    GraphCandidateSource,
    HmmMapMatcher,
    attribute_traversals,
)
from eta_forge.profiles.buckets import BucketSpec, bucket_label  # noqa: E402
from eta_forge.profiles.store import ProfileBuild, ProfileStore, build_profiles  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-trips", type=int, default=1_500)
    parser.add_argument("--ground-truth", action="store_true",
                        help="synthetic only: mine the generator's true traversals "
                             "instead of map-matched ones")
    parser.add_argument("--days", type=int, default=None,
                        help="mining window (default: profiles.max_profile_age_days)")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def print_build(build: ProfileBuild, spec: BucketSpec) -> None:
    report = build.report.as_dict()
    print(f"observations read   {report['observations_read']:,}")
    print(f"observations kept   {report['observations_kept']:,}  rejected {report['rejected']}")
    print(f"segments mined      {report['segments']:,}")
    print(f"cells observed      {report['cells']:,}  "
          f"(below min samples: {report['cells_below_threshold']:,})")
    counts = build.source_counts()
    total = sum(counts.values()) or 1
    print("smoothed cells by source:")
    for source, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<10}{count:>12,}  {100.0 * count / total:5.1f}%")

    # The slowest well-observed cells: a sanity check a human can eyeball.
    observed = [
        cell
        for cells in build.smoothed.values()
        for cell in cells.values()
        if cell.source == "observed" and cell.sample_count >= 5
    ]
    observed.sort(key=lambda c: c.speed_kph)
    if observed:
        print("slowest observed cells:")
        for cell in observed[:8]:
            print(f"  segment {cell.segment_id:>8}  "
                  f"{bucket_label(cell.dow_bucket, cell.tod_bucket, spec):<18}"
                  f"{cell.speed_kph:6.1f} km/h  (raw {cell.raw_speed_kph:5.1f}, "
                  f"n={cell.sample_count}, w={cell.shrink_weight:.2f})")


def run_synthetic(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=args.overrides)
    graph = SyntheticGraph()
    per_day = max(1, args.synthetic_trips // 14)
    source = SyntheticSource(graph, SyntheticConfig(days=14, trips_per_day=per_day))
    classes = {e.segment_id: e.highway_class for e in graph.edges}

    if args.ground_truth:
        observations = [obs for trip in source.trips() for obs in trip.true_traversals]
    else:
        matcher = HmmMapMatcher(GraphCandidateSource(graph))
        observations = []
        for trip in source.trips():
            result = matcher.match(trip.record.gps_trace)
            for attributed in attribute_traversals(
                result, trip_id=trip.record.trip_id, segment_classes=classes
            ):
                if attributed.confidence >= args.min_confidence:
                    observations.append(attributed.observation)

    store = ProfileStore(SqlRunner(), cfg.profiles) if args.dry_run else None
    build = build_profiles(
        observations,
        cfg.profiles,
        class_default=cfg.graph.default_speed_kph,
        all_segments=classes,
        store=store,
        swap=not args.no_swap,
    )
    print_build(build, BucketSpec.from_config(cfg.profiles))
    return 0


def run_database(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=args.overrides)
    days = args.days or cfg.profiles.max_profile_age_days
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=days)

    pool = None if args.dry_run else ConnectionPool(cfg.database).open()
    runner = SqlRunner(pool)
    try:
        store = ProfileStore(runner, cfg.profiles)
        all_segments = store.segment_classes()
        build = build_profiles(
            store.observations(since=since, until=until, min_confidence=args.min_confidence),
            cfg.profiles,
            class_default=cfg.graph.default_speed_kph,
            all_segments=all_segments,
            store=store,
            swap=not args.no_swap,
        )
        print(f"window {since:%Y-%m-%d} .. {until:%Y-%m-%d}  "
              f"segments in graph: {len(all_segments):,}")
        print_build(build, BucketSpec.from_config(cfg.profiles))
        if build.result is not None:
            print(f"written: {build.result.as_dict()}")
        if args.dry_run and runner.recorder is not None:
            print(runner.recorder.script())
    finally:
        if pool is not None:
            pool.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_synthetic(args) if args.synthetic else run_database(args)


if __name__ == "__main__":
    raise SystemExit(main())
