#!/usr/bin/env python3
"""Load NYC TLC trip files into the ``trip`` table.

    python scripts/load_tlc.py data/tlc/yellow_tripdata_2015-03.parquet
    python scripts/load_tlc.py data/tlc/*.parquet --zones-csv data/tlc/zone_centroids.csv
    python scripts/load_tlc.py data/tlc/green_tripdata_2015-03.csv --dry-run --limit 1000

Every file is streamed, normalised onto one schema, quality-filtered and inserted with
``ON CONFLICT DO NOTHING`` on a deterministic trip id, so re-running a file is safe.
The per-reason rejection tally is printed and recorded in ``ingest_batch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eta_forge.config import load_config  # noqa: E402
from eta_forge.data.quality import RejectionTally  # noqa: E402
from eta_forge.data.tlc import (  # noqa: E402
    INGEST_BATCH_SQL,
    TRIP_INSERT_SQL,
    TlcLoader,
    ZoneCentroids,
    file_checksum,
    trip_insert_row,
)
from eta_forge.db.dsn import redacted_dsn  # noqa: E402
from eta_forge.db.pool import ConnectionPool  # noqa: E402
from eta_forge.db.runner import SqlRunner  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*", help="files or globs (default: data.tlc_dir/*)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--dataset", choices=("yellow", "green", "fhv", "fhvhv"),
                        default=None, help="override schema detection")
    parser.add_argument("--zones-csv", default=None,
                        help="location_id,lat,lon centroids (default: taxi_zone table)")
    parser.add_argument("--limit", type=int, default=None, help="rows per file")
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and filter, print the tally and the insert SQL")
    return parser.parse_args(argv)


def expand(paths: List[str], default_dir: str) -> List[str]:
    patterns = paths or [os.path.join(default_dir, "*.parquet"),
                         os.path.join(default_dir, "*.csv"),
                         os.path.join(default_dir, "*.csv.gz")]
    files: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else ([pattern] if os.path.exists(pattern) else []))
    return files


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    files = expand(args.paths, cfg.data.tlc_dir)
    if not files:
        print(f"error: no TLC files found (looked in {cfg.data.tlc_dir})", file=sys.stderr)
        return 2

    pool = None if args.dry_run else ConnectionPool(cfg.database).open()
    runner = SqlRunner(pool)
    try:
        if args.zones_csv:
            zones = ZoneCentroids.from_csv(args.zones_csv)
        elif not args.dry_run:
            zones = ZoneCentroids.from_runner(runner)
        else:
            zones = ZoneCentroids()
        print(f"# target: {redacted_dsn(cfg.database)}  zones: {len(zones)}")

        loader = TlcLoader(cfg.data, zones=zones)
        overall = RejectionTally()
        for path in files:
            records, result = loader.load_file(path, dataset=args.dataset, limit=args.limit)
            rows = (trip_insert_row(r, result.dataset) for r in records)
            inserted = runner.execute_many(
                TRIP_INSERT_SQL, rows, label="tlc.insert", batch_size=args.batch_size
            )
            if args.dry_run:
                # execute_many stops after one batch in dry-run; drain for the tally.
                for _ in records:
                    pass
            runner.execute(
                INGEST_BATCH_SQL,
                (
                    os.path.basename(path),
                    None if args.dry_run else file_checksum(path),
                    result.tally.read,
                    result.tally.kept,
                    json.dumps(dict(result.tally.rejected)),
                ),
                label="tlc.ingest_batch",
            )
            overall.merge(result.tally)
            print(f"\n{path}  [{result.dataset}]  "
                  f"endpoints={result.endpoint_trips:,} zone={result.zone_trips:,} "
                  f"inserted={inserted:,}")
            for line in result.tally.summary_lines():
                print("  " + line)

        print("\n# all files")
        for line in overall.summary_lines():
            print("  " + line)
        if args.dry_run and runner.recorder is not None:
            print("\n# statements (first batch only)")
            print(runner.recorder.script())
    finally:
        if pool is not None:
            pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
