#!/usr/bin/env python3
"""Apply the SQL migrations under ``sql/``.

    python scripts/migrate.py --dry-run           # print the whole script
    python scripts/migrate.py --status            # what is applied, what is pending
    python scripts/migrate.py --config configs/eta_forge.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eta_forge.config import load_config  # noqa: E402
from eta_forge.db.dsn import redacted_dsn  # noqa: E402
from eta_forge.db.migrate import (  # noqa: E402
    applied_versions,
    discover_migrations,
    run_migrations,
)
from eta_forge.db.pool import ConnectionPool  # noqa: E402
from eta_forge.db.runner import SqlRunner  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--dir", dest="directory", default=None,
                        help="migrations directory (default: ./sql)")
    parser.add_argument("--target", type=int, default=None,
                        help="stop after this migration version")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify-checksums", action="store_true",
                        help="allow an edited already-applied migration (dangerous)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)

    if args.dry_run:
        runner = SqlRunner()
        applied = run_migrations(runner, directory=args.directory, target=args.target)
        print(f"-- {len(applied)} migration(s) would be applied to "
              f"{redacted_dsn(cfg.database)}")
        print(runner.recorder.script())
        return 0

    with ConnectionPool(cfg.database) as pool:
        runner = SqlRunner(pool)
        if args.status:
            known = applied_versions(runner)
            print(f"{'version':>8}  {'state':<9} slug")
            for migration in discover_migrations(args.directory):
                state = "applied" if migration.version in known else "pending"
                print(f"{migration.version:>8}  {state:<9} {migration.slug}")
            return 0

        applied = run_migrations(
            runner,
            directory=args.directory,
            target=args.target,
            verify_checksums=not args.no_verify_checksums,
        )
        if not applied:
            print("schema is up to date")
        for migration in applied:
            print(f"applied {migration.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
