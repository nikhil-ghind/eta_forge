#!/usr/bin/env python3
"""Evaluate a registered residual model on held-out trips and write the report.

    python scripts/evaluate.py --synthetic                          # fresh synthetic week
    python scripts/evaluate.py --model residual@shadow --synthetic
    python scripts/evaluate.py --since 2024-04-01 --until 2024-05-01
    python scripts/evaluate.py --model residual/20260824T1200Z-a1b2c3d4 --out-dir reports/

The held-out window must come *after* the model's training window; the script reads
the window from the manifest and refuses an overlapping one unless
``--allow-overlap`` is given, because evaluating on training-period trips grades the
model on questions it has seen the answers to.
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
from eta_forge.eval.harness import evaluate, records_from_rows  # noqa: E402
from eta_forge.eval.report import write_report  # noqa: E402
from eta_forge.graph.synthetic import SyntheticGraph  # noqa: E402
from eta_forge.models.dataset import build_rows, warehouse_trips  # noqa: E402
from eta_forge.models.registry import ModelRegistry  # noqa: E402
from eta_forge.models.residual import ResidualModel  # noqa: E402
from eta_forge.types import _parse_ts  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--model", default="residual", help="version, name@alias, or name")
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--trips-per-day", type=int, default=120)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--stem", default=None)
    return parser.parse_args(argv)


def training_end(manifest) -> Optional[datetime]:
    window = (manifest.get("training_window") or {}).get("train") or {}
    end = window.get("end")
    return _parse_ts(end) if end else None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    registry = ModelRegistry(args.registry_dir or cfg.model.registry_dir)
    version = registry.resolve(args.model, default_alias=cfg.service.model_alias)
    residual = ResidualModel.load(cfg, version.path)
    trained_until = training_end(version.manifest)
    print(f"# model {version.version}  backend={residual.backend}  trained until {trained_until}")

    pool = None
    if args.synthetic:
        start = (trained_until + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) \
            if trained_until else datetime(2024, 2, 5, tzinfo=timezone.utc)
        graph = SyntheticGraph()
        source = SyntheticSource(
            graph, SyntheticConfig(start=start, days=args.days,
                                   trips_per_day=args.trips_per_day, seed=args.seed)
        )
        trips = [t.record for t in source.trips()]
        router, weather, data_source = graph, source.weather_table(), "synthetic"
    else:
        from eta_forge.db.pool import ConnectionPool
        from eta_forge.db.runner import SqlRunner
        from eta_forge.features.weather import WeatherTable
        from eta_forge.graph.router import Router

        until = _parse_ts(args.until) if args.until else datetime.now(timezone.utc)
        since = _parse_ts(args.since) if args.since else until - timedelta(days=14)
        if trained_until and since <= trained_until and not args.allow_overlap:
            print(f"error: evaluation window starts {since.isoformat()}, inside the training "
                  f"window (ends {trained_until.isoformat()}); use --allow-overlap to override",
                  file=sys.stderr)
            return 2
        pool = ConnectionPool(cfg.database).open()
        runner = SqlRunner(pool)
        trips = list(warehouse_trips(runner, since=since, until=until, limit=args.limit))
        router = Router(cfg, runner)
        weather = WeatherTable.from_runner(runner, since, until) if cfg.features.use_weather else None
        data_source = "warehouse"

    try:
        rows, build_report = build_rows(
            trips, router, residual.builder, weather=weather,
            max_residual_abs_s=cfg.model.max_residual_abs_s,
        )
    finally:
        if pool is not None:
            pool.close()
    print(f"rows {build_report.rows_built:,}  skipped {dict(build_report.skipped)}")

    records = records_from_rows(rows, residual)
    result = evaluate(records, cfg.eval, meta={
        "residual_version": version.version,
        "backend": residual.backend,
        "feature_spec_hash": residual.spec_hash,
        "data_source": data_source,
    })
    stem = args.stem or "eval-" + version.version.replace("/", "-")
    paths = write_report(result, args.out_dir or cfg.eval.report_dir, stem=stem)

    o = result.overall
    lift = result.comparison[2].get("mae_s")
    print(f"p50 MAE {o['mae_s']:.1f}s  MAPE {100 * o['mape']:.1f}%  "
          f"p90 coverage {100 * o['coverage_p90']:.1f}%  sharpness {o['sharpness_s']:.0f}s  "
          f"MAE lift vs routing-only {'n/a' if lift is None else f'{100 * lift:.1f}%'}")
    for kind, path in paths.items():
        print(f"wrote {kind:<8} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
