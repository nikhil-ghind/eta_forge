#!/usr/bin/env python3
"""Train the residual, prep-time and dwell models, register them, and gate promotion.

    python scripts/train.py --synthetic                       # no database, no downloads
    python scripts/train.py --synthetic --backend stumps --days 21
    python scripts/train.py --since 2024-01-01 --until 2024-04-01
    python scripts/train.py --since 2024-01-01 --no-promote   # register only

Every run registers a new content-addressed version of each model with a manifest
(feature-spec hash, training window, row counts, parameters, held-out metrics, git
sha). Promotion to the serving alias is gated on the residual model's held-out p90
coverage: a "p90" that holds 75% of the time is not a p90, and one that holds 99% of
the time is padding. Outside ``model.promote_coverage_band`` the run exits with status
3 and production is left alone, unless ``--force-promote`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eta_forge import __version__  # noqa: E402
from eta_forge.config import Config, load_config  # noqa: E402
from eta_forge.data.synthetic import SyntheticConfig, SyntheticSource  # noqa: E402
from eta_forge.features.builder import FeatureBuilder  # noqa: E402
from eta_forge.features.weather import WeatherTable  # noqa: E402
from eta_forge.graph.synthetic import SyntheticGraph  # noqa: E402
from eta_forge.models.dataset import build_rows, warehouse_trips  # noqa: E402
from eta_forge.models.dwell import DwellEstimator  # noqa: E402
from eta_forge.models.prep_time import PrepTimeModel  # noqa: E402
from eta_forge.models.quantile import QuantilePrior, lightgbm_params, resolve_backend  # noqa: E402
from eta_forge.models.registry import ModelRegistry, ModelVersion  # noqa: E402
from eta_forge.models.residual import ResidualTrainer  # noqa: E402
from eta_forge.models.splits import time_split  # noqa: E402
from eta_forge.types import (  # noqa: E402
    DwellObservation,
    PrepObservation,
    TripRecord,
    _parse_ts,
)

PREP_SQL = """
SELECT order_id, merchant_id, cuisine_category, placed_at, prep_s, item_count,
       order_value_usd, open_orders_at_placement, prep_is_upper_bound
FROM merchant_prep_observation
WHERE placed_at >= %s AND placed_at < %s
ORDER BY placed_at
"""

DWELL_SQL = """
SELECT order_id, destination_h3_r8::text AS destination_cell, arrived_at, dwell_s
FROM handoff_dwell_observation
WHERE arrived_at >= %s AND arrived_at < %s
ORDER BY arrived_at
"""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--days", type=int, default=28, help="synthetic days")
    parser.add_argument("--trips-per-day", type=int, default=120)
    parser.add_argument("--orders-per-day", type=int, default=150)
    parser.add_argument("--since", default=None, help="warehouse window start (ISO-8601)")
    parser.add_argument("--until", default=None)
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--backend", choices=("auto", "lightgbm", "stumps"), default=None)
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--alias", default=None, help="alias to promote (default: service.model_alias)")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--force-promote", action="store_true")
    parser.add_argument("--report-json", default=None, help="write the full training report here")
    return parser.parse_args(argv)


def git_sha() -> Optional[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


# ----------------------------------------------------------------------------------
# data sources
# ----------------------------------------------------------------------------------


class Corpus:
    def __init__(self, trips: Sequence[TripRecord], router: Any, weather: Optional[WeatherTable],
                 prep: Sequence[PrepObservation], dwell: Sequence[DwellObservation], source: str):
        self.trips = trips
        self.router = router
        self.weather = weather
        self.prep = prep
        self.dwell = dwell
        self.source = source


def synthetic_corpus(cfg: Config, args: argparse.Namespace, builder: FeatureBuilder) -> Corpus:
    graph = SyntheticGraph()
    source = SyntheticSource(
        graph,
        SyntheticConfig(
            days=args.days,
            trips_per_day=args.trips_per_day,
            orders_per_day=args.orders_per_day,
        ),
    )
    trips = [t.record for t in source.trips()]
    return Corpus(
        trips=trips,
        router=graph,
        weather=source.weather_table() if cfg.features.use_weather else None,
        prep=source.prep_observations(),
        dwell=source.dwell_observations(
            resolution=cfg.features.h3_resolution_coarse, scheme=builder.scheme
        ),
        source="synthetic",
    )


def warehouse_corpus(cfg: Config, args: argparse.Namespace, runner: Any) -> Corpus:
    from eta_forge.graph.router import Router

    until = _parse_ts(args.until) if args.until else datetime.now(timezone.utc)
    since = _parse_ts(args.since) if args.since else until - timedelta(days=90)
    trips = list(warehouse_trips(runner, since=since, until=until, limit=args.limit))
    weather = (
        WeatherTable.from_runner(
            runner, since, until,
            max_staleness=timedelta(minutes=cfg.features.weather_max_staleness_min),
        )
        if cfg.features.use_weather else None
    )
    prep = []
    for row in runner.fetch_all(PREP_SQL, (since, until), label="train.prep"):
        try:
            prep.append(PrepObservation(
                order_id=str(row["order_id"]),
                merchant_id=str(row["merchant_id"]),
                placed_at=_parse_ts(row["placed_at"]),
                prep_s=float(row["prep_s"]),
                cuisine_category=str(row.get("cuisine_category") or "other"),
                item_count=int(row.get("item_count") or 1),
                order_value_usd=row.get("order_value_usd"),
                open_orders_at_merchant=row.get("open_orders_at_placement"),
                prep_is_upper_bound=bool(row.get("prep_is_upper_bound")),
            ))
        except (ValueError, TypeError):
            continue
    dwell = []
    for row in runner.fetch_all(DWELL_SQL, (since, until), label="train.dwell"):
        try:
            dwell.append(DwellObservation(
                order_id=str(row["order_id"]),
                destination_cell=str(row["destination_cell"]),
                arrived_at=_parse_ts(row["arrived_at"]),
                dwell_s=float(row["dwell_s"]),
            ))
        except (ValueError, TypeError):
            continue
    return Corpus(trips, Router(cfg, runner), weather, prep, dwell, source="warehouse")


# ----------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------


def fit_dwell(cfg: Config, observations: Sequence[DwellObservation]) -> Tuple[DwellEstimator, Dict[str, Any]]:
    fallback = QuantilePrior(cfg.service.default_handoff_p50_s, cfg.service.default_handoff_p90_s, 0)
    if not observations:
        return DwellEstimator(fallback, k=cfg.model.dwell_shrink_k), {"rows": 0}
    split = time_split(
        [o.arrived_at for o in observations],
        train_fraction=cfg.model.train_fraction,
        validation_fraction=cfg.model.validation_fraction,
    )
    train = [observations[i] for i in split.train + split.validation]
    test = [observations[i] for i in split.test]
    estimator = DwellEstimator.fit(train, k=cfg.model.dwell_shrink_k, fallback=fallback)
    return estimator, {"sizes": split.sizes(), "cells": len(estimator.cells),
                       "test": estimator.evaluate(test)}


def base_manifest(cfg: Config, backend: str, source: str) -> Dict[str, Any]:
    return {
        "eta_forge_version": __version__,
        "git_sha": git_sha(),
        "backend": backend,
        "data_source": source,
        "quantiles": list(cfg.model.quantiles),
        "params": lightgbm_params(cfg.model) if backend == "lightgbm" else {
            "stump_rounds": cfg.model.stump_rounds,
            "stump_learning_rate": cfg.model.stump_learning_rate,
            "subsample": cfg.model.subsample,
        },
    }


def in_band(value: Optional[float], band: Sequence[float]) -> bool:
    return value is not None and band[0] <= value <= band[1]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    backend = resolve_backend(args.backend or cfg.model.backend)
    registry = ModelRegistry(args.registry_dir or cfg.model.registry_dir)
    alias = args.alias or cfg.service.model_alias
    builder = FeatureBuilder(cfg)

    pool = None
    if args.synthetic:
        corpus = synthetic_corpus(cfg, args, builder)
    else:
        from eta_forge.db.pool import ConnectionPool
        from eta_forge.db.runner import SqlRunner

        pool = ConnectionPool(cfg.database).open()
        corpus = warehouse_corpus(cfg, args, SqlRunner(pool))

    try:
        print(f"# source={corpus.source} backend={backend} trips={len(corpus.trips):,} "
              f"prep={len(corpus.prep):,} dwell={len(corpus.dwell):,}")
        rows, build_report = build_rows(
            corpus.trips, corpus.router, builder,
            weather=corpus.weather, max_residual_abs_s=cfg.model.max_residual_abs_s,
        )
        print(f"rows built {build_report.rows_built:,}  skipped {dict(build_report.skipped)}")

        residual, report, _ = ResidualTrainer(cfg, backend=backend).fit(rows)
        prep, prep_report = PrepTimeModel.fit(corpus.prep, cfg, backend=backend) if corpus.prep else (None, {})
        dwell, dwell_report = fit_dwell(cfg, corpus.dwell)
    finally:
        if pool is not None:
            pool.close()

    for split_name in ("validation", "test"):
        m = report.metrics.get(split_name, {})
        if m:
            print(f"{split_name:<10} MAE p50 {m['mae_p50']:7.1f}s  (baseline {m['baseline_mae']:7.1f}s)  "
                  f"p90 coverage {m['coverage_p90']:.3f}  pinball p90 {m['pinball_p90']:.2f}  "
                  f"crossing {m['crossing_rate']:.3f}")
    if prep_report:
        print(f"prep       test {prep_report.get('test')}")
    print(f"dwell      test {dwell_report.get('test')}")

    common = base_manifest(cfg, backend, corpus.source)
    residual_version = registry.register(
        "residual",
        residual.save,
        dict(common, **{
            "feature_spec_hash": report.feature_spec_hash,
            "feature_names": list(residual.builder.spec.names),
            "training_window": report.window,
            "rows": report.sizes,
            "metrics": report.metrics,
            "build_report": build_report.as_dict(),
        }),
    )
    versions: List[ModelVersion] = [residual_version]
    if prep is not None:
        versions.append(registry.register("prep", prep.save, dict(common, **{"report": prep_report})))
    versions.append(registry.register("dwell", dwell.save, dict(common, **{"report": dwell_report})))
    for version in versions:
        print(f"registered {version.version}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump({
                "residual": report.as_dict(),
                "prep": prep_report,
                "dwell": dwell_report,
                "versions": [v.version for v in versions],
            }, handle, indent=2, default=str)

    if args.no_promote:
        print("not promoting (--no-promote)")
        return 0

    band = cfg.model.promote_coverage_band
    held_out = report.coverage("test")
    if not in_band(held_out, band) and not args.force_promote:
        print(f"REFUSING to promote: held-out p90 coverage {held_out} outside "
              f"[{band[0]:.2f}, {band[1]:.2f}]; production left unchanged")
        return 3
    reason = f"p90 coverage {held_out:.3f}" if held_out is not None else "forced"
    for version in versions:
        previous = registry.promote(version.version, alias=alias, reason=reason)
        print(f"promoted {version.version} -> {alias} (was {previous})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
