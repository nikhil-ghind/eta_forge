#!/usr/bin/env python3
"""End-to-end smoke run over the synthetic world: no database, no downloads.

    python scripts/smoke_end_to_end.py
    python scripts/smoke_end_to_end.py --days 14 --rounds 60 --out-dir /tmp/eta-smoke

generate trips -> map-match traces -> mine + shrink profiles -> build features ->
train residual / prep / dwell -> register + promote -> serve ride and delivery ETAs
-> evaluate held-out trips -> write the report.

Each stage asserts the invariants it is responsible for and the run exits non-zero
on the first violation, so this doubles as an integration check.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eta_forge.config import load_config  # noqa: E402
from eta_forge.data.synthetic import SyntheticConfig, SyntheticSource  # noqa: E402
from eta_forge.decompose import naive_sum  # noqa: E402
from eta_forge.eval.harness import evaluate, records_from_rows  # noqa: E402
from eta_forge.eval.report import write_report  # noqa: E402
from eta_forge.features.builder import FeatureBuilder  # noqa: E402
from eta_forge.graph.synthetic import SyntheticGraph  # noqa: E402
from eta_forge.mapmatch import GraphCandidateSource, HmmMapMatcher, attribute_traversals  # noqa: E402
from eta_forge.models.dataset import build_rows  # noqa: E402
from eta_forge.models.dwell import DwellEstimator  # noqa: E402
from eta_forge.models.prep_time import PrepTimeModel  # noqa: E402
from eta_forge.models.registry import ModelRegistry  # noqa: E402
from eta_forge.models.residual import ResidualTrainer  # noqa: E402
from eta_forge.profiles.store import build_profiles  # noqa: E402
from eta_forge.service import EtaService, ModelBundle  # noqa: E402
from eta_forge.types import EtaRequest  # noqa: E402


class SmokeFailure(SystemExit):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        raise SmokeFailure(1)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE")
    parser.add_argument("--grid", type=int, default=16, help="synthetic grid size (rows = cols)")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--trips-per-day", type=int, default=60)
    parser.add_argument("--match-trips", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=40, help="stump booster rounds")
    parser.add_argument("--backend", choices=("stumps", "lightgbm", "auto"), default="stumps")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def stage(name: str):
    started = time.perf_counter()
    print(f"-> {name} ...", flush=True)
    return lambda detail="": print(f"   done in {time.perf_counter() - started:.1f}s {detail}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    cfg.model.backend = args.backend
    cfg.model.stump_rounds = args.rounds
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="eta-forge-smoke-")
    os.makedirs(out_dir, exist_ok=True)

    done = stage("generate synthetic trips")
    graph = SyntheticGraph(rows=args.grid, cols=args.grid)
    source = SyntheticSource(graph, SyntheticConfig(
        days=args.days, trips_per_day=args.trips_per_day, orders_per_day=80, merchants=20,
    ))
    synthetic = list(source.trips())
    trips = [t.record for t in synthetic]
    check(len(trips) > 100, f"too few trips generated ({len(trips)})")
    done(f"({len(trips)} trips)")

    done = stage("map-match traces and mine profiles")
    matcher = HmmMapMatcher(GraphCandidateSource(graph))
    classes = {e.segment_id: e.highway_class for e in graph.edges}
    observations = []
    for trip in synthetic[: args.match_trips]:
        result = matcher.match(trip.record.gps_trace)
        observations += [
            t.observation
            for t in attribute_traversals(result, trip_id=trip.record.trip_id, segment_classes=classes)
        ]
    check(bool(observations), "map matching attributed no traversals")
    profiles = build_profiles(
        observations, cfg.profiles, class_default=cfg.graph.default_speed_kph, all_segments=classes,
    )
    sources = profiles.source_counts()
    check(sources.get("observed", 0) > 0, "no observed profile cells")
    done(f"({len(observations)} traversals, cells by source {sources})")

    done = stage("build features")
    builder = FeatureBuilder(cfg)
    weather = source.weather_table()
    rows, build_report = build_rows(trips, graph, builder, weather=weather)
    check(len(rows) >= 100, f"too few training rows ({len(rows)})")
    done(f"({len(rows)} rows, {len(builder.spec)} features, skipped {dict(build_report.skipped)})")

    done = stage(f"train ({cfg.model.backend})")
    residual, report, split = ResidualTrainer(cfg).fit(rows)
    prep, prep_report = PrepTimeModel.fit(source.prep_observations(), cfg)
    dwell = DwellEstimator.fit(
        source.dwell_observations(resolution=cfg.features.h3_resolution_coarse, scheme=builder.scheme),
        k=cfg.model.dwell_shrink_k,
    )
    test = report.metrics["test"]
    check(test["mae_p50"] < test["baseline_mae"], "residual model did not beat routing-only on MAE")
    done(f"(test MAE {test['mae_p50']:.1f}s vs baseline {test['baseline_mae']:.1f}s, "
         f"p90 coverage {test['coverage_p90']:.3f})")

    done = stage("register and promote")
    registry = ModelRegistry(os.path.join(out_dir, "registry"))
    versions = [
        registry.register("residual", residual.save, {"metrics": report.metrics,
                                                      "feature_spec_hash": report.feature_spec_hash,
                                                      "training_window": report.window}),
        registry.register("prep", prep.save, {"report": prep_report}),
        registry.register("dwell", dwell.save, {}),
    ]
    for version in versions:
        registry.promote(version.version, reason="smoke")
        check(registry.verify(version.version), f"{version.version} failed its content hash")
    done(f"({', '.join(v.version for v in versions)})")

    done = stage("serve")
    bundle = ModelBundle.load(registry, cfg, strict=True)
    service = EtaService(cfg, graph, bundle=bundle, registry=registry, weather=weather)
    sw, ne = graph.bounds()
    departure = trips[-1].pickup_at
    ride = service.estimate(EtaRequest(origin=sw, destination=ne, departure_at=departure))
    check(not ride.degraded, f"ride ETA degraded: {ride.degraded_reason}")
    check(ride.eta.p90 >= ride.eta.p50 > 0, "ride quantiles out of order")
    merchant = source.merchants()[0]
    delivery = service.estimate(EtaRequest(
        origin=merchant.location, destination=ne, departure_at=departure,
        merchant_id=merchant.merchant_id, order_item_count=3, open_orders_at_merchant=4,
    ))
    b = delivery.breakdown
    check(abs(delivery.eta.p50 - (b.prep.p50 + b.travel.p50 + b.handoff.p50)) < 1e-6,
          "delivery p50 is not the sum of component p50s")
    check(delivery.eta.p90 <= naive_sum([b.prep, b.travel, b.handoff]).p90 + 1e-6,
          "composed p90 exceeds the naive sum of p90s")
    pinned = service.estimate(EtaRequest(origin=sw, destination=ne, departure_at=departure,
                                         model_version=versions[0].version))
    check(pinned.model_versions["residual"] == versions[0].version, "model pinning ignored")
    service.close()
    done(f"(ride p50 {ride.eta.p50 / 60:.1f} min, delivery p50 {delivery.eta.p50 / 60:.1f} / "
         f"p90 {delivery.eta.p90 / 60:.1f} min)")

    done = stage("evaluate held-out trips")
    held_out = [rows[i] for i in split.test]
    result = evaluate(records_from_rows(held_out, residual), cfg.eval, meta={
        "residual_version": versions[0].version,
        "backend": residual.backend,
        "feature_spec_hash": residual.spec_hash,
        "data_source": "synthetic",
    })
    paths = write_report(result, os.path.join(out_dir, "eval"), stem="smoke",
                         title="ETA Forge smoke evaluation")
    done(f"(p90 coverage {100 * result.overall['coverage_p90']:.1f}%)")

    print("\nsmoke run OK")
    for kind, path in paths.items():
        print(f"  {kind:<8} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
