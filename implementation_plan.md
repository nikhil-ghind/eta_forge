# ETA Forge — Implementation Plan

An **ETA prediction service** for a ride-hail / food-delivery marketplace, built on the
two-stage architecture that Uber has publicly described: a **physical routing engine**
produces a baseline duration from a real road graph, and a **learned model corrects the
residual** between that baseline and what actually happened on the street.

The baseline alone is wrong in structured, learnable ways — it does not know that the
Williamsburg Bridge crawls at 18:00 on a Thursday, that a left turn across Broadway costs
40 seconds, or that this particular merchant takes 11 minutes to bag an order. The residual
model learns exactly those corrections, and it learns them as **quantiles**, because an ETA
is a promise: p90 coverage is a first-class metric, not a footnote.

```
ETA = merchant prep time  +  travel time  +  handoff dwell
                              │
                              └── baseline (pgRouting over profiled edge costs)
                                  + residual correction (LightGBM, quantile loss)
```

## Design constraints

The stack targets **PostgreSQL 16 + PostGIS 3.4 + pgRouting 3.6 + h3-pg**, fed by an OSM
extract and NYC TLC trip records. Neither the database nor the multi-gigabyte source data
is present in the development environment, which drives three rules applied everywhere:

1. **SQL is authored, not embedded ad hoc.** Schema, indexes, materialized views and the
   dynamic-cost functions live in versioned `sql/` migrations. Python composes queries
   through a thin typed layer; no query is built by string-concatenating user input.
2. **Heavy imports are lazy.** `psycopg`, `lightgbm`, `pandas`, `numpy`, `h3` and `fastapi`
   are imported inside functions or behind a guarded accessor, never at module import time.
   Importing any module in this repo on a bare interpreter must succeed.
3. **The numeric core is pure Python.** Pinball loss, coverage and calibration, MAE/MAPE,
   speed-profile shrinkage, turn classification, the decomposition arithmetic, feature
   encoding and the model registry carry no database or LightGBM dependency and are covered
   by stdlib `unittest` tests that run in milliseconds. A `SyntheticSource` produces
   deterministic trips and a fake graph so the whole pipeline — mine profiles → build
   features → train → evaluate → serve — can be exercised end to end with no PostGIS.

---

## Phase 1 — Config, domain types, database layer, and the PostGIS/pgRouting schema

The foundation: what an ETA request *is*, where the data lives, and the SQL that makes a
road network routable and a trip corpus queryable.

**Deliverables**
- Domain types: `Location`, `EtaRequest`, `EtaResponse`, `RouteLeg`, `BaselineRoute`,
  `TripRecord`, `SegmentObservation`, `EtaBreakdown` — validated (lat/lon ranges, non-empty
  waypoints, timezone-aware timestamps), all `to_dict`/`from_dict` round-trippable.
- Layered config: YAML file → `ETA_FORGE_*` environment overrides → CLI `--set` overrides,
  with typed sections (`database`, `graph`, `profiles`, `features`, `model`, `service`,
  `eval`) and precise error messages on bad keys or types.
- Database layer: DSN assembly, a lazily-imported `psycopg` connection pool, a
  `SqlRunner` that executes parameterised statements and migration files in order, and a
  `--dry-run` mode that prints the SQL it would run without a server.
- Versioned SQL migrations: extensions (`postgis`, `pgrouting`, `h3`, `h3_postgis`);
  `road_segment` / `road_vertex` topology tables written by `osm2pgrouting`, plus
  `segment_attr` (highway class, lanes, oneway, speed limit, length, bearing in/out);
  `trip` and `trip_gps_point` tables; `merchant` table; H3 cell columns with generated
  `h3_lat_lng_to_cell` expressions at resolutions 8 and 9; the full index set (GiST on
  geometry, btree on `source`/`target`, H3 cell indexes).
- `osm2pgrouting` driver script: config XML for the highway classes we keep, invocation
  with the right flags, post-import `segment_attr` derivation, and a strongly-worded
  `--dry-run` that emits the exact commands.
- Docker Compose: `postgis/postgis:16-3.4` with h3-pg installed, an `initdb` hook running
  the migrations, plus the API service; named volume for the cluster.

**Files**
`eta_forge/__init__.py`, `eta_forge/types.py`, `eta_forge/config.py`,
`eta_forge/db/__init__.py`, `eta_forge/db/dsn.py`, `eta_forge/db/pool.py`,
`eta_forge/db/runner.py`, `eta_forge/db/migrate.py`,
`sql/001_extensions.sql`, `sql/002_graph.sql`, `sql/003_trips.sql`,
`sql/004_merchants.sql`, `sql/005_indexes.sql`,
`scripts/import_osm.py`, `configs/eta_forge.yaml`, `configs/osm2pgrouting.xml`,
`deploy/docker-compose.yml`, `deploy/Dockerfile`, `deploy/initdb.sh`,
`requirements.txt`, `tests/test_types.py`, `tests/test_config.py`, `tests/test_db_dsn.py`

---

## Phase 2 — Baseline routing: pgRouting shortest path, turn analysis, dynamic edge costs

Stage one of the two-stage model. Everything here is *physics*: the graph, the path, the
free-flow duration, and the structural descriptors of that path the learner will need.

**Deliverables**
- `Router`: snap origin/destination to the nearest routable vertex (`<->` KNN against the
  GiST index with a search radius and a clear `NoRouteFound` when nothing is near),
  `pgr_dijkstra` over a cost SQL, and assembly of the ordered edge list into a
  `BaselineRoute` with per-leg geometry, length, class and duration.
- **Dynamic cost SQL builder**: the cost expression handed to `pgr_dijkstra` is not a
  constant. It is composed at query time as
  `length_m / COALESCE(profile_speed(segment, dow, tod_bucket), class_default_speed)`,
  with the reverse cost mirroring `oneway`. The builder emits the SQL text and binds the
  time bucket, so a route at 08:00 Monday and 03:00 Sunday traverse different graphs.
- Turn analysis: bearing delta between consecutive legs classified into
  `straight / slight / turn / sharp / u-turn`, left vs right by sign, plus signal-density
  and class-transition counts. Pure geometry — great-circle bearing, angle wrapping across
  the ±180° seam — and fully unit-tested without a database.
- Route summarisation: total distance, free-flow duration, **class mix** (fraction of
  distance on motorway / trunk / primary / secondary / residential / service), turn counts
  by type, and the count of distinct road classes traversed.
- `SyntheticGraph`: an in-memory grid graph with the same `Router` interface so routing,
  turn analysis and everything downstream runs with no PostGIS.

**Files**
`eta_forge/graph/__init__.py`, `eta_forge/graph/geometry.py`,
`eta_forge/graph/costs.py`, `eta_forge/graph/router.py`,
`eta_forge/graph/turns.py`, `eta_forge/graph/summary.py`,
`eta_forge/graph/synthetic.py`, `sql/006_routing_functions.sql`,
`tests/test_geometry.py`, `tests/test_turns.py`, `tests/test_costs.py`,
`tests/test_route_summary.py`

---

## Phase 3 — Trip ingestion, map matching, and time-of-day × day-of-week speed profiles

Turning a pile of historical trips into the thing that makes the baseline realistic:
observed speed per segment per time bucket, fed back into Phase 2's cost expression.

**Deliverables**
- NYC TLC loader: streaming Parquet/CSV reader for yellow/green/FHV schemas, column
  normalisation across the schema changes across years, unit fixes (miles → metres), and a
  quality filter (non-positive duration, implausible speed, zero distance, out-of-bbox
  coordinates, duration outside `[60s, 4h]`) with a rejection tally so data loss is visible.
- **HMM map matcher**: candidate segments per GPS point from a radius search, Gaussian
  emission probability on perpendicular distance, transition probability comparing route
  distance between candidates against great-circle distance, Viterbi decode over the
  sequence, then projection of matched points onto segment offsets to attribute a traversal
  time to each segment. Written as a pure algorithm over an injected `CandidateSource`, so
  the Viterbi and the probability model are tested against hand-built fixtures.
- Zone-centroid fallback: TLC records from later years carry `PULocationID`/`DOLocationID`
  instead of coordinates; those become centroid-to-centroid trips flagged
  `match_quality='zone'` and are excluded from segment-level mining while still training
  the residual model.
- **Speed-profile mining**: aggregate segment traversals into
  `(segment_id, dow_bucket, tod_bucket)` cells — 7 × 24 by default, with configurable
  coarser bucketings — computing harmonic-mean speed (the correct mean for speeds over a
  fixed distance), sample count, and p15/p50/p85 spread.
- **Hierarchical shrinkage**: a cell with 3 observations must not override the class prior.
  Each cell is shrunk toward its segment mean, then its (class × tod) mean, then the global
  class default, with weight `n / (n + k)` at each level. Pure function, unit-tested against
  the degenerate cases (zero samples → prior exactly; huge n → observation exactly).
- Materialised tables `segment_speed_profile` and `segment_speed_profile_smoothed`, a
  refresh routine, and the lookup function Phase 2's cost SQL calls.

**Files**
`eta_forge/data/__init__.py`, `eta_forge/data/tlc.py`, `eta_forge/data/quality.py`,
`eta_forge/data/synthetic.py`,
`eta_forge/mapmatch/__init__.py`, `eta_forge/mapmatch/candidates.py`,
`eta_forge/mapmatch/hmm.py`, `eta_forge/mapmatch/attribution.py`,
`eta_forge/profiles/__init__.py`, `eta_forge/profiles/buckets.py`,
`eta_forge/profiles/mining.py`, `eta_forge/profiles/shrinkage.py`,
`eta_forge/profiles/store.py`,
`sql/007_speed_profiles.sql`, `scripts/load_tlc.py`, `scripts/map_match.py`,
`scripts/build_profiles.py`,
`tests/test_tlc_loader.py`, `tests/test_quality.py`, `tests/test_hmm.py`,
`tests/test_buckets.py`, `tests/test_shrinkage.py`, `tests/test_mining.py`

---

## Phase 4 — Features, the quantile residual model, prep-time model, and the registry

Stage two. The learner never predicts duration directly — it predicts the *correction* to
the baseline, at two quantiles.

**Deliverables**
- Feature builder producing a named, ordered vector: cyclical hour (`sin`/`cos`), day of
  week, weekend and rush-hour flags, holiday flag, baseline duration and distance, mean
  baseline speed, class-mix fractions, turn counts by type, signal count, origin and
  destination H3 cells at r8 (as target-encoded historical residual means, with smoothing,
  fitted on train folds only) plus the raw cell for categorical splits, O-D bearing, and
  optional weather (temperature, precipitation rate, wind, a categorical condition).
  A `FeatureSpec` freezes name → index so train and serve can never drift; a version hash
  over the spec is stored with the model and checked at load.
- Weather join: hourly observations by station, nearest-station lookup, `as-of` join with a
  staleness bound, and complete graceful degradation to a `weather_missing` indicator.
- **Residual target**: `y = actual_duration − baseline_duration`, trained with LightGBM
  `objective='quantile'` at `alpha=0.5` and `alpha=0.9` (two models, one per quantile).
  Serving adds the prediction back onto the baseline and enforces monotonicity
  (`p90 >= p50`) after the fact, since independently-trained quantiles can cross.
- **Time-based splitting**: train/validation/test are split by trip timestamp, never
  randomly — a random split leaks the future through the target-encoded H3 cells and
  produces beautiful, meaningless offline numbers.
- **Prep-time model**: its own per-merchant predictor for `merchant_prep_seconds` —
  merchant historical p50/p90 prep with shrinkage toward the cuisine-category prior for
  cold-start merchants, item count, order value, hour-of-day, and a live
  `open_orders_at_merchant` load signal. Same quantile treatment.
- Handoff-dwell estimator: parking-difficulty proxy by destination H3 cell, dwell p50/p90.
- Model registry: content-addressed versions (`residual-p50/20260824T1200Z-a1b2c3d4`),
  a JSON manifest per version (feature-spec hash, training window, row counts, LightGBM
  params, offline metrics, git sha), atomic promotion of a `production` alias, listing,
  rollback, and load-by-version — all pure filesystem, no MLflow dependency.
- Training CLI: builds the dataset from the warehouse or the synthetic source, fits both
  quantiles plus the prep model, writes the manifest, and refuses to promote a version
  whose held-out p90 coverage falls outside the configured band.

**Files**
`eta_forge/features/__init__.py`, `eta_forge/features/spec.py`,
`eta_forge/features/builder.py`, `eta_forge/features/h3_cells.py`,
`eta_forge/features/target_encoding.py`, `eta_forge/features/calendar.py`,
`eta_forge/features/weather.py`,
`eta_forge/models/__init__.py`, `eta_forge/models/dataset.py`,
`eta_forge/models/splits.py`, `eta_forge/models/residual.py`,
`eta_forge/models/prep_time.py`, `eta_forge/models/dwell.py`,
`eta_forge/models/registry.py`, `eta_forge/models/quantile.py`,
`sql/008_weather.sql`, `scripts/train.py`,
`tests/test_feature_spec.py`, `tests/test_calendar.py`, `tests/test_target_encoding.py`,
`tests/test_splits.py`, `tests/test_registry.py`, `tests/test_quantile.py`,
`tests/test_prep_time.py`

---

## Phase 5 — Decomposed ETA service, FastAPI serving layer, and the evaluation harness

The two halves joined: a versioned service that answers in milliseconds, and an offline
harness that says whether it deserves to be believed.

**Deliverables**
- `EtaService`: route → features → residual p50/p90 → prep p50/p90 → dwell, composed as
  `ETA = prep + travel + handoff` with **quantile-aware composition** (adding p90s
  independently overstates the tail; the components are combined under a configurable
  correlation assumption, defaulting to a variance-based combination that reduces to the
  naive sum at ρ=1). Returns a full breakdown: baseline, each correction, each component's
  p50/p90, the model versions used, and the profile freshness.
- FastAPI app: `POST /v1/eta` (ride) and `POST /v1/eta/delivery` (prep + travel + handoff),
  `GET /v1/route` (baseline only, for debugging), `GET /v1/models` (loaded versions),
  `POST /v1/models/reload`, `GET /health`, `GET /ready`, `GET /metrics`. Model versioning is
  request-level: `model_version` in the body pins a specific version for shadow comparison.
- Serving safeguards: per-request timeout on the routing query with a haversine-plus-detour-
  factor fallback ETA, response-level `degraded` flags, and structured request logging with
  the feature vector hash for offline replay.
- **Evaluation harness**: MAE, MAPE (with the near-zero guard MAPE needs), RMSE, mean
  pinball loss at each quantile, **p90 coverage** (fraction of actuals ≤ predicted p90) and
  a calibration curve across nominal quantiles, sharpness (mean p90 − p50), plus
  **slice metrics** — by hour, day of week, distance bucket, borough/H3 cell, and merchant —
  because a service can look fine on average while being systematically 4 minutes late in
  Midtown at 18:00.
- Baseline comparison table: routing-only vs routing + residual p50, so the value the
  learned stage actually adds is a number and not a claim.
- Markdown + CSV report writer, and an end-to-end smoke path over the synthetic source.

**Files**
`eta_forge/service.py`, `eta_forge/decompose.py`, `eta_forge/server.py`,
`eta_forge/api/__init__.py`, `eta_forge/api/app.py`, `eta_forge/api/schemas.py`,
`eta_forge/api/deps.py`, `eta_forge/api/metrics.py`,
`eta_forge/eval/__init__.py`, `eta_forge/eval/metrics.py`, `eta_forge/eval/calibration.py`,
`eta_forge/eval/slices.py`, `eta_forge/eval/harness.py`, `eta_forge/eval/report.py`,
`scripts/evaluate.py`, `scripts/smoke_end_to_end.py`,
`tests/test_decompose.py`, `tests/test_eval_metrics.py`, `tests/test_calibration.py`,
`tests/test_slices.py`, `tests/test_api_schemas.py`, `tests/test_service.py`,
`tests/test_report.py`

---

## Verification strategy

No PostgreSQL, no OSM extract and no TLC data are available in the development environment,
so verification is:

- `python -m compileall eta_forge scripts` — every module parses.
- `python -m unittest discover -s tests -t .` — CPU-only, database-free unit tests over
  geometry and turn classification, cost-SQL composition, the Viterbi map matcher, bucketing
  and hierarchical shrinkage, the feature spec and target encoding, time-based splitting,
  quantile/pinball math and calibration, the decomposition arithmetic, the model registry,
  and the API schemas.
- An import-safety test asserting no module pulls in `psycopg`, `lightgbm`, `pandas`,
  `numpy`, `h3` or `fastapi` at import time.
- End-to-end smoke over `SyntheticSource` + `SyntheticGraph`: generate trips → mine
  profiles → build features → fit a trivially-small model → evaluate → render the report,
  with no database and no downloads.

Bringing up PostGIS, importing an OSM extract with `osm2pgrouting`, loading TLC Parquet and
running a real training job are documented in the README as operator steps and are
deliberately not executed here.
