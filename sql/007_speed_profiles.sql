-- 007: mined speed profiles.
--
-- Two tables, not one, and both are real tables rather than materialized views.
--
-- `segment_speed_profile` is the raw mined grid: only cells that were actually
-- observed, with their sample counts and spread. It is the audit trail — when a route
-- looks wrong, this is what says whether the profile behind it had 3 samples or 3000.
--
-- `segment_speed_profile_smoothed` is dense: every segment x every time cell, filled
-- through the shrinkage hierarchy. The routing cost expression reads only this one, so
-- the hot path is a single index lookup with no fallback branching.
--
-- A materialized view would be tempting, but the smoothing is not expressible as one
-- SQL statement (it walks a prior hierarchy per cell), and refreshing a matview locks
-- it against readers unless CONCURRENTLY is used, which requires a unique index and
-- doubles the storage during refresh. A plain table swapped in atomically is simpler
-- and gives readers a consistent view for free.

CREATE TABLE IF NOT EXISTS segment_speed_profile (
    segment_id        bigint  NOT NULL,
    dow_bucket        smallint NOT NULL,
    tod_bucket        smallint NOT NULL,
    speed_kph         double precision NOT NULL,
    sample_count      integer NOT NULL,
    p15_kph           double precision,
    p50_kph           double precision,
    p85_kph           double precision,
    total_length_m    double precision NOT NULL DEFAULT 0,
    total_traversal_s double precision NOT NULL DEFAULT 0,
    refreshed_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (segment_id, dow_bucket, tod_bucket),
    CONSTRAINT ssp_speed_ck   CHECK (speed_kph > 0),
    CONSTRAINT ssp_samples_ck CHECK (sample_count >= 0),
    CONSTRAINT ssp_dow_ck     CHECK (dow_bucket BETWEEN 0 AND 6),
    CONSTRAINT ssp_tod_ck     CHECK (tod_bucket BETWEEN 0 AND 23)
);

CREATE TABLE IF NOT EXISTS segment_speed_profile_smoothed (
    segment_id        bigint  NOT NULL,
    dow_bucket        smallint NOT NULL,
    tod_bucket        smallint NOT NULL,
    speed_kph         double precision NOT NULL,
    sample_count      integer NOT NULL DEFAULT 0,
    effective_sample_count double precision NOT NULL DEFAULT 0,
    prior_kph         double precision,
    shrink_weight     double precision NOT NULL DEFAULT 0,
    spread_kph        double precision NOT NULL DEFAULT 0,
    source            text NOT NULL DEFAULT 'default',
    refreshed_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (segment_id, dow_bucket, tod_bucket),
    CONSTRAINT sspm_speed_ck  CHECK (speed_kph > 0),
    CONSTRAINT sspm_source_ck CHECK (source IN
        ('observed','neighbour','segment','class','default'))
);

-- The routing lookup is (segment_id, dow, tod) -> speed, trusted when the cell's
-- effective sample count (observed + borrowed from adjacent hours) clears the floor. The primary key already
-- covers it; INCLUDE-ing the speed makes it index-only, which matters when a single
-- route probes a few hundred segments.
CREATE INDEX IF NOT EXISTS sspm_lookup_ix
    ON segment_speed_profile_smoothed (segment_id, dow_bucket, tod_bucket)
    INCLUDE (speed_kph, effective_sample_count, sample_count);

-- Freshness monitoring: a profile table nobody refreshed for three months is quietly
-- wrong, and the service reports profile staleness on every response.
CREATE OR REPLACE VIEW segment_profile_freshness AS
SELECT
    min(refreshed_at)                                  AS oldest_refresh,
    max(refreshed_at)                                  AS newest_refresh,
    count(*)                                           AS cells,
    count(*) FILTER (WHERE source = 'observed')        AS observed_cells,
    round(100.0 * count(*) FILTER (WHERE source = 'observed')
          / NULLIF(count(*), 0), 1)                    AS observed_pct
FROM segment_speed_profile_smoothed;

-- Where mining is thin. Read before trusting a slice of the offline evaluation: a
-- borough with 4% observed coverage is being routed on class defaults.
CREATE OR REPLACE VIEW segment_profile_coverage_by_class AS
SELECT
    a.highway_class,
    count(*)                                           AS cells,
    count(*) FILTER (WHERE s.source = 'observed')      AS observed_cells,
    round(avg(s.sample_count)::numeric, 2)             AS mean_samples,
    round(avg(s.speed_kph)::numeric, 1)                AS mean_speed_kph
FROM segment_speed_profile_smoothed s
JOIN segment_attr a USING (segment_id)
GROUP BY a.highway_class
ORDER BY observed_cells DESC;

-- Staging tables for an atomic swap. The refresh writes here, then one transaction
-- renames staging over live, so a reader never sees a half-rebuilt profile table.
CREATE TABLE IF NOT EXISTS segment_speed_profile_staging
    (LIKE segment_speed_profile INCLUDING ALL);
CREATE TABLE IF NOT EXISTS segment_speed_profile_smoothed_staging
    (LIKE segment_speed_profile_smoothed INCLUDING ALL);

-- Replace 006's lookup now that the smoothed table exists with its final shape: a
-- smoothed cell is trusted on its *effective* sample count, so a cell filled from a
-- well-observed adjacent hour is used, while one filled from a class average defers to
-- the speed-limit and class-default fallbacks exactly as the Python cost expression
-- (eta_forge/graph/costs.py) does.
CREATE OR REPLACE FUNCTION eta_segment_speed_kph(
    p_segment_id  bigint,
    p_dow_bucket  integer,
    p_tod_bucket  integer,
    p_min_samples integer DEFAULT 5
) RETURNS double precision
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT COALESCE(
        (SELECT s.speed_kph
           FROM segment_speed_profile_smoothed s
          WHERE s.segment_id = p_segment_id
            AND s.dow_bucket = p_dow_bucket
            AND s.tod_bucket = p_tod_bucket
            AND s.effective_sample_count >= p_min_samples),
        (SELECT a.speed_limit_kph * 0.75
           FROM segment_attr a
          WHERE a.segment_id = p_segment_id
            AND a.speed_limit_kph IS NOT NULL),
        (SELECT d.speed_kph
           FROM segment_attr a
           JOIN highway_class_default d USING (highway_class)
          WHERE a.segment_id = p_segment_id),
        30.0
    );
$$;
