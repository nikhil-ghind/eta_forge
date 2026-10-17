"""Reading and writing the profile tables.

The write path is a staging-table swap rather than a truncate-and-insert. Truncating
the live table takes an ACCESS EXCLUSIVE lock for the whole rebuild, during which every
routing query blocks; and if the rebuild fails halfway, the service is left routing on
a partially-populated table. Writing to staging and swapping in one transaction means
readers see the old table right up until they see the complete new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..config import ProfilesConfig
from ..db.runner import SqlRunner, safe_identifier
from ..types import HighwayClass, SegmentObservation, ValidationError
from .buckets import BucketSpec
from .mining import CellStats, MiningReport, SegmentProfile, mine_profiles
from .shrinkage import SmoothedCell, shrink_profiles

RAW_INSERT_SQL = """
INSERT INTO {table} (
    segment_id, dow_bucket, tod_bucket, speed_kph, sample_count,
    p15_kph, p50_kph, p85_kph, total_length_m, total_traversal_s
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

SMOOTHED_INSERT_SQL = """
INSERT INTO {table} (
    segment_id, dow_bucket, tod_bucket, speed_kph, sample_count,
    effective_sample_count, prior_kph, shrink_weight, spread_kph, source
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# One transaction, three renames per table. The old table is kept as `_previous` until
# the next refresh, which makes a rollback a rename rather than a re-mine.
SWAP_SQL = """
BEGIN;
DROP TABLE IF EXISTS {live}_previous;
ALTER TABLE IF EXISTS {live}     RENAME TO {live}_previous;
ALTER TABLE {staging}            RENAME TO {live};
CREATE TABLE {staging} (LIKE {live} INCLUDING ALL);
{rebind}
COMMIT;
"""

# PostgreSQL views bind to a table's OID, not its name. After the rename above, the
# monitoring views would quietly keep reading `_previous` — and the *next* refresh's
# DROP TABLE would then fail on the dependency. Re-issuing the definitions inside the
# swap transaction rebinds them to the freshly swapped-in table.
REBIND_SMOOTHED_VIEWS_SQL = """
CREATE OR REPLACE VIEW segment_profile_freshness AS
SELECT
    min(refreshed_at)                                  AS oldest_refresh,
    max(refreshed_at)                                  AS newest_refresh,
    count(*)                                           AS cells,
    count(*) FILTER (WHERE source = 'observed')        AS observed_cells,
    round(100.0 * count(*) FILTER (WHERE source = 'observed')
          / NULLIF(count(*), 0), 1)                    AS observed_pct
FROM {live};

CREATE OR REPLACE VIEW segment_profile_coverage_by_class AS
SELECT
    a.highway_class,
    count(*)                                           AS cells,
    count(*) FILTER (WHERE s.source = 'observed')      AS observed_cells,
    round(avg(s.sample_count)::numeric, 2)             AS mean_samples,
    round(avg(s.speed_kph)::numeric, 1)                AS mean_speed_kph
FROM {live} s
JOIN segment_attr a USING (segment_id)
GROUP BY a.highway_class
ORDER BY observed_cells DESC;
"""

TRUNCATE_STAGING_SQL = "TRUNCATE TABLE {staging}"

OBSERVATIONS_SQL = """
SELECT
    ts.segment_id,
    ts.entered_at,
    ts.traversal_s,
    ts.length_m,
    a.highway_class,
    ts.trip_id
FROM trip_segment ts
JOIN trip t          ON t.trip_id = ts.trip_id
LEFT JOIN segment_attr a ON a.segment_id = ts.segment_id
WHERE t.match_quality = 'matched'
  AND ts.match_confidence >= %s
  AND t.pickup_at >= %s
  AND t.pickup_at <  %s
ORDER BY ts.segment_id, ts.entered_at
"""

SEGMENT_CLASS_SQL = "SELECT segment_id, highway_class FROM segment_attr"


@dataclass
class RefreshResult:
    raw_rows: int
    smoothed_rows: int
    swapped: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "raw_rows": self.raw_rows,
            "smoothed_rows": self.smoothed_rows,
            "swapped": self.swapped,
        }


class ProfileStore:
    """Persistence for the mined and smoothed profile grids."""

    def __init__(self, runner: SqlRunner, cfg: ProfilesConfig):
        self._runner = runner
        self._cfg = cfg
        self._raw = safe_identifier(cfg.profile_table)
        self._smoothed = safe_identifier(cfg.smoothed_table)

    # -- read --------------------------------------------------------------------

    def iter_observations(
        self,
        *,
        since: datetime,
        until: Optional[datetime] = None,
        min_confidence: float = 0.5,
    ) -> Iterator[Mapping[str, object]]:
        """Map-matched traversals in a time window, oldest first."""
        until = until or datetime.now(timezone.utc)
        rows = self._runner.fetch_all(
            OBSERVATIONS_SQL,
            (min_confidence, since, until),
            label="profiles.observations",
        )
        for row in rows:
            yield row

    def observations(
        self,
        *,
        since: datetime,
        until: Optional[datetime] = None,
        min_confidence: float = 0.5,
    ) -> Iterator[SegmentObservation]:
        """:meth:`iter_observations`, converted to domain objects.

        A row that fails validation (zero traversal time from a degenerate match) is
        skipped here rather than aborting a multi-hour mining run.
        """
        for row in self.iter_observations(
            since=since, until=until, min_confidence=min_confidence
        ):
            obs = observation_from_row(row)
            if obs is not None:
                yield obs

    def segment_classes(self) -> Dict[int, HighwayClass]:
        rows = self._runner.fetch_all(SEGMENT_CLASS_SQL, label="profiles.classes")
        return {
            int(r["segment_id"]): HighwayClass.coerce(r["highway_class"]) for r in rows
        }

    def lookup(self, segment_id: int, dow: int, tod: int) -> Optional[float]:
        """Smoothed speed for one cell. Used by tooling, not by the routing hot path."""
        return self._runner.fetch_scalar(
            f"SELECT speed_kph FROM {self._smoothed} "
            "WHERE segment_id = %s AND dow_bucket = %s AND tod_bucket = %s",
            (segment_id, dow, tod),
        )

    def freshness(self) -> Mapping[str, object]:
        row = self._runner.fetch_one(
            "SELECT * FROM segment_profile_freshness", label="profiles.freshness"
        )
        return row or {}

    # -- write -------------------------------------------------------------------

    def write(
        self,
        mined: Mapping[int, SegmentProfile],
        smoothed: Mapping[int, Mapping[Tuple[int, int], SmoothedCell]],
        *,
        swap: bool = True,
    ) -> RefreshResult:
        """Write both grids to staging and (optionally) swap them into place."""
        raw_staging = f"{self._raw}_staging"
        smoothed_staging = f"{self._smoothed}_staging"

        self._runner.execute(
            TRUNCATE_STAGING_SQL.format(staging=raw_staging),
            label="profiles.truncate_raw_staging",
        )
        self._runner.execute(
            TRUNCATE_STAGING_SQL.format(staging=smoothed_staging),
            label="profiles.truncate_smoothed_staging",
        )

        raw_rows = self._runner.execute_many(
            RAW_INSERT_SQL.format(table=raw_staging),
            _raw_rows(mined),
            label="profiles.insert_raw",
        )
        smoothed_rows = self._runner.execute_many(
            SMOOTHED_INSERT_SQL.format(table=smoothed_staging),
            _smoothed_rows(smoothed),
            label="profiles.insert_smoothed",
        )

        if swap:
            self._runner.execute_script(
                SWAP_SQL.format(live=self._raw, staging=raw_staging, rebind=""),
                label="profiles.swap_raw",
            )
            self._runner.execute_script(
                SWAP_SQL.format(
                    live=self._smoothed,
                    staging=smoothed_staging,
                    rebind=REBIND_SMOOTHED_VIEWS_SQL.format(live=self._smoothed),
                ),
                label="profiles.swap_smoothed",
            )
            # The planner's row estimates for the new table are whatever the old one
            # had; without an ANALYZE the first routing queries can pick a sequential
            # scan over a few million rows.
            self._runner.execute(
                f"ANALYZE {self._smoothed}", label="profiles.analyze"
            )

        return RefreshResult(
            raw_rows=raw_rows, smoothed_rows=smoothed_rows, swapped=swap
        )


def observation_from_row(row: Mapping[str, object]) -> Optional[SegmentObservation]:
    """A ``trip_segment`` row -> :class:`SegmentObservation`, or ``None`` if invalid."""
    try:
        return SegmentObservation(
            segment_id=int(row["segment_id"]),  # type: ignore[arg-type]
            entered_at=row["entered_at"],  # type: ignore[arg-type]
            traversal_s=float(row["traversal_s"]),  # type: ignore[arg-type]
            length_m=float(row["length_m"]),  # type: ignore[arg-type]
            highway_class=HighwayClass.coerce(row.get("highway_class")),  # type: ignore[arg-type]
            trip_id=(None if row.get("trip_id") is None else str(row["trip_id"])),
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


@dataclass
class ProfileBuild:
    """Everything one refresh produced, kept together for the CLI's report."""

    mined: Dict[int, SegmentProfile]
    smoothed: Dict[int, Dict[Tuple[int, int], SmoothedCell]]
    report: MiningReport
    result: Optional[RefreshResult] = None

    def source_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cells in self.smoothed.values():
            for cell in cells.values():
                counts[cell.source] = counts.get(cell.source, 0) + 1
        return counts


def build_profiles(
    observations: Iterable[SegmentObservation],
    cfg: ProfilesConfig,
    *,
    class_default: Mapping[str, float],
    all_segments: Optional[Mapping[int, HighwayClass]] = None,
    store: Optional[ProfileStore] = None,
    swap: bool = True,
) -> ProfileBuild:
    """The refresh routine: mine -> shrink -> (optionally) write and swap.

    Pure up to the write: with ``store=None`` this runs entirely in memory, which is how
    the synthetic pipeline and the tests drive it.
    """
    spec = BucketSpec.from_config(cfg)
    mined, report = mine_profiles(
        observations, cfg, spec=spec, segment_classes=all_segments
    )
    smoothed = shrink_profiles(
        mined,
        cfg,
        spec=spec,
        class_default=class_default,
        all_segments=all_segments,
    )
    build = ProfileBuild(mined=mined, smoothed=smoothed, report=report)
    if store is not None:
        build.result = store.write(mined, smoothed, swap=swap)
    return build


def _raw_rows(mined: Mapping[int, SegmentProfile]) -> Iterator[Sequence]:
    for profile in mined.values():
        for cell in profile.cells.values():
            yield cell.to_row()


def _smoothed_rows(
    smoothed: Mapping[int, Mapping[Tuple[int, int], SmoothedCell]]
) -> Iterator[Sequence]:
    for cells in smoothed.values():
        for cell in cells.values():
            yield cell.to_row()


__all__ = [
    "ProfileStore",
    "RefreshResult",
    "ProfileBuild",
    "build_profiles",
    "observation_from_row",
]
