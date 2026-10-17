"""Streaming loader for NYC TLC trip records.

The TLC has published the same underlying data under at least five schemas:

* **yellow** 2009-2010: ``Trip_Pickup_DateTime``, ``Start_Lon``/``Start_Lat``...
* **yellow** 2011-mid 2016: ``tpep_pickup_datetime`` with ``pickup_longitude`` and
  ``pickup_latitude`` — real coordinates, the only years usable for endpoint routing;
* **yellow** mid 2016 onwards: coordinates replaced by ``PULocationID`` /
  ``DOLocationID`` taxi-zone ids;
* **green**: the yellow layout with an ``lpep_`` prefix;
* **fhv** / **fhvhv**: ``pickup_datetime``, ``PUlocationID`` (note the case), and for
  high-volume FHV a ``trip_miles`` / ``trip_time`` pair instead of ``trip_distance``.

Everything is normalised onto one :class:`NormalizedTrip`, distances are converted
from miles to metres, timestamps are localised (TLC timestamps are naive *local* New
York time — treating them as UTC shifts every trip by four or five hours), and each
row passes the quality filter before it becomes a :class:`TripRecord`.

Parquet is read with ``pyarrow`` in record batches and CSV with the stdlib reader, so
a 3 GB month never needs to fit in memory. ``pyarrow`` is imported only when a Parquet
file is actually opened.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..config import DataConfig
from ..graph.geometry import haversine_m
from ..types import EtaForgeError, Location, MatchQuality, TripRecord, ValidationError
from .quality import (
    MALFORMED,
    MISSING_LOCATION,
    UNKNOWN_ZONE,
    QualityRules,
    RejectionTally,
    check_trip,
    is_null_island,
)

METRES_PER_MILE = 1_609.344
TLC_TIMEZONE = "America/New_York"

# TLC zone ids 264 and 265 are "Unknown" and "Outside of NYC". They have no centroid
# anyone should route to.
UNKNOWN_ZONE_IDS = frozenset({264, 265})

# Zone-only trips carry no distance in the FHV schemas; a crow-line estimate is
# inflated by the usual urban detour so the feature is not systematically short.
ZONE_DISTANCE_DETOUR = 1.32


class TlcFormatError(EtaForgeError):
    """A file's columns match none of the known TLC schemas."""


# canonical name -> candidate source column names, first match wins.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "pickup_at": (
        "tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime",
        "Trip_Pickup_DateTime", "pickup_datetime_local",
    ),
    "dropoff_at": (
        "tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime",
        "dropOff_datetime", "Trip_Dropoff_DateTime",
    ),
    "distance_miles": ("trip_distance", "trip_miles", "Trip_Distance"),
    "trip_time_s": ("trip_time",),
    "pickup_lat": ("pickup_latitude", "Start_Lat"),
    "pickup_lon": ("pickup_longitude", "Start_Lon"),
    "dropoff_lat": ("dropoff_latitude", "End_Lat"),
    "dropoff_lon": ("dropoff_longitude", "End_Lon"),
    "pu_location_id": ("PULocationID", "PUlocationID", "pulocationid"),
    "do_location_id": ("DOLocationID", "DOlocationID", "dolocationid"),
    "passenger_count": ("passenger_count", "Passenger_Count"),
}

DATASETS = ("yellow", "green", "fhv", "fhvhv")

VEHICLE_TYPE = {"yellow": "taxi", "green": "taxi", "fhv": "fhv", "fhvhv": "fhv"}


@dataclass(frozen=True)
class ColumnMap:
    """Resolved canonical -> source column names for one file."""

    columns: Mapping[str, str]

    def get(self, row: Mapping[str, Any], canonical: str) -> Any:
        source = self.columns.get(canonical)
        if source is None:
            return None
        return row.get(source)

    @property
    def has_coordinates(self) -> bool:
        return all(
            k in self.columns for k in ("pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon")
        )

    @property
    def has_zones(self) -> bool:
        return "pu_location_id" in self.columns and "do_location_id" in self.columns


def resolve_columns(available: Iterable[str]) -> ColumnMap:
    """Map the canonical names onto whatever this file calls them.

    Matching is exact first and case-insensitive second: the TLC has changed the case
    of the same column between years (``PUlocationID`` vs ``PULocationID``).
    """
    names = list(available)
    lowered = {n.lower(): n for n in names}
    resolved: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in names:
                resolved[canonical] = alias
                break
            if alias.lower() in lowered:
                resolved[canonical] = lowered[alias.lower()]
                break
    if "pickup_at" not in resolved or "dropoff_at" not in resolved:
        raise TlcFormatError(
            f"no pickup/dropoff timestamp columns among {sorted(names)[:12]}..."
        )
    cmap = ColumnMap(resolved)
    if not cmap.has_coordinates and not cmap.has_zones:
        raise TlcFormatError("file has neither coordinates nor taxi-zone ids")
    return cmap


def detect_dataset(path: str, columns: Optional[Iterable[str]] = None) -> str:
    """``yellow`` / ``green`` / ``fhv`` / ``fhvhv`` from the file name, else the columns."""
    base = os.path.basename(path).lower()
    for name in ("fhvhv", "fhv", "green", "yellow"):  # fhvhv before fhv
        if base.startswith(name) or f"_{name}_" in base or f"{name}_tripdata" in base:
            return name
    cols = {c.lower() for c in (columns or ())}
    if "lpep_pickup_datetime" in cols:
        return "green"
    if "trip_miles" in cols or "hvfhs_license_num" in cols:
        return "fhvhv"
    if "dispatching_base_num" in cols:
        return "fhv"
    return "yellow"


# --------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------


@dataclass
class NormalizedTrip:
    """One TLC row on the canonical schema, before domain validation."""

    dataset: str
    row_index: int
    pickup_at: Optional[datetime]
    dropoff_at: Optional[datetime]
    distance_m: Optional[float]
    origin: Optional[Location]
    destination: Optional[Location]
    pu_location_id: Optional[int] = None
    do_location_id: Optional[int] = None
    distance_estimated: bool = False
    match_quality: MatchQuality = MatchQuality.ENDPOINTS


def parse_tlc_timestamp(value: Any, tz: str = TLC_TIMEZONE) -> Optional[datetime]:
    """Parse a TLC timestamp and attach the local timezone.

    pyarrow hands back naive ``datetime`` objects and CSV gives strings; both are wall
    clock time in New York. An already-aware value (a re-export) is respected as is.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%m/%d/%Y %I:%M:%S %p")
            except ValueError:
                return None
    if parsed.tzinfo is not None:
        return parsed
    try:
        from zoneinfo import ZoneInfo

        return parsed.replace(tzinfo=ZoneInfo(tz))
    except Exception:  # pragma: no cover - missing tzdata on a slim image
        return parsed.replace(tzinfo=timezone.utc)


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _to_int(value: Any) -> Optional[int]:
    number = _to_float(value)
    return None if number is None else int(number)


def _location(lat: Optional[float], lon: Optional[float]) -> Optional[Location]:
    if is_null_island(lat, lon):
        return None
    try:
        return Location(float(lat), float(lon))  # type: ignore[arg-type]
    except (ValidationError, TypeError):
        return None


class ZoneCentroids:
    """TLC taxi zone id -> centroid. Loaded from the ``taxi_zone`` table or a CSV."""

    def __init__(self, centroids: Optional[Mapping[int, Location]] = None):
        self._centroids: Dict[int, Location] = dict(centroids or {})

    def __len__(self) -> int:
        return len(self._centroids)

    def __contains__(self, zone_id: object) -> bool:
        return zone_id in self._centroids

    def get(self, zone_id: Optional[int]) -> Optional[Location]:
        if zone_id is None or zone_id in UNKNOWN_ZONE_IDS:
            return None
        return self._centroids.get(int(zone_id))

    @classmethod
    def from_csv(cls, path: str) -> "ZoneCentroids":
        """CSV with ``location_id,lat,lon`` columns (header required)."""
        out: Dict[int, Location] = {}
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                zone = _to_int(row.get("location_id") or row.get("LocationID"))
                loc = _location(_to_float(row.get("lat")), _to_float(row.get("lon")))
                if zone is not None and loc is not None:
                    out[zone] = loc
        return cls(out)

    @classmethod
    def from_runner(cls, runner: Any) -> "ZoneCentroids":
        rows = runner.fetch_all(
            "SELECT location_id, ST_Y(centroid) AS lat, ST_X(centroid) AS lon "
            "FROM taxi_zone WHERE centroid IS NOT NULL",
            label="tlc.zone_centroids",
        )
        return cls(
            {
                int(r["location_id"]): Location(float(r["lat"]), float(r["lon"]))
                for r in rows
            }
        )


def normalise_row(
    row: Mapping[str, Any],
    cmap: ColumnMap,
    *,
    dataset: str,
    row_index: int,
    zones: Optional[ZoneCentroids] = None,
    tz: str = TLC_TIMEZONE,
) -> NormalizedTrip:
    """Map one raw row onto :class:`NormalizedTrip`.

    Coordinates win over zones when both are present and valid. A zone-only trip gets
    centroid endpoints and ``match_quality='zone'``: usable for the residual model,
    which only needs an origin cell and a destination cell, but excluded from segment
    mining because a centroid-to-centroid path says nothing about any one road.
    """
    pickup = parse_tlc_timestamp(cmap.get(row, "pickup_at"), tz)
    dropoff = parse_tlc_timestamp(cmap.get(row, "dropoff_at"), tz)

    miles = _to_float(cmap.get(row, "distance_miles"))
    distance_m = None if miles is None else miles * METRES_PER_MILE

    # fhvhv reports trip_time in seconds; prefer it when the timestamps are coarse.
    trip_time = _to_float(cmap.get(row, "trip_time_s"))
    if pickup is not None and trip_time is not None and trip_time > 0 and dropoff is None:
        from datetime import timedelta

        dropoff = pickup + timedelta(seconds=trip_time)

    origin = destination = None
    quality = MatchQuality.ENDPOINTS
    if cmap.has_coordinates:
        origin = _location(
            _to_float(cmap.get(row, "pickup_lat")), _to_float(cmap.get(row, "pickup_lon"))
        )
        destination = _location(
            _to_float(cmap.get(row, "dropoff_lat")), _to_float(cmap.get(row, "dropoff_lon"))
        )

    pu = _to_int(cmap.get(row, "pu_location_id"))
    do = _to_int(cmap.get(row, "do_location_id"))
    estimated = False
    if (origin is None or destination is None) and cmap.has_zones and zones is not None:
        origin = zones.get(pu)
        destination = zones.get(do)
        quality = MatchQuality.ZONE
        if distance_m is None and origin is not None and destination is not None:
            distance_m = haversine_m(origin, destination) * ZONE_DISTANCE_DETOUR
            estimated = True

    return NormalizedTrip(
        dataset=dataset,
        row_index=row_index,
        pickup_at=pickup,
        dropoff_at=dropoff,
        distance_m=distance_m,
        origin=origin,
        destination=destination,
        pu_location_id=pu,
        do_location_id=do,
        distance_estimated=estimated,
        match_quality=quality,
    )


def trip_id_for(source_file: str, dataset: str, row_index: int) -> str:
    """Deterministic trip id: re-loading the same file yields the same ids, so the
    loader's ``ON CONFLICT DO NOTHING`` makes a re-run idempotent."""
    digest = hashlib.sha1(
        f"{os.path.basename(source_file)}:{row_index}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{dataset}-{digest}"


# --------------------------------------------------------------------------------------
# file readers
# --------------------------------------------------------------------------------------


def iter_csv_rows(path: str) -> Iterator[Dict[str, Any]]:
    """Stream rows from a ``.csv`` or ``.csv.gz`` file."""
    if path.endswith(".gz"):
        raw = gzip.open(path, "rb")
        handle: Any = io.TextIOWrapper(raw, encoding="utf-8", newline="")
    else:
        handle = open(path, "r", encoding="utf-8", newline="")
    with handle:
        reader = csv.DictReader(handle)
        # Some 2010 files pad column names with spaces.
        if reader.fieldnames:
            reader.fieldnames = [f.strip() for f in reader.fieldnames]
        for row in reader:
            yield row


def iter_parquet_rows(path: str, *, batch_size: int = 65_536) -> Iterator[Dict[str, Any]]:
    """Stream rows from Parquet in record batches."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TlcFormatError("pyarrow is required to read Parquet; pip install pyarrow") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def iter_rows(path: str, *, batch_size: int = 65_536) -> Iterator[Dict[str, Any]]:
    lowered = path.lower()
    if lowered.endswith(".parquet"):
        return iter_parquet_rows(path, batch_size=batch_size)
    if lowered.endswith(".csv") or lowered.endswith(".csv.gz"):
        return iter_csv_rows(path)
    raise TlcFormatError(f"unsupported TLC file type: {path}")


def parquet_columns(path: str) -> List[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise TlcFormatError("pyarrow is required to read Parquet") from exc
    return list(pq.ParquetFile(path).schema_arrow.names)


# --------------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------------


@dataclass
class LoadResult:
    source_file: str
    dataset: str
    tally: RejectionTally = field(default_factory=RejectionTally)
    zone_trips: int = 0
    endpoint_trips: int = 0

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "source_file": self.source_file,
            "dataset": self.dataset,
            "zone_trips": self.zone_trips,
            "endpoint_trips": self.endpoint_trips,
        }
        out.update(self.tally.as_dict())
        return out


class TlcLoader:
    """Normalise, filter and convert TLC rows into :class:`TripRecord` objects."""

    def __init__(
        self,
        cfg: DataConfig,
        *,
        zones: Optional[ZoneCentroids] = None,
        tz: str = TLC_TIMEZONE,
    ):
        self._rules = QualityRules.from_config(cfg)
        self._zones = zones or ZoneCentroids()
        self._tz = tz

    @property
    def rules(self) -> QualityRules:
        return self._rules

    def convert(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_file: str,
        dataset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[Iterator[TripRecord], LoadResult]:
        """Lazily convert raw rows. The :class:`LoadResult` fills in as you iterate."""
        iterator = iter(rows)
        try:
            first = next(iterator)
        except StopIteration:
            result = LoadResult(source_file, dataset or detect_dataset(source_file))
            return iter(()), result

        cmap = resolve_columns(first.keys())
        name = dataset or detect_dataset(source_file, first.keys())
        result = LoadResult(source_file=source_file, dataset=name)

        def generate() -> Iterator[TripRecord]:
            def chain() -> Iterator[Mapping[str, Any]]:
                yield first
                yield from iterator

            for index, row in enumerate(chain()):
                if limit is not None and index >= limit:
                    return
                record = self._convert_one(row, cmap, name, index, source_file, result)
                if record is not None:
                    yield record

        return generate(), result

    def load_file(
        self, path: str, *, dataset: Optional[str] = None, limit: Optional[int] = None
    ) -> Tuple[Iterator[TripRecord], LoadResult]:
        return self.convert(iter_rows(path), source_file=path, dataset=dataset, limit=limit)

    def _convert_one(
        self,
        row: Mapping[str, Any],
        cmap: ColumnMap,
        dataset: str,
        index: int,
        source_file: str,
        result: LoadResult,
    ) -> Optional[TripRecord]:
        try:
            trip = normalise_row(
                row, cmap, dataset=dataset, row_index=index, zones=self._zones, tz=self._tz
            )
        except (TypeError, ValueError):
            result.tally.reject(MALFORMED)
            return None

        if trip.origin is None or trip.destination is None:
            zone_only = trip.pu_location_id is not None and not cmap.has_coordinates
            result.tally.reject(UNKNOWN_ZONE if zone_only else MISSING_LOCATION)
            return None

        reason = check_trip(
            self._rules,
            pickup_at=trip.pickup_at,
            dropoff_at=trip.dropoff_at,
            distance_m=trip.distance_m,
            origin=trip.origin,
            destination=trip.destination,
            # A centroid-to-centroid crow line is not the trip's real crow line.
            check_crow_line=trip.match_quality is not MatchQuality.ZONE,
        )
        if reason is not None:
            result.tally.reject(reason)
            return None

        assert trip.pickup_at is not None and trip.dropoff_at is not None
        try:
            record = TripRecord(
                trip_id=trip_id_for(source_file, dataset, index),
                pickup_at=trip.pickup_at,
                dropoff_at=trip.dropoff_at,
                origin=trip.origin,
                destination=trip.destination,
                distance_m=float(trip.distance_m or 0.0),
                duration_s=(trip.dropoff_at - trip.pickup_at).total_seconds(),
                match_quality=trip.match_quality,
                vehicle_type=VEHICLE_TYPE.get(dataset, "car"),
            )
        except ValidationError:
            result.tally.reject(MALFORMED)
            return None

        result.tally.keep()
        if trip.match_quality is MatchQuality.ZONE:
            result.zone_trips += 1
        else:
            result.endpoint_trips += 1
        return record


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------

TRIP_INSERT_SQL = """
INSERT INTO trip (
    trip_id, source_dataset, vehicle_type, pickup_at, dropoff_at,
    duration_s, distance_m, origin, destination,
    pu_location_id, do_location_id, match_quality
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s,
    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
    ST_SetSRID(ST_MakePoint(%s, %s), 4326),
    %s, %s, %s
)
ON CONFLICT (trip_id) DO NOTHING
"""

INGEST_BATCH_SQL = """
INSERT INTO ingest_batch (source_file, file_checksum, rows_read, rows_kept, rejections)
VALUES (%s, %s, %s, %s, %s::jsonb)
ON CONFLICT (source_file) DO UPDATE SET
    file_checksum = EXCLUDED.file_checksum,
    rows_read     = EXCLUDED.rows_read,
    rows_kept     = EXCLUDED.rows_kept,
    rejections    = EXCLUDED.rejections,
    loaded_at     = now()
"""


def trip_insert_row(record: TripRecord, dataset: str) -> Sequence[Any]:
    """Parameters for :data:`TRIP_INSERT_SQL`. ST_MakePoint takes (lon, lat)."""
    return (
        record.trip_id,
        dataset,
        record.vehicle_type,
        record.pickup_at,
        record.dropoff_at,
        record.duration_s,
        record.distance_m,
        record.origin.lon,
        record.origin.lat,
        record.destination.lon,
        record.destination.lat,
        None,
        None,
        record.match_quality.value,
    )


def file_checksum(path: str, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "TlcLoader",
    "LoadResult",
    "NormalizedTrip",
    "ColumnMap",
    "ZoneCentroids",
    "TlcFormatError",
    "resolve_columns",
    "detect_dataset",
    "normalise_row",
    "parse_tlc_timestamp",
    "trip_id_for",
    "iter_rows",
    "iter_csv_rows",
    "iter_parquet_rows",
    "parquet_columns",
    "trip_insert_row",
    "file_checksum",
    "TRIP_INSERT_SQL",
    "INGEST_BATCH_SQL",
    "METRES_PER_MILE",
    "UNKNOWN_ZONE_IDS",
]
