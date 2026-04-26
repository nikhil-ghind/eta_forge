"""Statement execution with a first-class dry-run mode.

Every SQL path in this project can be exercised without a database by handing the
caller a :class:`DryRunRecorder` instead of a pool: the exact statements and their
bound parameters are recorded and can be printed or asserted on. That is how the
routing cost SQL, the migration ordering and the profile refresh are tested here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..types import EtaForgeError


class SqlError(EtaForgeError):
    """A statement failed, or was rejected before being sent."""


# An identifier we are willing to interpolate into SQL (table and column names come
# from config, not from a request, but config is still not a trusted SQL fragment).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def safe_identifier(name: str) -> str:
    """Validate a table/column identifier before it is interpolated into SQL.

    Values are always bound as parameters; identifiers cannot be, so they are
    whitelisted by shape instead. Anything with a quote, semicolon, space or comment
    marker is refused outright.
    """
    if not _IDENT_RE.match(name or ""):
        raise SqlError(f"unsafe SQL identifier: {name!r}")
    return name


@dataclass
class RecordedStatement:
    sql: str
    params: Optional[Sequence[Any]] = None
    label: Optional[str] = None

    def rendered(self) -> str:
        """Human-readable form for ``--dry-run`` output. Never executed."""
        text = self.sql.strip()
        if self.params:
            pretty = ", ".join(repr(p) for p in self.params)
            return f"{text}\n-- params: [{pretty}]"
        return text


@dataclass
class DryRunRecorder:
    """Stands in for a pool. Records instead of executing; returns no rows."""

    statements: List[RecordedStatement] = field(default_factory=list)

    def record(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        label: Optional[str] = None,
    ) -> None:
        self.statements.append(RecordedStatement(sql, params, label))

    def script(self) -> str:
        return "\n\n".join(s.rendered() + ";" for s in self.statements)

    def clear(self) -> None:
        self.statements.clear()

    def __len__(self) -> int:
        return len(self.statements)


class SqlRunner:
    """Executes statements against a pool, or records them in dry-run mode."""

    def __init__(self, pool: Any = None, *, recorder: Optional[DryRunRecorder] = None):
        if pool is None and recorder is None:
            recorder = DryRunRecorder()
        self._pool = pool
        self._recorder = recorder

    @property
    def dry_run(self) -> bool:
        return self._pool is None

    @property
    def recorder(self) -> Optional[DryRunRecorder]:
        return self._recorder

    # -- reads -------------------------------------------------------------------

    def fetch_all(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
    ) -> List[Mapping[str, Any]]:
        if self.dry_run:
            self._record(sql, params, label)
            return []
        with self._pool.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def fetch_one(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
    ) -> Optional[Mapping[str, Any]]:
        rows = self.fetch_all(sql, params, label=label)
        return rows[0] if rows else None

    def fetch_scalar(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        *,
        default: Any = None,
    ) -> Any:
        row = self.fetch_one(sql, params)
        if not row:
            return default
        return next(iter(row.values()))

    # -- writes ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        *,
        label: Optional[str] = None,
    ) -> int:
        if self.dry_run:
            self._record(sql, params, label)
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount if cur.rowcount is not None else 0

    def execute_many(
        self,
        sql: str,
        rows: Iterable[Sequence[Any]],
        *,
        label: Optional[str] = None,
        batch_size: int = 5_000,
    ) -> int:
        """Batched insert path used by the TLC loader and profile writer."""
        batch: List[Sequence[Any]] = []
        total = 0
        if self.dry_run:
            for row in rows:
                batch.append(row)
                total += 1
                if len(batch) >= batch_size:
                    break
            self._record(sql, batch[0] if batch else None, label)
            return total
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    batch.append(row)
                    if len(batch) >= batch_size:
                        cur.executemany(sql, batch)
                        total += len(batch)
                        batch.clear()
                if batch:
                    cur.executemany(sql, batch)
                    total += len(batch)
        return total

    def execute_script(self, sql_text: str, *, label: Optional[str] = None) -> None:
        """Run a whole migration file as one statement block, inside a transaction."""
        if self.dry_run:
            self._record(sql_text, None, label)
            return
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_text)

    # -- internals ---------------------------------------------------------------

    def _record(
        self, sql: str, params: Optional[Sequence[Any]], label: Optional[str]
    ) -> None:
        if self._recorder is None:  # pragma: no cover - constructor guarantees one
            self._recorder = DryRunRecorder()
        self._recorder.record(sql, params, label)


__all__ = [
    "SqlRunner",
    "DryRunRecorder",
    "RecordedStatement",
    "SqlError",
    "safe_identifier",
]
