"""Lazily-imported psycopg connection pool.

``psycopg`` is imported inside :meth:`ConnectionPool.open`, never at module import
time, so importing this module — and therefore the whole package — succeeds on a
machine with no PostgreSQL client library.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from ..config import DatabaseConfig
from ..types import EtaForgeError
from .dsn import build_dsn, redacted_dsn


class PoolUnavailable(EtaForgeError):
    """psycopg is missing, or the server could not be reached."""


class ConnectionPool:
    """A minimal wrapper around ``psycopg_pool.ConnectionPool``.

    The wrapper exists so the rest of the codebase depends on one small interface
    (``connection()``) that a fake can implement in a test, rather than on psycopg.
    """

    def __init__(self, cfg: DatabaseConfig, *, password_override: Optional[str] = None):
        self._cfg = cfg
        self._password_override = password_override
        self._pool: Any = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._pool is not None

    def open(self) -> "ConnectionPool":
        with self._lock:
            if self._pool is not None:
                return self
            try:
                from psycopg_pool import ConnectionPool as PsycopgPool
            except ImportError as exc:
                raise PoolUnavailable(
                    "psycopg[pool] is not installed; "
                    "pip install 'psycopg[binary,pool]'"
                ) from exc
            dsn = build_dsn(self._cfg, password_override=self._password_override)
            try:
                self._pool = PsycopgPool(
                    conninfo=dsn,
                    min_size=self._cfg.min_pool_size,
                    max_size=self._cfg.max_pool_size,
                    timeout=self._cfg.connect_timeout_s,
                    name=self._cfg.application_name,
                    open=True,
                )
            except Exception as exc:  # pragma: no cover - needs a live server
                raise PoolUnavailable(
                    f"could not open pool to {redacted_dsn(self._cfg)}: {exc}"
                ) from exc
        return self

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def __enter__(self) -> "ConnectionPool":
        return self.open()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- use ---------------------------------------------------------------------

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """Yield a pooled connection, opening the pool on first use."""
        if self._pool is None:
            self.open()
        assert self._pool is not None
        with self._pool.connection() as conn:
            yield conn

    @contextmanager
    def cursor(self, *, row_factory: Optional[str] = "dict") -> Iterator[Any]:
        """Yield a cursor; ``row_factory='dict'`` gives dict rows.

        Dict rows are worth the small overhead here: routing and profile queries
        return a dozen columns and positional access to them is how column-order
        bugs get written.
        """
        factory = None
        if row_factory == "dict":
            try:
                from psycopg.rows import dict_row

                factory = dict_row
            except ImportError as exc:  # pragma: no cover
                raise PoolUnavailable("psycopg is not installed") from exc
        with self.connection() as conn:
            with conn.cursor(row_factory=factory) if factory else conn.cursor() as cur:
                yield cur

    def describe(self) -> str:
        return redacted_dsn(self._cfg)


__all__ = ["ConnectionPool", "PoolUnavailable"]
