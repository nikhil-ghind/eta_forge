"""Thin database layer over PostgreSQL + PostGIS + pgRouting.

Deliberately not an ORM. The interesting work in this project is SQL — KNN snapping
against a GiST index, ``pgr_dijkstra`` over a composed cost expression, materialised
speed-profile aggregates — and an ORM would only obscure it.

What this package provides is: DSN assembly, a lazily-imported connection pool, a
runner that executes parameterised statements (and can print them instead of running
them), and an ordered file-based migration mechanism.
"""

from .dsn import build_dsn, redacted_dsn
from .pool import ConnectionPool, PoolUnavailable
from .runner import SqlRunner, DryRunRecorder
from .migrate import MIGRATIONS_DIR, discover_migrations, run_migrations

__all__ = [
    "build_dsn",
    "redacted_dsn",
    "ConnectionPool",
    "PoolUnavailable",
    "SqlRunner",
    "DryRunRecorder",
    "MIGRATIONS_DIR",
    "discover_migrations",
    "run_migrations",
]
