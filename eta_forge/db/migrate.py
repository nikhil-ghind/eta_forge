"""Ordered, idempotent-by-ledger SQL migrations.

Migrations are plain ``.sql`` files named ``NNN_description.sql`` under ``sql/``. They
are applied in numeric order and recorded in a ``schema_migration`` ledger with the
SHA-256 of the file, so an already-applied migration is skipped and an *edited*
already-applied migration is a loud error rather than a silent divergence between the
repository and the database.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .runner import SqlRunner, SqlError

MIGRATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sql"
)

_NAME_RE = re.compile(r"^(?P<version>\d{3})_(?P<slug>[a-z0-9_]+)\.sql$")

LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS schema_migration (
    version     integer      PRIMARY KEY,
    slug        text         NOT NULL,
    checksum    text         NOT NULL,
    applied_at  timestamptz  NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class Migration:
    version: int
    slug: str
    path: str

    @property
    def name(self) -> str:
        return f"{self.version:03d}_{self.slug}"

    def read(self) -> str:
        with open(self.path, "r", encoding="utf-8") as handle:
            return handle.read()

    def checksum(self) -> str:
        return hashlib.sha256(self.read().encode("utf-8")).hexdigest()


def discover_migrations(directory: Optional[str] = None) -> List[Migration]:
    """Return every migration in ``directory``, sorted by version.

    Duplicate version numbers are rejected: two files claiming ``004`` would apply in
    filesystem order, which is not reproducible across machines.
    """
    root = directory or MIGRATIONS_DIR
    if not os.path.isdir(root):
        raise SqlError(f"migrations directory not found: {root}")
    found: List[Migration] = []
    seen: dict = {}
    for entry in sorted(os.listdir(root)):
        match = _NAME_RE.match(entry)
        if not match:
            if entry.endswith(".sql"):
                raise SqlError(
                    f"migration {entry!r} does not match NNN_lower_snake.sql"
                )
            continue
        version = int(match.group("version"))
        if version in seen:
            raise SqlError(
                f"duplicate migration version {version:03d}: {seen[version]} and {entry}"
            )
        seen[version] = entry
        found.append(
            Migration(version=version, slug=match.group("slug"), path=os.path.join(root, entry))
        )
    return sorted(found, key=lambda m: m.version)


def applied_versions(runner: SqlRunner) -> dict:
    """Version -> checksum for everything already recorded in the ledger."""
    rows = runner.fetch_all(
        "SELECT version, checksum FROM schema_migration ORDER BY version",
        label="migrations.applied",
    )
    return {int(row["version"]): row["checksum"] for row in rows}


def run_migrations(
    runner: SqlRunner,
    *,
    directory: Optional[str] = None,
    target: Optional[int] = None,
    verify_checksums: bool = True,
) -> List[Migration]:
    """Apply every pending migration up to ``target``; return what was applied."""
    migrations = discover_migrations(directory)
    if target is not None:
        migrations = [m for m in migrations if m.version <= target]

    runner.execute_script(LEDGER_DDL, label="migrations.ledger")
    already = {} if runner.dry_run else applied_versions(runner)

    applied: List[Migration] = []
    for migration in migrations:
        checksum = migration.checksum()
        if migration.version in already:
            if verify_checksums and already[migration.version] != checksum:
                raise SqlError(
                    f"migration {migration.name} was modified after being applied "
                    f"(ledger {already[migration.version][:12]}, "
                    f"file {checksum[:12]}); write a new migration instead"
                )
            continue
        runner.execute_script(migration.read(), label=f"migration:{migration.name}")
        runner.execute(
            "INSERT INTO schema_migration (version, slug, checksum) VALUES (%s, %s, %s)",
            (migration.version, migration.slug, checksum),
            label=f"migration:{migration.name}:record",
        )
        applied.append(migration)
    return applied


def pending(runner: SqlRunner, *, directory: Optional[str] = None) -> List[Migration]:
    already = set(applied_versions(runner))
    return [m for m in discover_migrations(directory) if m.version not in already]


__all__ = [
    "MIGRATIONS_DIR",
    "Migration",
    "discover_migrations",
    "run_migrations",
    "applied_versions",
    "pending",
    "LEDGER_DDL",
]
