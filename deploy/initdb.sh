#!/bin/bash
# Applied once, by the postgres entrypoint, on an empty data directory.
#
# It runs the same ordered migrations that scripts/migrate.py runs, so a container
# brought up from scratch and a database migrated in place converge on one schema.
# The ledger insert mirrors what scripts/migrate.py records, so a later
# `scripts/migrate.py` run sees these as applied and does not re-run them.

set -euo pipefail

MIGRATIONS_DIR=${MIGRATIONS_DIR:-/migrations}
PSQL=(psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB")

echo "eta-forge: applying migrations from ${MIGRATIONS_DIR}"

"${PSQL[@]}" <<'SQL'
CREATE TABLE IF NOT EXISTS schema_migration (
    version     integer      PRIMARY KEY,
    slug        text         NOT NULL,
    checksum    text         NOT NULL,
    applied_at  timestamptz  NOT NULL DEFAULT now()
);
SQL

shopt -s nullglob
for file in "${MIGRATIONS_DIR}"/[0-9][0-9][0-9]_*.sql; do
    base="$(basename "$file" .sql)"
    version=$((10#${base%%_*}))   # 001 -> 1, without octal interpretation
    slug="${base#*_}"
    checksum="$(sha256sum "$file" | cut -d' ' -f1)"

    already="$("${PSQL[@]}" -tAc \
        "SELECT 1 FROM schema_migration WHERE version = ${version}" || true)"
    if [ "$already" = "1" ]; then
        echo "eta-forge:   skip ${base} (already applied)"
        continue
    fi

    echo "eta-forge:  apply ${base}"
    "${PSQL[@]}" -f "$file"
    "${PSQL[@]}" -c "INSERT INTO schema_migration (version, slug, checksum)
                     VALUES (${version}, '${slug}', '${checksum}')
                     ON CONFLICT (version) DO NOTHING"
done

echo "eta-forge: schema ready"
