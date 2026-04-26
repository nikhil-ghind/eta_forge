import os
import tempfile
import unittest

from eta_forge.config import DatabaseConfig
from eta_forge.db.dsn import build_dsn, build_uri, redacted_dsn
from eta_forge.db.migrate import discover_migrations, run_migrations
from eta_forge.db.runner import DryRunRecorder, SqlError, SqlRunner, safe_identifier


class TestDsn(unittest.TestCase):
    def setUp(self):
        self.cfg = DatabaseConfig(
            host="db.internal",
            port=5433,
            database="eta_forge",
            user="eta",
            password="s3cr3t",
        )

    def test_contains_expected_params(self):
        dsn = build_dsn(self.cfg)
        self.assertIn("host=db.internal", dsn)
        self.assertIn("port=5433", dsn)
        self.assertIn("dbname=eta_forge", dsn)
        self.assertIn("password=s3cr3t", dsn)

    def test_statement_timeout_is_pinned_on_the_connection(self):
        self.assertIn("statement_timeout=10000", build_dsn(self.cfg))

    def test_redacted_never_leaks_password(self):
        red = redacted_dsn(self.cfg)
        self.assertNotIn("s3cr3t", red)
        self.assertIn("***", red)

    def test_password_override(self):
        self.assertIn("password=other", build_dsn(self.cfg, password_override="other"))
        self.assertNotIn("password=", build_dsn(self.cfg, password_override=""))

    def test_values_needing_quotes_are_quoted(self):
        cfg = DatabaseConfig(password="a b'c", application_name="eta forge")
        dsn = build_dsn(cfg)
        self.assertIn("application_name='eta forge'", dsn)
        self.assertIn("\\'", dsn)

    def test_uri_percent_encodes(self):
        cfg = DatabaseConfig(user="e ta", password="p@ss", database="d b")
        uri = build_uri(cfg)
        self.assertIn("e%20ta", uri)
        self.assertIn("p%40ss", uri)


class TestSafeIdentifier(unittest.TestCase):
    def test_accepts_plain_and_qualified(self):
        self.assertEqual(safe_identifier("road_segment"), "road_segment")
        self.assertEqual(safe_identifier("public.trip"), "public.trip")

    def test_rejects_injection(self):
        for bad in ("road; DROP TABLE trip", "a'b", "a b", "", "1abc", "a--b"):
            with self.assertRaises(SqlError):
                safe_identifier(bad)


class TestSqlRunnerDryRun(unittest.TestCase):
    def test_records_instead_of_executing(self):
        runner = SqlRunner()
        self.assertTrue(runner.dry_run)
        runner.execute("UPDATE trip SET distance_m = %s WHERE trip_id = %s", (10.0, "t"))
        self.assertEqual(len(runner.recorder), 1)
        self.assertIn("UPDATE trip", runner.recorder.script())
        self.assertIn("params", runner.recorder.script())

    def test_reads_return_empty(self):
        runner = SqlRunner()
        self.assertEqual(runner.fetch_all("SELECT 1"), [])
        self.assertIsNone(runner.fetch_one("SELECT 1"))
        self.assertEqual(runner.fetch_scalar("SELECT 1", default=42), 42)

    def test_recorder_clear(self):
        rec = DryRunRecorder()
        runner = SqlRunner(recorder=rec)
        runner.execute("SELECT 1")
        rec.clear()
        self.assertEqual(len(rec), 0)


class TestMigrations(unittest.TestCase):
    def test_ships_migrations_in_order(self):
        found = discover_migrations()
        versions = [m.version for m in found]
        self.assertEqual(versions, sorted(versions))
        self.assertGreaterEqual(len(found), 5)
        self.assertEqual(found[0].slug, "extensions")

    def test_checksums_are_stable(self):
        first = discover_migrations()[0]
        self.assertEqual(first.checksum(), first.checksum())
        self.assertEqual(len(first.checksum()), 64)

    def test_dry_run_applies_everything_in_order(self):
        runner = SqlRunner()
        applied = run_migrations(runner)
        self.assertEqual([m.version for m in applied], sorted(m.version for m in applied))
        labels = [s.label for s in runner.recorder.statements]
        self.assertEqual(labels[0], "migrations.ledger")
        self.assertIn("migration:001_extensions", labels)

    def test_target_stops_early(self):
        runner = SqlRunner()
        applied = run_migrations(runner, target=2)
        self.assertEqual([m.version for m in applied], [1, 2])

    def test_rejects_badly_named_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "not-a-migration.sql"), "w").close()
            with self.assertRaises(SqlError):
                discover_migrations(tmp)

    def test_rejects_duplicate_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "001_alpha.sql"), "w").close()
            open(os.path.join(tmp, "001_beta.sql"), "w").close()
            with self.assertRaises(SqlError) as ctx:
                discover_migrations(tmp)
            self.assertIn("duplicate", str(ctx.exception))

    def test_missing_directory(self):
        with self.assertRaises(SqlError):
            discover_migrations("/nonexistent/sql")


if __name__ == "__main__":
    unittest.main()
