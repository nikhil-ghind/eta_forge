import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from eta_forge.models.registry import ModelRegistry, RegistryError, parse_version
from eta_forge.types import ModelNotFound

UTC = timezone.utc
WHEN = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def writer(payload):
    def write(directory):
        with open(os.path.join(directory, "model.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return {"kind": "test"}
    return write


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_version_id_is_timestamp_plus_content_hash(self):
        version = self.registry.register("residual", writer({"a": 1}), {"rows": 10}, created_at=WHEN)
        parts = parse_version(version.version)
        self.assertEqual(parts["name"], "residual")
        self.assertEqual(parts["stamp"], "20260824T1200Z")
        self.assertEqual(len(parts["digest"]), 8)
        self.assertTrue(version.content_hash.startswith(parts["digest"]))

    def test_manifest_records_provenance(self):
        version = self.registry.register("residual", writer({"a": 1}), {"rows": 10}, created_at=WHEN)
        manifest = self.registry.get(version.version).manifest
        self.assertEqual(manifest["rows"], 10)
        self.assertEqual(manifest["version"], version.version)
        self.assertEqual(manifest["files"], ["model.json"])
        self.assertEqual(manifest["artifact"], {"kind": "test"})

    def test_identical_content_is_not_a_new_version(self):
        a = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        b = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        self.assertEqual(a.version, b.version)
        self.assertEqual(len(self.registry.list_versions("residual")), 1)

    def test_different_content_is_a_different_version(self):
        a = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        b = self.registry.register("residual", writer({"a": 2}), {}, created_at=WHEN)
        self.assertNotEqual(a.version, b.version)
        self.assertEqual(len(self.registry.list_versions("residual")), 2)

    def test_verify_detects_tampering(self):
        version = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        self.assertTrue(self.registry.verify(version.version))
        with open(os.path.join(version.path, "model.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        self.assertFalse(self.registry.verify(version.version))

    def test_failed_writer_leaves_no_trace(self):
        def broken(directory):
            with open(os.path.join(directory, "half.json"), "w") as handle:
                handle.write("{")
            raise RuntimeError("disk full")

        with self.assertRaises(RuntimeError):
            self.registry.register("residual", broken, {}, created_at=WHEN)
        leftovers = os.listdir(os.path.join(self.root, "residual"))
        self.assertEqual(leftovers, [])
        self.assertEqual(self.registry.list_versions("residual"), [])

    def test_promote_resolve_and_rollback(self):
        first = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        second = self.registry.register(
            "residual", writer({"a": 2}), {}, created_at=datetime(2026, 8, 31, 9, 0, tzinfo=UTC)
        )
        self.assertIsNone(self.registry.promote(first.version))
        self.assertEqual(self.registry.promote(second.version, reason="better p90"), first.version)
        self.assertEqual(self.registry.resolve("residual").version, second.version)
        self.assertEqual(self.registry.resolve("residual@production").version, second.version)
        self.assertEqual(self.registry.resolve(first.version).version, first.version)
        self.assertEqual(self.registry.rollback("residual"), first.version)
        self.assertEqual(self.registry.resolve("residual").version, first.version)

    def test_promoting_the_current_version_is_a_no_op(self):
        first = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        self.registry.promote(first.version)
        self.assertEqual(self.registry.promote(first.version), first.version)
        self.assertEqual(self.registry.aliases("residual")["production"]["history"], [])

    def test_independent_aliases(self):
        first = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        second = self.registry.register("residual", writer({"a": 2}), {}, created_at=WHEN)
        self.registry.promote(first.version)
        self.registry.promote(second.version, alias="shadow")
        self.assertEqual(self.registry.resolve("residual").version, first.version)
        self.assertEqual(self.registry.resolve("residual@shadow").version, second.version)

    def test_latest(self):
        self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        later = self.registry.register(
            "residual", writer({"a": 2}), {}, created_at=datetime(2026, 9, 1, tzinfo=UTC)
        )
        self.assertEqual(self.registry.latest("residual").version, later.version)

    def test_errors(self):
        with self.assertRaises(ModelNotFound):
            self.registry.resolve("residual")
        with self.assertRaises(ModelNotFound):
            self.registry.get("residual/20260101T0000Z-deadbeef")
        with self.assertRaises(RegistryError):
            self.registry.get("not a version")
        with self.assertRaises(RegistryError):
            self.registry.register("Bad Name", writer({}), {})
        first = self.registry.register("residual", writer({"a": 1}), {}, created_at=WHEN)
        self.registry.promote(first.version)
        with self.assertRaises(RegistryError):
            self.registry.rollback("residual")
        with self.assertRaises(RegistryError):
            self.registry.promote(first.version, alias="Not Valid")

    def test_names(self):
        self.registry.register("prep", writer({"a": 1}), {}, created_at=WHEN)
        self.registry.register("dwell", writer({"a": 1}), {}, created_at=WHEN)
        self.assertEqual(self.registry.names(), ["dwell", "prep"])


if __name__ == "__main__":
    unittest.main()
