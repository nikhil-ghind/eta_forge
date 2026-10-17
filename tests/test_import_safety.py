"""Every module must import on a bare interpreter.

Checked in a *fresh* subprocess: other tests in the same run may legitimately import
LightGBM (via ``resolve_backend('auto')``), which would make an in-process check of
``sys.modules`` meaningless.
"""

import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEAVY = (
    "psycopg", "psycopg_pool", "lightgbm", "pandas", "numpy", "h3", "fastapi",
    "starlette", "uvicorn", "pydantic", "pyarrow", "yaml", "holidays", "prometheus_client",
)

PROBE = """
import importlib, json, pkgutil, sys
import eta_forge
names = sorted(m.name for m in pkgutil.walk_packages(eta_forge.__path__, "eta_forge."))
for name in names:
    importlib.import_module(name)
heavy = %r
print(json.dumps({"modules": names, "heavy": sorted(h for h in heavy if h in sys.modules)}))
"""


class TestImportSafety(unittest.TestCase):
    def test_no_heavy_dependency_is_imported_at_module_import(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        result = subprocess.run(
            [sys.executable, "-c", PROBE % (HEAVY,)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["heavy"], [], f"imported at module import: {report['heavy']}")
        for expected in ("eta_forge.service", "eta_forge.api.app", "eta_forge.eval.harness",
                         "eta_forge.mapmatch.hmm", "eta_forge.models.registry"):
            self.assertIn(expected, report["modules"])


if __name__ == "__main__":
    unittest.main()
