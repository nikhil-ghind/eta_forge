import csv
import io
import json
import os
import random
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from eta_forge.config import EvalConfig
from eta_forge.eval.harness import evaluate
from eta_forge.eval.report import SLICE_COLUMNS, render_markdown, slices_csv, write_report
from eta_forge.eval.slices import EvalRecord

UTC = timezone.utc
T0 = datetime(2024, 3, 4, tzinfo=UTC)


def records(n=400, seed=9):
    """A model that is better than its baseline, with a late evening-peak slice."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        hour = (7, 12, 18)[i % 3]
        actual = rng.uniform(400.0, 1800.0) * (1.25 if hour == 18 else 1.0)
        out.append(EvalRecord(
            trip_id=f"t{i}",
            departure_at=T0 + timedelta(minutes=17 * i),
            actual_s=actual,
            baseline_s=actual * rng.uniform(0.6, 1.0),
            p50_s=actual * rng.uniform(0.9, 1.08) / (1.15 if hour == 18 else 1.0),
            p90_s=actual * rng.uniform(1.05, 1.45),
            distance_m=rng.uniform(500.0, 15_000.0),
            hour=hour,
            dow=(i // 3) % 7,
            origin_cell=f"cell{i % 4}",
            dest_cell=f"cell{(i + 1) % 4}",
        ))
    return out


class TestHarnessAndReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluate(records(), EvalConfig(min_slice_size=10), meta={"residual_version": "residual/x"})

    def test_learned_stage_beats_routing_only(self):
        lift = self.result.comparison[2]
        self.assertEqual(lift["predictor"], "relative improvement")
        self.assertGreater(lift["mae_s"], 0.0)
        self.assertLess(self.result.overall["mae_s"], self.result.baseline["mae_s"])

    def test_worst_late_slice_is_the_evening_peak(self):
        self.assertTrue(self.result.worst_late)
        self.assertEqual(self.result.worst_late[0].key, "18:00")

    def test_calibration_points_cover_the_configured_quantiles(self):
        self.assertEqual(
            [p.nominal for p in self.result.calibration], EvalConfig().calibration_quantiles
        )

    def test_markdown_sections(self):
        text = render_markdown(self.result, title="Smoke")
        for heading in ("# Smoke", "## Headline", "## What the learned stage adds",
                        "## Calibration", "## Most systematically late slices", "## By hour"):
            self.assertIn(heading, text)
        self.assertIn("**p90 coverage**", text)
        self.assertIn("`residual/x`", text)

    def test_csv_has_one_row_per_slice(self):
        rows = list(csv.DictReader(io.StringIO(slices_csv(self.result))))
        self.assertEqual(tuple(rows[0].keys()), SLICE_COLUMNS)
        expected = sum(len(v) for v in self.result.slices.values())
        self.assertEqual(len(rows), expected)

    def test_write_report(self):
        directory = tempfile.mkdtemp()
        try:
            paths = write_report(self.result, directory, stem="unit")
            for path in paths.values():
                self.assertTrue(os.path.isfile(path))
            with open(paths["json"], encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertIn("coverage_p90", summary["overall"])
            self.assertEqual(summary["meta"]["residual_version"], "residual/x")
        finally:
            shutil.rmtree(directory)

    def test_empty_evaluation_still_renders(self):
        result = evaluate([], EvalConfig())
        text = render_markdown(result)
        self.assertIn("Held-out trips: **0**", text)


if __name__ == "__main__":
    unittest.main()
