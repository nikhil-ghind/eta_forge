"""Markdown + CSV + JSON evaluation reports.

The Markdown is for a human deciding whether to promote; the CSV is every slice row for
a notebook; the JSON is the machine-readable summary a CI gate or a dashboard reads.
All three are written side by side from one :class:`EvalResult`, so they can never
disagree.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .harness import EvalResult
from .slices import SliceMetrics

SLICE_COLUMNS = (
    "dimension", "key", "n", "mae_s", "mape", "bias_s", "late_rate",
    "coverage_p90", "pinball_p90", "sharpness_s",
)


def _fmt(value: Any, kind: str = "num") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if kind == "pct":
        return f"{100.0 * float(value):.1f}%"
    if kind == "s":
        return f"{float(value):.1f}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def render_markdown(result: EvalResult, *, title: str = "ETA Forge evaluation",
                    max_slice_rows: int = 24) -> str:
    o = result.overall
    lines: List[str] = [f"# {title}", ""]
    meta = result.meta
    lines.append(
        f"Held-out trips: **{int(o.get('n', 0)):,}**"
        + (f" · window {meta.get('window_start')} → {meta.get('window_end')}"
           if meta.get("window_start") else "")
    )
    for key in ("residual_version", "backend", "feature_spec_hash", "data_source"):
        if meta.get(key):
            lines.append(f"- {key.replace('_', ' ')}: `{meta[key]}`")
    lines.append("")

    lines += ["## Headline", ""]
    lines += _table(
        ["metric", "value"],
        [
            ["p50 MAE (s)", _fmt(o.get("mae_s"), "s")],
            ["p50 MAPE", _fmt(o.get("mape"), "pct")],
            ["p50 bias (s, negative = late)", _fmt(o.get("bias_s"), "s")],
            ["late rate vs p50", _fmt(o.get("late_rate"), "pct")],
            ["**p90 coverage**", f"**{_fmt(o.get('coverage_p90'), 'pct')}**"],
            ["p90 pinball loss", _fmt(o.get("pinball_p90"), "s")],
            ["sharpness, mean p90 − p50 (s)", _fmt(o.get("sharpness_s"), "s")],
            ["degraded answers", _fmt(o.get("degraded_rate"), "pct")],
        ],
    )
    lines.append("")

    lines += ["## What the learned stage adds", ""]
    rows = []
    for entry in result.comparison:
        is_lift = entry["predictor"] == "relative improvement"
        rows.append([
            entry["predictor"],
            _fmt(entry.get("mae_s"), "pct" if is_lift else "s"),
            _fmt(entry.get("mape"), "pct"),
            _fmt(entry.get("rmse_s"), "pct" if is_lift else "s"),
            _fmt(entry.get("bias_s"), "pct" if is_lift else "s"),
            _fmt(entry.get("pinball_p50"), "pct" if is_lift else "s"),
        ])
    lines += _table(["predictor", "MAE", "MAPE", "RMSE", "|bias|", "pinball p50"], rows)
    lines += ["", "_Relative improvement is 1 − model/baseline on each error metric._", ""]

    lines += ["## Calibration", ""]
    lines += _table(
        ["nominal quantile", "observed coverage", "gap"],
        [[f"{p.nominal:.2f}", _fmt(p.observed, "pct"), f"{100.0 * p.gap:+.1f} pp"]
         for p in result.calibration],
    )
    err = result.calibration_error
    lines += ["", f"Mean |gap| {_fmt(err.get('mean_abs_gap'), 'pct')}, "
                  f"worst {_fmt(err.get('max_abs_gap'), 'pct')}.", ""]

    if result.worst_late:
        lines += ["## Most systematically late slices", ""]
        lines += _table(
            ["dimension", "slice", "n", "bias (s)", "p90 coverage", "MAE (s)"],
            [[s.dimension, s.key, str(s.n), _fmt(s.get("bias_s"), "s"),
              _fmt(s.get("coverage_p90"), "pct"), _fmt(s.get("mae_s"), "s")]
             for s in result.worst_late],
        )
        lines.append("")

    for dimension, slices in result.slices.items():
        if not slices:
            continue
        lines += [f"## By {dimension.replace('_', ' ')}", ""]
        shown = sorted(slices, key=lambda s: -s.n)[:max_slice_rows]
        if dimension in ("hour", "dow", "distance"):
            shown = sorted(shown, key=lambda s: s.key)
        lines += _table(
            ["slice", "n", "MAE (s)", "MAPE", "bias (s)", "p90 coverage", "sharpness (s)"],
            [[s.key, str(s.n), _fmt(s.get("mae_s"), "s"), _fmt(s.get("mape"), "pct"),
              _fmt(s.get("bias_s"), "s"), _fmt(s.get("coverage_p90"), "pct"),
              _fmt(s.get("sharpness_s"), "s")] for s in shown],
        )
        dropped = result.dropped_slices.get(dimension, 0)
        if dropped:
            lines.append(f"\n_{dropped} slice(s) below the minimum size were omitted._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def slices_csv(result: EvalResult) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(SLICE_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for slices in result.slices.values():
        for s in slices:
            row = s.to_row()
            writer.writerow({k: row.get(k) for k in SLICE_COLUMNS})
    return buffer.getvalue()


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def write_report(result: EvalResult, directory: str, *, stem: str = "eval",
                 title: str = "ETA Forge evaluation") -> Dict[str, str]:
    """Write ``<stem>.report.md``, ``<stem>.report.csv`` and ``<stem>.summary.json``."""
    os.makedirs(directory, exist_ok=True)
    paths = {
        "markdown": os.path.join(directory, f"{stem}.report.md"),
        "csv": os.path.join(directory, f"{stem}.report.csv"),
        "json": os.path.join(directory, f"{stem}.summary.json"),
    }
    with open(paths["markdown"], "w", encoding="utf-8") as handle:
        handle.write(render_markdown(result, title=title))
    with open(paths["csv"], "w", encoding="utf-8", newline="") as handle:
        handle.write(slices_csv(result))
    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(_clean(result.summary()), handle, indent=2, default=_json_default)
    return paths


__all__ = ["render_markdown", "slices_csv", "write_report", "SLICE_COLUMNS"]
