"""Prometheus metrics in the text exposition format, without prometheus_client.

A dozen lines of counters and histograms do not justify a dependency in the serving
image, and owning the implementation means the metrics are unit-testable and render
identically whether or not ``prometheus_client`` happens to be installed.

What is measured is chosen for an ETA service specifically: not only request counts
and latency, but how often answers are *degraded* and why, and the distribution of the
ETAs themselves — a deploy that shifts median ETA by two minutes is visible on a
dashboard before it is visible in customer complaints.
"""

from __future__ import annotations

import bisect
import math
import threading
from typing import Dict, List, Optional, Sequence, Tuple

LabelKey = Tuple[Tuple[str, str], ...]

DEFAULT_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
ETA_BUCKETS_S = (120, 300, 600, 900, 1200, 1800, 2700, 3600, 5400)


def _label_key(labels: Optional[Dict[str, str]]) -> LabelKey:
    return tuple(sorted((k, str(v)) for k, v in (labels or {}).items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(key: LabelKey, extra: Sequence[Tuple[str, str]] = ()) -> str:
    pairs = list(key) + list(extra)
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in pairs) + "}"


def _format_value(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value):
        return str(int(value))
    return repr(float(value))


class Counter:
    kind = "counter"

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values: Dict[LabelKey, float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counters only go up")
        key = _label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(_label_key(labels), 0.0)

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for key, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_format_labels(key)} {_format_value(value)}")
        return lines


class Gauge(Counter):
    kind = "gauge"

    def set(self, value: float, **labels: str) -> None:
        with self._lock:
            self._values[_label_key(labels)] = float(value)

    def render(self) -> List[str]:
        lines = super().render()
        lines[1] = f"# TYPE {self.name} gauge"
        return lines


class Histogram:
    kind = "histogram"

    def __init__(self, name: str, help_text: str, buckets: Sequence[float]):
        if list(buckets) != sorted(buckets):
            raise ValueError("histogram buckets must be ascending")
        self.name = name
        self.help = help_text
        self.buckets = tuple(float(b) for b in buckets)
        self._counts: Dict[LabelKey, List[int]] = {}
        self._sums: Dict[LabelKey, float] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = _label_key(labels)
        index = bisect.bisect_left(self.buckets, value)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            counts[index] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value

    def count(self, **labels: str) -> int:
        return sum(self._counts.get(_label_key(labels), []))

    def render(self) -> List[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key in sorted(self._counts):
                counts = self._counts[key]
                cumulative = 0
                for bound, n in zip(self.buckets + (math.inf,), counts):
                    cumulative += n
                    le = "+Inf" if math.isinf(bound) else _format_value(bound)
                    lines.append(
                        f"{self.name}_bucket{_format_labels(key, [('le', le)])} {cumulative}"
                    )
                lines.append(f"{self.name}_sum{_format_labels(key)} {_format_value(self._sums[key])}")
                lines.append(f"{self.name}_count{_format_labels(key)} {cumulative}")
        return lines


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, object] = {}

    def _add(self, metric):
        if metric.name in self._metrics:
            raise ValueError(f"metric {metric.name} already registered")
        self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, help_text: str) -> Counter:
        return self._add(Counter(name, help_text))

    def gauge(self, name: str, help_text: str) -> Gauge:
        return self._add(Gauge(name, help_text))

    def histogram(self, name: str, help_text: str, buckets: Sequence[float]) -> Histogram:
        return self._add(Histogram(name, help_text, buckets))

    def render(self) -> str:
        lines: List[str] = []
        for name in sorted(self._metrics):
            lines.extend(self._metrics[name].render())  # type: ignore[attr-defined]
        return "\n".join(lines) + "\n"


class ServiceMetrics:
    """The metric set the API exports."""

    def __init__(self) -> None:
        self.registry = MetricsRegistry()
        self.requests = self.registry.counter(
            "eta_forge_requests_total", "HTTP requests by endpoint and status code."
        )
        self.latency = self.registry.histogram(
            "eta_forge_request_latency_seconds", "Request latency by endpoint.",
            DEFAULT_LATENCY_BUCKETS,
        )
        self.degraded = self.registry.counter(
            "eta_forge_degraded_total", "ETAs answered on a fallback path, by reason."
        )
        self.eta = self.registry.histogram(
            "eta_forge_eta_seconds", "Served ETA by kind (ride / delivery) and quantile.",
            ETA_BUCKETS_S,
        )
        self.reloads = self.registry.counter(
            "eta_forge_model_reloads_total", "Model reloads by outcome."
        )
        self.model_info = self.registry.gauge(
            "eta_forge_model_info", "1 for each model version currently served."
        )

    def record_eta(self, kind: str, p50: float, p90: float, degraded_reason: Optional[str]) -> None:
        self.eta.observe(p50, kind=kind, quantile="p50")
        self.eta.observe(p90, kind=kind, quantile="p90")
        if degraded_reason:
            for reason in degraded_reason.split("; "):
                # Reasons carry free text after a colon; keep the label cardinality bounded.
                self.degraded.inc(reason=reason.split(":", 1)[0])

    def set_models(self, versions: Dict[str, str]) -> None:
        with self.model_info._lock:
            self.model_info._values.clear()
        for name, version in versions.items():
            self.model_info.set(1.0, model=name, version=version)

    def render(self) -> str:
        return self.registry.render()


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "ServiceMetrics",
]
