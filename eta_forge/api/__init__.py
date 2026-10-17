"""HTTP serving layer.

``schemas``  pure-Python request validation and response shaping (the API contract)
``metrics``  Prometheus text-format counters and histograms, no client library
``deps``     builds router + registry + model bundle + service from config
``app``      the FastAPI application (fastapi imported lazily inside ``create_app``)
"""

from .schemas import SchemaError, error_body, eta_response_body, parse_eta_request
from .metrics import MetricsRegistry, ServiceMetrics
from .deps import AppState, build_state

__all__ = [
    "SchemaError",
    "error_body",
    "eta_response_body",
    "parse_eta_request",
    "MetricsRegistry",
    "ServiceMetrics",
    "AppState",
    "build_state",
]
