"""The ETA service: route -> features -> residual quantiles -> prep -> dwell -> compose.

``EtaService.estimate`` is the whole request path, and it is written so that every
failure has a defined, *labelled* answer rather than an exception:

* the routing query times out or finds no path -> a haversine-times-detour-factor
  travel estimate, with ``degraded=True`` and the reason;
* no residual model is loaded -> the baseline alone, with a nominal spread, flagged;
* no prep / dwell model -> the configured defaults, flagged.

A degraded ETA is still an ETA; a 500 is not. Callers that care can read the flag.

Model versions are request-level: ``model_version`` in a request pins a specific
residual version (for shadow comparison) without touching what everyone else is
served. Pinned bundles are cached, a handful at a time.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .config import Config
from .decompose import decompose
from .features.builder import feature_hash
from .features.weather import WeatherTable
from .graph.geometry import haversine_m
from .graph.router import RouterProtocol
from .models.dwell import DwellEstimator
from .models.prep_time import PrepTimeModel
from .models.registry import ModelRegistry
from .models.residual import ResidualModel
from .types import (
    BaselineRoute,
    EtaForgeError,
    EtaRequest,
    EtaResponse,
    Location,
    ModelNotFound,
    QuantilePair,
    ValidationError,
)

log = logging.getLogger("eta_forge.service")

MODEL_NAMES = ("residual", "prep", "dwell")
# Spread applied to a baseline-only travel estimate when no residual model is loaded.
BASELINE_ONLY_P90_FACTOR = 1.30
FALLBACK_P90_FACTOR = 1.50


@dataclass
class ModelBundle:
    """The three models served together, and the versions they came from."""

    residual: Optional[ResidualModel] = None
    prep: Optional[PrepTimeModel] = None
    dwell: Optional[DwellEstimator] = None
    versions: Dict[str, str] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def load(
        cls,
        registry: ModelRegistry,
        cfg: Config,
        *,
        alias: Optional[str] = None,
        pins: Optional[Mapping[str, str]] = None,
        strict: bool = False,
    ) -> "ModelBundle":
        """Resolve each model by pinned version, else by alias.

        A model that is not in the registry is left as ``None`` (and served on its
        fallback) unless ``strict``, so a fresh deployment with no trained models comes
        up degraded rather than not at all.
        """
        alias = alias or cfg.service.model_alias
        pins = dict(pins or {})
        bundle = cls()
        for name in MODEL_NAMES:
            ref = pins.get(name) or f"{name}@{alias}"
            try:
                version = registry.resolve(ref)
            except ModelNotFound:
                if strict or name in pins:
                    raise
                bundle.versions[name] = "none"
                continue
            if name == "residual":
                bundle.residual = ResidualModel.load(cfg, version.path)
            elif name == "prep":
                bundle.prep = PrepTimeModel.load(version.path)
            else:
                bundle.dwell = DwellEstimator.load(version.path)
            bundle.versions[name] = version.version
        return bundle

    def describe(self) -> Dict[str, Any]:
        return {
            "versions": dict(self.versions),
            "loaded_at": self.loaded_at.isoformat(),
            "residual_backend": self.residual.backend if self.residual else None,
            "feature_spec_hash": self.residual.spec_hash if self.residual else None,
        }


class EtaService:
    def __init__(
        self,
        cfg: Config,
        router: RouterProtocol,
        *,
        bundle: Optional[ModelBundle] = None,
        registry: Optional[ModelRegistry] = None,
        weather: Optional[WeatherTable] = None,
        profile_freshness: Optional[Callable[[], Mapping[str, Any]]] = None,
        pinned_cache_size: int = 4,
        max_workers: int = 8,
    ):
        self._cfg = cfg
        self._router = router
        self._registry = registry
        self._bundle = bundle or ModelBundle()
        self._weather = weather
        self._profile_freshness = profile_freshness
        self._pinned: "OrderedDict[str, ModelBundle]" = OrderedDict()
        self._pinned_size = pinned_cache_size
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="eta-route"
        )

    # -- models ------------------------------------------------------------------

    @property
    def bundle(self) -> ModelBundle:
        return self._bundle

    def reload(self) -> Dict[str, str]:
        """Re-resolve the serving alias; swap the bundle in one assignment."""
        if self._registry is None:
            raise EtaForgeError("no registry configured; nothing to reload from")
        fresh = ModelBundle.load(self._registry, self._cfg)
        with self._lock:
            self._bundle = fresh
            self._pinned.clear()
        return dict(fresh.versions)

    def _bundle_for(self, model_version: Optional[str]) -> ModelBundle:
        if not model_version:
            return self._bundle
        if model_version == self._bundle.versions.get("residual"):
            return self._bundle
        if self._registry is None:
            raise ModelNotFound("model pinning requires a registry")
        with self._lock:
            cached = self._pinned.get(model_version)
            if cached is not None:
                self._pinned.move_to_end(model_version)
                return cached
        # Only the residual model is pinned; prep and dwell stay on the serving alias.
        pins = {"residual": model_version}
        for name in ("prep", "dwell"):
            current = self._bundle.versions.get(name)
            if current and current != "none":
                pins[name] = current
        pinned = ModelBundle.load(self._registry, self._cfg, pins=pins)
        with self._lock:
            self._pinned[model_version] = pinned
            while len(self._pinned) > self._pinned_size:
                self._pinned.popitem(last=False)
        return pinned

    def ready(self) -> bool:
        return self._bundle.residual is not None

    # -- routing -----------------------------------------------------------------

    def route(self, origin: Location, destination: Location, departure_at: datetime) -> BaselineRoute:
        """Baseline route only, bounded by the routing timeout. Raises on failure."""
        future = self._executor.submit(self._router.route, origin, destination, departure_at)
        return future.result(timeout=self._cfg.service.route_timeout_ms / 1000.0)

    def fallback_travel(self, origin: Location, destination: Location) -> Tuple[QuantilePair, float]:
        """Crow-flight distance x detour factor at a fallback speed. Always answers."""
        distance = haversine_m(origin, destination) * self._cfg.graph.fallback_detour_factor
        p50 = distance / (self._cfg.graph.fallback_speed_kph / 3.6)
        return QuantilePair(p50, p50 * FALLBACK_P90_FACTOR), distance

    # -- the request path ------------------------------------------------------------

    def estimate(self, request: EtaRequest) -> EtaResponse:
        started = time.perf_counter()
        bundle = self._bundle_for(request.model_version)
        departure = request.departure_at or datetime.now(timezone.utc)
        reasons: List[str] = []

        route: Optional[BaselineRoute] = None
        try:
            route = self.route(request.origin, request.destination, departure)
        except concurrent.futures.TimeoutError:
            reasons.append("routing timed out")
        except ValidationError:
            raise
        except EtaForgeError as exc:
            reasons.append(f"routing failed: {exc}")
        except Exception as exc:  # a dropped database connection must not become a 500
            log.warning("routing error: %r", exc)
            reasons.append(f"routing error: {type(exc).__name__}")

        vector_hash: Optional[str] = None
        residual_p50 = residual_p90 = 0.0
        if route is None:
            travel, distance = self.fallback_travel(request.origin, request.destination)
            baseline_s = travel.p50
            coverage = 0.0
        else:
            baseline_s = route.duration_s
            distance = route.distance_m
            coverage = route.profile_coverage
            if bundle.residual is not None:
                observation = (
                    self._weather.as_of(request.origin, departure) if self._weather else None
                )
                vector, _ = bundle.residual.builder.vector(route, weather=observation)
                travel, residual_p50, residual_p90 = bundle.residual.predict_travel(baseline_s, vector)
                vector_hash = feature_hash(vector)
            else:
                travel = QuantilePair(baseline_s, baseline_s * BASELINE_ONLY_P90_FACTOR)
                reasons.append("no residual model; baseline only")

        if request.is_delivery:
            prep = self._prep(bundle, request, departure, reasons)
            handoff = self._handoff(bundle, request.destination, reasons)
        else:
            prep = handoff = QuantilePair.zero()

        parts = decompose(
            prep=prep,
            travel=travel,
            handoff=handoff,
            baseline_travel_s=baseline_s,
            residual_p50_s=residual_p50,
            residual_p90_s=residual_p90,
            rho=self._cfg.service.component_tail_correlation,
        )
        response = EtaResponse(
            eta=parts.total,
            breakdown=parts.breakdown,
            distance_m=distance,
            model_versions=dict(bundle.versions),
            degraded=bool(reasons),
            degraded_reason="; ".join(reasons) if reasons else None,
            profile_coverage=coverage,
            route=route if request.include_route else None,
            computed_at=datetime.now(timezone.utc),
        )
        self._log(request, response, vector_hash, time.perf_counter() - started)
        return response

    def _prep(self, bundle: ModelBundle, request: EtaRequest, departure: datetime,
              reasons: List[str]) -> QuantilePair:
        if bundle.prep is None:
            reasons.append("no prep model; default prep")
            return QuantilePair(self._cfg.service.default_prep_p50_s, self._cfg.service.default_prep_p90_s)
        return bundle.prep.predict(
            merchant_id=request.merchant_id,
            placed_at=departure,
            item_count=request.order_item_count,
            order_value_usd=request.order_value_usd,
            open_orders=request.open_orders_at_merchant,
        )

    def _handoff(self, bundle: ModelBundle, destination: Location, reasons: List[str]) -> QuantilePair:
        if bundle.dwell is None:
            reasons.append("no dwell model; default handoff")
            return QuantilePair(
                self._cfg.service.default_handoff_p50_s, self._cfg.service.default_handoff_p90_s
            )
        cell = bundle.residual.builder.cell(destination) if bundle.residual else None
        return bundle.dwell.predict(cell)

    def profile_freshness(self) -> Mapping[str, Any]:
        if self._profile_freshness is None:
            return {}
        try:
            return self._profile_freshness()
        except Exception as exc:  # pragma: no cover - needs a live database
            return {"error": str(exc)}

    # -- logging -----------------------------------------------------------------

    def _log(self, request: EtaRequest, response: EtaResponse, vector_hash: Optional[str],
             elapsed_s: float) -> None:
        """One structured line per request: enough to replay it offline."""
        if not log.isEnabledFor(logging.INFO):
            return
        payload = {
            "event": "eta",
            "request_hash": request_hash(request),
            "feature_hash": vector_hash if self._cfg.service.log_feature_hash else None,
            "delivery": request.is_delivery,
            "eta_p50_s": round(response.eta.p50, 1),
            "eta_p90_s": round(response.eta.p90, 1),
            "degraded": response.degraded,
            "residual_version": response.model_versions.get("residual"),
            "latency_ms": round(elapsed_s * 1000.0, 2),
        }
        log.info(json.dumps(payload, sort_keys=True))

    def close(self) -> None:
        self._executor.shutdown(wait=False)


def request_hash(request: EtaRequest) -> str:
    body = json.dumps(request.to_dict(), sort_keys=True, default=str)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


__all__ = ["EtaService", "ModelBundle", "request_hash", "MODEL_NAMES"]
