"""Wiring: build the router, registry, model bundle and service from a ``Config``.

Kept apart from ``app.py`` so the object graph can be constructed (and tested) without
FastAPI, and so the server can swap in the synthetic graph with one flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from ..config import Config
from ..models.registry import ModelRegistry
from ..service import EtaService, ModelBundle
from .metrics import ServiceMetrics


@dataclass
class AppState:
    cfg: Config
    service: EtaService
    registry: ModelRegistry
    metrics: ServiceMetrics
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    backend: str = "postgis"
    close: Optional[Callable[[], None]] = None


def build_state(
    cfg: Config,
    *,
    synthetic: bool = False,
    router: Any = None,
    registry: Optional[ModelRegistry] = None,
) -> AppState:
    """Assemble the serving object graph.

    With ``synthetic`` (or an injected ``router``) nothing touches PostgreSQL; otherwise
    the router and the profile-freshness probe share one lazily-opened pool.
    """
    registry = registry or ModelRegistry(cfg.model.registry_dir)
    closer: Optional[Callable[[], None]] = None
    freshness: Optional[Callable[[], Mapping[str, Any]]] = None
    backend = "injected"

    if router is None and synthetic:
        from ..graph.synthetic import SyntheticGraph

        router = SyntheticGraph()
        backend = "synthetic"
    elif router is None:
        from ..db.pool import ConnectionPool
        from ..db.runner import SqlRunner
        from ..graph.router import Router
        from ..profiles.store import ProfileStore

        pool = ConnectionPool(cfg.database)
        runner = SqlRunner(pool)
        router = Router(cfg, runner)
        freshness = ProfileStore(runner, cfg.profiles).freshness
        closer = pool.close
        backend = "postgis"

    bundle = ModelBundle.load(registry, cfg)
    service = EtaService(
        cfg, router, bundle=bundle, registry=registry, profile_freshness=freshness
    )
    metrics = ServiceMetrics()
    metrics.set_models(bundle.versions)
    return AppState(cfg=cfg, service=service, registry=registry, metrics=metrics,
                    backend=backend, close=closer)


__all__ = ["AppState", "build_state"]
