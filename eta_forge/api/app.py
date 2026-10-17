"""The FastAPI application.

    POST /v1/eta              ride ETA: travel only
    POST /v1/eta/delivery     delivery ETA: prep + travel + handoff
    GET  /v1/route            the baseline route alone, for debugging the first stage
    GET  /v1/models           which versions are being served
    POST /v1/models/reload    re-resolve the serving alias from the registry
    GET  /health              liveness: the process is up
    GET  /ready               readiness: a residual model is loaded
    GET  /metrics             Prometheus text format

``fastapi`` is imported inside :func:`create_app`, so this module imports on a bare
interpreter. Bodies are validated by ``api/schemas.py``, not by Pydantic models, so the
contract is testable without FastAPI and the error shape is the same everywhere.

The ETA computation itself is synchronous (it waits on the routing query) and runs in
the threadpool, so a slow route cannot stall the event loop for every other request.
"""

from __future__ import annotations

import time
from typing import Any

from .. import __version__
from ..types import EtaForgeError
from .deps import AppState
from .schemas import (
    SchemaError,
    error_body,
    eta_response_body,
    parse_eta_request,
    parse_route_query,
)


def create_app(state: AppState) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(
        title="ETA Forge",
        version=__version__,
        description="Two-stage ETA: pgRouting baseline + LightGBM quantile residuals.",
    )
    app.state.eta = state

    @app.middleware("http")
    async def measure(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or "unmatched"
        state.metrics.requests.inc(endpoint=endpoint, status=str(response.status_code))
        state.metrics.latency.observe(time.perf_counter() - started, endpoint=endpoint)
        return response

    @app.exception_handler(EtaForgeError)
    async def domain_error(_: Request, exc: EtaForgeError):
        status, body = error_body(exc)
        return JSONResponse(status_code=status, content=body)

    async def _json_body(request: Request) -> Any:
        try:
            return await request.json()
        except ValueError as exc:
            raise SchemaError([{"loc": [], "msg": "body is not valid JSON"}]) from exc

    async def _estimate(request: Request, *, delivery: bool):
        payload = await _json_body(request)
        eta_request = parse_eta_request(payload, delivery=delivery)
        response = await run_in_threadpool(state.service.estimate, eta_request)
        state.metrics.record_eta(
            "delivery" if delivery else "ride",
            response.eta.p50,
            response.eta.p90,
            response.degraded_reason,
        )
        return JSONResponse(eta_response_body(response))

    @app.post("/v1/eta")
    async def ride_eta(request: Request):
        return await _estimate(request, delivery=False)

    @app.post("/v1/eta/delivery")
    async def delivery_eta(request: Request):
        return await _estimate(request, delivery=True)

    @app.get("/v1/route")
    async def baseline_route(request: Request):
        import concurrent.futures
        from datetime import datetime, timezone

        origin, destination, departure = parse_route_query(dict(request.query_params))
        try:
            route = await run_in_threadpool(
                state.service.route, origin, destination, departure or datetime.now(timezone.utc)
            )
        except concurrent.futures.TimeoutError:
            # The ETA endpoints fall back; the debugging endpoint reports honestly.
            return JSONResponse(
                status_code=504,
                content={"error": "route_timeout",
                         "detail": [{"loc": [], "msg": "routing query exceeded the timeout"}]},
            )
        return JSONResponse(route.to_dict())

    @app.get("/v1/models")
    async def models():
        return {
            "served": state.service.bundle.describe(),
            "alias": state.cfg.service.model_alias,
            "registry": state.registry.root,
            "router_backend": state.backend,
            "profile_freshness": _jsonable(state.service.profile_freshness()),
        }

    @app.post("/v1/models/reload")
    async def reload_models():
        try:
            versions = await run_in_threadpool(state.service.reload)
        except EtaForgeError:
            state.metrics.reloads.inc(outcome="failed")
            raise
        state.metrics.reloads.inc(outcome="ok")
        state.metrics.set_models(versions)
        return {"reloaded": versions}

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__,
                "started_at": state.started_at.isoformat()}

    @app.get("/ready")
    async def ready():
        if state.service.ready():
            return {"status": "ready", "versions": state.service.bundle.versions}
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "no residual model loaded",
                     "versions": state.service.bundle.versions},
        )

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(state.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.on_event("shutdown")
    async def shutdown():
        state.service.close()
        if state.close is not None:
            state.close()

    return app


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["create_app"]
