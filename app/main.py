"""FastAPI application: wiring, middleware and health checks."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.observability import configure_logging, configure_tracing, request_id_var
from app.core.security import RateLimiter
from app.db import session as db

log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=False)
    app.state.rate_limiter = RateLimiter(app.state.redis, settings.rate_limit_per_minute)
    from app.api.v1.deps import build_services

    app.state.services = await build_services(settings, app.state.redis)
    log.info(
        "startup",
        extra={
            "environment": settings.environment,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        },
    )
    yield
    await app.state.redis.aclose()
    await db.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise RAG Engine", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path})
            response = JSONResponse({"detail": "Internal error", "request_id": rid}, status_code=500)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers["X-Request-ID"] = rid
        if request.url.path not in ("/health", "/health/ready"):
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": elapsed_ms,
                },
            )
        request_id_var.reset(token)
        return response

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready(request: Request) -> JSONResponse:
        checks: dict[str, str] = {}
        try:
            await db.ping()
            checks["postgres"] = "ok"
        except Exception as exc:
            checks["postgres"] = f"error: {type(exc).__name__}"
        try:
            await request.app.state.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {type(exc).__name__}"
        healthy = all(v == "ok" for v in checks.values())
        return JSONResponse({"status": "ok" if healthy else "degraded", **checks}, status_code=200 if healthy else 503)

    from app.api.v1.router import api_router

    app.include_router(api_router, prefix="/api/v1")
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    return app


app = create_app()
