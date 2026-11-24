"""API key authentication, per-principal rate limiting and the RBAC filter used by retrieval."""

from __future__ import annotations

import hmac
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from redis.asyncio import Redis

from app.core.config import AccessLevel, Principal, Settings, get_settings, hash_api_key
from app.core.observability import principal_var

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def authenticate(raw_key: str | None, settings: Settings) -> Principal:
    if not raw_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header")
    digest = hash_api_key(raw_key)
    for known, principal in settings.principals().items():
        if hmac.compare_digest(known, digest):
            return principal
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")


async def current_principal(
    request: Request,
    raw_key: Annotated[str | None, Depends(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Principal:
    principal = authenticate(raw_key, settings)
    principal_var.set(principal.name)
    request.state.principal = principal
    return principal


async def require_ingest(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    if not principal.can_ingest:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This key cannot ingest documents")
    return principal


class RateLimiter:
    """Fixed one-minute window per principal, kept in Redis so it holds across API replicas."""

    def __init__(self, redis: Redis, per_minute: int) -> None:
        self.redis = redis
        self.per_minute = per_minute

    async def hit(self, who: str) -> tuple[bool, int]:
        window = int(time.time() // 60)
        key = f"rl:{who}:{window}"
        async with self.redis.pipeline(transaction=True) as pipe:
            count, _ = await pipe.incr(key).expire(key, 65).execute()
        return count <= self.per_minute, max(0, self.per_minute - count)


async def rate_limited(request: Request, principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    limiter: RateLimiter = request.app.state.rate_limiter
    allowed, remaining = await limiter.hit(principal.name)
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded, retry in a minute", headers={"Retry-After": "60"}
        )
    request.state.rate_remaining = remaining
    return principal


def rbac_clause(alias: str = "c") -> str:
    """SQL fragment that keeps only chunks the caller may read. Bind :clearance and :departments."""
    return (
        f"{alias}.access_level <= :clearance "
        f"and ({alias}.department_id is null or {alias}.department_id = any(:departments))"
    )


def rbac_params(principal: Principal) -> dict[str, object]:
    return {"clearance": int(principal.clearance), "departments": sorted(principal.departments)}


def can_read(principal: Principal, department_id: str | None, access_level: int) -> bool:
    """The same rule as rbac_clause, for checks done in Python (cache entries, tests)."""
    return access_level <= principal.clearance and (department_id is None or department_id in principal.departments)


__all__ = [
    "AccessLevel",
    "Principal",
    "RateLimiter",
    "authenticate",
    "can_read",
    "current_principal",
    "rate_limited",
    "rbac_clause",
    "rbac_params",
    "require_ingest",
]
