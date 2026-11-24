"""Semantic answer cache in Redis.

A new question reuses a stored answer when their embeddings have cosine similarity above the
threshold (0.95 by default) and both were asked under the same permission set. Three things keep
it safe:
  * Entries are partitioned by the caller's scope key (clearance plus departments), so an answer
    built from confidential documents is never served to someone without that access.
  * Every ingest bumps an epoch counter, and lookups only match the current epoch, so new
    documents are never hidden behind an old answer.
  * Only first-turn questions are cached. A follow-up depends on the conversation before it.

With Redis Stack the lookup is a KNN query on a RediSearch HNSW index. On plain Redis it falls back
to a capped per-scope list scanned in Python, which is fine for development.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from redis.asyncio import Redis

from app.core.config import Principal, Settings
from app.core.observability import span

log = logging.getLogger(__name__)

INDEX = "idx:semcache"
PREFIX = "sc:"
EPOCH_KEY = "cache:epoch"


@dataclass(frozen=True)
class CacheHit:
    question: str
    similarity: float
    payload: dict[str, Any]


def _first_result(raw: Any) -> dict[str, bytes] | None:
    """FT.SEARCH replies as a dict under RESP3 and as a flat list under RESP2; read either."""
    if isinstance(raw, dict):
        results = raw.get(b"results") or raw.get("results") or []
        if not results:
            return None
        attrs = results[0].get(b"extra_attributes") or results[0].get("extra_attributes") or {}
        return {(k.decode() if isinstance(k, bytes) else k): v for k, v in attrs.items()}
    if not raw or raw[0] == 0:
        return None
    fields = raw[2]
    return {fields[i].decode(): fields[i + 1] for i in range(0, len(fields), 2)}


class SemanticCache:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.dim = settings.embedding_dim
        self.threshold = settings.cache_similarity_threshold
        self.ttl = settings.cache_ttl_s
        self.max_entries = settings.cache_max_entries_per_scope
        self.enabled = settings.cache_enabled
        self._search: bool | None = None

    async def _has_search(self) -> bool:
        if self._search is None:
            try:
                await self.redis.execute_command("FT._LIST")
                self._search = True
                await self._ensure_index()
            except Exception:
                self._search = False
                log.info("redis search module not available, semantic cache uses the list fallback")
        return self._search

    async def _ensure_index(self) -> None:
        try:
            await self.redis.execute_command("FT.INFO", INDEX)
        except Exception:
            await self.redis.execute_command(
                "FT.CREATE",
                INDEX,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                PREFIX,
                "SCHEMA",
                "scope",
                "TAG",
                "epoch",
                "TAG",
                "vec",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.dim),
                "DISTANCE_METRIC",
                "COSINE",
            )

    async def _epoch(self) -> str:
        value = await self.redis.get(EPOCH_KEY)
        return (value.decode() if isinstance(value, bytes) else value) or "0"

    @staticmethod
    def _blob(vector: list[float]) -> bytes:
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        return (array / norm if norm else array).tobytes()

    async def lookup(self, principal: Principal, question: str, vector: list[float]) -> CacheHit | None:
        if not self.enabled:
            return None
        with span("cache.lookup") as current:
            try:
                hit = await (self._lookup_search if await self._has_search() else self._lookup_list)(
                    principal.scope_key, await self._epoch(), vector
                )
            except Exception:
                log.warning("cache lookup failed", exc_info=True)
                return None
            current.set_attribute("cache.hit", hit is not None)
            if hit:
                current.set_attribute("cache.similarity", round(hit.similarity, 4))
            return hit

    async def _lookup_search(self, scope: str, epoch: str, vector: list[float]) -> CacheHit | None:
        query = f"(@scope:{{{scope}}} @epoch:{{{epoch}}})=>[KNN 1 @vec $v AS dist]"
        raw = await self.redis.execute_command(
            "FT.SEARCH",
            INDEX,
            query,
            "PARAMS",
            "2",
            "v",
            self._blob(vector),
            "RETURN",
            "3",
            "dist",
            "question",
            "payload",
            "DIALECT",
            "2",
        )
        values = _first_result(raw)
        if values is None:
            return None
        similarity = 1.0 - float(values["dist"])
        if similarity < self.threshold:
            return None
        question = values["question"]
        return CacheHit(
            question.decode() if isinstance(question, bytes) else question, similarity, json.loads(values["payload"])
        )

    async def _lookup_list(self, scope: str, epoch: str, vector: list[float]) -> CacheHit | None:
        entries = await self.redis.lrange(f"{PREFIX}list:{epoch}:{scope}", 0, self.max_entries - 1)
        if not entries:
            return None
        query = np.frombuffer(self._blob(vector), dtype=np.float32)
        best: tuple[float, dict[str, Any]] | None = None
        for raw in entries:
            entry = json.loads(raw)
            if entry["expires"] < time.time():
                continue
            similarity = float(np.dot(query, np.frombuffer(bytes.fromhex(entry["vec"]), dtype=np.float32)))
            if best is None or similarity > best[0]:
                best = (similarity, entry)
        if best is None or best[0] < self.threshold:
            return None
        return CacheHit(best[1]["question"], best[0], best[1]["payload"])

    async def store(self, principal: Principal, question: str, vector: list[float], payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            epoch = await self._epoch()
            blob = self._blob(vector)
            if await self._has_search():
                key = f"{PREFIX}{uuid.uuid4().hex}"
                await self.redis.hset(
                    key,
                    mapping={
                        "scope": principal.scope_key,
                        "epoch": epoch,
                        "question": question,
                        "payload": json.dumps(payload),
                        "vec": blob,
                    },
                )
                await self.redis.expire(key, self.ttl)
            else:
                key = f"{PREFIX}list:{epoch}:{principal.scope_key}"
                entry = {"question": question, "vec": blob.hex(), "payload": payload, "expires": time.time() + self.ttl}
                await self.redis.lpush(key, json.dumps(entry))
                await self.redis.ltrim(key, 0, self.max_entries - 1)
                await self.redis.expire(key, self.ttl)
        except Exception:
            log.warning("cache store failed", exc_info=True)

    async def invalidate(self) -> None:
        await self.redis.incr(EPOCH_KEY)
