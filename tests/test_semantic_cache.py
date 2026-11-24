"""Semantic cache: similarity threshold, permission scopes and invalidation on ingest."""

from __future__ import annotations

import os

import pytest
from redis.asyncio import Redis

from app.cache.semantic_cache import SemanticCache
from app.core.config import get_settings
from app.retrieval.embeddings import HashingEmbedder
from tests.conftest import principal

EMBED = HashingEmbedder(384)
FINANCE = principal("f", ("finance",), "confidential")
EMPLOYEE = principal("e", clearance="internal")
QUESTION = "What is the refund window for annual plans?"


async def _cache(redis: Redis) -> SemanticCache:
    return SemanticCache(redis, get_settings())


async def test_hit_on_same_question_and_scope(redis: Redis) -> None:
    cache = await _cache(redis)
    await cache.store(FINANCE, QUESTION, await EMBED.embed_query(QUESTION), {"answer": "14 days"})
    hit = await cache.lookup(FINANCE, QUESTION.lower(), await EMBED.embed_query(QUESTION.lower()))
    assert hit is not None and hit.payload == {"answer": "14 days"} and hit.similarity >= 0.95


async def test_different_question_misses(redis: Redis) -> None:
    cache = await _cache(redis)
    await cache.store(FINANCE, QUESTION, await EMBED.embed_query(QUESTION), {"answer": "14 days"})
    other = "How many days of paid vacation do employees get?"
    assert await cache.lookup(FINANCE, other, await EMBED.embed_query(other)) is None


async def test_scopes_never_share_answers(redis: Redis) -> None:
    cache = await _cache(redis)
    await cache.store(FINANCE, QUESTION, await EMBED.embed_query(QUESTION), {"answer": "confidential"})
    assert await cache.lookup(EMPLOYEE, QUESTION, await EMBED.embed_query(QUESTION)) is None
    same_rights = principal("another finance user", ("finance",), "confidential")
    assert await cache.lookup(same_rights, QUESTION, await EMBED.embed_query(QUESTION)) is not None


async def test_ingest_invalidates_previous_answers(redis: Redis) -> None:
    cache = await _cache(redis)
    await cache.store(EMPLOYEE, QUESTION, await EMBED.embed_query(QUESTION), {"answer": "old"})
    await cache.invalidate()
    assert await cache.lookup(EMPLOYEE, QUESTION, await EMBED.embed_query(QUESTION)) is None


@pytest.mark.skipif(
    not os.environ.get("TEST_REDIS_STACK_URL"),
    reason="set TEST_REDIS_STACK_URL (database 0: RediSearch indexes only live there)",
)
async def test_redisearch_backend() -> None:
    client = Redis.from_url(os.environ["TEST_REDIS_STACK_URL"])
    await client.flushdb()
    try:
        cache = await _cache(client)
        await cache.store(FINANCE, QUESTION, await EMBED.embed_query(QUESTION), {"answer": "14 days"})
        assert await cache._has_search()
        assert await cache.lookup(FINANCE, QUESTION, await EMBED.embed_query(QUESTION)) is not None
        assert await cache.lookup(EMPLOYEE, QUESTION, await EMBED.embed_query(QUESTION)) is None
        await cache.invalidate()
        assert await cache.lookup(FINANCE, QUESTION, await EMBED.embed_query(QUESTION)) is None
    finally:
        await client.flushdb()
        await client.aclose()
