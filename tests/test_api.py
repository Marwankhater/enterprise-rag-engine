"""HTTP API: auth, ingestion, streamed chat with cache, feedback, history and rate limits."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from redis.asyncio import Redis

from app.api.v1.deps import build_services
from app.core.config import get_settings
from app.core.security import RateLimiter
from app.main import create_app
from app.retrieval.reranker import PassthroughReranker
from tests.conftest import EXAMPLES, ScriptedLLM

ENGINEER = {"X-API-Key": "key-engineer"}
VISITOR = {"X-API-Key": "key-public"}


@pytest.fixture
async def client(db: None, redis: Redis) -> AsyncIterator[httpx.AsyncClient]:
    settings = get_settings()
    app = create_app()
    app.state.redis = redis
    app.state.rate_limiter = RateLimiter(redis, settings.rate_limit_per_minute)
    app.state.services = await build_services(settings, redis, llm=ScriptedLLM(), reranker=PassthroughReranker())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as http:
        yield http
    await app.state.services.sql_tool.engine.dispose()


def sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events, name = [], None
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1])))
    return events


async def upload(client: httpx.AsyncClient, name: str, headers: dict[str, str], **form: str) -> httpx.Response:
    with (EXAMPLES / name).open("rb") as handle:
        return await client.post("/api/v1/ingest", headers=headers, files={"file": (name, handle)}, data=form)


async def test_health(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).json() == {"status": "ok"}
    ready = await client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["postgres"] == "ok"


async def test_requests_need_a_valid_key(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/chat", json={"question": "hi"})).status_code == 401
    assert (
        await client.post("/api/v1/chat", json={"question": "hi"}, headers={"X-API-Key": "wrong"})
    ).status_code == 401


async def test_ingest_rules(client: httpx.AsyncClient) -> None:
    assert (await upload(client, "product-faq.txt", VISITOR)).status_code == 403  # no ingest right
    above = await upload(client, "product-faq.txt", ENGINEER, access_level="confidential")
    assert above.status_code == 403  # above own clearance
    other_dept = await upload(client, "product-faq.txt", ENGINEER, department_id="finance")
    assert other_dept.status_code == 403
    ok = await upload(client, "onboarding-guide.pdf", ENGINEER, department_id="engineering")
    assert ok.status_code == 202 and ok.json()["status"] == "done" and ok.json()["children"] > 0
    again = await upload(client, "onboarding-guide.pdf", ENGINEER, department_id="engineering")
    assert again.json()["status"] == "duplicate"
    job = await client.get(f"/api/v1/ingest/{ok.json()['job_id']}", headers=ENGINEER)
    assert job.json()["document_id"] == ok.json()["document_id"]
    assert (await client.get(f"/api/v1/ingest/{ok.json()['job_id']}", headers=VISITOR)).status_code == 404


async def test_streamed_chat_then_cache_then_feedback(client: httpx.AsyncClient) -> None:
    await upload(client, "onboarding-guide.pdf", ENGINEER, department_id="engineering")
    question = {"question": "How fast must an on-call engineer acknowledge a page?"}
    first = sse((await client.post("/api/v1/chat", json=question, headers=ENGINEER)).text)
    names = [name for name, _ in first]
    assert names[0] == "status" and "token" in names and names[-1] == "final"
    assert names.index("token") > max(i for i, n in enumerate(names) if n == "status")  # verified before sent
    final = first[-1][1]
    assert final["cached"] is False and final["citations"]
    assert "".join(d["text"] for n, d in first if n == "token") == final["answer"]

    second = sse((await client.post("/api/v1/chat", json=question, headers=ENGINEER)).text)
    assert second[0] == ("status", second[0][1]) and second[0][1]["step"] == "cache_hit"
    assert second[-1][1]["cached"] is True and second[-1][1]["answer"] == final["answer"]

    vote = await client.post(
        "/api/v1/feedback",
        headers=ENGINEER,
        json={"message_id": final["message_id"], "rating": -1, "comment": "too short"},
    )
    assert vote.status_code == 201
    stranger = await client.post(
        "/api/v1/feedback", headers=VISITOR, json={"message_id": final["message_id"], "rating": 1}
    )
    assert stranger.status_code == 404


async def test_json_mode_and_history(client: httpx.AsyncClient) -> None:
    await upload(client, "onboarding-guide.pdf", ENGINEER, department_id="engineering")
    body = {"question": "What is the git workflow?", "stream": False, "session_id": "session-123"}
    reply = await client.post("/api/v1/chat", json=body, headers=ENGINEER)
    assert reply.status_code == 200 and reply.json()["session_id"] == "session-123"
    history = (await client.get("/api/v1/chat/session-123/history", headers=ENGINEER)).json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert (await client.get("/api/v1/chat/session-123/history", headers=VISITOR)).status_code == 404


async def test_rate_limit(client: httpx.AsyncClient, redis: Redis) -> None:
    client._transport.app.state.rate_limiter = RateLimiter(redis, 2)  # type: ignore[attr-defined]
    codes = [(await client.get("/api/v1/chat/nope-1234/history", headers=VISITOR)).status_code for _ in range(3)]
    assert codes == [404, 404, 429]
