"""Test setup: a real PostgreSQL (pgvector) and Redis, an offline embedder and a scripted model.

Set TEST_DATABASE_URL / TEST_REDIS_URL to point elsewhere. Tests that need the database are skipped
when it is not reachable, so the pure unit tests still run anywhere.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

API_KEYS = {
    "key-public": {"name": "visitor", "clearance": "public"},
    "key-employee": {"name": "employee", "clearance": "internal"},
    "key-engineer": {"name": "engineer", "departments": ["engineering"], "clearance": "internal", "can_ingest": True},
    "key-finance": {"name": "finance", "departments": ["finance"], "clearance": "confidential", "can_ingest": True},
}

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag_test")

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATABASE_URL": TEST_DB,
        # The SQL tool connects as the read-only role created by the migrations, as in production.
        "SQL_TOOL_DATABASE_URL": re.sub(r"//[^@]+@", "//rag_reader:rag_reader@", TEST_DB),
        "REDIS_URL": os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15"),
        "EMBEDDING_BACKEND": "hashing",
        "EMBEDDING_DIM": "384",
        "RERANK_ENABLED": "false",
        "INGEST_EAGER": "true",
        "API_KEYS": json.dumps(API_KEYS),
        "RATE_LIMIT_PER_MINUTE": "1000",
        "UPLOAD_DIR": str(ROOT / ".pytest-uploads"),
        "LLM_API_KEY": "test-not-used",
    }
)

from app.agents.llm import LLMResult  # noqa: E402
from app.core.config import AccessLevel, Principal, get_settings  # noqa: E402


def principal(name: str = "p", departments: tuple[str, ...] = (), clearance: str = "public") -> Principal:
    return Principal(name=name, departments=frozenset(departments), clearance=AccessLevel[clearance.upper()])


class ScriptedLLM:
    """A deterministic stand-in for the chat model, keyed on the node that calls it.

    Overrides let a test script specific replies: {"verify_hallucination": [reply1, reply2]}.
    """

    def __init__(self, overrides: dict[str, list[str]] | None = None) -> None:
        self.overrides = {k: list(v) for k, v in (overrides or {}).items()}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        name: str = "llm",
    ) -> LLMResult:
        self.calls.append((name, messages))
        if self.overrides.get(name):
            return LLMResult(self.overrides[name].pop(0), model or "scripted")
        user = messages[-1]["content"]
        return LLMResult(
            json.dumps(self._default(name, user, messages))
            if json_mode or name != "generate"
            else self._answer(messages),
            model or "scripted",
        )

    def _default(self, name: str, user: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        lower = user.lower()
        if name == "rewrite_query":
            latest = user.split("Latest message:")[-1].strip()
            return {"standalone_question": latest, "search_queries": [latest]}
        if name == "route_intent":
            if any(w in lower for w in ("weather", "poem", "football", "recipe")):
                return {"intent": "out_of_scope", "reason": "unrelated"}
            if "how many" in lower and "refund window" in lower:
                return {"intent": "hybrid_both", "reason": "data and policy"}
            if "how many" in lower or "revenue" in lower:
                return {"intent": "structured_sql", "reason": "counting"}
            return {"intent": "unstructured_rag", "reason": "policy"}
        if name == "grade_context":
            count = len(re.findall(r"^\[\d+\] ", user, flags=re.M))
            question = user.split("\n", 1)[0].lower()
            words = [w for w in re.findall(r"[a-z0-9\-]{4,}", question) if w not in {"question", "what", "does"}]
            relevant = [
                i
                for i, block in enumerate(re.split(r"^\[\d+\] ", user, flags=re.M)[1:], start=1)
                if any(w in block.lower() for w in words)
            ]
            return {
                "relevant": relevant,
                "sufficient": bool(relevant) and count > 0,
                "missing": "" if relevant else "nothing relevant",
            }
        if name == "verify_hallucination":
            return {"unsupported": [], "grounded": True}
        if name == "sql.generate":
            if "refunded" in lower:
                return {
                    "sql": "select count(*) as refunded_orders from sales.orders where status = 'refunded'",
                    "explanation": "count",
                }
            return {"sql": None, "explanation": "not covered"}
        return {}

    @staticmethod
    def _answer(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        sources = system.split("Sources:\n", 1)[-1]
        first = re.search(r"^\[1\] (\[[^\]]+\])[^\n]*\n(.+?)(?:\n\n\[\d+\] |\Z)", sources, flags=re.S | re.M)
        if not first:
            return "No sources."
        body = first.group(2)
        sentence = re.split(r"(?<=[.!?])\s", body.replace("\n", " ").strip())[0]
        return f"{sentence} [1]"

    async def stream(self, messages, *, model=None, max_tokens=None):
        yield (await self.chat(messages, model=model, name="generate")).text


async def _db_available() -> bool:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect():
            return True
    except Exception:
        return False
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def migrated() -> None:
    if not asyncio.run(_db_available()):
        pytest.skip("PostgreSQL with pgvector is not reachable")
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


@pytest.fixture
async def db(migrated: None) -> AsyncIterator[None]:
    from sqlalchemy import text

    from app.db.session import dispose, get_engine

    async with get_engine().begin() as conn:
        await conn.execute(text("truncate documents, ingest_jobs, chat_messages, audit_logs restart identity cascade"))
    yield
    await dispose()


@pytest.fixture
async def redis() -> AsyncIterator[Any]:
    from redis.asyncio import Redis

    client = Redis.from_url(get_settings().redis_url)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis is not reachable")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


ACLS = {
    "product-faq.txt": (None, 0),
    "company-handbook.md": (None, 1),
    "onboarding-guide.pdf": ("engineering", 1),
}


@pytest.fixture
async def corpus(db: None) -> dict[str, str]:
    """The three bundled example documents, ingested with their access rules."""
    from app.db.session import get_sessionmaker
    from app.ingestion.pipeline import ingest_file
    from app.retrieval.embeddings import build_embedder

    settings = get_settings()
    embedder = build_embedder(settings)
    ids: dict[str, str] = {}
    for name, (department, level) in ACLS.items():
        async with get_sessionmaker()() as session:
            result = await ingest_file(
                session, embedder, settings, EXAMPLES / name, department_id=department, access_level=level
            )
            ids[name] = result.document_id
    return ids
