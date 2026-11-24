"""Long-lived services, built once at startup and shared by all requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.graph import build_graph
from app.agents.llm import LLM, OpenAICompatibleLLM
from app.agents.nodes import AgentNodes
from app.cache.semantic_cache import SemanticCache
from app.core.config import Settings
from app.db.session import get_sessionmaker
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import Reranker, build_reranker
from app.retrieval.sql_tool import SQLTool


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    llm: LLM
    reranker: Reranker
    searcher: HybridSearcher
    sql_tool: SQLTool
    cache: SemanticCache
    graph: Any
    sessions: async_sessionmaker[AsyncSession]


async def build_services(
    settings: Settings,
    redis: Redis,
    *,
    llm: LLM | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> Services:
    embedder = embedder or build_embedder(settings)
    llm = llm or OpenAICompatibleLLM(settings)
    reranker = reranker or build_reranker(settings.rerank_enabled, settings.reranker_model)
    searcher = HybridSearcher(embedder, settings)
    sql_tool = SQLTool(llm, settings)
    sessions = get_sessionmaker()
    nodes = AgentNodes(llm, searcher, reranker, sql_tool, sessions, settings)
    return Services(
        settings=settings,
        embedder=embedder,
        llm=llm,
        reranker=reranker,
        searcher=searcher,
        sql_tool=sql_tool,
        cache=SemanticCache(redis, settings),
        graph=build_graph(nodes, settings),
        sessions=sessions,
    )


def get_services(request: Request) -> Services:
    return request.app.state.services
