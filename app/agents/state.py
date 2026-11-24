"""State carried through the agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from app.core.config import Principal
from app.retrieval.hybrid_search import ContextBlock
from app.retrieval.sql_tool import SQLResult

Intent = Literal["unstructured_rag", "structured_sql", "hybrid_both", "out_of_scope"]
RefusalReason = Literal["out_of_scope", "not_found", "unverified", "error"]


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Citation(TypedDict):
    label: str
    title: str
    source: str
    page: int | None
    section: str


class AgentState(TypedDict, total=False):
    # input
    question: str
    history: list[Message]
    principal: Principal
    language: Literal["ar", "en"]
    # query understanding
    standalone_question: str
    search_queries: list[str]
    intent: Intent
    # evidence
    contexts: list[ContextBlock]
    sql_result: SQLResult | None
    missing: str
    rewrites: int
    # answer
    answer: str
    citations: list[Citation]
    unsupported_claims: list[str]
    regenerations: int
    refusal: RefusalReason | None
    # bookkeeping, appended by every node
    steps: Annotated[list[str], operator.add]
    usage: Annotated[list[dict[str, Any]], operator.add]
