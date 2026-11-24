"""Chat endpoint: semantic cache, then the agent graph, streamed as Server-Sent Events.

Events, in order:
  status  {"step": "...", "label": "..."}   one per graph step (or "cache_hit")
  token   {"text": "..."}                   the answer, sent only after it passed verification
  final   {...}                             message id, citations, intent, refusal, timings, usage
  error   {"detail": "..."}                 if something failed after the stream started
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.agents.state import AgentState, Message
from app.api.v1.deps import Services, get_services
from app.core.config import Principal
from app.core.observability import request_id_var, span
from app.core.security import rate_limited
from app.db.models import AuditLog, ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])
log = logging.getLogger(__name__)

STEP_LABELS = {
    "rewrite_query": "Understanding the question",
    "route_intent": "Choosing sources",
    "retrieve_context": "Searching documents and data",
    "grade_context": "Checking the evidence",
    "generate_with_citations": "Writing the answer",
    "verify_hallucination": "Verifying claims against sources",
    "refuse": "Preparing the reply",
}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{8,100}$")
    stream: bool = True


class ChatResponse(BaseModel):
    message_id: str
    session_id: str
    answer: str
    citations: list[dict[str, Any]]
    intent: str | None
    refusal: str | None
    cached: bool
    latency_ms: float
    usage: dict[str, int]


async def load_history(services: Services, session_id: str, principal: Principal) -> list[Message]:
    async with services.sessions() as session:
        rows = (
            await session.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id, ChatMessage.principal == principal.name)
                .order_by(ChatMessage.created_at.desc())
                .limit(services.settings.history_turns * 2)
            )
        ).all()
    return [Message(role=r.role, content=r.content) for r in reversed(rows)]  # type: ignore[typeddict-item]


async def save_turn(
    services: Services, session_id: str, principal: Principal, question: str, result: dict[str, Any]
) -> str:
    asked_at = datetime.now(UTC)
    async with services.sessions() as session:
        session.add(
            ChatMessage(
                session_id=session_id, principal=principal.name, role="user", content=question, created_at=asked_at
            )
        )
        assistant = ChatMessage(
            session_id=session_id,
            principal=principal.name,
            role="assistant",
            created_at=asked_at + timedelta(milliseconds=1),
            content=result["answer"],
            meta={k: result[k] for k in ("citations", "intent", "refusal", "cached", "latency_ms", "usage")},
        )
        session.add(assistant)
        session.add(
            AuditLog(
                principal=principal.name,
                action="chat",
                request_id=request_id_var.get(),
                detail={
                    "session_id": session_id,
                    "intent": result["intent"],
                    "refusal": result["refusal"],
                    "cached": result["cached"],
                    "sources": [c["source"] for c in result["citations"]],
                },
            )
        )
        await session.commit()
        return str(assistant.id)


def _chunks(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text) or [text]


async def run_chat(services: Services, principal: Principal, request: ChatRequest) -> AsyncIterator[tuple[str, dict]]:
    started = time.perf_counter()
    session_id = request.session_id or uuid.uuid4().hex
    history = await load_history(services, session_id, principal)
    question = request.question.strip()
    vector: list[float] | None = None

    with span("chat", session_id=session_id, first_turn=not history):
        if not history:
            vector = await services.embedder.embed_query(question)
            hit = await services.cache.lookup(principal, question, vector)
            if hit is not None:
                yield (
                    "status",
                    {"step": "cache_hit", "label": "Answered from cache", "similarity": round(hit.similarity, 4)},
                )
                result = {
                    **hit.payload,
                    "cached": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                for piece in _chunks(result["answer"]):
                    yield "token", {"text": piece}
                result["message_id"] = await save_turn(services, session_id, principal, question, result)
                yield "final", {"session_id": session_id, **result}
                return

        state: AgentState = {"question": question, "history": history, "principal": principal, "steps": [], "usage": []}
        final: dict[str, Any] = dict(state)
        seen = 0
        async for snapshot in services.graph.astream(state, stream_mode="values"):
            final = snapshot
            steps = snapshot.get("steps", [])
            for step in steps[seen:]:
                yield "status", {"step": step, "label": STEP_LABELS.get(step, step)}
            seen = len(steps)

    usage = {
        "input_tokens": sum(u["input_tokens"] for u in final.get("usage", [])),
        "output_tokens": sum(u["output_tokens"] for u in final.get("usage", [])),
    }
    result = {
        "answer": final.get("answer", ""),
        "citations": list(final.get("citations", [])),
        "intent": final.get("intent"),
        "refusal": final.get("refusal"),
        "cached": False,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "usage": usage,
    }
    for piece in _chunks(result["answer"]):
        yield "token", {"text": piece}
    result["message_id"] = await save_turn(services, session_id, principal, question, result)
    if vector is not None and not result["refusal"]:
        await services.cache.store(
            principal, question, vector, {k: result[k] for k in ("answer", "citations", "intent", "refusal")}
        )
    log.info(
        "chat answered",
        extra={"intent": result["intent"], "refusal": result["refusal"], "latency_ms": result["latency_ms"], **usage},
    )
    yield "final", {"session_id": session_id, **result}


@router.post("", response_model=None, summary="Ask a question (SSE stream, or JSON with stream=false)")
async def chat(
    body: ChatRequest,
    principal: Annotated[Principal, Depends(rate_limited)],
    services: Annotated[Services, Depends(get_services)],
) -> EventSourceResponse | ChatResponse:
    if not body.stream:
        final: dict[str, Any] | None = None
        async for event, data in run_chat(services, principal, body):
            if event == "final":
                final = data
        assert final is not None
        return ChatResponse(**final)

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            async for event, data in run_chat(services, principal, body):
                yield {"event": event, "data": json.dumps(data, ensure_ascii=False, default=str)}
        except Exception:
            log.exception("chat stream failed")
            yield {
                "event": "error",
                "data": json.dumps({"detail": "Internal error", "request_id": request_id_var.get()}),
            }

    return EventSourceResponse(events(), ping=15)


@router.get("/{session_id}/history", summary="Messages of one of your chat sessions")
async def history(
    session_id: str,
    principal: Annotated[Principal, Depends(rate_limited)],
    services: Annotated[Services, Depends(get_services)],
) -> list[dict[str, Any]]:
    async with services.sessions() as session:
        rows = (
            await session.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id, ChatMessage.principal == principal.name)
                .order_by(ChatMessage.created_at)
            )
        ).all()
    if not rows:
        raise HTTPException(404, "Session not found")
    return [
        {"id": str(r.id), "role": r.role, "content": r.content, "meta": r.meta, "created_at": r.created_at.isoformat()}
        for r in rows
    ]
