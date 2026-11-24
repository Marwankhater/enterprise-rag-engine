"""The agent graph with a scripted model: routing, refusals, access control and self-correction."""

from __future__ import annotations

import json
from typing import Any

from app.agents.graph import build_graph
from app.agents.nodes import AgentNodes, numbers_missing_from_sources
from app.core.config import get_settings
from app.db.session import get_sessionmaker
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import PassthroughReranker
from app.retrieval.sql_tool import SQLTool
from tests.conftest import ScriptedLLM, principal

ENGINEER = principal("engineer", ("engineering",), "internal")
VISITOR = principal("visitor", clearance="public")


async def run(question: str, who, llm: ScriptedLLM | None = None, history=None) -> dict[str, Any]:
    settings = get_settings()
    llm = llm or ScriptedLLM()
    embedder = build_embedder(settings)
    sql_tool = SQLTool(llm, settings)
    nodes = AgentNodes(
        llm, HybridSearcher(embedder, settings), PassthroughReranker(), sql_tool, get_sessionmaker(), settings
    )
    try:
        return await build_graph(nodes, settings).ainvoke(
            {"question": question, "history": history or [], "principal": who, "steps": [], "usage": []}
        )
    finally:
        await sql_tool.engine.dispose()


async def test_answer_cites_a_readable_source(corpus: dict[str, str]) -> None:
    state = await run("How fast must an on-call engineer acknowledge a page?", ENGINEER)
    assert state.get("refusal") is None
    assert state["citations"] and state["citations"][0]["label"] in state["answer"]
    assert state["steps"][:3] == ["rewrite_query", "route_intent", "retrieve_context"]
    assert state["steps"][-1] == "verify_hallucination"


async def test_out_of_scope_is_refused_without_retrieval(corpus: dict[str, str]) -> None:
    state = await run("Write me a poem about the weather", ENGINEER)
    assert state["refusal"] == "out_of_scope"
    assert "retrieve_context" not in state["steps"]


async def test_restricted_document_is_not_found_rather_than_forbidden(corpus: dict[str, str]) -> None:
    # The onboarding guide is engineering-only. A visitor gets the same reply as for a missing fact,
    # so the reply does not reveal that a restricted document exists.
    state = await run("What does make bootstrap install for new engineers?", VISITOR)
    assert state["refusal"] == "not_found"
    assert all(c["source"] != "onboarding-guide.pdf" for c in state.get("citations", []))


async def test_insufficient_evidence_triggers_rewrite_then_refusal(corpus: dict[str, str]) -> None:
    llm = ScriptedLLM({"grade_context": [json.dumps({"relevant": [], "sufficient": False, "missing": "x"})] * 3})
    state = await run("What is the parking reimbursement rate?", ENGINEER, llm)
    assert state["steps"].count("retrieve_context") == get_settings().max_rewrites + 1
    assert state["rewrites"] == get_settings().max_rewrites
    assert state["refusal"] == "not_found"


async def test_unsupported_claims_are_regenerated_then_refused(corpus: dict[str, str]) -> None:
    bad = json.dumps({"unsupported": ["a claim"], "grounded": False})
    llm = ScriptedLLM({"verify_hallucination": [bad, bad]})
    state = await run("How fast must an on-call engineer acknowledge a page?", ENGINEER, llm)
    assert state["steps"].count("generate_with_citations") == 2
    assert state["refusal"] == "unverified"
    regen_prompt = [m for name, m in llm.calls if name == "generate"][-1][0]["content"]
    assert "a claim" in regen_prompt


async def test_follow_up_is_rewritten_with_history(corpus: dict[str, str]) -> None:
    llm = ScriptedLLM(
        {
            "rewrite_query": [
                json.dumps(
                    {
                        "standalone_question": "What is the Pro plan API rate limit?",
                        "search_queries": ["Pro plan API rate limit"],
                    }
                )
            ]
        }
    )
    history = [
        {"role": "user", "content": "What plans are there?"},
        {"role": "assistant", "content": "Starter, Pro and Enterprise."},
    ]
    state = await run("And what is its rate limit?", VISITOR, llm, history)
    assert state["standalone_question"] == "What is the Pro plan API rate limit?"
    rewrite_input = next(m for name, m in llm.calls if name == "rewrite_query")[-1]["content"]
    assert "Starter, Pro and Enterprise." in rewrite_input


async def test_sql_route_answers_from_the_sales_database(corpus: dict[str, str]) -> None:
    state = await run("How many refunded orders do we have?", ENGINEER)
    assert state["intent"] == "structured_sql"
    assert state["sql_result"] is not None and state["sql_result"].columns == ["refunded_orders"]


async def test_visitor_cannot_reach_sales_data(corpus: dict[str, str]) -> None:
    state = await run("How many refunded orders do we have?", VISITOR)
    assert state["intent"] == "unstructured_rag"
    assert state.get("sql_result") is None


def test_number_check_flags_invented_figures() -> None:
    sources = "Refunds are processed within 5 to 7 business days. Pro is $29 per user per month."
    assert numbers_missing_from_sources("Pro costs $29 [FAQ | Plans].", sources) == []
    assert numbers_missing_from_sources("Pro costs $35 and refunds take 10 days.", sources) == ["35", "10"]
