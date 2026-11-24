"""The agent as a LangGraph state machine.

rewrite_query -> route_intent --out_of_scope--------------------------------> refuse
                              \\-> retrieve_context -> grade_context
                                    ^                    |  sufficient -> generate_with_citations
                                    |                    |                     |
                                    '-- rewrite_query <--' retry (max_rewrites) v
                                                         |            verify_hallucination
                                                         '-- nothing -> refuse  |   grounded -> END
                                                                                |   regenerate (max_regenerations)
                                                                                '-> refuse
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import AgentNodes
from app.agents.state import AgentState
from app.core.config import Settings


def build_graph(nodes: AgentNodes, settings: Settings) -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("route_intent", nodes.route_intent)
    graph.add_node("retrieve_context", nodes.retrieve_context)
    graph.add_node("grade_context", nodes.grade_context)
    graph.add_node("generate_with_citations", nodes.generate_with_citations)
    graph.add_node("verify_hallucination", nodes.verify_hallucination)
    graph.add_node("refuse", nodes.refuse)

    def after_route(state: AgentState) -> str:
        return "refuse" if state["intent"] == "out_of_scope" else "retrieve_context"

    def after_grade(state: AgentState) -> str:
        has_evidence = bool(state.get("contexts")) or bool(state.get("sql_result"))
        if not state.get("missing"):
            return "generate_with_citations"
        if state.get("rewrites", 0) < settings.max_rewrites and state["intent"] != "structured_sql":
            return "rewrite_query"
        return "generate_with_citations" if has_evidence else "refuse"

    def after_generate(state: AgentState) -> str:
        return "refuse" if state.get("refusal") else "verify_hallucination"

    def after_verify(state: AgentState) -> str:
        if not state.get("unsupported_claims"):
            return END
        if state.get("regenerations", 0) < settings.max_regenerations:
            return "generate_with_citations"
        return "refuse"

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "route_intent")
    graph.add_conditional_edges("route_intent", after_route, ["refuse", "retrieve_context"])
    graph.add_edge("retrieve_context", "grade_context")
    graph.add_conditional_edges("grade_context", after_grade, ["generate_with_citations", "rewrite_query", "refuse"])
    graph.add_conditional_edges("generate_with_citations", after_generate, ["refuse", "verify_hallucination"])
    graph.add_conditional_edges("verify_hallucination", after_verify, [END, "generate_with_citations", "refuse"])
    graph.add_edge("refuse", END)
    return graph.compile()
