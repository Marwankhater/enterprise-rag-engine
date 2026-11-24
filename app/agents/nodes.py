"""Graph nodes. Each takes the state and returns only the keys it changes.

Model calls use JSON mode with small, fixed schemas, and every reply is validated before use; a
malformed reply degrades to a safe default (search the documents, refuse to answer) instead of
raising inside the graph.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.llm import LLM, LLMError, LLMResult
from app.agents.state import AgentState, Citation, Intent
from app.core.config import Settings
from app.core.observability import span
from app.retrieval.hybrid_search import ChildHit, ContextBlock, HybridSearcher
from app.retrieval.reranker import Reranker, rerank
from app.retrieval.sql_tool import SQLResult, SQLTool

log = logging.getLogger(__name__)

INTENTS: tuple[Intent, ...] = ("unstructured_rag", "structured_sql", "hybrid_both", "out_of_scope")
_ARABIC = re.compile(r"[؀-ۿ]")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_CITATION = re.compile(r"\[([^\[\]|]{2,200})\|\s*([^\[\]]{1,120})\]")

REWRITE_PROMPT = """You turn the user's latest message into a standalone search question.
Resolve pronouns and references using the conversation. Keep names, numbers, product terms and
the user's language. If the message asks several things, also list each part as its own query.
{feedback}
Reply with JSON: {{"standalone_question": "...", "search_queries": ["...", "..."]}}"""

ROUTE_PROMPT = """Classify the question for a company assistant that has two sources:
1. Company documents: policies, handbook, product FAQ, onboarding, runbooks, playbooks.
2. A sales database: customers (country, region, plan, seats), products and prices, orders (date,
   billing, status paid/refunded/cancelled, totals) and order items.
Labels:
- unstructured_rag: answered from documents (policies, how things work, rules, definitions).
- structured_sql: answered by counting, summing or listing sales data.
- hybrid_both: needs a figure from the sales data and a rule or fact from the documents.
- out_of_scope: unrelated to the company (general knowledge, chit-chat, opinions, general coding help).
Internal tools, commands, onboarding steps, runbooks and engineering practices are company topics,
so a question like "what does make bootstrap do?" is unstructured_rag.
Reply with JSON: {"intent": "<label>", "reason": "<few words>"}"""

GRADE_PROMPT = """You check whether retrieved evidence can answer a question.
For each numbered source, decide if it contains information that helps answer the question.
Then decide if the relevant sources together are enough for a complete answer.
Reply with JSON: {"relevant": [<source numbers>], "sufficient": true|false,
"missing": "<what is still missing, empty if nothing>"}"""

ANSWER_PROMPT = """You answer questions for employees using only the sources below.
Rules:
- Use only facts stated in the sources. Do not add outside knowledge or assumptions.
- After each sentence that uses a source, cite it with the source number in square brackets,
  for example [1] or [1][2]. Use only the numbers listed below.
- Copy numbers, amounts, dates and limits exactly as they appear in the sources.
- If the sources answer only part of the question, answer that part and say plainly what is not
  covered.
- Answer in {language}. Be concise: a short paragraph or a few bullet points.
{feedback}
Sources:
{sources}"""

VERIFY_PROMPT = """You verify an answer against its sources.
List every factual claim in the answer (each number, amount, date, limit, name or rule) and check
whether the sources state it. Paraphrase is fine; a claim the sources do not state is unsupported.
Reply with JSON: {"unsupported": ["<claim>", ...], "grounded": true|false}
grounded is true only when the unsupported list is empty."""

REFUSALS = {
    "en": {
        "out_of_scope": "I can only help with questions about the company's documents and sales data.",
        "not_found": (
            "I couldn't find this in the documents and data you have access to. "
            "If you think it should be there, ask the document owner or your manager."
        ),
        "unverified": (
            "I found related information but couldn't verify an answer against it, "
            "so I won't guess. Try asking a narrower question."
        ),
        "error": "Something went wrong while answering. Please try again.",
    },
    "ar": {
        "out_of_scope": "أستطيع المساعدة فقط في الأسئلة المتعلقة بمستندات الشركة وبيانات المبيعات.",
        "not_found": (
            "لم أجد هذه المعلومة في المستندات والبيانات المتاحة لك. "
            "إذا كنت ترى أنها يجب أن تكون موجودة، تواصل مع صاحب المستند أو مديرك."
        ),
        "unverified": "وجدت معلومات قريبة لكن لم أستطع التحقق من الإجابة منها، لذلك لن أخمن. جرّب سؤالًا أكثر تحديدًا.",
        "error": "حدث خطأ أثناء الإجابة. حاول مرة أخرى.",
    },
}


def detect_language(text: str) -> str:
    return "ar" if len(_ARABIC.findall(text)) > len(re.findall(r"[A-Za-z]", text)) else "en"


def _usage(result: LLMResult, node: str) -> list[dict[str, Any]]:
    return [
        {
            "node": node,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
    ]


def numbered_sources(contexts: list[ContextBlock], sql: SQLResult | None) -> tuple[str, list[str]]:
    """Render sources for a prompt, and return the citation label of each one."""
    parts: list[str] = []
    labels: list[str] = []
    for i, block in enumerate(contexts, start=1):
        label = block.citation
        labels.append(label)
        section = f" (section: {block.heading_path})" if block.heading_path else ""
        parts.append(f"[{i}] {label}{section}\n{block.content}")
    if sql is not None and sql.sql and not sql.error:
        label = "[Sales database | query]"
        labels.append(label)
        parts.append(f"[{len(labels)}] {label}\n{sql.as_context()}")
    return "\n\n".join(parts), labels


_MARKER = re.compile(r"\[(\d{1,2})\]")


def resolve_markers(answer: str, labels: list[str]) -> tuple[str, list[int]]:
    """Replace the model's [n] markers with the readable labels, and list which sources were cited.

    Asking the model for numbers is more reliable than asking it to copy a long label exactly.
    Markers pointing at a source that does not exist are dropped, so they cannot pass as citations.
    """
    used: list[int] = []

    def swap(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if 0 <= index < len(labels):
            if index not in used:
                used.append(index)
            return labels[index]
        return ""

    text = _MARKER.sub(swap, answer)
    for label in labels:  # collapse "[A][A]" left by "[1][1]"
        text = text.replace(label + label, label)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text).strip(), sorted(used)


def numbers_missing_from_sources(answer: str, sources: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the sources. A cheap check run before the LLM one."""
    normalized = sources.replace(",", "")
    missing = []
    for raw in _NUMBER.findall(_CITATION.sub("", answer)):
        value = raw.replace(",", "").rstrip("%")
        if value and value not in normalized and raw not in sources:
            missing.append(raw)
    return missing


class AgentNodes:
    def __init__(
        self,
        llm: LLM,
        searcher: HybridSearcher,
        reranker: Reranker,
        sql_tool: SQLTool,
        sessions: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.searcher = searcher
        self.reranker = reranker
        self.sql_tool = sql_tool
        self.sessions = sessions
        self.settings = settings

    async def _json(
        self, messages: list[dict[str, str]], name: str, *, grader: bool = True, max_tokens: int = 400
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        model = self.settings.grader_model if grader else None
        try:
            result = await self.llm.chat(messages, model=model, json_mode=True, max_tokens=max_tokens, name=name)
            return result.json(), _usage(result, name)
        except LLMError as exc:
            log.warning("model reply unusable", extra={"node": name, "error": str(exc)[:300]})
            return None, []

    # 1. rewrite_query -------------------------------------------------------------------------
    async def rewrite_query(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        history = state.get("history") or []
        rewrites = state.get("rewrites", 0)
        retrying = "contexts" in state  # we have been through retrieval and grading already
        update: dict[str, Any] = {
            "language": state.get("language") or detect_language(question),
            "steps": ["rewrite_query"],
        }
        if not history and not retrying:
            update.update(standalone_question=question, search_queries=[question])
            return update
        feedback = ""
        if retrying:
            feedback = (
                f"A search for {state.get('standalone_question')!r} did not find enough. "
                f"Missing: {state.get('missing') or 'unknown'}. Write different search queries "
                "using synonyms and the specific terms a policy document would use."
            )
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-self.settings.history_turns :])
        messages = [
            {"role": "system", "content": REWRITE_PROMPT.format(feedback=feedback)},
            {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"},
        ]
        data, usage = await self._json(messages, "rewrite_query")
        standalone = str((data or {}).get("standalone_question") or question).strip()
        queries = [q for q in (data or {}).get("search_queries", []) if isinstance(q, str) and q.strip()][:4]
        update.update(standalone_question=standalone, search_queries=queries or [standalone], usage=usage)
        if retrying:
            update["rewrites"] = rewrites + 1
        return update

    # 2. route_intent --------------------------------------------------------------------------
    async def route_intent(self, state: AgentState) -> dict[str, Any]:
        if state.get("intent"):  # a retry keeps the original route
            return {"steps": ["route_intent"]}
        messages = [
            {"role": "system", "content": ROUTE_PROMPT},
            {"role": "user", "content": state["standalone_question"]},
        ]
        data, usage = await self._json(messages, "route_intent", max_tokens=80)
        intent = (data or {}).get("intent")
        if intent not in INTENTS:
            intent = "unstructured_rag"
        if intent in ("structured_sql", "hybrid_both") and not self.sql_tool.allowed_for(state["principal"]):
            intent = "unstructured_rag"  # no sales access: answer from documents or refuse
        return {"intent": intent, "steps": ["route_intent"], "usage": usage}

    # 3. retrieve_context ----------------------------------------------------------------------
    async def retrieve_context(self, state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        principal = state["principal"]
        contexts: list[ContextBlock] = []
        sql_result: SQLResult | None = None
        if intent in ("unstructured_rag", "hybrid_both"):
            async with self.sessions() as session:
                merged: dict[str, ChildHit] = {}
                for query in state.get("search_queries") or [state["standalone_question"]]:
                    for hit in await self.searcher.search_children(session, query, principal, limit=20):
                        known = merged.get(hit.child_id)
                        if known is None or hit.rrf_score > known.rrf_score:
                            merged[hit.child_id] = hit
                candidates = sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)[:30]
                top = await rerank(
                    self.reranker, state["standalone_question"], candidates, self.settings.reranker_top_n
                )
                if self.reranker.name != "passthrough":
                    top = [h for h in top if (h.rerank_score or 0.0) >= self.settings.min_rerank_score]
                contexts = await self.searcher.parents_for(session, top, principal, self.settings.context_parents)
        if intent in ("structured_sql", "hybrid_both"):
            sql_result = await self.sql_tool.run(state["standalone_question"], principal)
        return {"contexts": contexts, "sql_result": sql_result, "steps": ["retrieve_context"]}

    # 4. grade_context -------------------------------------------------------------------------
    async def grade_context(self, state: AgentState) -> dict[str, Any]:
        contexts = state.get("contexts") or []
        sql = state.get("sql_result")
        sources, _ = numbered_sources(contexts, sql)
        if not sources:
            return {"contexts": [], "missing": "no sources found", "steps": ["grade_context"]}
        messages = [
            {"role": "system", "content": GRADE_PROMPT},
            {"role": "user", "content": f"Question: {state['standalone_question']}\n\n{sources}"},
        ]
        data, usage = await self._json(messages, "grade_context", max_tokens=200)
        if data is None:  # grader unavailable: keep everything and let verification catch problems
            return {"missing": "", "steps": ["grade_context"]}
        relevant = {int(i) for i in data.get("relevant", []) if str(i).isdigit()}
        kept = [c for i, c in enumerate(contexts, start=1) if i in relevant]
        sql_index = len(contexts) + 1
        keep_sql = sql if (sql is not None and sql.sql and not sql.error and sql_index in relevant) else None
        sufficient = bool(data.get("sufficient")) and (kept or keep_sql is not None)
        missing = "" if sufficient else str(data.get("missing") or "the answer")
        return {
            "contexts": kept,
            "sql_result": keep_sql if sql is not None else None,
            "missing": missing,
            "steps": ["grade_context"],
            "usage": usage,
        }

    # 5. generate_with_citations ---------------------------------------------------------------
    async def generate_with_citations(self, state: AgentState) -> dict[str, Any]:
        contexts = state.get("contexts") or []
        sources, labels = numbered_sources(contexts, state.get("sql_result"))
        feedback = ""
        if state.get("unsupported_claims"):
            claims = "; ".join(state["unsupported_claims"][:5])
            feedback = (
                f"Your previous answer included claims the sources do not support: {claims}. "
                "Remove them or correct them from the sources."
            )
        language = "Arabic" if state.get("language") == "ar" else "English"
        prompt = ANSWER_PROMPT.format(language=language, feedback=feedback, sources=sources)
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": state["standalone_question"]}]
        try:
            result = await self.llm.chat(messages, max_tokens=700, name="generate")
        except LLMError:
            log.exception("generation failed")
            return {"refusal": "error", "steps": ["generate_with_citations"]}
        answer, used = resolve_markers(result.text.strip(), labels)
        return {
            "answer": answer,
            "citations": self._citations(used, contexts, labels),
            "regenerations": state.get("regenerations", -1) + 1,
            "steps": ["generate_with_citations"],
            "usage": _usage(result, "generate"),
        }

    @staticmethod
    def _citations(used_indexes: list[int], contexts: list[ContextBlock], labels: list[str]) -> list[Citation]:
        used: list[Citation] = []
        for i, label in enumerate(labels):
            if i not in used_indexes:
                continue
            if i < len(contexts):
                block = contexts[i]
                page = next((h.page for h in block.hits if h.page is not None), block.page_start)
                used.append(
                    Citation(
                        label=label,
                        title=block.title,
                        source=block.source,
                        page=page,
                        section=block.heading_path,
                        parent_id=block.parent_id,
                    )
                )
            else:
                used.append(
                    Citation(label=label, title="Sales database", source="sales", page=None, section="", parent_id=None)
                )
        return used

    # 6. verify_hallucination ------------------------------------------------------------------
    async def verify_hallucination(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        sources, _ = numbered_sources(state.get("contexts") or [], state.get("sql_result"))
        with span("verify.numbers"):
            unsupported = [f"the number {n}" for n in numbers_missing_from_sources(answer, sources)]
        usage: list[dict[str, Any]] = []
        if not unsupported:
            messages = [
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": f"Sources:\n{sources}\n\nAnswer:\n{answer}"},
            ]
            data, usage = await self._json(messages, "verify_hallucination", max_tokens=300)
            if data is None:
                unsupported = ["verification unavailable"]
            else:
                unsupported = [str(c) for c in data.get("unsupported", []) if str(c).strip()]
                if not data.get("grounded", not unsupported) and not unsupported:
                    unsupported = ["answer marked as not grounded"]
        if not state.get("citations") and answer:
            unsupported.append("the answer cites no source")
        return {"unsupported_claims": unsupported, "steps": ["verify_hallucination"], "usage": usage}

    # Terminal ---------------------------------------------------------------------------------
    async def refuse(self, state: AgentState) -> dict[str, Any]:
        reason = state.get("refusal")
        if reason is None:
            if state.get("intent") == "out_of_scope":
                reason = "out_of_scope"
            elif state.get("unsupported_claims"):
                reason = "unverified"
            else:
                reason = "not_found"
        language = state.get("language") or detect_language(state["question"])
        return {"answer": REFUSALS[language][reason], "citations": [], "refusal": reason, "steps": ["refuse"]}
