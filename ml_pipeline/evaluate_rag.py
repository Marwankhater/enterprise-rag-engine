"""Evaluation against tests/golden_dataset.json, in two stages.

retrieval   No model calls. For every answerable case, checks whether the passage that holds the
            answer is retrieved, for dense only, sparse only, hybrid, and hybrid plus reranker.
            Reports hit@1, hit@5 and MRR@10, and counts access-control leaks (a forbidden document
            retrieved for a case's principal). Cheap enough to run on every push.

end-to-end  Runs the full agent for every case, then scores it:
            * Ragas faithfulness, context recall and answer relevancy on answered cases,
              with the judge model given by EVAL_JUDGE_MODEL;
            * refusal accuracy (unanswerable and restricted questions must be refused, others not);
            * SQL accuracy (the exact value from the seeded database appears in the answer);
            * access-control leaks in citations, latency percentiles and token use.
            With --gate the process exits non-zero when a threshold is missed, which fails CI.

    python -m ml_pipeline.evaluate_rag retrieval
    python -m ml_pipeline.evaluate_rag end-to-end --gate

Both need a database with migrations applied, the sales seed and the example corpus ingested
(alembic upgrade head; python -m scripts.seed_sales; python -m scripts.ingest_examples).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import AccessLevel, Principal, Settings, get_settings
from app.core.observability import configure_logging
from app.db.session import dispose, get_sessionmaker
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.hybrid_search import ChildHit, HybridSearcher
from app.retrieval.reranker import PassthroughReranker, Reranker, build_reranker, rerank

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden_dataset.json"
REPORTS = ROOT / "reports"

THRESHOLDS = {
    "faithfulness": float(os.environ.get("GATE_FAITHFULNESS", "0.85")),
    "context_recall": float(os.environ.get("GATE_CONTEXT_RECALL", "0.80")),
    "refusal_accuracy": float(os.environ.get("GATE_REFUSAL_ACCURACY", "0.85")),
    "rbac_leaks": 0,
}


def load_golden(path: Path = GOLDEN) -> tuple[dict[str, Principal], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    principals = {
        name: Principal(
            name=f"eval:{name}",
            departments=frozenset(spec.get("departments", [])),
            clearance=AccessLevel[spec.get("clearance", "public").upper()],
        )
        for name, spec in data["principals"].items()
    }
    return principals, data["cases"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_reference(text: str, phrases: list[str]) -> bool:
    body = _norm(text)
    return any(_norm(p) in body for p in phrases)


# Retrieval -----------------------------------------------------------------------------------


@dataclass
class RetrievalScores:
    hits_at_1: int = 0
    hits_at_5: int = 0
    reciprocal_ranks: list[float] = field(default_factory=list)

    def add(self, rank: int | None) -> None:
        self.hits_at_1 += int(rank == 1)
        self.hits_at_5 += int(rank is not None and rank <= 5)
        self.reciprocal_ranks.append(1.0 / rank if rank and rank <= 10 else 0.0)

    def summary(self) -> dict[str, float]:
        n = max(len(self.reciprocal_ranks), 1)
        return {
            "hit@1": round(self.hits_at_1 / n, 3),
            "hit@5": round(self.hits_at_5 / n, 3),
            "mrr@10": round(sum(self.reciprocal_ranks) / n, 3),
            "cases": len(self.reciprocal_ranks),
        }


def first_relevant_rank(hits: list[ChildHit], parents: dict[str, str], case: dict[str, Any]) -> int | None:
    """Rank (1-based) of the first retrieved parent from an expected source that holds a reference phrase."""
    seen: list[str] = []
    for hit in hits:
        if hit.parent_id in seen:
            continue
        seen.append(hit.parent_id)
        if hit.source in case["expected_sources"] and contains_reference(
            parents[hit.parent_id], case["reference_contexts"]
        ):
            return len(seen)
    return None


async def evaluate_retrieval(settings: Settings, embedder: Embedder, reranker: Reranker) -> dict[str, Any]:
    from sqlalchemy import text

    principals, cases = load_golden()
    searcher = HybridSearcher(embedder, settings)
    modes: dict[str, RetrievalScores] = {m: RetrievalScores() for m in ("dense", "sparse", "hybrid")}
    reranked_name = f"hybrid+rerank ({reranker.name})"
    if not isinstance(reranker, PassthroughReranker):
        modes[reranked_name] = RetrievalScores()
    leaks: list[dict[str, Any]] = []
    sessions = get_sessionmaker()
    async with sessions() as session:
        parents = {str(r.id): r.content for r in (await session.execute(text("select id, content from parent_chunks")))}
        if not parents:
            raise SystemExit("No documents in the database. Run: python -m scripts.ingest_examples")
        for case in cases:
            who = principals[case["principal"]]
            question = case["question"]
            if case.get("forbidden_sources"):
                for hit in await searcher.search_children(session, question, who, limit=30):
                    if hit.source in case["forbidden_sources"]:
                        leaks.append({"case": case["id"], "source": hit.source})
            if case["expect"] != "answer" or not case.get("reference_contexts") or case.get("history"):
                continue
            hits = {
                m: await searcher.search_children(session, question, who, limit=30, mode=m)
                for m in ("dense", "sparse", "hybrid")
            }
            for mode in ("dense", "sparse", "hybrid"):
                modes[mode].add(first_relevant_rank(hits[mode], parents, case))
            if reranked_name in modes:
                reranked = await rerank(reranker, question, hits["hybrid"], top_n=30)
                modes[reranked_name].add(first_relevant_rank(reranked, parents, case))
    await dispose()
    return {
        "embedding_model": embedder.model_name,
        "modes": {m: s.summary() for m, s in modes.items()},
        "rbac_leaks": leaks,
    }


# End to end ----------------------------------------------------------------------------------


def _number_forms(value: Any) -> list[str]:
    if isinstance(value, (int, float)) or re.fullmatch(r"-?\d+(\.\d+)?", str(value)):
        number = float(value)
        forms = {f"{number:,.2f}", f"{number:.2f}"}
        if number.is_integer():
            forms |= {f"{int(number):,}", str(int(number))}
        return sorted(forms)
    return [str(value)]


def sql_value_in_answer(answer: str, value: Any) -> bool:
    plain = answer.replace("$", "")
    return any(form in plain or form.replace(",", "") in plain.replace(",", "") for form in _number_forms(value))


def stratified_sample(cases: list[dict[str, Any]], size: int, seed: int = 13) -> list[dict[str, Any]]:
    """A fixed, category-balanced subset, so a smaller run still covers every kind of question."""
    import random

    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)
    rng = random.Random(seed)
    for group in by_category.values():
        rng.shuffle(group)
    picked: list[dict[str, Any]] = []
    while len(picked) < min(size, len(cases)):
        for group in by_category.values():
            if group and len(picked) < size:
                picked.append(group.pop())
    return sorted(picked, key=lambda c: c["id"])


async def run_agent_cases(
    settings: Settings, embedder: Embedder, reranker: Reranker, sample: int | None
) -> list[dict[str, Any]]:
    from redis.asyncio import Redis

    from app.api.v1.deps import build_services

    principals, cases = load_golden()
    if sample:
        cases = stratified_sample(cases, sample)
    redis = Redis.from_url(settings.redis_url)
    services = await build_services(settings, redis, embedder=embedder, reranker=reranker)
    rows = []
    for case in cases:
        started = time.perf_counter()
        state = await services.graph.ainvoke(
            {
                "question": case["question"],
                "history": case.get("history") or [],
                "principal": principals[case["principal"]],
                "steps": [],
                "usage": [],
            }
        )
        contexts = [c.content for c in state.get("contexts") or []]
        if state.get("sql_result") is not None and state["sql_result"].sql:
            contexts.append(state["sql_result"].as_context())
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expect": case["expect"],
                "question": case["question"],
                "reference": case["reference_answer"],
                "answer": state.get("answer", ""),
                "contexts": contexts,
                "refusal": state.get("refusal"),
                "intent": state.get("intent"),
                "citations": [c["source"] for c in state.get("citations", [])],
                "forbidden": case.get("forbidden_sources", []),
                "sql_check": case.get("sql_check"),
                "latency_s": round(time.perf_counter() - started, 2),
                "tokens": sum(u["input_tokens"] + u["output_tokens"] for u in state.get("usage", [])),
                "unsupported": state.get("unsupported_claims") or [],
                "intent": state.get("intent"),
            }
        )
        row = rows[-1]
        print(
            f"{case['id']:>5} {case['category']:<16} intent={row['intent']!s:<17} refusal={row['refusal']!s:<12} "
            f"{row['latency_s']:>5}s",
            flush=True,
        )
        if row["unsupported"]:
            print(f"        flagged: {'; '.join(map(str, row['unsupported']))[:300]}", flush=True)
    await services.sql_tool.engine.dispose()
    await redis.aclose()
    await dispose()
    return rows


class _LocalEmbeddings:
    """Adapts the app's embedder to the interface Ragas expects for answer relevancy."""

    def __init__(self, embedder: Embedder) -> None:
        from ragas.embeddings import BaseRagasEmbeddings

        outer = embedder

        class Wrapped(BaseRagasEmbeddings):
            def embed_query(self, text: str) -> list[float]:
                return asyncio.run(outer.embed_query(text))

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return asyncio.run(outer.embed_documents(texts))

            async def aembed_query(self, text: str) -> list[float]:
                return await outer.embed_query(text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return await outer.embed_documents(texts)

        self.wrapped = Wrapped()


def ragas_scores(rows: list[dict[str, Any]], settings: Settings, embedder: Embedder) -> dict[str, float]:
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

    answered = [r for r in rows if r["expect"] == "answer" and not r["refusal"] and r["contexts"]]
    if not answered:
        return {"faithfulness": 0.0, "context_recall": 0.0, "answer_relevancy": 0.0}
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"],
                "reference": r["reference"],
            }
            for r in answered
        ]
    )
    # The judge can be a different provider from the app (for example to use a second free quota).
    app_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
    judge = ChatOpenAI(
        model=os.environ.get("EVAL_JUDGE_MODEL") or settings.llm_model,
        temperature=0,
        base_url=os.environ.get("EVAL_JUDGE_BASE_URL") or settings.llm_base_url,
        api_key=os.environ.get("EVAL_JUDGE_API_KEY") or app_key,
    )
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), LLMContextRecall(), ResponseRelevancy()],
        llm=LangchainLLMWrapper(judge),
        embeddings=_LocalEmbeddings(embedder).wrapped,
        run_config=RunConfig(max_workers=int(os.environ.get("EVAL_WORKERS", "2")), timeout=180, max_retries=6),
        show_progress=False,
    )
    frame = result.to_pandas()
    for i, row in enumerate(answered):
        row["ragas"] = {
            k: (None if frame[k].isna().iloc[i] else round(float(frame[k].iloc[i]), 3))
            for k in ("faithfulness", "context_recall", "answer_relevancy")
            if k in frame
        }
    return {
        k: round(float(frame[k].mean(skipna=True)), 3)
        for k in ("faithfulness", "context_recall", "answer_relevancy")
        if k in frame
    }


def deterministic_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    should_refuse = [r for r in rows if r["expect"] == "refuse"]
    should_answer = [r for r in rows if r["expect"] == "answer"]
    sql_cases = [r for r in rows if r["sql_check"]]
    leaks = [{"case": r["id"], "source": s} for r in rows for s in r["citations"] if s in r["forbidden"]]
    latencies = sorted(r["latency_s"] for r in rows)
    by_category: dict[str, dict[str, int]] = {}
    for r in rows:
        ok = bool(r["refusal"]) == (r["expect"] == "refuse")
        bucket = by_category.setdefault(r["category"], {"cases": 0, "correct_behaviour": 0})
        bucket["cases"] += 1
        bucket["correct_behaviour"] += int(ok)
    return {
        "cases": len(rows),
        "refusal_accuracy": round(sum(bool(r["refusal"]) for r in should_refuse) / max(len(should_refuse), 1), 3),
        "answer_rate": round(sum(not r["refusal"] for r in should_answer) / max(len(should_answer), 1), 3),
        "citation_rate": round(
            sum(bool(r["citations"]) for r in should_answer if not r["refusal"])
            / max(sum(not r["refusal"] for r in should_answer), 1),
            3,
        ),
        "sql_accuracy": round(
            sum(sql_value_in_answer(r["answer"], r["sql_check"]["value"]) for r in sql_cases) / max(len(sql_cases), 1),
            3,
        ),
        "rbac_leaks": leaks,
        "latency_p50_s": statistics.median(latencies) if latencies else 0,
        "latency_p95_s": latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0,
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows), 1) if rows else 0,
        "by_category": by_category,
    }


def check_gate(scores: dict[str, Any]) -> list[str]:
    failures = []
    for metric in ("faithfulness", "context_recall", "refusal_accuracy"):
        if scores.get(metric, 0.0) < THRESHOLDS[metric]:
            failures.append(f"{metric} {scores.get(metric, 0.0):.3f} < {THRESHOLDS[metric]:.2f}")
    if scores["rbac_leaks"]:
        failures.append(f"{len(scores['rbac_leaks'])} access-control leak(s)")
    return failures


# Reports -------------------------------------------------------------------------------------


def write_report(name: str, payload: dict[str, Any]) -> Path:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
    ]
    if "modes" in payload:
        lines += [
            f"Embedding model: `{payload['embedding_model']}`",
            "",
            "| Retrieval | hit@1 | hit@5 | MRR@10 | cases |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {m} | {s['hit@1']} | {s['hit@5']} | {s['mrr@10']} | {s['cases']} |" for m, s in payload["modes"].items()
        ]
        lines += ["", f"Access-control leaks in top 30: {len(payload['rbac_leaks'])}"]
    if "scores" in payload:
        s = payload["scores"]
        lines += ["| Metric | Value | Gate |", "|---|---|---|"]
        for key in (
            "faithfulness",
            "context_recall",
            "answer_relevancy",
            "refusal_accuracy",
            "answer_rate",
            "citation_rate",
            "sql_accuracy",
            "latency_p50_s",
            "latency_p95_s",
            "avg_tokens",
        ):
            if key in s:
                gate = f">= {THRESHOLDS[key]}" if key in THRESHOLDS else ""
                lines.append(f"| {key} | {s[key]} | {gate} |")
        lines += [
            f"| rbac_leaks | {len(s['rbac_leaks'])} | 0 |",
            "",
            "| Category | Cases | Correct answer/refusal |",
            "|---|---|---|",
        ]
        lines += [f"| {c} | {v['cases']} | {v['correct_behaviour']} |" for c, v in sorted(s["by_category"].items())]
        if payload.get("gate_failures") is not None:
            lines += [
                "",
                "Gate: "
                + ("passed" if not payload["gate_failures"] else "FAILED: " + "; ".join(payload["gate_failures"])),
            ]
    path = REPORTS / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["retrieval", "end-to-end"])
    parser.add_argument("--sample", type=int, help="a category-balanced subset of N cases (end-to-end)")
    parser.add_argument("--gate", action="store_true", help="exit 1 when a quality threshold is missed")
    parser.add_argument("--reranker", help="reranker model id or local path (default: settings)")
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()
    embedder = build_embedder(settings)
    reranker = build_reranker(settings.rerank_enabled, args.reranker or settings.reranker_model)

    if args.stage == "retrieval":
        payload = asyncio.run(evaluate_retrieval(settings, embedder, reranker))
        print(write_report("retrieval_eval", payload).read_text())
        return 1 if args.gate and payload["rbac_leaks"] else 0

    rows = asyncio.run(run_agent_cases(settings, embedder, reranker, args.sample))
    scores = deterministic_scores(rows)
    scores.update(ragas_scores(rows, settings, embedder))
    failures = check_gate(scores)
    payload = {
        "embedding_model": embedder.model_name,
        "reranker": reranker.name,
        "llm": settings.llm_model,
        "scores": scores,
        "gate_failures": failures,
        "cases": rows,
    }
    print(write_report("end_to_end_eval", payload).read_text())
    if args.gate and failures:
        print("Quality gate failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
