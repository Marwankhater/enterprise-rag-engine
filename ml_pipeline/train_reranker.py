"""Fine-tune the cross-encoder reranker on this corpus.

1. build-data   Writes (query, positive, hard negatives) triplets to a JSONL file:
                * synthetic questions: the LLM writes questions that a child chunk answers;
                * hard negatives: chunks hybrid search ranks highly for that question but that come
                  from a different section, which are exactly the near misses a reranker must learn;
                * feedback: rated answers from the feedback table. A thumbs-up makes the cited
                  passages positives for that question, a thumbs-down makes them negatives.
                Questions in tests/golden_dataset.json are never used, so evaluation stays clean.
2. train        Splits by document (no document appears in both splits), measures the base model on
                the held-out split, fine-tunes with binary cross-entropy on positive and negative
                pairs, measures again, and saves the model plus before/after metrics.

    python -m ml_pipeline.train_reranker build-data --questions-per-chunk 3
    python -m ml_pipeline.train_reranker train --epochs 2

The trained directory can be used directly: RERANKER_MODEL=ml_pipeline/output/reranker-ft.
Training runs on CPU, but a GPU (for example a free Kaggle T4) makes it a few minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

from app.core.config import AccessLevel, Principal, get_settings
from app.core.observability import configure_logging

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ml_pipeline" / "output"
DATA = OUTPUT / "reranker_triplets.jsonl"
GOLDEN = ROOT / "tests" / "golden_dataset.json"
# Reads everything: training data is built offline by an operator, never served to users.
OPERATOR = Principal(
    name="trainer", departments=frozenset({"finance", "hr", "engineering", "sales"}), clearance=AccessLevel.RESTRICTED
)

QUESTION_PROMPT = """Write {n} different questions that an employee might ask and that the passage
below answers. Use the vocabulary a real employee would use, not the passage's exact wording.
Mix short and specific questions. If the passage is in Arabic, write the questions in Arabic.
Reply with JSON: {{"questions": ["...", "..."]}}

Passage:
{passage}"""


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


async def build_data(questions_per_chunk: int, negatives: int, limit: int | None) -> int:
    from sqlalchemy import text

    from app.agents.llm import LLMError, OpenAICompatibleLLM
    from app.db.session import dispose, get_sessionmaker
    from app.retrieval.embeddings import build_embedder
    from app.retrieval.hybrid_search import HybridSearcher

    settings = get_settings()
    llm = OpenAICompatibleLLM(settings)
    searcher = HybridSearcher(build_embedder(settings), settings)
    golden = {_normalise(c["question"]) for c in json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    written = 0
    async with get_sessionmaker()() as session:
        chunks = (
            await session.execute(
                text(
                    "select c.id, c.parent_id, c.document_id, c.content from child_chunks c "
                    "order by c.document_id, c.position"
                )
            )
        ).all()
        if limit:
            chunks = chunks[:limit]
        with DATA.open("w", encoding="utf-8") as out:
            for chunk in chunks:
                try:
                    reply = await llm.chat(
                        [
                            {
                                "role": "user",
                                "content": QUESTION_PROMPT.format(n=questions_per_chunk, passage=chunk.content),
                            }
                        ],
                        json_mode=True,
                        model=settings.grader_model,
                        name="train.questions",
                        max_tokens=400,
                    )
                    questions = [q for q in reply.json().get("questions", []) if isinstance(q, str)]
                except LLMError as exc:
                    print(f"skipped chunk {chunk.id}: {exc}")
                    continue
                for question in questions[:questions_per_chunk]:
                    if _normalise(question) in golden:
                        continue
                    hits = await searcher.search_children(session, question, OPERATOR, limit=20)
                    hard = [h.content for h in hits if h.parent_id != str(chunk.parent_id)][:negatives]
                    if not hard:
                        continue
                    out.write(
                        json.dumps(
                            {
                                "query": question,
                                "positive": [chunk.content],
                                "negative": hard,
                                "group": str(chunk.document_id),
                                "origin": "synthetic",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
            written += await _feedback_rows(session, out)
    await dispose()
    print(f"wrote {written} training queries to {DATA}")
    return written


async def _feedback_rows(session: Any, out: Any) -> int:
    """Rated answers become extra training queries, grouped by their document."""
    from sqlalchemy import text

    rows = (
        await session.execute(
            text("""
        select f.rating, q.content as question, a.meta
        from feedback f
        join chat_messages a on a.id = f.message_id
        join lateral (
            select content from chat_messages u
            where u.session_id = a.session_id and u.role = 'user' and u.created_at < a.created_at
            order by u.created_at desc limit 1
        ) q on true
    """)
        )
    ).all()
    count = 0
    for row in rows:
        parent_ids = [c.get("parent_id") for c in (row.meta or {}).get("citations", []) if c.get("parent_id")]
        if not parent_ids:
            continue
        found = (
            await session.execute(
                text("select id, document_id, content from parent_chunks where id = any(cast(:ids as uuid[]))"),
                {"ids": parent_ids},
            )
        ).all()
        passages = [p.content for p in found]
        if not passages:
            continue
        record = {
            "query": row.question,
            "positive": passages if row.rating > 0 else [],
            "negative": passages if row.rating < 0 else [],
            "group": str(found[0].document_id),
            "origin": "feedback",
        }
        if record["positive"] or record["negative"]:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def split_by_group(rows: list[dict[str, Any]], eval_share: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups = sorted({r["group"] for r in rows})
    random.Random(seed).shuffle(groups)
    held_out = set(groups[: max(1, round(len(groups) * eval_share))])
    train = [r for r in rows if r["group"] not in held_out]
    evaluation = [r for r in rows if r["group"] in held_out and r["positive"] and r["negative"]]
    return train, evaluation


def train(base_model: str, epochs: int, batch_size: int, lr: float, eval_share: float, seed: int) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 20:
        raise SystemExit(f"Only {len(rows)} training queries in {DATA}; run build-data first.")
    train_rows, eval_rows = split_by_group(rows, eval_share, seed)
    pairs = [(r["query"], p, 1.0) for r in train_rows for p in r["positive"]]
    pairs += [(r["query"], n, 0.0) for r in train_rows for n in r["negative"]]
    random.Random(seed).shuffle(pairs)
    positives = sum(1 for p in pairs if p[2] == 1.0)
    train_set = Dataset.from_dict(
        {"query": [p[0] for p in pairs], "passage": [p[1] for p in pairs], "label": [p[2] for p in pairs]}
    )
    samples = [{"query": r["query"], "positive": r["positive"], "negative": r["negative"]} for r in eval_rows]

    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    evaluator = CrossEncoderRerankingEvaluator(
        samples=samples, at_k=10, name="held_out", always_rerank_positives=True, write_csv=False
    )
    before = evaluator(model)
    print("base model:", {k: round(v, 4) for k, v in before.items()})

    out_dir = OUTPUT / "reranker-ft"
    args = CrossEncoderTrainingArguments(
        output_dir=str(OUTPUT / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=20,
        seed=seed,
        report_to="none",
    )
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor((len(pairs) - positives) / max(positives, 1)))
    CrossEncoderTrainer(model=model, args=args, train_dataset=train_set, loss=loss).train()
    after = evaluator(model)
    print("fine-tuned:", {k: round(v, 4) for k, v in after.items()})
    model.save_pretrained(str(out_dir))
    metrics = {
        "base_model": base_model,
        "epochs": epochs,
        "train_queries": len(train_rows),
        "train_pairs": len(pairs),
        "eval_queries": len(eval_rows),
        "eval_documents": sorted({r["group"] for r in eval_rows}),
        "before": {k: round(v, 4) for k, v in before.items()},
        "after": {k: round(v, 4) for k, v in after.items()},
    }
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("build-data", help="generate training triplets from the indexed corpus")
    data.add_argument("--questions-per-chunk", type=int, default=3)
    data.add_argument("--negatives", type=int, default=4)
    data.add_argument("--limit", type=int, help="only the first N chunks (for a quick run)")
    fit = sub.add_parser("train", help="fine-tune and compare with the base model")
    fit.add_argument("--base-model", default="BAAI/bge-reranker-base")
    fit.add_argument("--epochs", type=int, default=2)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--lr", type=float, default=2e-5)
    fit.add_argument("--eval-share", type=float, default=0.25, help="share of documents held out for evaluation")
    fit.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    configure_logging("WARNING")
    if args.command == "build-data":
        asyncio.run(build_data(args.questions_per_chunk, args.negatives, args.limit))
    else:
        train(args.base_model, args.epochs, args.batch_size, args.lr, args.eval_share, args.seed)


if __name__ == "__main__":
    main()
