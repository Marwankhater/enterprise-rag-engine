"""Cross-encoder reranking of fused candidates.

A cross-encoder reads the question and the chunk together, which is slower than comparing vectors
but much more precise, so it only sees the top candidates from hybrid search. RERANKER_MODEL can be
a Hugging Face id (BAAI/bge-reranker-base) or a local directory written by
ml_pipeline/train_reranker.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from app.core.observability import span
from app.retrieval.hybrid_search import ChildHit

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    async def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str, max_length: int = 512, batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name_or_path
        self.model = CrossEncoder(model_name_or_path, max_length=max_length, device="cpu")
        self.batch_size = batch_size

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        scores = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._predict, query, passages)


class PassthroughReranker:
    """Keeps the fused order. Used when reranking is disabled and in tests."""

    name = "passthrough"

    async def score(self, query: str, passages: list[str]) -> list[float]:
        n = len(passages)
        return [1.0 - i / max(n, 1) for i in range(n)]


async def rerank(reranker: Reranker, query: str, hits: list[ChildHit], top_n: int) -> list[ChildHit]:
    if not hits:
        return []
    with span("retrieval.rerank", model=reranker.name, candidates=len(hits)):
        scores = await reranker.score(query, [h.content for h in hits])
    for hit, value in zip(hits, scores, strict=True):
        hit.rerank_score = value
    return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)[:top_n]


def build_reranker(enabled: bool, model: str) -> Reranker:
    if not enabled:
        return PassthroughReranker()
    try:
        return CrossEncoderReranker(model)
    except Exception:
        log.exception("reranker unavailable, falling back to fused order", extra={"model": model})
        return PassthroughReranker()
