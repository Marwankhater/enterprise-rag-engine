"""Hybrid retrieval in PostgreSQL: dense (pgvector HNSW) plus sparse (tsvector GIN), fused with RRF.

Access control is part of both candidate queries' WHERE clause, so a chunk the caller cannot read
is never ranked, never returned and never counted. Child chunks are matched; their parents are
returned as context (small-to-big retrieval).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Principal, Settings
from app.core.observability import span
from app.core.security import rbac_clause, rbac_params
from app.retrieval.embeddings import Embedder

_WORD = re.compile(r"[\w\-\.]+", re.UNICODE)
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "my",
        "me",
        "any",
        "about",
        "after",
        "before",
        "into",
        "than",
        "then",
        "they",
        "them",
        "في",
        "من",
        "على",
        "الى",
        "إلى",
        "عن",
        "مع",
        "هل",
        "ما",
        "ماذا",
        "كيف",
        "متى",
        "أين",
        "اين",
        "هو",
        "هي",
        "هذا",
        "هذه",
        "ذلك",
        "التي",
        "الذي",
        "او",
        "أو",
        "و",
        "ثم",
        "كم",
        "لماذا",
    ]
)


def keyword_query(question: str, max_terms: int = 24) -> str | None:
    """The question without stop words. Postgres tokenises it and ORs the lexemes (see _CANDIDATES_SQL),
    so tokens like AES-256 or HTTP 429 are matched the same way they were indexed."""
    kept: list[str] = []
    for word in _WORD.findall(question):
        if word.lower().strip(".-") in _STOP or not word.strip(".-"):
            continue
        kept.append(word)
    return " ".join(kept[:max_terms]) or None


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


@dataclass
class ChildHit:
    child_id: str
    parent_id: str
    document_id: str
    content: str
    page: int | None
    title: str
    source: str
    heading_path: str
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    rerank_score: float | None = None


@dataclass
class ContextBlock:
    parent_id: str
    document_id: str
    title: str
    source: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    content: str
    score: float
    hits: list[ChildHit] = field(default_factory=list)

    @property
    def citation(self) -> str:
        page = next((h.page for h in self.hits if h.page is not None), self.page_start)
        where = f"p. {page}" if page is not None else (self.heading_path.split(" > ")[-1] or "document")
        return f"[{self.title} | {where}]"


_CANDIDATES_SQL = """
with dense as (
    select c.id, row_number() over (order by c.embedding <=> cast(:qvec as vector)) as rnk
    from child_chunks c
    where {acl}
    order by c.embedding <=> cast(:qvec as vector)
    limit :n_dense
),
terms as (
    select to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) as q
    from unnest(to_tsvector('simple', :qtext))
),
sparse as (
    select c.id, row_number() over (order by ts_rank_cd(c.tsv, t.q) desc) as rnk
    from child_chunks c, terms t
    where :use_sparse and c.tsv @@ t.q and {acl}
    order by ts_rank_cd(c.tsv, t.q) desc
    limit :n_sparse
),
fused as (
    select coalesce(d.id, s.id) as id, d.rnk as dense_rank, s.rnk as sparse_rank,
           coalesce(1.0 / (:k + d.rnk), 0) + coalesce(1.0 / (:k + s.rnk), 0) as rrf
    from dense d full outer join sparse s on s.id = d.id
)
select f.id, c.parent_id, c.document_id, c.content, c.page, doc.title, doc.source, p.heading_path,
       f.dense_rank, f.sparse_rank, f.rrf
from fused f
join child_chunks c on c.id = f.id
join parent_chunks p on p.id = c.parent_id
join documents doc on doc.id = c.document_id
order by f.rrf desc
limit :limit
"""

_PARENTS_SQL = """
select p.id, p.document_id, doc.title, doc.source, p.heading_path, p.page_start, p.page_end, p.content
from parent_chunks p
join documents doc on doc.id = p.document_id
where p.id = any(:ids)
  and doc.access_level <= :clearance
  and (doc.department_id is null or doc.department_id = any(:departments))
"""


class HybridSearcher:
    def __init__(self, embedder: Embedder, settings: Settings) -> None:
        self.embedder = embedder
        self.settings = settings

    async def search_children(
        self, session: AsyncSession, question: str, principal: Principal, limit: int = 20, mode: str = "hybrid"
    ) -> list[ChildHit]:
        """mode: 'hybrid', 'dense' or 'sparse' (the last two exist for evaluation baselines)."""
        terms = keyword_query(question)
        with span("retrieval.hybrid", mode=mode, has_terms=terms is not None):
            qvec = await self.embedder.embed_query(question)
            params = {
                "qvec": vector_literal(qvec),
                "qtext": terms or "",
                "use_sparse": terms is not None and mode != "dense",
                "n_dense": self.settings.dense_candidates if mode != "sparse" else 0,
                "n_sparse": self.settings.sparse_candidates,
                "k": self.settings.rrf_k,
                "limit": limit,
                **rbac_params(principal),
            }
            rows = (await session.execute(text(_CANDIDATES_SQL.format(acl=rbac_clause("c"))), params)).all()
        return [
            ChildHit(
                child_id=str(r.id),
                parent_id=str(r.parent_id),
                document_id=str(r.document_id),
                content=r.content,
                page=r.page,
                title=r.title,
                source=r.source,
                heading_path=r.heading_path,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
                rrf_score=float(r.rrf),
            )
            for r in rows
        ]

    async def parents_for(
        self, session: AsyncSession, hits: list[ChildHit], principal: Principal, max_parents: int
    ) -> list[ContextBlock]:
        """Group child hits by parent, keep the best-scoring parents, and load their full text."""
        best: dict[str, list[ChildHit]] = {}
        for hit in hits:
            best.setdefault(hit.parent_id, []).append(hit)

        def score(group: list[ChildHit]) -> float:
            return max((h.rerank_score if h.rerank_score is not None else h.rrf_score) for h in group)

        ranked = sorted(best.items(), key=lambda kv: score(kv[1]), reverse=True)[:max_parents]
        if not ranked:
            return []
        ids = [uuid.UUID(pid) for pid, _ in ranked]
        rows = {
            str(r.id): r
            for r in (await session.execute(text(_PARENTS_SQL), {"ids": ids, **rbac_params(principal)})).all()
        }
        blocks = []
        for pid, group in ranked:
            row = rows.get(pid)
            if row is None:  # access changed between the two queries
                continue
            blocks.append(
                ContextBlock(
                    parent_id=pid,
                    document_id=str(row.document_id),
                    title=row.title,
                    source=row.source,
                    heading_path=row.heading_path,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    content=row.content,
                    score=score(group),
                    hits=group,
                )
            )
        return blocks
