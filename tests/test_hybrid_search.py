"""Reciprocal Rank Fusion and access-controlled hybrid retrieval."""

from __future__ import annotations

import re

import pytest

from app.core.config import get_settings
from app.db.session import get_sessionmaker
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid_search import HybridSearcher, keyword_query, rrf_fuse
from tests.conftest import principal

# RRF -----------------------------------------------------------------------------------------


def test_rrf_scores_follow_the_formula() -> None:
    scores = rrf_fuse([["a", "b", "c"], ["b", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_between_retrievers() -> None:
    # "d" is only second in each list, but both lists have it, so it beats each list's single winner.
    scores = rrf_fuse([["a", "d"], ["b", "d"]], k=60)
    assert next(iter(scores)) == "d"


def test_rrf_handles_empty_and_single_rankings() -> None:
    assert rrf_fuse([]) == {}
    assert list(rrf_fuse([["x", "y"]])) == ["x", "y"]


def test_keyword_query_drops_stop_words_and_keeps_technical_tokens() -> None:
    query = keyword_query("What is the rate limit and does it return HTTP 429 or AES-256?")
    assert query is not None
    for token in ("rate", "limit", "HTTP", "429", "AES-256"):
        assert token in query
    for stop in (" the ", " is ", " and "):
        assert stop not in f" {query} "
    assert keyword_query("what is the") is None


def test_keyword_query_keeps_arabic_content_words() -> None:
    query = keyword_query("ما هي ساعات العمل الأساسية؟")
    assert query is not None and "ساعات" in query and "ما" not in query.split()


# Retrieval against PostgreSQL ------------------------------------------------------------------

EMPLOYEE = principal("employee", clearance="internal")
ENGINEER = principal("engineer", ("engineering",), "internal")
VISITOR = principal("visitor", clearance="public")


async def search(who, question, mode="hybrid", limit=10):
    settings = get_settings()
    searcher = HybridSearcher(build_embedder(settings), settings)
    async with get_sessionmaker()() as session:
        return await searcher.search_children(session, question, who, limit=limit, mode=mode)


async def test_exact_token_is_found_by_the_sparse_side(corpus: dict[str, str]) -> None:
    hits = await search(VISITOR, "AES-256")
    assert hits, "expected a keyword match"
    assert hits[0].source == "product-faq.txt"
    assert hits[0].sparse_rank == 1
    assert "AES-256" in hits[0].content


async def test_both_retrievers_contribute_to_fused_results(corpus: dict[str, str]) -> None:
    hits = await search(EMPLOYEE, "refund within 30 days of purchase")
    assert any(h.dense_rank is not None and h.sparse_rank is not None for h in hits)
    assert hits == sorted(hits, key=lambda h: h.rrf_score, reverse=True)


async def test_dense_and_sparse_modes_for_baselines(corpus: dict[str, str]) -> None:
    dense = await search(EMPLOYEE, "encryption at rest", mode="dense")
    sparse = await search(EMPLOYEE, "encryption at rest", mode="sparse")
    assert dense and all(h.sparse_rank is None for h in dense)
    assert sparse and all(h.dense_rank is None for h in sparse)


async def test_public_visitor_only_sees_public_documents(corpus: dict[str, str]) -> None:
    for question in ("paid parental leave weeks", "acknowledge a page within 15 minutes", "Data is encrypted"):
        hits = await search(VISITOR, question, limit=50)
        assert {h.source for h in hits} <= {"product-faq.txt"}


async def test_department_documents_need_membership(corpus: dict[str, str]) -> None:
    question = "make bootstrap toolchain onboarding buddy"
    employee_sources = {h.source for h in await search(EMPLOYEE, question, limit=50)}
    engineer_sources = {h.source for h in await search(ENGINEER, question, limit=50)}
    assert "onboarding-guide.pdf" not in employee_sources
    assert "onboarding-guide.pdf" in engineer_sources


async def test_filter_applies_before_ranking_so_limits_are_filled(corpus: dict[str, str]) -> None:
    # Ask for an engineering-only phrase as a visitor: every returned row must still be readable,
    # and the result is not empty just because the best matches were filtered out.
    hits = await search(VISITOR, "trunk-based development pull request", limit=5)
    assert hits and all(h.source == "product-faq.txt" for h in hits)


async def test_parents_are_loaded_with_access_checks(corpus: dict[str, str]) -> None:
    settings = get_settings()
    searcher = HybridSearcher(build_embedder(settings), settings)
    async with get_sessionmaker()() as session:
        hits = await searcher.search_children(session, "on-call rotation postmortem", ENGINEER, limit=10)
        blocks = await searcher.parents_for(session, hits, ENGINEER, max_parents=3)
        assert blocks and len(blocks) <= 3
        onboarding = [b for b in blocks if b.source == "onboarding-guide.pdf"]
        assert onboarding and re.fullmatch(r"\[Acme Engineering .+ \| p\. [12]\]", onboarding[0].citation)
        # The same hits cannot be turned into parents by someone without access.
        assert await searcher.parents_for(session, hits, VISITOR, max_parents=3) == [] or all(
            b.source == "product-faq.txt" for b in await searcher.parents_for(session, hits, VISITOR, 3)
        )
