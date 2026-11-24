"""Parsing and parent/child chunking."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_document, count_tokens
from app.ingestion.parser import Element, ParsedDocument, UnsupportedDocument, parse, parse_markdown
from tests.conftest import EXAMPLES


def test_pdf_keeps_pages_and_headings() -> None:
    doc = parse(EXAMPLES / "onboarding-guide.pdf")
    assert doc.pages == 2
    headings = [e.text for e in doc.elements if e.kind == "heading"]
    assert "Git Workflow" in headings and "On-Call" in headings
    assert {e.page for e in doc.elements} == {1, 2}


def test_markdown_tables_stay_whole_and_headings_nest() -> None:
    text = "# Policy\n\n## Limits\n\nIntro text.\n\n| Plan | Requests |\n|---|---|\n| Free | 60 |\n| Pro | 600 |\n"
    doc = parse_markdown(text, "x")
    kinds = [e.kind for e in doc.elements]
    assert kinds == ["heading", "heading", "text", "table"]
    table = doc.elements[-1].text
    assert table.splitlines()[0] == "| Plan | Requests |" and "| Pro | 600 |" in table


def test_unsupported_type_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"MZ")
    with pytest.raises(UnsupportedDocument):
        parse(bad)


def test_children_respect_token_limits_and_stay_inside_their_parent() -> None:
    doc = parse(EXAMPLES / "company-handbook.md")
    parents = chunk_document(doc, parent_tokens=400, child_tokens=80, child_overlap=20)
    assert len(parents) > 1
    for parent in parents:
        assert parent.tokens <= 400
        for child in parent.children:
            assert child.tokens <= 80 + 40  # a single unit may exceed slightly; never by much
            assert child.content.split()[0] in parent.content


def test_overlap_repeats_the_end_of_the_previous_child() -> None:
    sentences = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    doc = ParsedDocument("t", [Element("heading", "T", None, 1), Element("text", sentences)])
    children = chunk_document(doc, 1500, 60, 20)[0].children
    assert len(children) > 2
    for prev, nxt in itertools.pairwise(children):
        last_sentence = prev.content.rsplit(".", 2)[-2].strip()
        assert last_sentence in nxt.content


def test_large_table_is_split_by_rows_with_its_header_repeated() -> None:
    rows = "\n".join(f"| row {i} | value {i} with some extra words to take space |" for i in range(80))
    table = "| Key | Value |\n|---|---|\n" + rows
    doc = ParsedDocument("t", [Element("heading", "Data", None, 1), Element("table", table)])
    children = [c for p in chunk_document(doc, 1500, 150, 30) for c in p.children]
    assert len(children) > 3
    assert all("| Key | Value |\n|---|---|" in c.content for c in children)


def test_token_counter_is_monotonic() -> None:
    assert count_tokens("one") < count_tokens("one two three four")
