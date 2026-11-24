"""Hierarchical chunking: section-sized parents for context, small children for matching.

Parents (about 1500 tokens) follow the document's sections and are what the LLM reads. Children
(about 300 tokens, 50 overlapping) are cut on sentence or table-row boundaries inside one parent and
are what gets embedded and searched. A table never loses its header: when it is split, each piece
repeats the header row.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from app.ingestion.parser import Element, ParsedDocument

log = logging.getLogger(__name__)


@lru_cache
def _encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        log.warning("tiktoken encoding unavailable, falling back to an approximate token count")
        return None


_APPROX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    return int(len(_APPROX.findall(text)) * 1.15) + 1


_SENTENCE_END = re.compile(r"(?<=[.!?؟。])\s+(?=[\"'(\[A-Z؀-ۿ0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


@dataclass
class Unit:
    """The smallest piece the chunker moves around: a sentence, a heading, or a group of table rows."""

    text: str
    tokens: int
    page: int | None


@dataclass
class ChildDraft:
    content: str
    page: int | None
    tokens: int


@dataclass
class ParentDraft:
    heading_path: str
    elements: list[Element] = field(default_factory=list)
    tokens: int = 0
    children: list[ChildDraft] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n\n".join(e.text for e in self.elements)

    @property
    def pages(self) -> tuple[int | None, int | None]:
        pages = [e.page for e in self.elements if e.page is not None]
        return (min(pages), max(pages)) if pages else (None, None)


def _split_long_text(element: Element, limit: int) -> list[Element]:
    """Break a paragraph longer than `limit` into sentence groups (words as a last resort)."""
    pieces: list[Element] = []
    current: list[str] = []
    used = 0
    for sentence in split_sentences(element.text):
        size = count_tokens(sentence)
        if size > limit:
            words = sentence.split()
            step = max(1, int(len(words) * limit / size))
            for start in range(0, len(words), step):
                pieces.append(Element("text", " ".join(words[start : start + step]), element.page))
            continue
        if used + size > limit and current:
            pieces.append(Element("text", " ".join(current), element.page))
            current, used = [], 0
        current.append(sentence)
        used += size
    if current:
        pieces.append(Element("text", " ".join(current), element.page))
    return pieces


def _table_units(element: Element, limit: int) -> list[Unit]:
    lines = element.text.splitlines()
    header, rows = lines[:2], lines[2:]
    header_text = "\n".join(header)
    header_tokens = count_tokens(header_text)
    units: list[Unit] = []
    group: list[str] = []
    used = header_tokens
    for row in rows:
        size = count_tokens(row)
        if group and used + size > limit:
            units.append(Unit(header_text + "\n" + "\n".join(group), used, element.page))
            group, used = [], header_tokens
        group.append(row)
        used += size
    if group or not rows:
        units.append(Unit(header_text + ("\n" + "\n".join(group) if group else ""), used, element.page))
    return units


def _units(elements: Iterable[Element], child_limit: int) -> list[Unit]:
    units: list[Unit] = []
    for element in elements:
        if element.kind == "table":
            units.extend(_table_units(element, child_limit))
        elif element.kind == "heading":
            units.append(Unit(element.text, count_tokens(element.text), element.page))
        else:
            for sentence in split_sentences(element.text):
                for piece in _split_long_text(Element("text", sentence, element.page), child_limit):
                    units.append(Unit(piece.text, count_tokens(piece.text), element.page))
    return units


def _children(parent: ParentDraft, limit: int, overlap: int) -> list[ChildDraft]:
    units = _units(parent.elements, limit)
    children: list[ChildDraft] = []
    window: list[Unit] = []
    used = 0
    fresh = 0  # units added since the last child was cut, so overlap-only windows are not emitted
    for unit in units:
        if window and used + unit.tokens > limit and fresh:
            children.append(ChildDraft(" ".join(u.text for u in window), window[0].page, used))
            carried: list[Unit] = []
            carried_tokens = 0
            for previous in reversed(window):
                if carried_tokens + previous.tokens > overlap or "\n|" in previous.text:
                    break
                carried.insert(0, previous)
                carried_tokens += previous.tokens
            window, used, fresh = carried, carried_tokens, 0
        window.append(unit)
        used += unit.tokens
        fresh += 1
    if window and fresh:
        children.append(ChildDraft(" ".join(u.text for u in window), window[0].page, used))
    return children


def chunk_document(
    doc: ParsedDocument, parent_tokens: int = 1500, child_tokens: int = 300, child_overlap: int = 50
) -> list[ParentDraft]:
    if child_tokens >= parent_tokens:
        raise ValueError("child_tokens must be smaller than parent_tokens")
    parents: list[ParentDraft] = []
    path: list[tuple[int, str]] = []
    current: ParentDraft | None = None

    def start(heading_path: str) -> ParentDraft:
        nonlocal current
        if current and current.elements:
            parents.append(current)
        current = ParentDraft(heading_path=heading_path)
        return current

    for element in doc.elements:
        if element.kind == "heading":
            path = [(lvl, txt) for lvl, txt in path if lvl < element.level] + [(element.level, element.text)]
            heading_path = " > ".join(txt for _, txt in path)
            # A top-level section starts a new parent unless the current one is still tiny.
            if current is None or (element.level <= 2 and current.tokens >= parent_tokens * 0.25):
                start(heading_path)
        pieces = [element]
        if element.kind == "text" and count_tokens(element.text) > parent_tokens:
            pieces = _split_long_text(element, child_tokens)
        for piece in pieces:
            size = count_tokens(piece.text)
            if current is None or (current.tokens + size > parent_tokens and current.elements):
                start(" > ".join(txt for _, txt in path))
            assert current is not None
            current.elements.append(piece)
            current.tokens += size
    if current and current.elements:
        parents.append(current)
    for parent in parents:
        parent.children = _children(parent, child_tokens, child_overlap)
    return [p for p in parents if p.children]
