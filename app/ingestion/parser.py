"""Layout-aware parsing of PDF, DOCX, Markdown and plain text into ordered elements.

Each element is a heading, a paragraph or a table, with its page number where the format has
pages. Tables are kept whole and rendered as Markdown so their rows survive chunking and stay
readable to the LLM. Scanned PDFs (no text layer) are reported, not OCR'd.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ElementKind = Literal["heading", "text", "table"]


@dataclass
class Element:
    kind: ElementKind
    text: str
    page: int | None = None
    level: int = 0  # heading level, 1 = top


@dataclass
class ParsedDocument:
    title: str
    elements: list[Element]
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(e.text for e in self.elements)


class UnsupportedDocument(ValueError):
    pass


SUPPORTED = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [[re.sub(r"\s+", " ", (cell or "")).strip().replace("|", "\\|") for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


# PDF -----------------------------------------------------------------------------------------


def _inside(bbox: tuple[float, float, float, float], x: float, top: float) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 - 1 <= x <= x1 + 1 and y0 - 1 <= top <= y1 + 1


def parse_pdf(path: Path) -> ParsedDocument:
    import pdfplumber

    elements: list[Element] = []
    warnings: list[str] = []
    with pdfplumber.open(path) as pdf:
        sizes = [round(c["size"], 1) for page in pdf.pages for c in page.chars if c.get("text", "").strip()]
        if not sizes:
            raise UnsupportedDocument("PDF has no text layer (scanned?). Run OCR before ingesting it.")
        body = statistics.median(sizes)
        for number, page in enumerate(pdf.pages, start=1):
            placed: list[tuple[float, Element]] = []
            tables = page.find_tables()
            boxes = [t.bbox for t in tables]
            for table in tables:
                markdown = table_to_markdown(table.extract())
                if markdown:
                    placed.append((table.bbox[1], Element("table", markdown, number)))
            lines = [
                ln
                for ln in page.extract_text_lines(keep_blank_chars=False)
                if ln["text"].strip() and not any(_inside(b, ln["x0"], ln["top"]) for b in boxes)
            ]
            paragraph: list[str] = []
            para_top: float | None = None
            last_bottom: float | None = None

            def flush(placed: list[tuple[float, Element]] = placed, number: int = number) -> None:
                nonlocal paragraph, para_top
                if paragraph:
                    text = re.sub(r"\s+", " ", " ".join(paragraph)).strip()
                    placed.append((para_top or 0.0, Element("text", text, number)))
                paragraph, para_top = [], None

            for line in lines:
                chars = [c for c in line["chars"] if c.get("text", "").strip()]
                size = statistics.median(round(c["size"], 1) for c in chars) if chars else body
                bold = bool(chars) and sum("Bold" in c.get("fontname", "") for c in chars) > len(chars) / 2
                text = line["text"].strip()
                is_heading = len(text) < 120 and (
                    size >= body * 1.15 or (bold and len(text.split()) <= 10 and not text.endswith("."))
                )
                gap = line["top"] - last_bottom if last_bottom is not None else 0.0
                if is_heading:
                    flush()
                    level = 1 if size >= body * 1.6 else 2 if size >= body * 1.15 else 3
                    placed.append((line["top"], Element("heading", text, number, level)))
                else:
                    if paragraph and gap > (line["bottom"] - line["top"]) * 0.9:
                        flush()
                    if para_top is None:
                        para_top = line["top"]
                    paragraph.append(text)
                last_bottom = line["bottom"]
            flush()
            placed.sort(key=lambda item: item[0])
            elements.extend(e for _, e in placed)
        pages = len(pdf.pages)
    elements = _merge_split_headings(elements)
    title = next((e.text for e in elements if e.kind == "heading"), path.stem)
    return ParsedDocument(title=title, elements=elements, pages=pages, warnings=warnings)


def _merge_split_headings(elements: list[Element]) -> list[Element]:
    """Join a heading that wrapped onto two lines (same page, same level, consecutive)."""
    out: list[Element] = []
    for element in elements:
        prev = out[-1] if out else None
        if (
            prev
            and element.kind == prev.kind == "heading"
            and element.level == prev.level
            and element.page == prev.page
            and len(prev.text) + len(element.text) < 120
        ):
            prev.text = f"{prev.text} {element.text}"
            continue
        out.append(element)
    return out


# DOCX ----------------------------------------------------------------------------------------


def parse_docx(path: Path) -> ParsedDocument:
    from docx import Document as load_docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = load_docx(str(path))
    elements: list[Element] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, doc)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "") if paragraph.style is not None else ""
            match = re.match(r"Heading (\d)", style)
            if style == "Title":
                elements.append(Element("heading", text, None, 1))
            elif match:
                elements.append(Element("heading", text, None, int(match.group(1))))
            else:
                elements.append(Element("text", text))
        elif tag == "tbl":
            table = Table(child, doc)
            markdown = table_to_markdown([[cell.text for cell in row.cells] for row in table.rows])
            if markdown:
                elements.append(Element("table", markdown))
    title = (doc.core_properties.title or "").strip() or next(
        (e.text for e in elements if e.kind == "heading"), path.stem
    )
    return ParsedDocument(title=title, elements=elements)


# Markdown and text ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def parse_markdown(text: str, fallback_title: str) -> ParsedDocument:
    elements: list[Element] = []
    block: list[str] = []
    table: list[str] = []

    def flush_block() -> None:
        if block:
            elements.append(Element("text", re.sub(r"\s+", " ", " ".join(block)).strip()))
            block.clear()

    def flush_table() -> None:
        if table:
            rows = [
                [c.strip() for c in line.strip().strip("|").split("|")]
                for line in table
                if not re.fullmatch(r"\|?[\s:\-|]+\|?", line.strip())
            ]
            markdown = table_to_markdown(rows)
            if markdown:
                elements.append(Element("table", markdown))
            table.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_block()
            table.append(stripped)
            continue
        flush_table()
        if heading := _MD_HEADING.match(stripped):
            flush_block()
            elements.append(Element("heading", heading.group(2), None, len(heading.group(1))))
        elif not stripped:
            flush_block()
        else:
            block.append(stripped)
    flush_block()
    flush_table()
    title = next((e.text for e in elements if e.kind == "heading"), fallback_title)
    return ParsedDocument(title=title, elements=elements)


def parse_text(text: str, fallback_title: str) -> ParsedDocument:
    """Plain text: short upper-case lines are treated as section headings, blank lines end paragraphs."""
    elements: list[Element] = []
    block: list[str] = []
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.isupper() and len(stripped) <= 60 and len(stripped.split()) <= 8:
            if block:
                elements.append(Element("text", " ".join(block)))
                block = []
            elements.append(Element("heading", stripped.title(), None, 2))
        elif not stripped:
            if block:
                elements.append(Element("text", " ".join(block)))
                block = []
        else:
            block.append(stripped)
    if block:
        elements.append(Element("text", " ".join(block)))
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    title = first if first and len(first) <= 100 else fallback_title
    return ParsedDocument(title=title, elements=elements)


def parse(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedDocument(f"Unsupported file type {suffix!r}; supported: {', '.join(sorted(SUPPORTED))}")
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".md", ".markdown"):
        return parse_markdown(text, path.stem)
    return parse_text(text, path.stem)
