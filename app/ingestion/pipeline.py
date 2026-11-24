"""Parse, chunk, embed and store one file, in a single transaction."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.observability import span
from app.db.models import ChildChunk, Document, ParentChunk
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import SUPPORTED, parse
from app.retrieval.embeddings import Embedder

log = logging.getLogger(__name__)

_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    title: str
    parents: int
    children: int
    duplicate: bool = False


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_language(text: str) -> str:
    arabic, latin = len(_ARABIC.findall(text)), len(_LATIN.findall(text))
    return "ar" if arabic > latin else "en"


def embedding_text(title: str, heading_path: str, content: str) -> str:
    """What the embedder sees: the child text with its document and section, so short chunks keep their topic."""
    where = " > ".join(part for part in (title, heading_path) if part and part != title) or title
    return f"{title}\n{where}\n{content}" if where != title else f"{title}\n{content}"


async def ingest_file(
    session: AsyncSession,
    embedder: Embedder,
    settings: Settings,
    path: Path,
    *,
    source: str | None = None,
    department_id: str | None = None,
    access_level: int = 0,
    title: str | None = None,
) -> IngestResult:
    content_hash = file_hash(path)
    existing = await session.scalar(select(Document).where(Document.content_hash == content_hash))
    if existing is not None:
        return IngestResult(str(existing.id), existing.title, 0, 0, duplicate=True)

    with span("ingest.parse", file=path.name):
        parsed = parse(path)
    with span("ingest.chunk"):
        parents = chunk_document(
            parsed, settings.parent_chunk_tokens, settings.child_chunk_tokens, settings.child_chunk_overlap
        )
    if not parents:
        raise ValueError(f"{path.name} produced no text to index")

    doc_title = title or parsed.title
    document = Document(
        title=doc_title[:500],
        source=source or path.name,
        mime_type=SUPPORTED[path.suffix.lower()],
        content_hash=content_hash,
        department_id=department_id,
        access_level=access_level,
        language=detect_language(parsed.text[:5000]),
        embedding_model=embedder.model_name,
        pages=parsed.pages,
    )
    texts = [embedding_text(doc_title, p.heading_path, c.content) for p in parents for c in p.children]
    with span("ingest.embed", chunks=len(texts)):
        vectors = await embedder.embed_documents(texts)

    session.add(document)
    await session.flush()
    vector_iter = iter(vectors)
    child_count = 0
    for p_index, draft in enumerate(parents):
        page_start, page_end = draft.pages
        parent = ParentChunk(
            document_id=document.id,
            position=p_index,
            heading_path=draft.heading_path,
            page_start=page_start,
            page_end=page_end,
            token_count=draft.tokens,
            content=draft.content,
        )
        session.add(parent)
        await session.flush()
        for c_index, child in enumerate(draft.children):
            session.add(
                ChildChunk(
                    parent_id=parent.id,
                    document_id=document.id,
                    position=c_index,
                    page=child.page,
                    content=child.content,
                    embedding=next(vector_iter),
                    department_id=department_id,
                    access_level=access_level,
                )
            )
            child_count += 1
    await session.commit()
    log.info(
        "ingested",
        extra={"document": doc_title, "parents": len(parents), "children": child_count, "warnings": parsed.warnings},
    )
    return IngestResult(str(document.id), doc_title, len(parents), child_count)
