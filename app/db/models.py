"""Tables.

Access control fields (department_id, access_level) live on the document and are copied onto every
child chunk, so the retrieval query can filter inside the same index scan instead of joining.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings

EMBEDDING_DIM = get_settings().embedding_dim
# 'simple' does no stemming, which keeps Arabic and English tokens intact in one column.
TS_CONFIG = "simple"


class Base(DeclarativeBase):
    type_annotation_map: ClassVar = {dict[str, Any]: JSONB}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(1000))
    mime_type: Mapped[str] = mapped_column(String(100))
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    department_id: Mapped[str | None] = mapped_column(String(64), index=True)  # None = company-wide
    access_level: Mapped[int] = mapped_column(SmallInteger, default=0)
    language: Mapped[str | None] = mapped_column(String(8))
    embedding_model: Mapped[str] = mapped_column(String(200))
    pages: Mapped[int | None] = mapped_column(Integer)

    parents: Mapped[list[ParentChunk]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (CheckConstraint("access_level between 0 and 3", name="ck_documents_access_level"),)


class ParentChunk(Base):
    """A section-sized span (about 1500 tokens) that is handed to the LLM as context."""

    __tablename__ = "parent_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[str] = mapped_column(Text, default="")
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="parents")
    children: Mapped[list[ChildChunk]] = relationship(back_populates="parent", cascade="all, delete-orphan")


class ChildChunk(Base):
    """A small span (about 300 tokens) that is embedded and searched."""

    __tablename__ = "child_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("parent_chunks.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str] = mapped_column(TSVECTOR, Computed(f"to_tsvector('{TS_CONFIG}', content)", persisted=True))
    department_id: Mapped[str | None] = mapped_column(String(64))
    access_level: Mapped[int] = mapped_column(SmallInteger, default=0)

    parent: Mapped[ParentChunk] = relationship(back_populates="children")

    __table_args__ = (
        Index(
            "ix_child_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_child_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_child_chunks_acl", "access_level", "department_id"),
    )


class IngestJob(TimestampMixin, Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # queued | running | done | failed | duplicate
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    filename: Mapped[str] = mapped_column(String(500))
    stored_path: Mapped[str] = mapped_column(String(1000))
    requested_by: Mapped[str] = mapped_column(String(200))
    department_id: Mapped[str | None] = mapped_column(String(64))
    access_level: Mapped[int] = mapped_column(SmallInteger, default=0)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    parents: Mapped[int] = mapped_column(Integer, default=0)
    children: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChatMessage(TimestampMixin, Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    principal: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)  # citations, intent, cache hit, latency

    __table_args__ = (Index("ix_chat_messages_session_created", "session_id", "created_at"),)


class Feedback(TimestampMixin, Base):
    """Ratings on answers. Thumbs-down rows with their contexts become hard negatives for the reranker."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"), index=True)
    principal: Mapped[str] = mapped_column(String(200))
    rating: Mapped[int] = mapped_column(SmallInteger)  # 1 up, -1 down
    comment: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (CheckConstraint("rating in (-1, 1)", name="ck_feedback_rating"),)


class AuditLog(TimestampMixin, Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    principal: Mapped[str] = mapped_column(String(200), index=True)
    action: Mapped[str] = mapped_column(String(50), index=True)  # chat | ingest | feedback | sql | denied
    request_id: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[dict[str, Any]] = mapped_column(default=dict)
