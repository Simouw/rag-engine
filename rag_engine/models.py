"""ORM models: documents and their parent/child chunks."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_engine.config import settings
from rag_engine.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _embedding_column_type():
    # SQLite has no vector type: store the list as JSON and compute cosine
    # similarity in Python. PostgreSQL uses pgvector directly.
    return Vector(settings.EMBEDDING_DIMENSION).with_variant(JSON(), "sqlite")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("source_key", name="uq_documents_source_key"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500))
    # Stable identity used by the CLI to re-ingest a file in place. NULL for
    # one-off API uploads, and SQL unique constraints allow several NULLs.
    source_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    collection: Mapped[str] = mapped_column(String(100), default="default", index=True)
    doc_type: Mapped[str] = mapped_column(String(50), default="document", index=True)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    file_path: Mapped[str] = mapped_column(String(1000))
    mime_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="uploaded", index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(_embedding_column_type(), nullable=True)
    # "parent" = full section sent to the LLM, "child" = retrieval unit.
    chunk_role: Mapped[str] = mapped_column(String(16), default="child", index=True)
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True, index=True
    )
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")
