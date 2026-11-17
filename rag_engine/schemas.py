"""Pydantic request/response models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    source_name: str | None
    collection: str
    doc_type: str
    tags: list[str] | None
    language: str | None
    mime_type: str | None
    status: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    metadata: dict[str, Any] | None = Field(
        default=None, validation_alias="metadata_json", serialization_alias="metadata"
    )
    chunk_role: str
    parent_chunk_id: uuid.UUID | None
    is_enabled: bool


class ChunkUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=100_000)
    is_enabled: bool | None = None


class UploadResponse(BaseModel):
    document: DocumentOut
    chunks_indexed: int
    embedding_model: str


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class SearchFilters(BaseModel):
    collection: str | None = Field(default=None, max_length=100)
    doc_type: str | None = Field(default=None, max_length=50)
    language: str | None = Field(default=None, max_length=16)
    tags: list[str] | None = Field(default=None, max_length=20)
    document_ids: list[uuid.UUID] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: SearchFilters | None = None


class RetrievedChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    content: str
    score: float
    metadata: dict[str, Any] | None = None
    page_start: int | None = None
    page_end: int | None = None
    parent_content: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    total: int
    embedding_model: str


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20_000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: SearchFilters | None = None
    history: list[ChatTurn] | None = None
    system_prompt: str | None = Field(default=None, max_length=20_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=16_000)
    include_context: bool = True


class SourceOut(BaseModel):
    index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    content: str | None
    score: float
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] | None = None


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    model: str
    usage: Usage | None = None
    retrieved: int
    rewritten_query: str | None = None


# ---------------------------------------------------------------------------
# Health / stats
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    database: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    llm_configured: bool
    llm_model: str


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    collections: list[str]
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
