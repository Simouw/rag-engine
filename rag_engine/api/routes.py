"""HTTP API: documents, search and ask."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.api.auth import require_api_key
from rag_engine.api.errors import api_error
from rag_engine.config import settings
from rag_engine.db import get_db, ping_db
from rag_engine.llm.chat import LLMError, LLMNotConfiguredError, chat_client
from rag_engine.llm.embeddings import EmbeddingError, embedding_client
from rag_engine.logging import logger
from rag_engine.models import Document, DocumentChunk
from rag_engine.retrieval.pipeline import retrieval_pipeline
from rag_engine.schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    ChunkUpdate,
    DocumentOut,
    HealthResponse,
    ReadyResponse,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UploadResponse,
)
from rag_engine.services.ask import ask_service
from rag_engine.services.ingestion import IngestionError, ingestion_service
from rag_engine.storage import storage_service

router = APIRouter(dependencies=[Depends(require_api_key)])
public_router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@public_router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@public_router.get("/ready", response_model=ReadyResponse, tags=["health"])
async def ready() -> ReadyResponse:
    try:
        await ping_db()
        database = "ok"
    except Exception:
        logger.exception("Database readiness check failed")
        raise HTTPException(
            status_code=503,
            detail=api_error("database_unavailable", "The database is not reachable."),
        ) from None

    return ReadyResponse(
        status=database,
        database=settings.DATABASE_URL.split("://", 1)[0],
        embedding_provider=embedding_client.provider,
        embedding_model=embedding_client.model_name,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        llm_configured=chat_client.is_configured,
        llm_model=chat_client.model_name,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@router.get("/stats", response_model=StatsResponse, tags=["stats"])
async def stats(db: AsyncSession = Depends(get_db)) -> StatsResponse:
    documents = (await db.execute(select(func.count()).select_from(Document))).scalar_one()
    chunks = (
        await db.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.chunk_role == "child")
        )
    ).scalar_one()
    collections = list(
        (await db.execute(select(Document.collection).distinct().order_by(Document.collection)))
        .scalars()
        .all()
    )
    return StatsResponse(
        documents=int(documents),
        chunks=int(chunks),
        collections=collections,
        embedding_provider=embedding_client.provider,
        embedding_model=embedding_client.model_name,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def _split_tags(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    tags = [part.strip() for part in raw.split(",") if part.strip()]
    return tags or None


@router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=201,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    collection: str = Form(default="default"),
    doc_type: str = Form(default="document"),
    tags: str | None = Form(default=None),
    language: str | None = Form(default=None),
    embed: bool = Form(default=True),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload a file and index it synchronously.

    For large batches, use the CLI (`python -m rag_engine ingest ./docs`).
    """
    content = await file.read()
    try:
        document, chunks = await ingestion_service.ingest_bytes(
            db,
            content=content,
            original_filename=file.filename or "document.txt",
            title=title,
            collection=collection.strip() or "default",
            doc_type=doc_type.strip() or "document",
            tags=_split_tags(tags),
            language=language.strip() if language else None,
            embed=embed,
        )
    except IngestionError as exc:
        raise HTTPException(
            status_code=422,
            detail=api_error("ingestion_failed", str(exc)),
        ) from exc

    await db.refresh(document)
    return UploadResponse(
        document=DocumentOut.model_validate(document),
        chunks_indexed=chunks,
        embedding_model=embedding_client.model_name,
    )


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
async def list_documents(
    collection: str | None = Query(default=None, max_length=100),
    doc_type: str | None = Query(default=None, max_length=50),
    status: str | None = Query(default=None, max_length=20),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc())
    if collection:
        stmt = stmt.where(Document.collection == collection)
    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    if status:
        stmt = stmt.where(Document.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list((await db.execute(stmt)).scalars().all())


@router.get("/documents/{document_id}", response_model=DocumentOut, tags=["documents"])
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=api_error("document_not_found", "Document not found."),
        )
    return document


@router.get("/documents/{document_id}/file", tags=["documents"])
async def download_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=api_error("document_not_found", "Document not found."),
        )
    path = Path(document.file_path)
    if not storage_service.exists(path):
        raise HTTPException(
            status_code=404,
            detail=api_error("file_not_found", "The stored file is missing."),
        )
    return FileResponse(
        path,
        media_type=document.mime_type or "application/octet-stream",
        filename=document.source_name or path.name,
    )


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=api_error("document_not_found", "Document not found."),
        )
    await ingestion_service.delete(db, document)


@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[ChunkOut],
    tags=["documents"],
)
async def list_chunks(
    document_id: uuid.UUID,
    include_parents: bool = Query(default=False),
    include_disabled: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentChunk]:
    if await db.get(Document, document_id) is None:
        raise HTTPException(
            status_code=404,
            detail=api_error("document_not_found", "Document not found."),
        )
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    if not include_parents:
        stmt = stmt.where(DocumentChunk.chunk_role == "child")
    if not include_disabled:
        stmt = stmt.where(DocumentChunk.is_enabled.is_(True))
    stmt = stmt.order_by(DocumentChunk.chunk_index.asc())
    return list((await db.execute(stmt)).scalars().all())


@router.patch(
    "/documents/{document_id}/chunks/{chunk_id}",
    response_model=ChunkOut,
    tags=["documents"],
)
async def update_chunk(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    payload: ChunkUpdate,
    db: AsyncSession = Depends(get_db),
) -> DocumentChunk:
    """Enable/disable or edit a child chunk.

    Editing content clears its embedding and parent link; reindex the document
    to make the edited chunk searchable again.
    """
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=api_error("document_not_found", "Document not found."),
        )
    chunk = await db.get(DocumentChunk, chunk_id)
    if chunk is None or chunk.document_id != document_id:
        raise HTTPException(
            status_code=404,
            detail=api_error("chunk_not_found", "Chunk not found."),
        )
    if chunk.chunk_role != "child":
        raise HTTPException(
            status_code=400,
            detail=api_error("chunk_not_editable", "Only child chunks can be edited."),
        )

    if payload.content is not None:
        cleaned = payload.content.strip()
        if cleaned != chunk.content:
            chunk.content = cleaned
            chunk.parent_chunk_id = None
            chunk.embedding = None
            document.status = "parsed"
    if payload.is_enabled is not None:
        chunk.is_enabled = payload.is_enabled

    await db.flush()
    await db.refresh(chunk)
    return chunk


@router.post(
    "/documents/{document_id}/reindex",
    response_model=UploadResponse,
    tags=["documents"],
)
async def reindex_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=api_error("document_not_found", "Document not found."),
        )
    try:
        chunks = await ingestion_service.reindex(db, document)
    except IngestionError as exc:
        raise HTTPException(
            status_code=422,
            detail=api_error("ingestion_failed", str(exc)),
        ) from exc
    await db.refresh(document)
    return UploadResponse(
        document=DocumentOut.model_validate(document),
        chunks_indexed=chunks,
        embedding_model=embedding_client.model_name,
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    payload: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    filters = payload.filters.model_dump() if payload.filters else {}
    try:
        results = await retrieval_pipeline.search(
            db=db,
            query=payload.query,
            top_k=payload.top_k,
            filters=filters,
        )
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=502,
            detail=api_error("embedding_failed", str(exc)),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=api_error("invalid_query", str(exc)),
        ) from exc

    return SearchResponse(
        query=payload.query,
        results=[_retrieved_chunk(item) for item in results],
        total=len(results),
        embedding_model=embedding_client.model_name,
    )


def _retrieved_chunk(item: dict) -> RetrievedChunk:
    metadata = item.get("metadata") or {}
    return RetrievedChunk(
        chunk_id=item["chunk_id"],
        document_id=item["document_id"],
        title=item["title"],
        content=item["content"],
        score=round(float(item["score"]), 6),
        metadata=metadata,
        page_start=item.get("parent_page_start") or metadata.get("page_start"),
        page_end=item.get("parent_page_end") or metadata.get("page_end"),
        parent_content=item.get("parent_content"),
    )


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------
@router.post("/ask", response_model=AskResponse, tags=["ask"])
async def ask(payload: AskRequest, db: AsyncSession = Depends(get_db)) -> AskResponse:
    try:
        return await ask_service.ask(db, payload)
    except LLMNotConfiguredError as exc:
        raise HTTPException(
            status_code=503,
            detail=api_error("llm_not_configured", str(exc)),
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=502,
            detail=api_error("embedding_failed", str(exc)),
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail=api_error("llm_failed", str(exc)),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=api_error("invalid_request", str(exc)),
        ) from exc
