"""Dense vector search, with pgvector on PostgreSQL and Python cosine on SQLite."""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import cast, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.models import Document, DocumentChunk


class VectorStore:
    async def similarity_search(
        self,
        db: AsyncSession,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            raise ValueError("Query embedding is empty.")

        top_k = max(1, min(top_k, 100))
        filters = filters or {}
        if db.get_bind().dialect.name == "postgresql":
            return await self._postgres_search(db, query_embedding, top_k, filters)
        return await self._python_search(db, query_embedding, top_k, filters)

    async def _postgres_search(
        self,
        db: AsyncSession,
        query_embedding: list[float],
        top_k: int,
        filters: dict,
    ) -> list[dict[str, Any]]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(DocumentChunk, Document, distance.label("distance"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.embedding.is_not(None))
        )
        stmt = self.apply_filters(stmt, filters)
        stmt = self._apply_tag_containment(stmt, filters)
        # Over-fetch so that Python-side tag filtering does not starve the result.
        stmt = stmt.order_by(distance.asc()).limit(max(top_k * 3, top_k))

        rows = (await db.execute(stmt)).all()
        matches: list[dict[str, Any]] = []
        for chunk, document, distance_value in rows:
            if not self.document_matches_python_filters(document, filters):
                continue
            matches.append(self.format_result(chunk, document, 1.0 - float(distance_value)))
            if len(matches) >= top_k:
                break
        return matches

    async def _python_search(
        self,
        db: AsyncSession,
        query_embedding: list[float],
        top_k: int,
        filters: dict,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.embedding.is_not(None))
        )
        stmt = self.apply_filters(stmt, filters)

        rows = (await db.execute(stmt)).all()
        scored: list[dict[str, Any]] = []
        for chunk, document in rows:
            if chunk.embedding is None:
                continue
            if not self.document_matches_python_filters(document, filters):
                continue
            score = self.cosine_similarity(query_embedding, list(chunk.embedding))
            scored.append(self.format_result(chunk, document, score))

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def apply_filters(self, stmt, filters: dict):
        # Only enabled child chunks are searchable.
        stmt = stmt.where(
            DocumentChunk.chunk_role == "child",
            DocumentChunk.is_enabled.is_(True),
        )
        if filters.get("collection"):
            stmt = stmt.where(Document.collection == filters["collection"])
        if filters.get("doc_type"):
            stmt = stmt.where(Document.doc_type == filters["doc_type"])
        if filters.get("language"):
            stmt = stmt.where(Document.language == filters["language"])
        if filters.get("document_ids"):
            stmt = stmt.where(Document.id.in_(filters["document_ids"]))
        return stmt

    @staticmethod
    def _apply_tag_containment(stmt, filters: dict):
        tags = filters.get("tags") or []
        if not tags:
            return stmt
        # PostgreSQL: JSONB containment means "has all requested tags".
        return stmt.where(cast(Document.tags, JSONB).contains(tags))

    @staticmethod
    def clean_tags(tags: list[str] | None) -> list[str]:
        if not tags:
            return []
        cleaned: list[str] = []
        for tag in tags:
            value = str(tag).strip().lower()
            if value and value not in cleaned:
                cleaned.append(value)
        return cleaned

    def document_matches_python_filters(self, document: Document, filters: dict) -> bool:
        wanted = set(self.clean_tags(filters.get("tags")))
        if not wanted:
            return True
        document_tags = set(self.clean_tags(document.tags))
        return wanted.issubset(document_tags)

    @staticmethod
    def format_result(
        chunk: DocumentChunk,
        document: Document,
        score: float,
    ) -> dict[str, Any]:
        metadata = dict(chunk.metadata_json or {})
        metadata.update(
            {
                "collection": document.collection,
                "doc_type": document.doc_type,
                "tags": document.tags,
                "language": document.language,
                "source_name": document.source_name,
            }
        )
        section_path = metadata.get("section_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]
        heading = " > ".join(str(part).strip() for part in section_path if str(part).strip())
        rerank_text = f"{heading}\n{chunk.content}" if heading else chunk.content
        return {
            "chunk_id": chunk.id,
            "document_id": document.id,
            "title": document.title,
            "content": chunk.content,
            "rerank_text": rerank_text,
            "parent_chunk_id": chunk.parent_chunk_id,
            "score": float(score),
            "metadata": metadata,
        }

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)


vector_store = VectorStore()
