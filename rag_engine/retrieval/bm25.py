"""
Sparse lexical search.

PostgreSQL uses native full-text search (to_tsvector + ts_rank_cd), which is
much faster than tokenizing thousands of chunks in Python. SQLite and tests use
a compact BM25 Okapi implementation as a fallback.
"""

from __future__ import annotations

import math
import re
from typing import Any

from sqlalchemy import cast, func, literal_column, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.config import settings
from rag_engine.models import Document, DocumentChunk
from rag_engine.retrieval.vector_store import vector_store

BM25_MAX_CANDIDATES = 5000
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if corpus else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}

        document_frequency: dict[str, int] = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies: dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                document_frequency[word] = document_frequency.get(word, 0) + 1

        for word, frequency in document_frequency.items():
            self.idf[word] = math.log(
                1.0 + (self.corpus_size - frequency + 0.5) / (frequency + 0.5)
            )
        self.k1 = k1
        self.b = b

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for index in range(self.corpus_size):
            doc_len = self.doc_len[index]
            frequencies = self.doc_freqs[index]
            for word in query:
                if word not in self.idf:
                    continue
                frequency = frequencies.get(word, 0)
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
                scores[index] += self.idf[word] * numerator / denominator
        return scores


def run_bm25(raw_data: list[dict], query: str, top_k: int) -> list[dict[str, Any]]:
    corpus = [tokenize(item["content"]) for item in raw_data]
    if not corpus:
        return []

    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    scored: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        if score <= 0.0:
            continue
        item = raw_data[index]
        metadata = dict(item["metadata_json"] or {})
        metadata.update(
            {
                "collection": item["collection"],
                "doc_type": item["doc_type"],
                "tags": item["tags"],
                "language": item["language"],
                "source_name": item["source_name"],
            }
        )
        scored.append(
            {
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "title": item["title"],
                "content": item["content"],
                "rerank_text": item["content"],
                "parent_chunk_id": item["parent_chunk_id"],
                "score": float(score),
                "metadata": metadata,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


class BM25Service:
    async def search(
        self,
        db: AsyncSession,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        filters = filters or {}
        top_k = max(1, min(top_k, 100))
        if db.get_bind().dialect.name == "postgresql":
            return await self._postgres_search(db, query, top_k, filters)
        return await self._python_search(db, query, top_k, filters)

    async def _postgres_search(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        filters: dict,
    ) -> list[dict[str, Any]]:
        language = literal_column(f"'{settings.fts_language}'")
        tsvector = func.to_tsvector(language, DocumentChunk.content)
        tsquery = func.websearch_to_tsquery(language, query)
        rank = func.ts_rank_cd(tsvector, tsquery)

        stmt = (
            select(DocumentChunk, Document, rank.label("rank"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(tsvector.op("@@")(tsquery))
        )
        stmt = vector_store.apply_filters(stmt, filters)
        tags = vector_store.clean_tags(filters.get("tags"))
        if tags:
            stmt = stmt.where(cast(Document.tags, JSONB).contains(tags))
        stmt = stmt.order_by(rank.desc()).limit(max(top_k * 2, top_k))

        rows = (await db.execute(stmt)).all()
        return [
            vector_store.format_result(chunk, document, score) for chunk, document, score in rows
        ][:top_k]

    async def _python_search(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        filters: dict,
    ) -> list[dict[str, Any]]:
        stmt = select(DocumentChunk, Document).join(
            Document, Document.id == DocumentChunk.document_id
        )
        stmt = vector_store.apply_filters(stmt, filters)
        # Deterministic ordering makes candidate truncation reproducible.
        stmt = stmt.order_by(DocumentChunk.id).limit(BM25_MAX_CANDIDATES)
        rows = (await db.execute(stmt)).all()
        if not rows:
            return []

        raw_data: list[dict] = []
        for chunk, document in rows:
            if not vector_store.document_matches_python_filters(document, filters):
                continue
            raw_data.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "title": document.title,
                    "content": chunk.content or "",
                    "metadata_json": chunk.metadata_json,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "collection": document.collection,
                    "doc_type": document.doc_type,
                    "tags": document.tags,
                    "language": document.language,
                    "source_name": document.source_name,
                }
            )
        if not raw_data:
            return []

        import asyncio

        return await asyncio.to_thread(run_bm25, raw_data, query, top_k)


bm25_service = BM25Service()
