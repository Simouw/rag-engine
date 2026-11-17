"""Hybrid retrieval: dense + lexical candidates, RRF fusion, rerank, parent context."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.config import settings
from rag_engine.llm.chat import chat_client
from rag_engine.llm.embeddings import embedding_client
from rag_engine.models import DocumentChunk
from rag_engine.retrieval.bm25 import bm25_service
from rag_engine.retrieval.vector_store import vector_store

RRF_K = 60


class RetrievalPipeline:
    async def search(
        self,
        db: AsyncSession,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("The search query is empty.")

        filters = filters or {}
        limit = max(1, min(top_k or settings.TOP_K, 50))
        query_embedding = await embedding_client.embed_query(query)

        results = await self._base_search(db, query, query_embedding, filters, limit)

        # Optional LLM reranking. It degrades gracefully to the fused order.
        if settings.RERANK_ENABLED and results:
            results = await chat_client.rerank(query, results)

        if settings.PARENT_CONTEXT_ENABLED and results:
            results = await self._attach_parent_context(db, results)

        return results[:limit]

    async def _base_search(
        self,
        db: AsyncSession,
        query: str,
        query_embedding: list[float],
        filters: dict,
        limit: int,
    ) -> list[dict]:
        candidates = max(limit, settings.CANDIDATES_PER_MODE)

        if not settings.HYBRID_SEARCH:
            return await vector_store.similarity_search(
                db=db,
                query_embedding=query_embedding,
                top_k=limit,
                filters=filters,
            )

        dense_results = await vector_store.similarity_search(
            db=db,
            query_embedding=query_embedding,
            top_k=candidates,
            filters=filters,
        )
        sparse_results = await bm25_service.search(
            db=db,
            query=query,
            top_k=candidates,
            filters=filters,
        )
        return self.reciprocal_rank_fusion(dense_results, sparse_results, limit=candidates)

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[dict],
        sparse_results: list[dict],
        limit: int,
    ) -> list[dict]:
        """RRF score = sum(1 / (k + rank)) across result lists."""
        scores: dict = {}
        chunks: dict = {}

        for ranked_list in (dense_results, sparse_results):
            for rank, item in enumerate(ranked_list):
                chunk_id = item["chunk_id"]
                chunks[chunk_id] = item
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        fused: list[dict] = []
        for chunk_id in ordered[:limit]:
            item = dict(chunks[chunk_id])
            item["score"] = scores[chunk_id]
            fused.append(item)
        return fused

    async def _attach_parent_context(
        self,
        db: AsyncSession,
        results: list[dict],
    ) -> list[dict]:
        """Swap child context for the full parent section when available."""
        parent_ids = {
            item.get("parent_chunk_id")
            for item in results
            if item.get("parent_chunk_id") is not None
        }
        if not parent_ids:
            return results

        rows = await db.execute(
            select(DocumentChunk).where(
                DocumentChunk.id.in_(parent_ids),
                DocumentChunk.chunk_role == "parent",
                DocumentChunk.is_enabled.is_(True),
            )
        )
        parents = {chunk.id: chunk for chunk in rows.scalars().all()}
        if not parents:
            return results

        for item in results:
            parent = parents.get(item.get("parent_chunk_id"))
            if parent is None:
                continue
            metadata = parent.metadata_json or {}
            item["parent_content"] = parent.content
            if metadata.get("page_start") is not None:
                item["parent_page_start"] = metadata.get("page_start")
                item["parent_page_end"] = metadata.get("page_end")
        return results


retrieval_pipeline = RetrievalPipeline()
