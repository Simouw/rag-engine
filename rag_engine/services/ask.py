"""Question answering: rewrite -> retrieve -> ground -> answer with citations."""

from __future__ import annotations

import re

from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.llm.chat import chat_client
from rag_engine.llm.prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt, format_context
from rag_engine.logging import logger
from rag_engine.retrieval.pipeline import retrieval_pipeline
from rag_engine.schemas import AskRequest, AskResponse, SourceOut, Usage

_FINAL_SOURCES_RE = re.compile(
    r"(?:\r?\n){1,3}\s*(?:#{1,6}\s*)?(?:sources?|references?)\s*:?\s*"
    r"(?:\r?\n[\s\S]*)?$",
    re.IGNORECASE,
)

NO_RESULTS_ANSWER = (
    "No sufficiently relevant passage was found in the indexed documents. "
    "Try reformulating the question, or ingest more documents first."
)


class AskService:
    async def ask(self, db: AsyncSession, request: AskRequest) -> AskResponse:
        filters = request.filters.model_dump() if request.filters else {}
        history = [turn.model_dump() for turn in request.history] if request.history else []

        rewritten_query: str | None = None
        query = request.question
        if history:
            rewritten = await chat_client.rewrite_query(request.question, history)
            if rewritten != request.question:
                query = rewritten
                rewritten_query = rewritten

        results = await retrieval_pipeline.search(
            db=db,
            query=query,
            top_k=request.top_k,
            filters=filters,
        )

        if not results:
            return AskResponse(
                answer=NO_RESULTS_ANSWER,
                sources=[],
                model=chat_client.model_name,
                usage=None,
                retrieved=0,
                rewritten_query=rewritten_query,
            )

        context = format_context(results)
        system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        try:
            completion = await chat_client.generate(
                system=system_prompt,
                user=build_user_prompt(request.question, context),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception:
            logger.exception("Answer generation failed")
            raise

        answer = self._strip_sources_section(completion.text)
        sources = [
            self._source_out(index, chunk, request.include_context)
            for index, chunk in enumerate(results, start=1)
        ]
        usage = Usage(
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
        )
        return AskResponse(
            answer=answer,
            sources=sources,
            model=completion.model,
            usage=usage,
            retrieved=len(results),
            rewritten_query=rewritten_query,
        )

    @staticmethod
    def _source_out(index: int, chunk: dict, include_context: bool) -> SourceOut:
        metadata = chunk.get("metadata") or {}
        page_start = chunk.get("parent_page_start") or metadata.get("page_start")
        page_end = chunk.get("parent_page_end") or metadata.get("page_end")
        return SourceOut(
            index=index,
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            title=chunk["title"],
            content=chunk["content"] if include_context else None,
            score=round(float(chunk["score"]), 6),
            page_start=page_start,
            page_end=page_end,
            metadata=metadata,
        )

    @staticmethod
    def _strip_sources_section(answer: str) -> str:
        cleaned = _FINAL_SOURCES_RE.sub("", answer or "").rstrip()
        return cleaned or answer


ask_service = AskService()
