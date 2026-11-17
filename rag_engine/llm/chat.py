"""Generation and reranking through any OpenAI-compatible chat API."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI

from rag_engine.config import settings
from rag_engine.llm.prompts import (
    RERANK_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)
from rag_engine.logging import logger


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    @property
    def is_configured(self) -> bool:
        return settings.llm_configured

    @property
    def model_name(self) -> str:
        return settings.LLM_MODEL

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        if not self.is_configured:
            raise LLMNotConfiguredError(
                "The generation LLM is not configured. Set LLM_API_KEY and LLM_MODEL "
                "(or point LLM_BASE_URL at a local OpenAI-compatible server)."
            )
        self._client = AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY or "not-needed",
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=1,
        )
        logger.info(
            "LLM client ready (model=%s, base_url=%s)",
            settings.LLM_MODEL,
            settings.LLM_BASE_URL,
        )
        return self._client

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        client = self._get_client()
        kwargs: dict = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": (settings.LLM_TEMPERATURE if temperature is None else temperature),
        }
        tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        if tokens:
            kwargs["max_tokens"] = tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.exception("LLM generation failed")
            raise LLMError(f"LLM request failed: {exc}") from exc

        if not response.choices:
            raise LLMError("LLM returned no choices.")
        text = response.choices[0].message.content
        if not text:
            raise LLMError("LLM returned an empty answer.")

        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            model=getattr(response, "model", None) or settings.LLM_MODEL,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        )

    async def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Score candidate chunks with the LLM and blend with the RRF score."""
        if not chunks or not self.is_configured or not settings.RERANK_ENABLED:
            return chunks

        top_n = max(1, settings.RERANK_TOP_N)
        candidates = chunks[:top_n]
        remaining = chunks[top_n:]

        payload = [
            {
                "index": index,
                "chunk_id": str(item["chunk_id"]),
                "content": (item.get("rerank_text") or item["content"])[:2500],
            }
            for index, item in enumerate(candidates)
        ]
        prompt = (
            f"Query: {query}\n\n"
            f"Excerpts:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            'Return exactly: {"scores": [{"chunk_id": "...", "score": 0.0}]} '
            "with a score from 0 to 10 for every excerpt. Return JSON only."
        )

        try:
            result = await self.generate(
                system=RERANK_SYSTEM_PROMPT,
                user=prompt,
                temperature=0.0,
                max_tokens=800,
                json_mode=True,
            )
            data = json.loads(result.text)
        except Exception:
            logger.exception("Reranking failed; keeping the fused order")
            return chunks

        scores: dict[str, float] = {}
        for item in data.get("scores", []):
            try:
                scores[str(uuid.UUID(str(item.get("chunk_id"))))] = float(item.get("score", 0.0))
            except Exception:
                continue

        for item in candidates:
            llm_score = scores.get(str(item["chunk_id"]))
            if llm_score is None:
                item["score"] = item.get("score", 0.5) * 0.1
            else:
                item["score"] = (llm_score / 10.0) * 0.9 + item.get("score", 0.5) * 0.1

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates + remaining

    async def rewrite_query(self, current: str, history: list[dict]) -> str:
        """Turn a follow-up into a standalone search query. Falls back to the input."""
        if not history or not self.is_configured:
            return current
        recent = history[-6:]
        transcript = "\n".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
            for turn in recent
        )
        try:
            result = await self.generate(
                system=REWRITE_SYSTEM_PROMPT,
                user=(
                    f"Conversation:\n{transcript}\n\n"
                    f"Follow-up question: {current}\n\n"
                    "Standalone search query:"
                ),
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = result.text.strip().strip('"').strip("'")
            return rewritten[:500] if rewritten else current
        except Exception:
            logger.warning("Query rewrite failed; using the raw question")
            return current


chat_client = ChatClient()
