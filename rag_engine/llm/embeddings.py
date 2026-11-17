"""
Embeddings with a bring-your-own-keys design.

- `EMBEDDING_PROVIDER=local`  -> fastembed ONNX model, no API key, runs on CPU.
- `EMBEDDING_PROVIDER=openai` -> any OpenAI-compatible /embeddings endpoint
  (OpenAI, Gemini OpenAI-compat, Mistral, Ollama, LM Studio, vLLM...).

The vector dimension is fixed when the database schema is created. Changing the
model or the dimension therefore requires a fresh database (or a migration).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from openai import AsyncOpenAI

from rag_engine.config import settings
from rag_engine.logging import logger


class EmbeddingError(RuntimeError):
    pass


class LocalEmbeddingBackend:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError(
                    "fastembed is not installed. Install it with "
                    "`pip install 'rag-engine[local]'` or set EMBEDDING_PROVIDER=openai."
                ) from exc
            logger.info(
                "Loading local embedding model %s (cache=%s)",
                settings.EMBEDDING_MODEL,
                settings.EMBEDDING_CACHE_DIR,
            )
            self._model = TextEmbedding(
                model_name=settings.EMBEDDING_MODEL,
                cache_dir=settings.EMBEDDING_CACHE_DIR,
            )
            return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.embed(texts)
        return [[float(value) for value in vector] for vector in vectors]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)


class OpenAIEmbeddingBackend:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        if (
            not settings.EMBEDDING_API_KEY.strip()
            and settings.EMBEDDING_BASE_URL.strip().rstrip("/") == "https://api.openai.com/v1"
        ):
            raise EmbeddingError(
                "EMBEDDING_API_KEY is missing. Set it, or use EMBEDDING_PROVIDER=local."
            )
        self._client = AsyncOpenAI(
            base_url=settings.EMBEDDING_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY or "not-needed",
            timeout=90.0,
            max_retries=1,
        )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in ordered]


class EmbeddingClient:
    def __init__(self) -> None:
        self._local = LocalEmbeddingBackend()
        self._openai = OpenAIEmbeddingBackend()

    @property
    def provider(self) -> str:
        return settings.embedding_provider

    @property
    def model_name(self) -> str:
        return settings.EMBEDDING_MODEL

    def _backend(self):
        if settings.embedding_provider in {"local", "fastembed", "onnx"}:
            return self._local
        if settings.embedding_provider in {"openai", "api", "openai-compatible"}:
            return self._openai
        raise EmbeddingError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. Use 'local' or 'openai'."
        )

    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        prefix = settings.EMBEDDING_QUERY_PREFIX.strip()
        payload = f"{prefix}{text}" if prefix else text
        vectors = await self._embed_batch([payload])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefix = settings.EMBEDDING_DOCUMENT_PREFIX.strip()
        payloads = [f"{prefix}{text}" if prefix else text for text in texts]
        return await self._embed_batch(payloads)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        backend = self._backend()
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._with_retries(backend, batch))
        self._validate_dimensions(vectors)
        return vectors

    async def _with_retries(self, backend, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await backend.embed(batch)
            except EmbeddingError:
                raise
            except Exception as exc:  # network, rate limit, provider error
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise EmbeddingError(f"Embedding request failed: {last_error}") from last_error

    @staticmethod
    def _validate_dimensions(vectors: list[list[float]]) -> None:
        expected = settings.EMBEDDING_DIMENSION
        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: got {len(vector)}, expected {expected}. "
                    "Check EMBEDDING_MODEL/EMBEDDING_DIMENSION and reindex in a fresh database."
                )


embedding_client = EmbeddingClient()
