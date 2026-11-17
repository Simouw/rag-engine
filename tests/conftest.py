"""Shared test setup: temp SQLite DB and deterministic embeddings (no network)."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="rag-engine-tests-"))
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP / 'test.db'}"
os.environ["DATA_DIR"] = str(_TMP / "data")
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["AUTO_CREATE_TABLES"] = "false"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_DIMENSION"] = "64"
os.environ["RERANK_ENABLED"] = "false"
os.environ["LLM_API_KEY"] = ""

import httpx  # noqa: E402

from rag_engine.db import Base, engine  # noqa: E402
from rag_engine.llm.chat import ChatResult, chat_client  # noqa: E402
from rag_engine.llm.embeddings import embedding_client  # noqa: E402
from rag_engine.main import app  # noqa: E402


def fake_vector(text: str, dimension: int = 64) -> list[float]:
    """Deterministic bag-of-words vector: shared words => similar vectors."""
    vector = [0.0] * dimension
    for word in re.findall(r"[^\W_]+", text.lower(), re.UNICODE):
        digest = hashlib.sha256(word.encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        vector[bucket] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else [0.0] * dimension


@pytest.fixture(autouse=True)
async def clean_database():
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture(autouse=True)
def offline_providers(monkeypatch):
    async def embed_documents(texts: list[str]) -> list[list[float]]:
        return [fake_vector(text) for text in texts]

    async def embed_query(text: str) -> list[float]:
        return fake_vector(text)

    async def generate(**kwargs) -> ChatResult:
        return ChatResult(
            text="The documents explain photosynthesis [1].",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
        )

    async def identity_rerank(query: str, chunks: list[dict]) -> list[dict]:
        return chunks

    monkeypatch.setattr(embedding_client, "embed_documents", embed_documents)
    monkeypatch.setattr(embedding_client, "embed_query", embed_query)
    monkeypatch.setattr(chat_client, "generate", generate)
    monkeypatch.setattr(chat_client, "rerank", identity_rerank)
    yield


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
