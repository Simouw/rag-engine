"""Optional server-level API key protection."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from rag_engine.api.errors import api_error
from rag_engine.config import settings


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """Accept `Authorization: Bearer <key>` or `X-API-Key: <key>`.

    When RAG_API_KEY is empty, the API is open. This is convenient for local
    use; set a key before exposing the service on a network.
    """
    expected = settings.RAG_API_KEY.strip()
    if not expected:
        return

    provided = (x_api_key or "").strip()
    if not provided and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided = value.strip()

    if not provided or not secrets.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail=api_error("unauthorized", "Missing or invalid API key."),
            headers={"WWW-Authenticate": "Bearer"},
        )
