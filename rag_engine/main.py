"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from rag_engine import __version__
from rag_engine.api import public_router, router
from rag_engine.config import ensure_directories, settings
from rag_engine.db import init_db
from rag_engine.logging import clear_request_id, logger, set_request_id


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_directories()
    if settings.AUTO_CREATE_TABLES:
        await init_db()
        logger.info("Database schema ready")

    if not settings.llm_configured:
        logger.warning(
            "The generation LLM is not configured. /search works, /ask will return 503 "
            "until LLM_API_KEY (or a local LLM_BASE_URL) is set."
        )
    if settings.embedding_provider == "local":
        logger.info(
            "Embeddings: local model %s (%d dims)",
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DIMENSION,
        )
    elif not settings.EMBEDDING_API_KEY.strip():
        logger.warning("EMBEDDING_PROVIDER=openai but EMBEDDING_API_KEY is empty")

    if settings.APP_ENV.lower() == "production" and not settings.RAG_API_KEY.strip():
        logger.warning("RAG_API_KEY is empty: the API is public on the network")

    yield
    logger.info("rag-engine stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="rag-engine",
        description=(
            "Self-hosted RAG engine with hybrid retrieval, parent/child chunking "
            "and source-grounded answers. Bring your own LLM and embedding keys."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    origins = settings.cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        set_request_id(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("%s %s failed", request.method, request.url.path)
            clear_request_id()
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%sms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        clear_request_id()
        return response

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": "rag-engine",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
            "api": "/api",
        }

    app.include_router(public_router, prefix="/api")
    app.include_router(router, prefix="/api")
    return app


app = create_app()
