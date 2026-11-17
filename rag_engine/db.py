"""Database engine, session factory and ORM base."""

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from rag_engine.config import ensure_directories, settings
from rag_engine.logging import logger

ensure_directories()

_engine_kwargs: dict = {"echo": False, "future": True}
if settings.is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Supabase/poolers do not support asyncpg prepared statement caching.
    if "pooler.supabase.com:6543" in settings.DATABASE_URL:
        _engine_kwargs["poolclass"] = NullPool
        _engine_kwargs["connect_args"] = {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        }

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, commit on success."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create tables and the optional PostgreSQL indexes/extensions."""
    # Importing models registers them on Base.metadata.
    from rag_engine import models  # noqa: F401

    async with engine.begin() as connection:
        if settings.is_postgres:
            try:
                # Savepoint: a missing extension permission must not abort the
                # transaction that creates the tables.
                async with connection.begin_nested():
                    await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception:
                logger.warning("Could not enable the pgvector extension")
        await connection.run_sync(Base.metadata.create_all)

        if settings.is_postgres:
            await _create_postgres_indexes(connection)


async def _create_postgres_indexes(connection) -> None:
    language = settings.fts_language
    statements = [
        (
            "ix_document_chunks_content_fts",
            f"CREATE INDEX IF NOT EXISTS ix_document_chunks_content_fts "
            f"ON document_chunks USING gin (to_tsvector('{language}', content))",
        ),
        (
            "ix_document_chunks_embedding_hnsw",
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding vector_cosine_ops)",
        ),
    ]
    for name, statement in statements:
        try:
            # One savepoint per index so a failure cannot poison the others.
            async with connection.begin_nested():
                await connection.execute(text(statement))
        except Exception:
            logger.warning("Could not create index %s", name)


async def ping_db() -> bool:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    return True
