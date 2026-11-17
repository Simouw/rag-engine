from sqlalchemy import func, select

from rag_engine.db import async_session_factory
from rag_engine.models import Document, DocumentChunk
from rag_engine.services.ingestion import ingestion_service


async def test_reingesting_the_same_source_updates_in_place(tmp_path):
    source = tmp_path / "course.txt"
    source.write_text("# Section\n\nFirst version of the content.", encoding="utf-8")

    async with async_session_factory() as db:
        first, first_chunks = await ingestion_service.ingest_path(
            db,
            source,
            collection="courses",
            source_key="stable-key",
        )
        await db.commit()
        first_id = first.id

        source.write_text(
            "# Section\n\nSecond version of the content, with more detail.",
            encoding="utf-8",
        )

        second, second_chunks = await ingestion_service.ingest_path(
            db,
            source,
            collection="courses",
            source_key="stable-key",
        )
        await db.commit()

        documents = (await db.execute(select(Document))).scalars().all()
        assert len(documents) == 1
        assert second.id == first_id
        assert second_chunks >= 1
        assert first_chunks >= 1

        children = (
            await db.execute(
                select(func.count())
                .select_from(DocumentChunk)
                .where(
                    DocumentChunk.document_id == first_id,
                    DocumentChunk.chunk_role == "child",
                )
            )
        ).scalar_one()
        assert int(children) == second_chunks
