"""Document ingestion: store -> parse -> structure-aware chunk -> embed."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_engine.config import settings
from rag_engine.documents.parser import SUPPORTED_EXTENSIONS, get_parser
from rag_engine.documents.structure_chunker import ChunkNode, StructureChunker
from rag_engine.llm.embeddings import embedding_client
from rag_engine.logging import logger
from rag_engine.models import Document, DocumentChunk
from rag_engine.storage import storage_service

PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]")

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class IngestionError(RuntimeError):
    pass


class IngestionService:
    async def ingest_path(
        self,
        db: AsyncSession,
        path: Path,
        *,
        collection: str = "default",
        doc_type: str = "document",
        tags: list[str] | None = None,
        language: str | None = None,
        title: str | None = None,
        source_key: str | None = None,
        embed: bool = True,
    ) -> tuple[Document, int]:
        if not path.exists() or not path.is_file():
            raise IngestionError(f"File not found: {path}")
        content = path.read_bytes()
        return await self.ingest_bytes(
            db,
            content=content,
            original_filename=path.name,
            collection=collection,
            doc_type=doc_type,
            tags=tags,
            language=language,
            title=title or path.stem.replace("_", " ").strip() or path.name,
            source_key=source_key,
            embed=embed,
        )

    async def ingest_bytes(
        self,
        db: AsyncSession,
        *,
        content: bytes,
        original_filename: str,
        collection: str = "default",
        doc_type: str = "document",
        tags: list[str] | None = None,
        language: str | None = None,
        title: str | None = None,
        source_key: str | None = None,
        embed: bool = True,
    ) -> tuple[Document, int]:
        suffix = Path(original_filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise IngestionError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if not content:
            raise IngestionError("The file is empty.")
        if len(content) > settings.max_upload_bytes:
            raise IngestionError(
                f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
                f"Limit: {settings.MAX_UPLOAD_MB} MB."
            )

        saved_path = storage_service.save(content, original_filename)
        old_path: str | None = None
        try:
            document, old_path = await self._upsert_document(
                db,
                saved_path=saved_path,
                original_filename=original_filename,
                collection=collection,
                doc_type=doc_type,
                tags=tags,
                language=language,
                title=title or Path(original_filename).stem,
                source_key=source_key,
            )
            # Savepoint: parsing/embedding failures leave the session usable and
            # can be rolled back without touching the document upsert.
            async with db.begin_nested():
                await self._parse_and_chunk(db, document)
                chunks = await self._embed_children(db, document) if embed else document.chunk_count
        except Exception as exc:
            # Restore the previous state and remove the freshly written file.
            storage_service.delete(saved_path)
            await db.rollback()
            logger.exception("Ingestion failed for %s", original_filename)
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(str(exc)) from exc

        # Only remove the previous file once the new version is fully indexed.
        if old_path and old_path != str(saved_path):
            storage_service.delete(old_path)
        return document, chunks

    async def _upsert_document(
        self,
        db: AsyncSession,
        *,
        saved_path: Path,
        original_filename: str,
        collection: str,
        doc_type: str,
        tags: list[str] | None,
        language: str | None,
        title: str,
        source_key: str | None,
    ) -> tuple[Document, str | None]:
        existing = None
        if source_key:
            existing = (
                await db.execute(select(Document).where(Document.source_key == source_key))
            ).scalar_one_or_none()

        normalized_tags = self._normalize_tags(tags)

        if existing is not None:
            old_path = existing.file_path
            existing.title = title
            existing.source_name = original_filename
            existing.collection = collection
            existing.doc_type = doc_type
            existing.tags = normalized_tags
            existing.language = language
            existing.file_path = str(saved_path)
            existing.mime_type = MIME_TYPES.get(Path(original_filename).suffix.lower())
            existing.status = "uploaded"
            existing.chunk_count = 0
            document = existing
        else:
            old_path = None
            document = Document(
                title=title,
                source_key=source_key,
                source_name=original_filename,
                collection=collection,
                doc_type=doc_type,
                tags=normalized_tags,
                language=language,
                file_path=str(saved_path),
                mime_type=MIME_TYPES.get(Path(original_filename).suffix.lower()),
                status="uploaded",
            )
            db.add(document)

        await db.flush()
        await db.refresh(document)
        return document, old_path

    async def _parse_and_chunk(self, db: AsyncSession, document: Document) -> int:
        with storage_service.local_path(document.file_path) as local_path:
            text = await asyncio.to_thread(self._parse_text, str(local_path))
            is_pdf = Path(local_path).suffix.lower() == ".pdf"

        nodes = StructureChunker(
            child_chunk_size=settings.CHUNK_SIZE,
            child_overlap=settings.CHUNK_OVERLAP,
            parent_chunk_size=settings.PARENT_CHUNK_SIZE,
        ).split(text)
        if not nodes:
            raise IngestionError("No usable text could be extracted from this document.")

        page_ranges = (
            self._extract_node_page_ranges(nodes) if is_pdf else [(None, None)] * len(nodes)
        )

        await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))

        node_ids = [uuid.uuid4() for _ in nodes]
        base_metadata = {
            "source_file": document.source_name,
            "doc_type": document.doc_type,
            "collection": document.collection,
            "tags": document.tags,
            "language": document.language,
        }

        parents: list[DocumentChunk] = []
        children: list[DocumentChunk] = []
        for index, (node, node_id) in enumerate(zip(nodes, node_ids, strict=True)):
            page_start, page_end = page_ranges[index]
            metadata = {
                **base_metadata,
                "section_title": node.section_title,
                "section_path": node.section_path,
                "heading_level": node.heading_level,
                "chunk_role": node.role,
            }
            if page_start is not None:
                metadata["page_start"] = page_start
                metadata["page_end"] = page_end if page_end is not None else page_start

            model = DocumentChunk(
                id=node_id,
                document_id=document.id,
                chunk_index=index,
                content=node.content,
                metadata_json=metadata,
                chunk_role=node.role,
                parent_chunk_id=node_ids[node.parent_key] if node.parent_key is not None else None,
                is_enabled=True,
            )
            (parents if node.role == "parent" else children).append(model)

        # Parents must be inserted before children (self-referencing FK).
        db.add_all(parents)
        await db.flush()
        db.add_all(children)
        await db.flush()

        document.status = "parsed"
        document.chunk_count = len(children)
        document.error_message = None
        await db.flush()
        logger.info(
            "Parsed document %s: %d children, %d parents",
            document.id,
            len(children),
            len(parents),
        )
        return len(children)

    async def _embed_children(self, db: AsyncSession, document: Document) -> int:
        rows = await db.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == document.id,
                DocumentChunk.chunk_role == "child",
            )
            .order_by(DocumentChunk.chunk_index.asc())
        )
        chunks = list(rows.scalars().all())
        if not chunks:
            raise IngestionError("No child chunks to embed.")

        texts = [self._embedding_text(document, chunk) for chunk in chunks]
        vectors = await embedding_client.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise IngestionError("Embedding provider returned an unexpected vector count.")

        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector

        document.status = "indexed"
        document.chunk_count = len(chunks)
        await db.flush()
        await db.refresh(document)
        logger.info("Indexed document %s: %d chunks", document.id, len(chunks))
        return len(chunks)

    async def reindex(self, db: AsyncSession, document: Document) -> int:
        if not storage_service.exists(document.file_path):
            raise IngestionError("The original file is missing from storage.")
        try:
            async with db.begin_nested():
                await self._parse_and_chunk(db, document)
                return await self._embed_children(db, document)
        except Exception as exc:
            await db.refresh(document)
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(str(exc)) from exc

    async def delete(self, db: AsyncSession, document: Document) -> None:
        storage_service.delete(document.file_path)
        await db.delete(document)
        await db.flush()

    async def delete_collection(self, db: AsyncSession, collection: str | None) -> int:
        stmt = select(Document)
        if collection:
            stmt = stmt.where(Document.collection == collection)
        documents = list((await db.execute(stmt)).scalars().all())
        for document in documents:
            storage_service.delete(document.file_path)
            await db.delete(document)
        await db.flush()
        return len(documents)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_text(file_path: str) -> str:
        return get_parser(file_path).parse(file_path)

    @staticmethod
    def _normalize_tags(tags: list[str] | None) -> list[str] | None:
        if not tags:
            return None
        cleaned: list[str] = []
        for tag in tags:
            value = str(tag).strip().lower()
            if value and value not in cleaned:
                cleaned.append(value)
        return cleaned or None

    @staticmethod
    def _embedding_text(document: Document, chunk: DocumentChunk) -> str:
        metadata = chunk.metadata_json or {}
        section_path = metadata.get("section_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]
        heading = " > ".join(str(part).strip() for part in section_path if str(part).strip())

        parts: list[str] = []
        if document.title.strip():
            parts.append(document.title.strip())
        if heading:
            parts.append(heading)
        parts.append(chunk.content)
        return "\n".join(parts)

    @staticmethod
    def _extract_node_page_ranges(
        nodes: list[ChunkNode],
    ) -> list[tuple[int | None, int | None]]:
        ranges: list[tuple[int | None, int | None]] = []
        last_page_by_role: dict[str, int | None] = {"parent": None, "child": None}
        for node in nodes:
            matches = [int(match) for match in PAGE_MARKER_RE.findall(node.content)]
            previous = last_page_by_role[node.role]
            if matches:
                start = min(previous, matches[0]) if previous is not None else matches[0]
                end = max(matches)
                last_page_by_role[node.role] = end
            else:
                start = end = previous
            ranges.append((start, end))
        return ranges


ingestion_service = IngestionService()
