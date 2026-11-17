"""Command line interface: initdb, ingest, reset, stats."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import func, select

from rag_engine.config import ensure_directories, settings
from rag_engine.db import async_session_factory, init_db
from rag_engine.documents.parser import SUPPORTED_EXTENSIONS
from rag_engine.logging import logger
from rag_engine.models import Document, DocumentChunk
from rag_engine.services.ingestion import ingestion_service


def _source_key(collection: str, relative_path: str) -> str:
    return hashlib.sha256(f"{collection}:{relative_path}".encode()).hexdigest()


def _discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    files: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if any(part.startswith(".") for part in candidate.relative_to(path).parts):
            continue
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(candidate)
    return files


def _tags_for(path: Path, root: Path, extra: list[str]) -> list[str]:
    tags = list(extra)
    if path.is_file() and root.is_dir():
        for part in path.relative_to(root).parts[:-1]:
            value = part.strip().lower()
            if value and value not in tags:
                tags.append(value)
    return tags


async def _cmd_initdb(_: argparse.Namespace) -> dict:
    ensure_directories()
    await init_db()
    return {"ok": True, "database": "initialized"}


async def _cmd_ingest(args: argparse.Namespace) -> dict:
    ensure_directories()
    if args.init_db:
        await init_db()

    if args.embedding_provider:
        settings.EMBEDDING_PROVIDER = args.embedding_provider

    paths = [Path(value).expanduser().resolve() for value in args.paths]
    files: list[tuple[Path, Path, str]] = []
    skipped: list[str] = []
    for root in paths:
        if not root.exists():
            skipped.append(f"{root} (not found)")
            continue
        if root.is_file() and root.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(f"{root} (unsupported)")
            continue
        collection = args.collection or (root.parent.name if root.is_file() else root.name)
        for file_path in _discover(root):
            files.append((file_path, root, collection))

    if args.limit is not None:
        files = files[: max(0, args.limit)]

    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "files": len(files),
            "documents": [
                {
                    "file": str(file_path),
                    "collection": collection,
                    "doc_type": args.doc_type,
                    "tags": _tags_for(file_path, root, args.tags or []),
                }
                for file_path, root, collection in files
            ],
            "skipped": skipped,
        }

    if not files:
        return {"ok": False, "error": "No supported files found.", "skipped": skipped}

    ingested = 0
    failed: list[dict[str, str]] = []

    async with async_session_factory() as db:
        for index, (file_path, root, collection) in enumerate(files, start=1):
            relative = str(file_path.relative_to(root)) if root.is_dir() else file_path.name
            try:
                document, chunks = await ingestion_service.ingest_path(
                    db,
                    file_path,
                    collection=collection,
                    doc_type=args.doc_type,
                    tags=_tags_for(file_path, root, args.tags or []),
                    language=args.language,
                    source_key=_source_key(collection, relative),
                    embed=not args.no_embed,
                )
                await db.commit()
                ingested += 1
                print(
                    f"[{index}/{len(files)}] {relative} -> {document.status} ({chunks} chunks)",
                    file=sys.stderr,
                )
            except Exception as exc:
                await db.rollback()
                logger.exception("Ingestion failed for %s", file_path)
                failed.append({"file": str(file_path), "error": str(exc)})
                print(f"[{index}/{len(files)}] FAILED {relative}: {exc}", file=sys.stderr)

    return {
        "ok": not failed,
        "ingested": ingested,
        "failed": len(failed),
        "failures": failed,
        "skipped": skipped,
        "embedding_provider": settings.embedding_provider,
        "no_embed": args.no_embed,
    }


async def _cmd_reset(args: argparse.Namespace) -> dict:
    if not args.yes:
        return {
            "ok": False,
            "error": "Refusing to delete documents without --yes.",
        }
    async with async_session_factory() as db:
        deleted = await ingestion_service.delete_collection(db, args.collection)
        await db.commit()
    return {"ok": True, "deleted_documents": deleted, "collection": args.collection}


async def _cmd_stats(_: argparse.Namespace) -> dict:
    async with async_session_factory() as db:
        documents = (await db.execute(select(func.count()).select_from(Document))).scalar_one()
        chunks = (
            await db.execute(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.chunk_role == "child")
            )
        ).scalar_one()
        collections = list(
            (await db.execute(select(Document.collection).distinct().order_by(Document.collection)))
            .scalars()
            .all()
        )
        by_status = {
            status: count
            for status, count in (
                await db.execute(select(Document.status, func.count()).group_by(Document.status))
            ).all()
        }
    return {
        "ok": True,
        "documents": int(documents),
        "child_chunks": int(chunks),
        "collections": collections,
        "by_status": by_status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-engine",
        description="Self-hosted RAG engine: ingest documents, search, ask.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initdb = subparsers.add_parser("initdb", help="Create database tables.")
    initdb.set_defaults(func=_cmd_initdb)

    ingest = subparsers.add_parser("ingest", help="Ingest files or directories.")
    ingest.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    ingest.add_argument(
        "--collection", default=None, help="Collection name (default: folder name)."
    )
    ingest.add_argument("--doc-type", default="document", help="Document type label.")
    ingest.add_argument("--tags", nargs="*", default=[], help="Extra tags applied to every file.")
    ingest.add_argument("--language", default=None, help="Optional language code (en, fr, ro...).")
    ingest.add_argument("--no-embed", action="store_true", help="Chunk but do not embed.")
    ingest.add_argument("--limit", type=int, default=None, help="Maximum number of files.")
    ingest.add_argument("--dry-run", action="store_true", help="List files without writing.")
    ingest.add_argument("--init-db", action="store_true", help="Create tables before ingesting.")
    ingest.add_argument(
        "--embedding-provider",
        default=None,
        choices=["local", "openai"],
        help="Override EMBEDDING_PROVIDER for this run.",
    )
    ingest.set_defaults(func=_cmd_ingest)

    reset = subparsers.add_parser("reset", help="Delete documents (and their files).")
    reset.add_argument("--collection", default=None, help="Only delete this collection.")
    reset.add_argument("--yes", action="store_true", help="Required confirmation flag.")
    reset.set_defaults(func=_cmd_reset)

    stats = subparsers.add_parser("stats", help="Show document/chunk counts.")
    stats.set_defaults(func=_cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = asyncio.run(args.func(args))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(payload, ensure_ascii=False, default=str))
    if not payload.get("ok", True):
        raise SystemExit(1)
