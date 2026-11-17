"""Local file storage. Keeping it simple keeps deployment simple."""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from rag_engine.config import ensure_directories, settings

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class StorageService:
    def __init__(self) -> None:
        ensure_directories()
        self.root = Path(settings.UPLOAD_DIR).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, content: bytes, original_filename: str) -> Path:
        suffix = Path(original_filename).suffix.lower()
        safe_stem = _SAFE_STEM.sub("_", Path(original_filename).stem)[:80] or "document"
        destination = self.root / f"{uuid.uuid4()}_{safe_stem}{suffix}"
        destination.write_bytes(content)
        return destination

    def exists(self, file_path: str | Path) -> bool:
        return Path(file_path).exists()

    def delete(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if path.exists():
            path.unlink()

    def read_bytes(self, file_path: str | Path) -> bytes:
        return Path(file_path).read_bytes()

    @contextmanager
    def local_path(self, file_path: str | Path):
        yield Path(file_path)


storage_service = StorageService()
