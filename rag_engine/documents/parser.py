"""Text extraction for PDF, DOCX, DOC, PPTX, TXT and Markdown."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
except ImportError:  # pragma: no cover - optional dependency at runtime
    Presentation = None
    MSO_SHAPE_TYPE = None

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".doc", ".docx", ".pptx"}


def _normalize_block(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _pptx_dependencies():
    global Presentation, MSO_SHAPE_TYPE
    if Presentation is not None and MSO_SHAPE_TYPE is not None:
        return Presentation, MSO_SHAPE_TYPE
    try:
        from pptx import Presentation as PptxPresentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE as PptxShapeType
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PowerPoint support requires python-pptx. Install it with `pip install python-pptx`."
        ) from exc
    Presentation = PptxPresentation
    MSO_SHAPE_TYPE = PptxShapeType
    return Presentation, MSO_SHAPE_TYPE


class TextParser:
    def parse(self, file_path: str) -> str:
        path = Path(file_path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")


class PDFParser:
    def parse(self, file_path: str) -> str:
        pages: list[str] = []
        with pymupdf.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(f"[Page {page_number}]\n{text}")
        return "\n\n".join(pages)


class DocxParser:
    def parse(self, file_path: str) -> str:
        doc = DocxDocument(file_path)
        parts: list[str] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)


class DocParser:
    """Legacy .doc files require LibreOffice (`soffice`) on the PATH."""

    def parse(self, file_path: str) -> str:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            raise RuntimeError(
                "Legacy .doc support requires LibreOffice in headless mode. "
                "Install it or convert the file to .docx first."
            )

        source = Path(file_path).resolve()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            command = [
                soffice,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(output_dir),
                str(source),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                details = (exc.stderr or exc.stdout or "").strip()
                raise RuntimeError(
                    f"Could not convert '{source.name}' to .docx."
                    + (f" Details: {details}" if details else "")
                ) from exc

            converted = output_dir / f"{source.stem}.docx"
            if not converted.exists():
                raise RuntimeError(f"Conversion produced no output for '{source.name}'.")
            return DocxParser().parse(str(converted))


class PptxParser:
    def parse(self, file_path: str) -> str:
        presentation_class = _pptx_dependencies()[0]
        presentation = presentation_class(file_path)
        slides: list[str] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            parts = self._extract_shapes(slide.shapes)
            notes = self._extract_notes(slide)
            if notes:
                parts.append(f"Notes:\n{notes}")
            slide_text = "\n\n".join(part for part in parts if part).strip()
            if slide_text:
                slides.append(f"[Slide {slide_number}]\n{slide_text}")
        return "\n\n".join(slides)

    def _extract_shapes(self, shapes) -> list[str]:
        parts: list[str] = []
        for shape in shapes:
            parts.extend(self._extract_shape(shape))
        return parts

    def _extract_shape(self, shape) -> list[str]:
        if self._is_group_shape(shape):
            return self._extract_shapes(shape.shapes)
        if getattr(shape, "has_table", False):
            return self._extract_table(shape.table)
        if getattr(shape, "has_text_frame", False):
            text = _normalize_block(shape.text_frame.text)
            return [text] if text else []
        return []

    def _extract_table(self, table) -> list[str]:
        rows: list[str] = []
        for row in table.rows:
            cells = [_normalize_block(cell.text) for cell in row.cells]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        return rows

    def _extract_notes(self, slide) -> str | None:
        try:
            notes = _normalize_block(slide.notes_slide.notes_text_frame.text)
        except AttributeError:
            return None
        if not notes:
            return None
        if notes.lower() in {"click to add notes", "cliquez pour ajouter des notes"}:
            return None
        return notes

    def _is_group_shape(self, shape) -> bool:
        if MSO_SHAPE_TYPE is None:
            return False
        return getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.GROUP


def get_parser(file_path: str):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PDFParser()
    if suffix == ".docx":
        return DocxParser()
    if suffix == ".doc":
        return DocParser()
    if suffix == ".pptx":
        return PptxParser()
    return TextParser()
