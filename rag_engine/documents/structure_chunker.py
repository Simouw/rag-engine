"""
Structure-aware chunking.

A document is first split into sections (markdown headings, numbered headings,
keywords, uppercase titles). Each section becomes one or more "parent" chunks
(~2000 chars) that are sent to the LLM, and each parent is split into smaller
"child" chunks (~500 chars) that are embedded and searched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_engine.documents.chunker import Chunker

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,3})[.)]?\s+(.{2,100})$")
_ROMAN_HEADING_RE = re.compile(r"^([IVXLC]{1,5}(?:\.\d{1,3}){0,3})[.)]?\s+(.{2,100})$")
_ALPHA_HEADING_RE = re.compile(r"^([A-Z])[.)]\s+(.{2,100})$")
_KEYWORD_HEADING_RE = re.compile(
    r"^(chapters?|parts?|sections?|modules?|units?|titles?|appendi(?:x|ces)|"
    r"introductions?|conclusions?|overview|objectives?|definitions?|theorems?|"
    r"propert(?:y|ies)|remarks?|examples?|methods?|exercises?|solutions?|"
    r"summar(?:y|ies)|notes?|chapitre|partie|annexe|objectifs?|"
    r"d[ée]finitions?|th[ée]or[èe]mes?|propri[ée]t[ée]s?|remarques?|exemples?|"
    r"m[ée]thodes?|exercices?|corrections?|r[ée]sum[ée]s?|fiches?)\b[\s:.\-]*(.{0,100})$",
    re.IGNORECASE,
)
_PAGE_OR_SLIDE_MARKER_RE = re.compile(
    r"^\[(?:Page|Slide)\s+\d+\]\s*$", re.IGNORECASE | re.MULTILINE
)
_SENTENCE_END_RE = re.compile(r"[.!?;]\s*$")


@dataclass(frozen=True)
class ChunkNode:
    content: str
    role: str  # "parent" | "child"
    parent_key: int | None
    section_title: str | None
    section_path: list[str] = field(default_factory=list)
    heading_level: int | None = None


@dataclass
class _Section:
    title: str | None
    path: list[str]
    level: int | None
    lines: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n".join(self.lines).strip()


class StructureChunker:
    def __init__(
        self,
        child_chunk_size: int = 500,
        child_overlap: int = 50,
        parent_chunk_size: int = 2000,
    ) -> None:
        if parent_chunk_size < child_chunk_size:
            raise ValueError("parent_chunk_size must be >= child_chunk_size")
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.parent_chunk_size = parent_chunk_size

    def split(self, text: str) -> list[ChunkNode]:
        if not text or not text.strip():
            return []

        nodes: list[ChunkNode] = []
        for section in self._split_sections(text):
            content = section.content
            if not self._is_substantive(content):
                continue

            parents = Chunker(chunk_size=self.parent_chunk_size, overlap=0).split(content)
            for parent_content in parents:
                parent_content = parent_content.strip()
                if not self._is_substantive(parent_content):
                    continue

                parent_key = len(nodes)
                nodes.append(
                    ChunkNode(
                        content=parent_content,
                        role="parent",
                        parent_key=None,
                        section_title=section.title,
                        section_path=list(section.path),
                        heading_level=section.level,
                    )
                )

                children = Chunker(
                    chunk_size=self.child_chunk_size,
                    overlap=self.child_overlap,
                ).split(parent_content)
                for child_content in children:
                    child_content = child_content.strip()
                    if not child_content:
                        continue
                    nodes.append(
                        ChunkNode(
                            content=child_content,
                            role="child",
                            parent_key=parent_key,
                            section_title=section.title,
                            section_path=list(section.path),
                            heading_level=section.level,
                        )
                    )
        return nodes

    # ------------------------------------------------------------------
    # Section detection
    # ------------------------------------------------------------------
    def _split_sections(self, text: str) -> list[_Section]:
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []
        current = _Section(title=None, path=[], level=None)
        has_content = False
        # A page marker alone (e.g. "[Page 1]" emitted before a heading) must
        # survive so that chunk metadata can report page numbers. Carry it into
        # the next real section instead of creating a marker-only chunk.
        pending_markers: list[str] = []

        def flush() -> None:
            nonlocal has_content
            if has_content and current.content:
                if self._is_substantive(current.content):
                    sections.append(
                        _Section(
                            title=current.title,
                            path=list(current.path),
                            level=current.level,
                            lines=list(current.lines),
                        )
                    )
                elif _PAGE_OR_SLIDE_MARKER_RE.search(current.content):
                    pending_markers.extend(current.lines)
            has_content = False

        for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.rstrip()
            heading = self._detect_heading(line)
            if heading is not None:
                flush()
                level, title = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current = _Section(
                    title=title,
                    path=[item[1] for item in stack],
                    level=level,
                    lines=[*pending_markers, line.strip()],
                )
                pending_markers.clear()
                has_content = True
                continue

            current.lines.append(line)
            if line.strip():
                has_content = True

        flush()
        return sections

    def _detect_heading(self, raw_line: str) -> tuple[int, str] | None:
        line = raw_line.strip()
        if not line or len(line) > 140:
            return None
        if _PAGE_OR_SLIDE_MARKER_RE.match(line):
            return None
        if line.startswith(("-", "•", "*", "–", "—")):
            return None

        markdown = _MARKDOWN_HEADING_RE.match(line)
        if markdown:
            title = markdown.group(2).strip()
            return (len(markdown.group(1)), title) if title else None

        keyword = _KEYWORD_HEADING_RE.match(line)
        if keyword and len(line.split()) <= 14:
            return (1, line)

        numbered = _NUMBERED_HEADING_RE.match(line)
        if numbered and self._is_heading_title(numbered.group(2)):
            return (numbered.group(1).count(".") + 1, line)

        roman = _ROMAN_HEADING_RE.match(line)
        if roman and self._is_heading_title(roman.group(2)):
            return (1 + roman.group(1).count("."), line)

        alpha = _ALPHA_HEADING_RE.match(line)
        if alpha and self._is_heading_title(alpha.group(2)):
            return (2, line)

        if self._is_uppercase_heading(line):
            return (1, line)
        return None

    @staticmethod
    def _is_heading_title(title: str) -> bool:
        title = title.strip()
        if not title or len(title.split()) > 12:
            return False
        if _SENTENCE_END_RE.search(title):
            return False
        if "=" in title or "→" in title:
            return False
        return title[0].isupper()

    @staticmethod
    def _is_substantive(content: str) -> bool:
        without_markers = _PAGE_OR_SLIDE_MARKER_RE.sub("", content)
        return any(char.isalnum() for char in without_markers)

    @staticmethod
    def _is_uppercase_heading(line: str) -> bool:
        if not line.isupper():
            return False
        words = line.split()
        if len(words) < 2 or len(words) > 10:
            return False
        if sum(1 for char in line if char.isalpha()) < 4:
            return False
        return not _SENTENCE_END_RE.search(line)
