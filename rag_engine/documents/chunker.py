"""Recursive text splitter: paragraphs -> lines -> sentences -> words -> chars."""

from __future__ import annotations

from rag_engine.config import settings


class Chunker:
    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be zero or positive")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        if not separators:
            chunks = []
            step = self.chunk_size - self.overlap
            for i in range(0, len(text), step):
                chunk = text[i : i + self.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            splits = list(text)
        elif separator in {". ", "? ", "! "}:
            # Keep sentence-ending punctuation attached to the sentence.
            splits = []
            start = 0
            while True:
                index = text.find(separator, start)
                if index == -1:
                    splits.append(text[start:])
                    break
                splits.append(text[start : index + len(separator)])
                start = index + len(separator)
        else:
            splits = text.split(separator)

        final_chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in splits:
            if not part:
                continue

            if len(part) > self.chunk_size:
                if current:
                    final_chunks.append(self._merge(current, separator))
                    current = []
                    current_len = 0
                final_chunks.extend(self._recursive_split(part, next_separators))
                continue

            separator_len = len(separator) if current else 0
            if current_len + separator_len + len(part) > self.chunk_size:
                if current:
                    final_chunks.append(self._merge(current, separator))

                # Carry the tail of the previous chunk as overlap.
                overlap_parts: list[str] = []
                overlap_len = 0
                for previous in reversed(current):
                    part_sep_len = len(separator) if overlap_parts else 0
                    if overlap_len + part_sep_len + len(previous) <= self.overlap:
                        overlap_parts.insert(0, previous)
                        overlap_len += part_sep_len + len(previous)
                    else:
                        break
                current = overlap_parts
                current_len = overlap_len

            current.append(part)
            current_len += (len(separator) if len(current) > 1 else 0) + len(part)

        if current:
            final_chunks.append(self._merge(current, separator))

        return [chunk.strip() for chunk in final_chunks if chunk.strip()]

    @staticmethod
    def _merge(splits: list[str], separator: str) -> str:
        if separator in {". ", "? ", "! "}:
            return "".join(splits)
        return separator.join(splits)
