"""Split pages into overlapping chunks that respect sentence boundaries.

Chunks never cross a page boundary, so every retrieved chunk maps to exactly
one page for citations. Overlap keeps a sentence that straddles two chunks
retrievable from either side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.loader import Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n\n+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    source: str
    page_number: int


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _hard_split(sentence: str, size: int) -> list[str]:
    """Break an over-long 'sentence' (tables, code, no punctuation) on spaces."""
    words, parts, current = sentence.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > size and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    units: list[str] = []
    for sentence in split_sentences(text):
        units.extend(_hard_split(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and length + added > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap.
            carried: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) + 1 > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, max(carried_len - 1, 0)
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        length += added
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 800, overlap: int = 150, start_id: int = 0) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = start_id
    for page in pages:
        for text in chunk_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(next_id, text, page.source, page.page_number))
            next_id += 1
    return chunks
