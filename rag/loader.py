"""Extract clean text from PDFs, one record per page so answers can cite pages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str  # file name shown to the user
    page_number: int  # 1-based, as people read it
    text: str


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # "infor-\nmation" -> "information"
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")  # line wraps inside a paragraph
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SINGLE_NEWLINE.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def load_pdf(path: str | Path, max_pages: int | None = None) -> list[Page]:
    """Return the non-empty pages of a PDF. Scanned (image-only) pages have no
    text layer and are skipped; the caller reports that to the user."""
    path = Path(path)
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf raises several types
            raise ValueError(f"{path.name} is password-protected") from exc

    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        if max_pages is not None and i > max_pages:
            break
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(Page(source=path.name, page_number=i, text=text))
    return pages
