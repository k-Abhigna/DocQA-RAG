"""Load documents from disk and split them into overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    text: str
    source: str  # file name
    page: int | None = None  # 1-indexed page for PDFs
    section: str | None = None  # nearest Markdown heading, if any


def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return (text, page) pairs. Text files are a single page=None entry."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _split_blocks(text: str) -> list[str]:
    """Split on blank lines, but never inside a ``` code fence."""
    blocks, current, in_fence = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"^#{1,6}\s", line):
            # A heading is always its own block, even with no blank line after it.
            if current:
                blocks.append("\n".join(current))
                current = []
            blocks.append(line.strip())
            continue
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def chunk_text(text: str, max_chars: int = 700, overlap_chars: int = 120) -> list[tuple[str, str | None]]:
    """Group paragraph blocks into chunks of about max_chars.

    Each chunk carries the most recent Markdown heading so a chunk like
    "Employees receive 12 days..." still knows it belongs to "Sick Leave".
    Consecutive chunks share overlap_chars of trailing text for context.
    """
    chunks: list[tuple[str, str | None]] = []
    buf, heading = "", None
    for block in _split_blocks(text):
        match = re.match(r"^#{1,6}\s+(.*)", block)
        if match and "\n" not in block:
            if buf.strip():
                chunks.append((buf.strip(), heading))
                buf = ""
            heading = match.group(1).strip()
            continue
        if buf and len(buf) + len(block) > max_chars:
            chunks.append((buf.strip(), heading))
            buf = buf[-overlap_chars:] if overlap_chars else ""
        buf = f"{buf}\n\n{block}" if buf else block
    if buf.strip():
        chunks.append((buf.strip(), heading))
    return chunks


def load_directory(docs_dir: Path, **chunk_kwargs) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        for text, page in load_file(path):
            for piece, section in chunk_text(text, **chunk_kwargs):
                chunks.append(Chunk(text=piece, source=path.name, page=page, section=section))
    return chunks
