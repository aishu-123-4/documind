"""Read documents and split them into retrievable chunks.

Supports `.txt` and `.md` natively, and `.pdf` when the optional `pypdf`
package is installed. Chunking is paragraph-aware with a target size and
overlap, and every chunk keeps its source document and an approximate location
so answers can cite `document#chunk`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}
TARGET_CHARS = 900       # ~200-250 tokens per chunk
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    doc: str             # source document name
    chunk_id: int        # 0-based index within the document
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc}#{self.chunk_id}"


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Reading PDFs needs the optional 'pypdf' package: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, target: int = TARGET_CHARS,
               overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks that respect paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > target:
            # Flush, then hard-split the oversized paragraph into windows.
            if buf:
                chunks.append(buf.strip())
                buf = ""
            step = max(1, target - overlap)
            for i in range(0, len(para), step):
                chunks.append(para[i:i + target].strip())
            continue
        if len(buf) + len(para) + 2 <= target:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf.strip())
            # Start the next buffer with a tail overlap of the previous one.
            tail = buf[-overlap:] if buf else ""
            buf = (tail + "\n\n" + para).strip() if tail else para
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def ingest_path(path: str | Path) -> list[Chunk]:
    """Ingest a file or a directory of documents into chunks."""
    path = Path(path)
    files: list[Path] = []
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and (p.suffix.lower() in TEXT_EXTENSIONS
                                or p.suffix.lower() == ".pdf"):
                files.append(p)
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(path)

    base = path if path.is_dir() else path.parent
    chunks: list[Chunk] = []
    for f in files:
        try:
            text = read_document(f)
        except (RuntimeError, ValueError):
            continue
        name = str(f.relative_to(base)) if path.is_dir() else f.name
        for i, body in enumerate(chunk_text(text)):
            chunks.append(Chunk(doc=name, chunk_id=i, text=body))
    return chunks
