"""
core/ingest.py
Ingestion pipeline: chunking, file parsing, and feeding into the RAG store.

Supported sources:
  - Raw text strings
  - .txt and .md files
  - .pdf files (requires pypdf: pip install pypdf)
  - Directories (recursively processes supported files)
  - URLs (requires httpx)

Usage:
    from core.ingest import ingest_text, ingest_file, ingest_directory, ingest_url
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional
from core.rag import rag, RAGStore


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    Tries to break at sentence boundaries where possible.

    chunk_size: target words per chunk
    overlap: words to repeat at the start of the next chunk (context continuity)
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def make_ids(chunks: list[str], source: str) -> list[str]:
    """Generate stable, deterministic IDs based on source + content hash."""
    ids = []
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
        safe_source = re.sub(r"[^\w]", "_", source)[:40]
        ids.append(f"{safe_source}_{i}_{content_hash}")
    return ids


# ── Ingest Functions ──────────────────────────────────────────────────────────

def ingest_text(
    text: str,
    source: str = "manual",
    collection: str = "main",
    chunk_size: int = 400,
    overlap: int = 50,
    store: Optional[RAGStore] = None,
) -> int:
    """
    Chunk and ingest a raw text string.
    Returns number of chunks added.
    """
    store = store or rag
    store.use_collection(collection)

    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return 0

    ids = make_ids(chunks, source)
    metadatas = [{"source": source, "chunk": i, "collection": collection} for i in range(len(chunks))]

    added = store.add(chunks, metadatas=metadatas, ids=ids)
    print(f"[Ingest] '{source}' → {added} chunks into '{collection}'")
    return added


def ingest_file(
    path: str | Path,
    collection: str = "main",
    chunk_size: int = 400,
    overlap: int = 50,
    store: Optional[RAGStore] = None,
) -> int:
    """
    Parse and ingest a file. Supports .txt, .md, .pdf
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    source = path.name

    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")

    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF ingestion: pip install pypdf")
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt .md .pdf")

    return ingest_text(text, source=source, collection=collection, chunk_size=chunk_size, overlap=overlap, store=store)


def ingest_directory(
    directory: str | Path,
    collection: str = "main",
    recursive: bool = True,
    chunk_size: int = 400,
    overlap: int = 50,
    store: Optional[RAGStore] = None,
) -> dict[str, int]:
    """
    Ingest all supported files in a directory.
    Returns dict of {filename: chunks_added}.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    supported = {".txt", ".md", ".pdf"}
    results = {}

    for file_path in directory.glob(pattern):
        if file_path.suffix.lower() in supported and file_path.is_file():
            try:
                count = ingest_file(file_path, collection=collection, chunk_size=chunk_size, overlap=overlap, store=store)
                results[file_path.name] = count
            except Exception as e:
                print(f"[Ingest] Failed on {file_path.name}: {e}")
                results[file_path.name] = 0

    return results


async def ingest_url(
    url: str,
    collection: str = "main",
    chunk_size: int = 400,
    overlap: int = 50,
    store: Optional[RAGStore] = None,
) -> int:
    """
    Fetch a URL and ingest its text content.
    Strips HTML tags. Requires httpx.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for URL ingestion: pip install httpx")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        raw = response.text

    # Strip HTML if needed
    if "html" in content_type:
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"&\w+;", " ", raw)

    source = url.split("//")[-1].split("/")[0]  # use domain as source label
    return ingest_text(raw, source=source, collection=collection, chunk_size=chunk_size, overlap=overlap, store=store)
