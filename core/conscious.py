"""
core/conscious.py
Gizmo's conscious layer — a semantic index of everything he's written.

Not content storage. Pointers.

Every file Gizmo writes gets registered here with:
  - path
  - a one-sentence description of what's in it
  - who it's about (subject)
  - when it was written
  - an embedding of the description

When Mind needs to know what Gizmo knows, it queries conscious first.
Cheap lookup → file paths → read actual files only when needed.

This mirrors how memory actually works — you don't remember everything
verbatim, you remember that you know something and roughly where it lives.

ChromaDB collection: "conscious"
Each document = the one-line description
Each metadata = {path, subject, written_at, file_type, collection}

Usage:
    from core.conscious import conscious
    await conscious.register(path, content, subject, llm)
    results = conscious.search("jess emotional state", n=5)
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from core.log import log, log_event, log_error

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")
COLLECTION_NAME    = "conscious"
PERSONALITY_DIR    = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))

# Reuse the same embedding function singleton from rag.py if available,
# otherwise create one. Either way it's loaded once.
_embed_fn = None

def _get_embed_fn():
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from core.rag import _EMBED_FN
        _embed_fn = _EMBED_FN
        log("Conscious", "reusing embedding function from rag.py")
    except ImportError:
        from chromadb.utils import embedding_functions
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        log("Conscious", "loaded embedding function independently")
    return _embed_fn


# ── Description generation ────────────────────────────────────────────────────

async def _generate_description(path: str, content: str, subject: str, llm) -> str:
    """
    Ask the LLM for a one-sentence description of this file's content.
    Used as the searchable document in ChromaDB.
    Falls back to a simple path-based description on failure.
    """
    # Trim content for the prompt
    sample = content[:800].strip()
    subject_hint = f" about {subject}" if subject else ""

    prompt = [{
        "role": "user",
        "content": (
            f"File: {path}\n"
            f"Content sample:\n\"\"\"\n{sample}\n\"\"\"\n\n"
            f"Write ONE sentence describing what this file contains{subject_hint}. "
            f"Be specific — capture the actual subject matter, mood, or insight. "
            f"No preamble, just the sentence."
        )
    }]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You write one-sentence descriptions of file contents. "
                "Specific, accurate, no preamble."
            ),
            max_new_tokens=80,
            temperature=0.2,
        )
        return result.strip().strip('"').strip("'")
    except Exception as e:
        log_error("Conscious", "description generation failed", exc=e)
        # Fallback — use filename and subject
        name = Path(path).stem.replace("-", " ").replace("_", " ")
        return f"{name}{' — about ' + subject if subject else ''}"


# ── Conscious store ───────────────────────────────────────────────────────────

class ConsciousStore:

    def __init__(self):
        self._client     = None
        self._collection = None
        log("Conscious", "initialised (lazy — connects on first use)")

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=_get_embed_fn(),
            )
            log_event("Conscious", "COLLECTION_READY",
                count=self._collection.count(),
            )
        except Exception as e:
            log_error("Conscious", "failed to connect to ChromaDB", exc=e)
            raise
        return self._collection

    # ── Register a file ───────────────────────────────────────────────────────

    async def register(
        self,
        path: str,
        content: str,
        subject: str = "",
        llm = None,
        description: str = "",
    ) -> bool:
        """
        Register a file in the conscious index.
        Generates a one-sentence description and embeds it.

        path:        absolute or relative path to the file
        content:     the file's content (used to generate description)
        subject:     who this file is about (headmate name, "self", etc.)
        llm:         LLM instance for description generation
        description: if provided, skips LLM generation

        Returns True on success.
        """
        if not path or not content:
            return False

        # Normalize path
        path_str = str(path).strip()

        # Generate description if not provided
        if not description:
            if llm is None:
                try:
                    from core.llm import llm as _llm
                    llm = _llm
                except Exception:
                    pass

            if llm:
                description = await _generate_description(path_str, content, subject, llm)
            else:
                name = Path(path_str).stem.replace("-", " ").replace("_", " ")
                description = f"{name}{' — about ' + subject if subject else ''}"

        now = datetime.now().isoformat(timespec="seconds")

        # Use path as the stable ID — re-registering updates the entry
        doc_id = path_str.replace("/", "_").replace("\\", "_").lstrip("_")

        try:
            collection = self._get_collection()

            # Delete existing entry for this path if present
            try:
                collection.delete(ids=[doc_id])
            except Exception:
                pass

            collection.add(
                documents=[description],
                metadatas=[{
                    "path":       path_str,
                    "subject":    subject or "",
                    "written_at": now,
                    "file_type":  Path(path_str).suffix.lstrip(".") or "txt",
                    "description": description,
                }],
                ids=[doc_id],
            )

            log_event("Conscious", "REGISTERED",
                path=path_str,
                subject=subject or "unknown",
                description=description[:80],
            )
            return True

        except Exception as e:
            log_error("Conscious", f"register failed for {path_str}", exc=e)
            return False

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 5,
        subject: str = "",
    ) -> list[dict]:
        """
        Search the conscious index for files relevant to a query.
        Returns list of {path, description, subject, written_at, distance}.

        subject: if provided, filters to files about that person.
        """
        if not query:
            return []

        try:
            collection = self._get_collection()
            count = collection.count()
            if count == 0:
                return []

            n = min(n, count)
            kwargs = {
                "query_texts": [query],
                "n_results":   n,
            }
            if subject:
                kwargs["where"] = {"subject": {"$eq": subject.lower()}}

            results = collection.query(**kwargs)

            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            output = []
            for doc, meta, dist in zip(docs, metas, dists):
                output.append({
                    "path":        meta.get("path", ""),
                    "description": meta.get("description", doc),
                    "subject":     meta.get("subject", ""),
                    "written_at":  meta.get("written_at", ""),
                    "distance":    dist,
                })

            log_event("Conscious", "SEARCH",
                query=query[:60],
                subject=subject or "any",
                results=len(output),
            )
            return output

        except Exception as e:
            log_error("Conscious", "search failed", exc=e)
            return []

    # ── Read a file by path ───────────────────────────────────────────────────

    def read_file(self, path: str) -> Optional[str]:
        """
        Read the actual content of a registered file.
        Called by Mind after search returns a path.
        """
        try:
            p = Path(path)
            if not p.is_absolute():
                p = PERSONALITY_DIR / path
            if p.exists():
                return p.read_text(encoding="utf-8")
            return None
        except Exception as e:
            log_error("Conscious", f"read_file failed for {path}", exc=e)
            return None

    # ── List all registered files ─────────────────────────────────────────────

    def list_all(self, subject: str = "") -> list[dict]:
        """Return all registered files, optionally filtered by subject."""
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return []

            kwargs = {}
            if subject:
                kwargs["where"] = {"subject": {"$eq": subject.lower()}}

            results = collection.get(**kwargs)
            metas   = results.get("metadatas", [])
            return [
                {
                    "path":        m.get("path", ""),
                    "description": m.get("description", ""),
                    "subject":     m.get("subject", ""),
                    "written_at":  m.get("written_at", ""),
                }
                for m in metas
            ]
        except Exception as e:
            log_error("Conscious", "list_all failed", exc=e)
            return []

    # ── Count ─────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0


# ── Singleton ─────────────────────────────────────────────────────────────────
conscious = ConsciousStore()
