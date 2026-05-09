"""
tools/memory_tool.py
Direct read/write access to Gizmo's two memory collections.

conscious — Gizmo's own thoughts, reflections, impressions, relationship
            notes. He owns this. Reads and writes freely.

memory    — Structured datapoints about people and events. Facts,
            observations, corrections, moments of note. System writes,
            Gizmo reads.

Operations:
  write   — add an entry to conscious or memory
  read    — semantic search across conscious, memory, or both
  delete  — remove an entry by ID (conscious only)
  list    — list recent entries for a subject

This is the primary way Gizmo accesses what he knows and records what
he thinks. No file system, no indirection. Just direct ChromaDB.
"""

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from tools.base_tool import BaseTool, ToolResult

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")

CONSCIOUS_COLLECTION = "conscious"
MEMORY_COLLECTION    = "memory"
VALID_COLLECTIONS    = {CONSCIOUS_COLLECTION, MEMORY_COLLECTION}

# ── Shared ChromaDB client and embedding function ─────────────────────────────

_client    = None
_embed_fn  = None


def _get_client():
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from core.rag import _EMBED_FN
        _embed_fn = _EMBED_FN
    except Exception:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embed_fn


def _get_collection(name: str):
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_get_embed_fn(),
    )


# ── Write tool ────────────────────────────────────────────────────────────────

class MemoryWriteTool(BaseTool):
    @property
    def name(self) -> str:
        return "memory_write"

    @property
    def description(self) -> str:
        return (
            "Write a thought, observation, or fact directly into memory. "
            "Use 'conscious' for your own thoughts, reflections, impressions, "
            "and relationship notes — things you've noticed or felt. "
            "Use 'memory' for factual datapoints about people or events. "
            "This is your primary way to remember things. Use it freely. "
            "Args: "
            "collection (str) — 'conscious' or 'memory'. Default: 'conscious'. "
            "content (str) — what to remember. "
            "subject (str) — who this is about (headmate name, 'self', etc). "
            "type (str) — category: reflection, observation, relationship, fact, "
            "question, private, moment, thought, note. Default: note. "
            "tags (str) — optional comma-separated tags."
        )

    async def run(
        self,
        collection: str = "conscious",
        content: str = "",
        subject: str = "",
        type: str = "",
        tags: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not content:
            return ToolResult(success=False, output="Need content to write.")

        collection = collection.lower().strip()
        if collection not in VALID_COLLECTIONS:
            collection = CONSCIOUS_COLLECTION

        # Normalize type
        valid_types = {
            "reflection", "observation", "relationship", "fact",
            "question", "private", "moment", "thought", "note"
        }
        entry_type = type.lower().strip() if type else "note"
        if entry_type not in valid_types:
            entry_type = "note"

        now      = datetime.now().isoformat(timespec="seconds")
        entry_id = f"{collection[:3]}_{uuid.uuid4().hex[:12]}"

        try:
            col = _get_collection(collection)
            col.add(
                documents=[content],
                metadatas=[{
                    "subject":    subject.lower().strip() if subject else "",
                    "type":       entry_type,
                    "tags":       tags.strip() if tags else "",
                    "written_at": now,
                    "session_id": session_id or "",
                    "collection": collection,
                }],
                ids=[entry_id],
            )

            print(f"[Memory] wrote to {collection} | subject={subject or 'none'} | {content[:60]}")

            return ToolResult(
                success=True,
                output=f"Remembered. [{collection}] {content[:80]}{'...' if len(content) > 80 else ''}",
                data={"id": entry_id, "collection": collection},
            )

        except Exception as e:
            return ToolResult(success=False, output=f"Write failed: {e}")


# ── Read tool ─────────────────────────────────────────────────────────────────

class MemoryReadTool(BaseTool):
    @property
    def name(self) -> str:
        return "memory_read"

    @property
    def description(self) -> str:
        return (
            "Search your memory for anything relevant to a query. "
            "Searches your thoughts and reflections (conscious) and "
            "factual datapoints (memory), or either one specifically. "
            "Use this whenever you want to know what you already know "
            "or have thought about something or someone. "
            "Args: "
            "query (str) — what to search for. "
            "collection (str) — 'conscious', 'memory', or 'both'. Default: 'both'. "
            "subject (str) — optional filter by person. "
            "n (int) — number of results per collection. Default: 5."
        )

    async def run(
        self,
        query: str = "",
        collection: str = "both",
        subject: str = "",
        n: int = 5,
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not query:
            return ToolResult(success=False, output="Need a query to search.")

        # Coerce string args from marker system
        if isinstance(n, str):
            try: n = int(n)
            except: n = 5

        collection = collection.lower().strip()
        subject    = subject.lower().strip() if subject else ""

        collections_to_search = (
            [CONSCIOUS_COLLECTION, MEMORY_COLLECTION]
            if collection == "both"
            else [collection] if collection in VALID_COLLECTIONS
            else [CONSCIOUS_COLLECTION, MEMORY_COLLECTION]
        )

        all_results = []

        for col_name in collections_to_search:
            try:
                col   = _get_collection(col_name)
                count = col.count()
                if count == 0:
                    continue

                k = min(n, count)
                kwargs_q = {"query_texts": [query], "n_results": k}
                if subject:
                    kwargs_q["where"] = {"subject": {"$eq": subject}}

                results = col.query(**kwargs_q)

                docs  = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]
                ids   = results.get("ids", [[]])[0]

                for doc, meta, dist, rid in zip(docs, metas, dists, ids):
                    if dist < 1.2:  # relevance threshold
                        all_results.append({
                            "id":         rid,
                            "collection": col_name,
                            "content":    doc,
                            "subject":    meta.get("subject", ""),
                            "written_at": meta.get("written_at", ""),
                            "tags":       meta.get("tags", ""),
                            "distance":   dist,
                        })

            except Exception as e:
                print(f"[Memory] search failed in {col_name}: {e}")

        if not all_results:
            return ToolResult(
                success=True,
                output=f"Nothing found for '{query}'" + (f" about {subject}" if subject else "") + ".",
                data={"results": []},
            )

        # Sort by relevance
        all_results.sort(key=lambda r: r["distance"])

        # Format output
        lines = []
        for r in all_results[:n * len(collections_to_search)]:
            col_label = "💭" if r["collection"] == CONSCIOUS_COLLECTION else "📌"
            when      = r["written_at"][:10] if r["written_at"] else ""
            subj      = f" [{r['subject']}]" if r["subject"] else ""
            lines.append(f"{col_label}{subj} {when}\n{r['content']}")

        output = "\n\n---\n\n".join(lines)

        print(f"[Memory] read | query='{query[:40]}' | {len(all_results)} results")

        return ToolResult(
            success=True,
            output=output,
            data={"results": all_results},
        )


# ── List tool ─────────────────────────────────────────────────────────────────

class MemoryListTool(BaseTool):
    @property
    def name(self) -> str:
        return "memory_list"

    @property
    def description(self) -> str:
        return (
            "List recent memory entries for a subject or collection. "
            "Useful for reviewing what you know about someone or "
            "what you've been thinking about lately. "
            "Args: "
            "subject (str) — filter by person (optional). "
            "collection (str) — 'conscious', 'memory', or 'both'. Default: 'both'. "
            "n (int) — max entries to return. Default: 10."
        )

    async def run(
        self,
        subject: str = "",
        collection: str = "both",
        n: int = 10,
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        # Coerce string args from marker system
        if isinstance(n, str):
            try: n = int(n)
            except: n = 10

        collection = collection.lower().strip()
        subject    = subject.lower().strip() if subject else ""

        collections_to_list = (
            [CONSCIOUS_COLLECTION, MEMORY_COLLECTION]
            if collection == "both"
            else [collection] if collection in VALID_COLLECTIONS
            else [CONSCIOUS_COLLECTION, MEMORY_COLLECTION]
        )

        all_entries = []

        for col_name in collections_to_list:
            try:
                col   = _get_collection(col_name)
                if col.count() == 0:
                    continue

                kwargs_g = {"limit": n}
                if subject:
                    kwargs_g["where"] = {"subject": {"$eq": subject}}

                results = col.get(**kwargs_g)
                docs    = results.get("documents", [])
                metas   = results.get("metadatas", [])
                ids     = results.get("ids", [])

                for doc, meta, rid in zip(docs, metas, ids):
                    all_entries.append({
                        "id":         rid,
                        "collection": col_name,
                        "content":    doc,
                        "subject":    meta.get("subject", ""),
                        "written_at": meta.get("written_at", ""),
                        "tags":       meta.get("tags", ""),
                    })

            except Exception as e:
                print(f"[Memory] list failed in {col_name}: {e}")

        if not all_entries:
            msg = f"Nothing stored"
            if subject:
                msg += f" about {subject}"
            if collection != "both":
                msg += f" in {collection}"
            return ToolResult(success=True, output=msg + ".", data={"entries": []})

        # Sort by written_at descending
        all_entries.sort(key=lambda e: e.get("written_at", ""), reverse=True)
        all_entries = all_entries[:n]

        lines = []
        for e in all_entries:
            col_label = "💭" if e["collection"] == CONSCIOUS_COLLECTION else "📌"
            when      = e["written_at"][:10] if e["written_at"] else ""
            subj      = f" [{e['subject']}]" if e["subject"] else ""
            lines.append(f"{col_label}{subj} {when}\n{e['content'][:200]}")

        return ToolResult(
            success=True,
            output="\n\n---\n\n".join(lines),
            data={"entries": all_entries},
        )


# ── Delete tool ───────────────────────────────────────────────────────────────

class MemoryDeleteTool(BaseTool):
    @property
    def name(self) -> str:
        return "memory_delete"

    @property
    def description(self) -> str:
        return (
            "Delete a specific entry from conscious memory by ID. "
            "Only works on conscious — memory datapoints are system-managed. "
            "Use when something you wrote is wrong, outdated, or shouldn't be kept. "
            "Get the ID from memory_read or memory_list results. "
            "Args: "
            "entry_id (str) — the ID of the entry to delete."
        )

    async def run(
        self,
        entry_id: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not entry_id:
            return ToolResult(success=False, output="Need an entry ID to delete.")

        # Only allow deleting from conscious
        if not entry_id.startswith("con_"):
            return ToolResult(
                success=False,
                output="Can only delete from conscious memory. Memory datapoints are system-managed."
            )

        try:
            col = _get_collection(CONSCIOUS_COLLECTION)
            col.delete(ids=[entry_id])
            return ToolResult(
                success=True,
                output=f"Deleted entry {entry_id} from conscious.",
                data={"deleted": entry_id},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Delete failed: {e}")
