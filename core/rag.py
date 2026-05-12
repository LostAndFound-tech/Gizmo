"""
core/rag.py
Chroma-backed RAG. Persists to disk at CHROMA_PERSIST_DIR.

Added:
  - retrieve_by_topic(): filter by topic tags in metadata
  - retrieve_by_timerange(): filter by date/hour for temporal queries
  - retrieve_recent(): convenience for "earlier today" type queries

IMPORTANT: CHROMA_PERSIST_DIR defaults to /data/chroma (absolute) so
it always lands on Render's persistent disk. Override via env var if needed.
"""

import os
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# Default to /data/chroma (Render persistent disk).
# ./data/chroma is a relative path that resolves to the ephemeral
# app directory on Render and will be wiped on every redeploy.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGStore:
    def __init__(
        self,
        collection_name: str = "main",
        persist_dir: str = CHROMA_PERSIST_DIR,
        embed_model: str = EMBED_MODEL,
    ):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        self._embed_model = embed_model
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
        )

    # ── Collection management ─────────────────────────────────────────────────

    def use_collection(self, name: str) -> None:
        """Switch the active collection, creating it if it doesn't exist."""
        if self.collection.name != name:
            self.collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embed_fn,
            )

    def list_collections(self) -> list[str]:
        """Return names of all collections in the store."""
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, name: str = None) -> None:
        """Delete a collection by name. Defaults to active collection."""
        target = name or self.collection.name
        self.client.delete_collection(target)
        if target == self.collection.name:
            self.collection = self.client.get_or_create_collection(
                name="main",
                embedding_function=self.embed_fn,
            )

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks tagged with a specific source."""
        results = self.collection.get(where={"source": source})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"[RAG] Deleted {len(results['ids'])} chunks from source '{source}'")
        else:
            print(f"[RAG] No chunks found for source '{source}'")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """Add documents to the active collection."""
        if ids is None:
            existing = self.collection.count()
            ids = [f"doc_{existing + i}" for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            metadatas=metadatas if metadatas else None,
            ids=ids,
        )
        print(f"[RAG] Ingested {len(texts)} chunks into '{self.collection.name}'")

    # ── Standard Retrieval ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n_results: int = 4,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve top-k relevant documents for a query.
        Returns list of {"text", "metadata", "distance"}
        """
        n_results = min(n_results, self.collection.count())
        if n_results == 0:
            return []

        kwargs = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs = results.get("documents") or []
        if not docs or not docs[0]:
            return []

        docs = []
        for i, doc in enumerate(results["documents"][0]):
            docs.append({
                "text": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i],
            })
        return docs

    # ── Topic Retrieval ───────────────────────────────────────────────────────

    def retrieve_by_topic(
        self,
        topic: str,
        query: str = "",
        n_results: int = 8,
    ) -> list[dict]:
        """
        Retrieve chunks whose topic metadata contains the given topic string.
        ChromaDB stores topics as comma-separated strings, so we use $contains.

        If query is provided, also scores by semantic similarity.
        Otherwise returns by insertion order (most recent first via ID).
        """
        n_results = min(n_results, self.collection.count())
        if n_results == 0:
            return []

        where = {"topics": {"$contains": topic}}

        try:
            if query:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where,
                )
                raw = results.get("documents") or []
                if not raw or not raw[0]:
                    return []
                docs = []
                for i, doc in enumerate(raw[0]):
                    docs.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i],
                    })
                return docs
            else:
                results = self.collection.get(
                    where=where,
                    limit=n_results,
                )
                return [
                    {"text": doc, "metadata": meta, "distance": 0.0}
                    for doc, meta in zip(results["documents"], results["metadatas"])
                ]
        except Exception as e:
            print(f"[RAG] Topic retrieval failed for '{topic}': {e}")
            return []

    # ── Temporal Retrieval ────────────────────────────────────────────────────

    def retrieve_by_timerange(
        self,
        query: str,
        date: Optional[str] = None,       # "YYYY-MM-DD", defaults to today
        hour_start: Optional[str] = None,  # "HH" 24h format
        hour_end: Optional[str] = None,    # "HH" 24h format
        n_results: int = 8,
        collection_override: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve chunks filtered by date and optionally hour range.
        Useful for "what were we talking about this morning" type queries.

        Uses ChromaDB's $and/$gte/$lte operators on metadata fields.
        """
        if collection_override:
            self.use_collection(collection_override)

        n_results = min(n_results, self.collection.count())
        if n_results == 0:
            return []

        today = date or datetime.now().strftime("%Y-%m-%d")
        conditions = [{"date": {"$eq": today}}]

        if hour_start and hour_end:
            conditions.append({"hour": {"$gte": hour_start}})
            conditions.append({"hour": {"$lte": hour_end}})
        elif hour_start:
            conditions.append({"hour": {"$gte": hour_start}})
        elif hour_end:
            conditions.append({"hour": {"$lte": hour_end}})

        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
            raw = results.get("documents") or []
            if not raw or not raw[0]:
                return []
            docs = []
            for i, doc in enumerate(raw[0]):
                docs.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i],
                })
            return docs
        except Exception as e:
            print(f"[RAG] Temporal retrieval failed: {e}")
            return []

    def retrieve_recent(
        self,
        query: str,
        hours_back: int = 8,
        n_results: int = 8,
        collection_override: Optional[str] = None,
    ) -> list[dict]:
        """
        Convenience: retrieve from the past N hours.
        Covers the "earlier today" / "this morning" use case.
        """
        now = datetime.now()
        current_hour = int(now.strftime("%H"))
        start_hour = max(0, current_hour - hours_back)

        return self.retrieve_by_timerange(
            query=query,
            date=now.strftime("%Y-%m-%d"),
            hour_start=str(start_hour).zfill(2),
            hour_end=now.strftime("%H"),
            n_results=n_results,
            collection_override=collection_override,
        )

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_context(self, docs: list[dict]) -> str:
        """Format retrieved docs into a context block for the LLM prompt."""
        if not docs:
            return ""
        sections = []
        for i, d in enumerate(docs, 1):
            meta = d.get("metadata") or {}
            source = meta.get("source", f"doc {i}")
            sections.append(f"[{source}]\n{d['text']}")
        return "\n\n".join(sections)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self.collection.count()


# Singleton
rag = RAGStore()