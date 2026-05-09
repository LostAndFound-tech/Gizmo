"""
core/mind.py
The Mind — facts, retrieval, world knowledge. No personality. No relationships.

The Mind's only job is answering: what do we know about what's being asked?

It never fires an LLM just because it can. LLM only fires when:
  - The Librarian (stub for now) has insufficient confidence AND
  - Novel synthesis is genuinely needed

Query priority (always in this order):
  1. Librarian       — structured knowledge built from lived experience
                       (stub in v1 — returns empty, grows into real component)
  2. RAG / ChromaDB  — archived conversations and ingested knowledge
  3. Web search      — last resort, only when confidence is still insufficient
                       (stub in v1 — returns instantly with nothing)

The confidence score returned tells the Ego how much to trust the facts block.
Low confidence = Ego should caveat. High confidence = use it directly.
"""

import time
from typing import Optional, TYPE_CHECKING

from core.log import log, log_event, log_error

if TYPE_CHECKING:
    from core.archivist import Brief

# ── Confidence thresholds ─────────────────────────────────────────────────────

LIBRARIAN_SUFFICIENT  = 0.75
RAG_SUFFICIENT        = 0.45
WEB_SEARCH_THRESHOLD  = 0.20
RAG_DISTANCE_THRESHOLD = 1.2


# ── Empty facts ───────────────────────────────────────────────────────────────

def _empty_facts(topics: list) -> dict:
    return {
        "synthesis":      "",
        "confidence":     0.0,
        "source":         "none",
        "topics_queried": topics,
        "chunks":         [],
    }


# ── Librarian stub ────────────────────────────────────────────────────────────

async def _query_librarian(
    topics: list,
    query: str,
    headmate: Optional[str],
) -> dict:
    """Stub — always returns empty. Grows into real component later."""
    return {"text": "", "confidence": 0.0, "domain": ""}


# ── RAG retrieval ─────────────────────────────────────────────────────────────

async def _query_rag(
    topics: list,
    query: str,
    fronters: list,
    session_id: str,
) -> dict:
    """
    Query ChromaDB across relevant collections.
    Collections queried: each fronter's personal collection + main.
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb

        collections_to_query = set(["main"])
        for f in fronters:
            if f:
                collections_to_query.add(f.lower().strip())

        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            existing = {c.name for c in client.list_collections()}
            collections_to_query &= existing
        except Exception:
            collections_to_query = {"main"}

        if not collections_to_query:
            return {"text": "", "confidence": 0.0, "chunks": []}

        all_chunks = []

        for collection_name in collections_to_query:
            try:
                store = RAGStore(collection_name=collection_name)
                if store.count == 0:
                    continue
                chunks = store.retrieve(query=query, n_results=4)
                for chunk in chunks:
                    chunk["collection"] = collection_name
                all_chunks.extend(chunks)
            except Exception as e:
                log_error("Mind", f"RAG query failed for collection {collection_name}", exc=e)

        if not all_chunks:
            return {"text": "", "confidence": 0.0, "chunks": []}

        all_chunks.sort(key=lambda c: c.get("distance", 999))

        good_chunks = [
            c for c in all_chunks
            if c.get("distance", 999) < RAG_DISTANCE_THRESHOLD
        ]

        if not good_chunks:
            return {"text": "", "confidence": 0.0, "chunks": all_chunks}

        best_distance = good_chunks[0].get("distance", 1.0)
        confidence = max(0.0, min(1.0, 1.0 - (best_distance / RAG_DISTANCE_THRESHOLD)))

        seen = set()
        text_parts = []
        for chunk in good_chunks[:5]:
            text = chunk.get("text", "").strip()
            if text and text not in seen:
                seen.add(text)
                text_parts.append(text)

        assembled = "\n\n".join(text_parts)

        return {
            "text":       assembled,
            "confidence": confidence,
            "chunks":     good_chunks,
        }

    except Exception as e:
        log_error("Mind", "RAG retrieval failed", exc=e)
        return {"text": "", "confidence": 0.0, "chunks": []}


# ── Web search stub ───────────────────────────────────────────────────────────

def _query_web_stub(query: str, topics: list) -> dict:
    """
    Instant stub — returns empty immediately, zero latency.
    Wire in web_search.py when ready.
    """
    log_event("Mind", "WEB_SEARCH_STUB",
        query=query[:60],
        note="web search not yet wired — skipping",
    )
    return {"text": "", "confidence": 0.0}


# ── LLM synthesis ─────────────────────────────────────────────────────────────

async def _synthesize(
    query: str,
    chunks: list,
    topics: list,
    headmate: Optional[str],
) -> str:
    """
    LLM synthesis — only fires when retrieval alone isn't enough.
    Takes raw chunks and assembles them into a coherent knowledge block.
    """
    try:
        from core.llm import llm

        if not chunks:
            return ""

        chunk_text = "\n\n".join(
            c.get("text", "")[:300]
            for c in chunks[:4]
            if c.get("text")
        )

        if not chunk_text.strip():
            return ""

        addressing = f"The person asking is {headmate}. " if headmate else ""

        prompt = [{
            "role": "user",
            "content": (
                f"{addressing}"
                f"Synthesize these retrieved memory chunks into a single coherent "
                f"knowledge block relevant to: '{query}'\n\n"
                f"Chunks:\n{chunk_text}\n\n"
                f"Rules:\n"
                f"- Past tense, factual, no invented details\n"
                f"- Only include what's directly relevant to the query\n"
                f"- 3-5 sentences maximum\n"
                f"- No preamble, no explanation, just the facts"
            )
        }]

        result = await llm.generate(
            prompt,
            system_prompt=(
                "You synthesize retrieved memory chunks into concise factual knowledge blocks. "
                "Never invent. Never expand beyond what the chunks contain."
            ),
            max_new_tokens=300,
            temperature=0.1,
        )
        return result.strip()

    except Exception as e:
        log_error("Mind", "LLM synthesis failed", exc=e)
        return ""


# ── Correction veto stub ──────────────────────────────────────────────────────

def _apply_correction_veto(text: str, topics: list) -> str:
    """Stub — passthrough until Id exists."""
    return text


# ── Mind ──────────────────────────────────────────────────────────────────────

class Mind:
    """
    Singleton. Stateless between queries — all state lives in the archive.

    Public API:
        await mind.query(brief) → facts dict
    """

    async def query(self, brief: "Brief") -> dict:
        """
        Main entry point. Takes an Archivist brief, returns a facts dict.

        Tiered retrieval:
          Librarian → RAG → Web search (stub)
        Each tier only fires if the previous tier's confidence is insufficient.
        """
        t_start = time.monotonic()

        topics   = brief.topics
        query    = brief.message
        fronters = brief.fronters
        headmate = brief.headmate
        session_id = brief.session_id

        # Augment query with hot topic context
        hot = brief.field_snapshot.get("hot", [])
        if hot and not any(t in query.lower() for t in hot):
            query = f"{query} (context: {', '.join(hot)})"

        log_event("Mind", "QUERY_START",
            session=session_id[:8],
            topics=topics,
            hot=hot,
            headmate=headmate or "unknown",
            query_len=len(query),
        )

        # ── Tier 1: Librarian ─────────────────────────────────────────────────
        librarian_result = await _query_librarian(topics, query, headmate)
        librarian_confidence = librarian_result.get("confidence", 0.0)

        if librarian_confidence >= LIBRARIAN_SUFFICIENT:
            text = _apply_correction_veto(librarian_result["text"], topics)
            log_event("Mind", "QUERY_COMPLETE",
                session=session_id[:8],
                source="librarian",
                confidence=librarian_confidence,
                duration_ms=round((time.monotonic() - t_start) * 1000),
            )
            return {
                "synthesis":      text,
                "confidence":     librarian_confidence,
                "source":         "librarian",
                "topics_queried": topics,
                "chunks":         [],
            }
        
        # ── Tier 1.5: Conscious layer — semantic file index ───────────────────
        # Searches Gizmo's written files by description embedding.
        # Returns file paths → reads actual content → adds to RAG chunks.
        # Cheap: searches descriptions only, reads files on hit.
        conscious_text = ""
        try:
            from core.conscious import conscious
            conscious_results = conscious.search(
                query=query,
                n=3,
                subject=(headmate or "").lower(),
            )
            if not conscious_results:
                # Also try without subject filter
                conscious_results = conscious.search(query=query, n=3)
 
            if conscious_results:
                parts = []
                for r in conscious_results:
                    if r.get("distance", 1.0) < 1.0:  # relevance threshold
                        file_content = conscious.read_file(r["path"])
                        if file_content:
                            parts.append(
                                f"[{r['description']}]\n{file_content[:600]}"
                            )
                if parts:
                    conscious_text = "\n\n".join(parts)
                    log_event("Mind", "CONSCIOUS_HIT",
                        session=session_id[:8],
                        files=len(parts),
                        query=query[:60],
                    )
        except Exception as e:
            log_error("Mind", "conscious layer query failed", exc=e)
 
        if conscious_text:
            return {
                "synthesis":      conscious_text,
                "confidence":     0.75,
                "source":         "conscious",
                "topics_queried": topics,
                "chunks":         [],
            }

        # ── Tier 2: RAG ───────────────────────────────────────────────────────
        rag_result     = await _query_rag(topics, query, fronters, session_id)
        rag_confidence = rag_result.get("confidence", 0.0)
        rag_chunks     = rag_result.get("chunks", [])
        rag_text       = rag_result.get("text", "")

        if rag_confidence >= RAG_SUFFICIENT and rag_text:
            text = _apply_correction_veto(rag_text, topics)
            log_event("Mind", "QUERY_COMPLETE",
                session=session_id[:8],
                source="rag",
                confidence=rag_confidence,
                chunks=len(rag_chunks),
                duration_ms=round((time.monotonic() - t_start) * 1000),
            )
            return {
                "synthesis":      text,
                "confidence":     rag_confidence,
                "source":         "rag",
                "topics_queried": topics,
                "chunks":         rag_chunks,
            }

        # ── Tier 2.5: RAG with synthesis ──────────────────────────────────────
        if rag_chunks and rag_confidence >= WEB_SEARCH_THRESHOLD:
            synthesized = await _synthesize(query, rag_chunks, topics, headmate)
            if synthesized:
                synthesized = _apply_correction_veto(synthesized, topics)
                confidence = min(0.7, rag_confidence + 0.15)
                log_event("Mind", "QUERY_COMPLETE",
                    session=session_id[:8],
                    source="rag+synthesis",
                    confidence=confidence,
                    chunks=len(rag_chunks),
                    duration_ms=round((time.monotonic() - t_start) * 1000),
                )
                return {
                    "synthesis":      synthesized,
                    "confidence":     confidence,
                    "source":         "rag+synthesis",
                    "topics_queried": topics,
                    "chunks":         rag_chunks,
                }

        # ── Tier 3: Web search (stub — instant, no latency) ──────────────────
        personal_topics = {
            "identity", "relationship", "distress", "sadness",
            "anger", "anxiety", "happiness", "sleep", "health",
        }
        skip_web = bool(set(topics) & personal_topics)

        if not skip_web:
            # Stub — synchronous, zero latency, returns immediately
            web_result = _query_web_stub(query, topics)
            web_text   = web_result.get("text", "")

            if web_text:
                web_text = _apply_correction_veto(web_text, topics)
                log_event("Mind", "QUERY_COMPLETE",
                    session=session_id[:8],
                    source="web",
                    confidence=web_result.get("confidence", 0.0),
                    duration_ms=round((time.monotonic() - t_start) * 1000),
                )
                return {
                    "synthesis":      web_text,
                    "confidence":     web_result.get("confidence", 0.0),
                    "source":         "web",
                    "topics_queried": topics,
                    "chunks":         [],
                }

        # ── Nothing found ─────────────────────────────────────────────────────
        log_event("Mind", "QUERY_EMPTY",
            session=session_id[:8],
            topics=topics,
            skip_web=skip_web,
            duration_ms=round((time.monotonic() - t_start) * 1000),
        )
        return _empty_facts(topics)

    async def query_topic(
        self,
        topic: str,
        query: str,
        fronters: Optional[list] = None,
        session_id: str = "direct",
    ) -> dict:
        """Direct topic query — bypasses brief, useful for background processes."""

        class _MiniBrief:
            message        = query
            topics         = [topic]
            fronters       = fronters or []
            headmate       = None
            session_id     = session_id
            field_snapshot = {"hot": [topic], "warm": []}

        return await self.query(_MiniBrief())  # type: ignore


# ── Singleton ─────────────────────────────────────────────────────────────────
mind = Mind()