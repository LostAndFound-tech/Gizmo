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

The confidence score returned tells the Ego how much to trust the facts block.
Low confidence = Ego should caveat. High confidence = use it directly.

Correction veto:
  Before anything leaves Mind, it checks the Id's correction log.
  If a retrieved fact contradicts a known correction, it gets flagged or dropped.
  (v1: correction veto is a stub — wired but empty until Id exists)

Usage:
    from core.mind import mind
    facts = await mind.query(brief)
    # facts = {
    #     "synthesis":       str,    — assembled knowledge block
    #     "confidence":      float,  — 0.0 to 1.0
    #     "source":          str,    — "librarian" | "rag" | "web" | "none"
    #     "topics_queried":  list,   — what was searched
    #     "chunks":          list,   — raw retrieved chunks (for Archivist)
    # }
"""

import asyncio
import time
from typing import Optional, TYPE_CHECKING

from core.log import log, log_event, log_error

if TYPE_CHECKING:
    from core.archivist import Brief

# ── Confidence thresholds ──────────────────────────────────────────────────────
# These determine when Mind escalates to the next retrieval tier.

LIBRARIAN_SUFFICIENT  = 0.75   # trust Librarian, skip RAG
RAG_SUFFICIENT        = 0.45   # trust RAG, skip web search
WEB_SEARCH_THRESHOLD  = 0.20   # below this, try web search

# Distance threshold for RAG results — lower = more similar
# Results above this distance are considered low confidence
RAG_DISTANCE_THRESHOLD = 1.2


# ── Empty facts ────────────────────────────────────────────────────────────────

def _empty_facts(topics: list) -> dict:
    return {
        "synthesis":      "",
        "confidence":     0.0,
        "source":         "none",
        "topics_queried": topics,
        "chunks":         [],
    }


# ── Librarian stub ─────────────────────────────────────────────────────────────
# Returns empty until the Librarian component exists.
# Interface is fixed — Librarian will implement this exact contract.

async def _query_librarian(
    topics: list,
    query: str,
    headmate: Optional[str],
) -> dict:
    """
    Stub — always returns empty with zero confidence.
    When Librarian exists it returns:
      {"text": str, "confidence": float, "domain": str}
    """
    return {"text": "", "confidence": 0.0, "domain": ""}


# ── RAG retrieval ──────────────────────────────────────────────────────────────

async def _query_rag(
    topics: list,
    query: str,
    fronters: list,
    session_id: str,
) -> dict:
    """
    Query ChromaDB across relevant collections.
    Collections queried: each fronter's personal collection + main.
    Returns assembled text and a confidence score derived from distances.
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb

        # Determine which collections to query
        collections_to_query = set(["main"])
        for f in fronters:
            if f:
                collections_to_query.add(f.lower().strip())

        # Check which collections actually exist
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

        # Sort by distance (ascending — lower = more relevant)
        all_chunks.sort(key=lambda c: c.get("distance", 999))

        # Filter by distance threshold
        good_chunks = [
            c for c in all_chunks
            if c.get("distance", 999) < RAG_DISTANCE_THRESHOLD
        ]

        if not good_chunks:
            return {"text": "", "confidence": 0.0, "chunks": all_chunks}

        # Confidence from best distance — closer to 0 = higher confidence
        best_distance = good_chunks[0].get("distance", 1.0)
        confidence = max(0.0, min(1.0, 1.0 - (best_distance / RAG_DISTANCE_THRESHOLD)))

        # Assemble text block
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


# ── Web search ─────────────────────────────────────────────────────────────────

async def _query_web(query: str, topics: list) -> dict:
    """
    Web search — last resort. Uses existing epistemic synthesis.
    Only fires when RAG confidence is below threshold.
    """
    try:
        from core.epistemic_synthesis import synthesize as epistemic_synthesize
        from core.llm import llm

        result = await epistemic_synthesize(
            query=query,
            llm=llm,
        )

        if not result:
            return {"text": "", "confidence": 0.0}

        return {
            "text":       result,
            "confidence": 0.6,   # web search is decent but not ground truth
        }

    except Exception as e:
        log_error("Mind", "web search failed", exc=e)
        return {"text": "", "confidence": 0.0}


# ── Correction veto stub ───────────────────────────────────────────────────────

def _apply_correction_veto(text: str, topics: list) -> str:
    """
    Stub — when Id exists, checks retrieved text against known corrections.
    If a retrieved fact contradicts a correction, it gets flagged or dropped.

    For now: passthrough.
    Future: Id.check_corrections(text) → filtered text
    """
    return text


# ── LLM synthesis ──────────────────────────────────────────────────────────────

async def _synthesize(
    query: str,
    chunks: list,
    topics: list,
    headmate: Optional[str],
) -> str:
    """
    LLM synthesis — only fires when retrieval alone isn't enough.
    Takes raw chunks and assembles them into a coherent knowledge block.
    Tight prompt, focused output, minimal tokens.
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


# ── Mind ───────────────────────────────────────────────────────────────────────

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
          Librarian → RAG → Web search
        Each tier only fires if the previous tier's confidence is insufficient.

        Never fires LLM unless synthesis is genuinely needed.
        """
        t_start = time.monotonic()

        topics  = brief.topics
        query   = brief.message
        fronters = brief.fronters
        headmate = brief.headmate
        session_id = brief.session_id

        # Build a richer query from hot topics in the conversation field
        # Hot topics are what the conversation has been about — they're the context
        hot = brief.field_snapshot.get("hot", [])
        if hot and not any(t in query.lower() for t in hot):
            # Augment query with hot topic context
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

        # ── Tier 2: RAG ───────────────────────────────────────────────────────
        rag_result = await _query_rag(topics, query, fronters, session_id)
        rag_confidence = rag_result.get("confidence", 0.0)
        rag_chunks     = rag_result.get("chunks", [])
        rag_text       = rag_result.get("text", "")

        if rag_confidence >= RAG_SUFFICIENT and rag_text:
            # Good enough — apply veto and return without LLM
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
        # We have some RAG results but not enough confidence for raw text.
        # Try synthesizing what we have before going to web.
        if rag_chunks and rag_confidence >= WEB_SEARCH_THRESHOLD:
            synthesized = await _synthesize(query, rag_chunks, topics, headmate)
            if synthesized:
                synthesized = _apply_correction_veto(synthesized, topics)
                # Synthesis bumps confidence slightly — we at least have something
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

        # ── Tier 3: Web search ────────────────────────────────────────────────
        # Only if RAG genuinely couldn't help
        # Skip web search for personal/relational topics — RAG is authoritative
        personal_topics = {
            "identity", "relationship", "distress", "sadness",
            "anger", "anxiety", "happiness", "sleep", "health",
        }
        skip_web = bool(set(topics) & personal_topics)

        if not skip_web:
            web_result = await _query_web(query, topics)
            web_text       = web_result.get("text", "")
            web_confidence = web_result.get("confidence", 0.0)

            if web_text:
                web_text = _apply_correction_veto(web_text, topics)
                log_event("Mind", "QUERY_COMPLETE",
                    session=session_id[:8],
                    source="web",
                    confidence=web_confidence,
                    duration_ms=round((time.monotonic() - t_start) * 1000),
                )
                return {
                    "synthesis":      web_text,
                    "confidence":     web_confidence,
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
        """
        Direct topic query — bypasses brief, useful for background processes.
        Same tiered retrieval, simpler interface.
        """
        from core.archivist import Brief
        import time as _time

        # Build a minimal brief-like structure
        class _MiniBrief:
            message        = query
            topics         = [topic]
            fronters       = fronters or []
            headmate       = None
            session_id     = session_id
            field_snapshot = {"hot": [topic], "warm": []}

        return await self.query(_MiniBrief())  # type: ignore


# ── Singleton ──────────────────────────────────────────────────────────────────
mind = Mind()