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

READ SIDE FIXES (vs original):
  - _query_rag now queries headmate collection first with higher n_results,
    then main — headmate facts always surface before generic noise
  - Speaker filter applied to extracted_fact type entries so memory_writer
    facts are retrieved with precision
  - Tier 1.5 memory query uses scored fallback instead of silent unfiltered
    retry — unfiltered hits get a distance penalty so they don't crowd out
    speaker-specific results
  - RAG_DISTANCE_THRESHOLD kept at 1.2 (correct) — synthesis.py's 2.0 was
    the broken one (fixed separately)
"""

import time
from typing import Optional, TYPE_CHECKING

from core.log import log, log_event, log_error

if TYPE_CHECKING:
    from core.archivist import Brief

# ── Confidence thresholds ─────────────────────────────────────────────────────

LIBRARIAN_SUFFICIENT   = 0.75
RAG_SUFFICIENT         = 0.45
WEB_SEARCH_THRESHOLD   = 0.20
RAG_DISTANCE_THRESHOLD = 1.2

# How many results to pull per collection in _query_rag.
# Headmate collection gets more because it's more signal-dense.
RAG_N_HEADMATE = 8
RAG_N_MAIN     = 5


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
    """
    Query the knowledge graph for structured relationship edges.
    Returns formatted knowledge block if strong edges exist.
    This is Tier 1 — fires before RAG, returns high-confidence structured facts.
    """
    if not headmate:
        return {"text": "", "confidence": 0.0, "domain": ""}
 
    try:
        from core.knowledge_graph import query_context, format_for_prompt
 
        edges = query_context(
            subject     = headmate,
            topics      = topics,
            min_strength= 0.25,
            limit       = 15,
        )
 
        if not edges:
            return {"text": "", "confidence": 0.0, "domain": ""}
 
        # Score confidence based on average edge strength and count
        avg_strength = sum(e["strength"] for e in edges) / len(edges)
        count_bonus  = min(0.2, len(edges) * 0.02)
        confidence   = min(0.85, avg_strength + count_bonus)
 
        text = format_for_prompt(edges, subject=headmate)
 
        if not text:
            return {"text": "", "confidence": 0.0, "domain": ""}
 
        log_event("Mind", "LIBRARIAN_HIT",
            headmate = headmate,
            edges    = len(edges),
            confidence = round(confidence, 2),
            topics   = topics,
        )
 
        return {
            "text":       text,
            "confidence": confidence,
            "domain":     "knowledge_graph",
        }
 
    except Exception as e:
        log_error("Mind", "librarian query failed", exc=e)
        return {"text": "", "confidence": 0.0, "domain": ""}


# ── RAG retrieval ─────────────────────────────────────────────────────────────

async def _query_rag(
    topics: list,
    query: str,
    fronters: list,
    session_id: str,
    headmate: Optional[str] = None,
) -> dict:
    """
    Query ChromaDB with speaker-prioritized retrieval.

    Order:
      1. Headmate's named collection (speaker-filtered extracted_facts first,
         then all other entries) — RAG_N_HEADMATE results
      2. Each additional fronter's collection — RAG_N_HEADMATE results each
      3. main — RAG_N_MAIN results, deduplicated against above

    Chunks from the headmate collection are always ranked above main chunks
    at equal distance, so specific facts about the current speaker surface
    before generic noise.
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = {c.name for c in client.list_collections()}

        # Build ordered collection list: headmate first, then other fronters, then main
        ordered = []
        if headmate:
            h = headmate.lower().strip()
            if h and h in existing:
                ordered.append((h, RAG_N_HEADMATE, True))   # (name, n, is_primary_speaker)
        for f in (fronters or []):
            name = f.lower().strip() if isinstance(f, str) else str(f).lower().strip()
            if name and name != (headmate or "").lower() and name in existing:
                ordered.append((name, RAG_N_HEADMATE, False))
        if "main" in existing:
            ordered.append(("main", RAG_N_MAIN, False))

        if not ordered:
            return {"text": "", "confidence": 0.0, "chunks": []}

        all_chunks = []
        seen_texts = set()

        for collection_name, n_results, is_primary in ordered:
            try:
                store = RAGStore(collection_name=collection_name)
                if store.count == 0:
                    continue

                # For the primary speaker's collection: pull extracted_facts
                # first (speaker-filtered), then general entries.
                # For everything else: standard retrieve.
                if is_primary and headmate:
                    # Pass 1 — extracted facts for this speaker specifically
                    try:
                        fact_chunks = store.retrieve(
                            query=query,
                            n_results=min(n_results, store.count),
                            where={
                                "$and": [
                                    {"type":    {"$eq": "extracted_fact"}},
                                    {"speaker": {"$eq": headmate.lower()}},
                                ]
                            },
                        )
                        for chunk in fact_chunks:
                            text = chunk.get("text", "").strip()
                            if text and text not in seen_texts:
                                seen_texts.add(text)
                                all_chunks.append({
                                    **chunk,
                                    "collection": collection_name,
                                    "priority":   "extracted_fact",
                                })
                    except Exception:
                        pass  # where clause may fail if collection has no extracted_facts yet

                    # Pass 2 — everything else in the headmate collection
                    try:
                        general_chunks = store.retrieve(
                            query=query,
                            n_results=min(n_results, store.count),
                        )
                        for chunk in general_chunks:
                            text = chunk.get("text", "").strip()
                            if text and text not in seen_texts:
                                seen_texts.add(text)
                                all_chunks.append({
                                    **chunk,
                                    "collection": collection_name,
                                    "priority":   "headmate",
                                })
                    except Exception as e:
                        log_error("Mind", f"RAG general query failed for {collection_name}", exc=e)

                else:
                    chunks = store.retrieve(query=query, n_results=n_results)
                    for chunk in chunks:
                        text = chunk.get("text", "").strip()
                        if text and text not in seen_texts:
                            seen_texts.add(text)
                            all_chunks.append({
                                **chunk,
                                "collection": collection_name,
                                "priority":   "general",
                            })

            except Exception as e:
                log_error("Mind", f"RAG query failed for collection {collection_name}", exc=e)

        if not all_chunks:
            return {"text": "", "confidence": 0.0, "chunks": []}

        # Sort: extracted_facts first, then headmate, then general — within
        # each priority bucket, sort by distance ascending.
        priority_order = {"extracted_fact": 0, "headmate": 1, "general": 2}
        all_chunks.sort(key=lambda c: (
            priority_order.get(c.get("priority", "general"), 2),
            c.get("distance", 999),
        ))

        good_chunks = [
            c for c in all_chunks
            if c.get("distance", 999) < RAG_DISTANCE_THRESHOLD
        ]

        log_event("Mind", "RAG_RETRIEVED",
            session=session_id[:8],
            total=len(all_chunks),
            good=len(good_chunks),
            collections=[c["collection"] for c in all_chunks[:6]],
            priorities=[c.get("priority") for c in all_chunks[:6]],
        )

        if not good_chunks:
            return {"text": "", "confidence": 0.0, "chunks": all_chunks}

        best_distance = good_chunks[0].get("distance", 1.0)
        confidence = max(0.0, min(1.0, 1.0 - (best_distance / RAG_DISTANCE_THRESHOLD)))

        seen_text = set()
        text_parts = []
        for chunk in good_chunks[:6]:   # raised from 5 → 6
            text = chunk.get("text", "").strip()
            if text and text not in seen_text:
                seen_text.add(text)
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
          Librarian → Memory/Conscious → RAG → Web search (stub)
        Each tier only fires if the previous tier's confidence is insufficient.
        """
        t_start = time.monotonic()

        topics     = brief.topics
        query      = brief.message
        fronters   = brief.fronters
        headmate   = brief.headmate
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
        librarian_result     = await _query_librarian(topics, query, headmate)
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

        # ── Tier 1.5: Memory/Conscious collections ────────────────────────────
        # Query Gizmo's own memory and conscious collections first.
        # Speaker-filtered where possible; unfiltered hits get a distance
        # penalty so they don't crowd out speaker-specific results.
        memory_text = ""
        try:
            from tools.memory_tool import _get_collection, CONSCIOUS_COLLECTION, MEMORY_COLLECTION

            all_hits = []

            for col_name in [CONSCIOUS_COLLECTION, MEMORY_COLLECTION]:
                try:
                    col   = _get_collection(col_name)
                    count = col.count()
                    if count == 0:
                        continue

                    # Filtered pass — entries about this headmate specifically
                    if headmate:
                        k = min(4, count)
                        results = col.query(
                            query_texts=[query],
                            n_results=k,
                            where={"subject": {"$eq": headmate.lower()}},
                        )
                        docs  = results.get("documents", [[]])[0]
                        metas = results.get("metadatas", [[]])[0]
                        dists = results.get("distances", [[]])[0]
                        for doc, meta, dist in zip(docs, metas, dists):
                            if dist < 1.1:
                                all_hits.append({
                                    "text":       doc,
                                    "collection": col_name,
                                    "subject":    meta.get("subject", ""),
                                    "distance":   dist,
                                    "filtered":   True,
                                })

                    # Unfiltered pass — general entries, penalised distance
                    k = min(3, count)
                    results = col.query(query_texts=[query], n_results=k)
                    docs  = results.get("documents", [[]])[0]
                    metas = results.get("metadatas", [[]])[0]
                    dists = results.get("distances", [[]])[0]
                    for doc, meta, dist in zip(docs, metas, dists):
                        # Apply distance penalty so unfiltered hits rank below
                        # speaker-specific ones unless they're meaningfully closer
                        penalised = dist + 0.3
                        if penalised < 1.1:
                            all_hits.append({
                                "text":       doc,
                                "collection": col_name,
                                "subject":    meta.get("subject", ""),
                                "distance":   penalised,
                                "filtered":   False,
                            })

                except Exception as e:
                    log_error("Mind", f"memory collection query failed for {col_name}", exc=e)

            if all_hits:
                # Deduplicate and rank
                seen = set()
                unique_hits = []
                for h in sorted(all_hits, key=lambda x: x["distance"]):
                    if h["text"] not in seen:
                        seen.add(h["text"])
                        unique_hits.append(h)

                memory_text = "\n\n".join(h["text"] for h in unique_hits[:6])
                log_event("Mind", "MEMORY_HIT",
                    session=session_id[:8],
                    hits=len(unique_hits),
                    filtered=[h["filtered"] for h in unique_hits[:6]],
                    query=query[:60],
                )

        except Exception as e:
            log_error("Mind", "memory collection query failed", exc=e)

        if memory_text:
            return {
                "synthesis":      memory_text,
                "confidence":     0.8,
                "source":         "memory",
                "topics_queried": topics,
                "chunks":         [],
            }

        # ── Tier 2: RAG ───────────────────────────────────────────────────────
        rag_result     = await _query_rag(topics, query, fronters, session_id, headmate=headmate)
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
                confidence  = min(0.7, rag_confidence + 0.15)
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