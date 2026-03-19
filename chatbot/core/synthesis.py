"""
core/synthesis.py
Multi-collection RAG retrieval + combined overview/synthesis in one LLM call.

Changes from original:
  - ambient_log is always included in retrieval (lowest priority, separate label)
  - web_knowledge included — results from epistemic_synthesis searches
    labelled with truth_confidence so the LLM knows how reliable they are
  - Temporal query detection: "earlier today", "this morning" etc. trigger
    time-filtered retrieval from ambient_log before standard semantic search
  - Topic cluster surfacing: synthesis includes related topics when available
"""

import re
from datetime import datetime
from typing import Optional
from core.rag import RAGStore, CHROMA_PERSIST_DIR
import chromadb

DISTANCE_THRESHOLD = 2.0
AMBIENT_COLLECTION = "ambient_log"
WEB_COLLECTION = "web_knowledge"

# Personality collections — never retrieved as memory
from core.personality_growth import PERSONALITY_COLLECTIONS

# Temporal signal patterns — trigger time-filtered ambient retrieval
_TEMPORAL_PATTERNS = re.compile(
    r"\b(earlier today|this morning|this afternoon|this evening|"
    r"a (little |while )?(ago|earlier)|just now|recently|"
    r"before|last hour|few (minutes|hours) ago|"
    r"we were talking|i was (saying|talking|thinking)|"
    r"i mentioned|you heard|did (i|you) (say|mention|talk))\b",
    re.IGNORECASE,
)


def _is_temporal_query(query: str) -> bool:
    return bool(_TEMPORAL_PATTERNS.search(query))


def _get_collection_store(name: str) -> Optional[RAGStore]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [c.name for c in client.list_collections()]
        if name not in existing:
            return None
        store = RAGStore(collection_name=name)
        return store if store.count > 0 else None
    except Exception:
        return None


def _get_all_collections() -> list[str]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        all_names = [c.name for c in client.list_collections()]
        # Filter out personality collections — those are injected separately
        excluded = set(PERSONALITY_COLLECTIONS) | {AMBIENT_COLLECTION, WEB_COLLECTION}
        return [n for n in all_names if n not in excluded]
    except Exception:
        return ["main"]


def _extract_topics_from_chunks(chunks: list[dict]) -> list[str]:
    """
    Pull all unique topic tags from retrieved chunks' metadata.
    """
    all_topics = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        topics_str = meta.get("topics", "")
        if topics_str:
            for t in topics_str.split(","):
                t = t.strip()
                if t:
                    all_topics.add(t)
    return sorted(all_topics)


def _confidence_label(confidence: float) -> str:
    """Human-readable confidence label for web knowledge chunks."""
    if confidence >= 0.85:
        return "well-established"
    elif confidence >= 0.60:
        return "generally agreed"
    elif confidence >= 0.30:
        return "contested"
    else:
        return "uncertain"


async def retrieve_and_synthesize(
    query: str,
    current_host: Optional[str] = None,
    fronters: Optional[list] = None,
    history_summary: Optional[str] = None,
    n_per_collection: int = 5,
    llm=None,
) -> str:
    """
    Query all collections including ambient_log and web_knowledge,
    synthesize into one context block.

    Priority ordering:
      1. Host collection
      2. Fronter collections
      3. main + other personal collections
      4. web_knowledge (labelled with confidence level)
      5. ambient_log (lowest priority, time-labelled)

    Temporal queries get additional time-filtered ambient retrieval.
    Web knowledge chunks are labelled with their truth_confidence so
    the synthesis LLM knows how much weight to give them.
    """
    all_known = _get_all_collections()
    is_temporal = _is_temporal_query(query)

    if is_temporal:
        print(f"[Synthesis] Temporal query detected: '{query[:60]}'")

    # Priority ordering: host → fronters → everything else
    priority = []
    if current_host:
        h = current_host.lower().strip()
        if h:
            priority.append(h)
    if fronters:
        for f in fronters:
            name = f.lower().strip() if isinstance(f, str) else str(f).lower().strip()
            if name and name not in priority:
                priority.append(name)

    non_priority = [c for c in all_known if c not in priority]
    collections_to_search = priority + non_priority
    if "main" not in collections_to_search:
        collections_to_search.append("main")

    # ── Standard semantic retrieval across personal collections ──────────────
    all_chunks = []
    for collection_name in collections_to_search:
        store = _get_collection_store(collection_name)
        if store is None:
            continue
        try:
            results = store.retrieve(query, n_results=n_per_collection)
            for r in results:
                all_chunks.append({
                    "collection": collection_name,
                    "text": r["text"],
                    "distance": r["distance"],
                    "metadata": r.get("metadata", {}),
                    "source": "memory",
                })
        except Exception as e:
            print(f"[Synthesis] Error querying '{collection_name}': {e}")

    # ── Web knowledge retrieval ───────────────────────────────────────────────
    web_store = _get_collection_store(WEB_COLLECTION)
    if web_store is not None:
        try:
            web_results = web_store.retrieve(query, n_results=4)
            for r in web_results:
                meta = r.get("metadata", {})
                confidence = float(meta.get("truth_confidence", 0.5))
                consensus_level = meta.get("consensus_level", "unknown")
                retrieved_at = meta.get("retrieved_at", "")
                label = (
                    f"web [{_confidence_label(confidence)}]"
                    + (f" retrieved {retrieved_at[:10]}" if retrieved_at else "")
                )
                all_chunks.append({
                    "collection": label,
                    "text": r["text"],
                    "distance": r["distance"],
                    "metadata": meta,
                    "source": "web",
                    "truth_confidence": confidence,
                    "consensus_level": consensus_level,
                })
        except Exception as e:
            print(f"[Synthesis] Error querying web_knowledge: {e}")

    # ── Ambient retrieval ─────────────────────────────────────────────────────
    ambient_store = _get_collection_store(AMBIENT_COLLECTION)
    if ambient_store is not None:
        try:
            if is_temporal:
                ambient_results = ambient_store.retrieve_recent(
                    query=query,
                    hours_back=8,
                    n_results=6,
                )
                print(f"[Synthesis] Temporal ambient results: {len(ambient_results)}")
            else:
                ambient_results = ambient_store.retrieve(query, n_results=4)

            for r in ambient_results:
                meta = r.get("metadata", {})
                time_label = meta.get("time", "")
                label = f"ambient_log ({time_label})" if time_label else AMBIENT_COLLECTION
                all_chunks.append({
                    "collection": label,
                    "text": r["text"],
                    "distance": r["distance"],
                    "metadata": meta,
                    "source": "ambient",
                })
        except Exception as e:
            print(f"[Synthesis] Error querying ambient_log: {e}")

    if not all_chunks:
        return ""

    # ── Filter and rank ───────────────────────────────────────────────────────
    all_chunks.sort(key=lambda x: x["distance"])
    relevant = [c for c in all_chunks if c["distance"] < DISTANCE_THRESHOLD]

    if not relevant:
        print(f"[Synthesis] Nothing relevant found for: {query[:60]}")
        return ""

    relevant = relevant[:12]

    print(
        f"[Synthesis] Found {len(relevant)} relevant chunks across "
        f"{len({c['collection'] for c in relevant})} collections "
        f"({'temporal' if is_temporal else 'semantic'})"
    )

    # ── Extract related topics ────────────────────────────────────────────────
    related_topics = _extract_topics_from_chunks(relevant)
    topics_hint = ""
    if related_topics:
        topics_hint = f"\nRelated topics found: {', '.join(related_topics[:8])}"

    # ── Build chunk text with attribution ─────────────────────────────────────
    chunk_lines = []
    for c in relevant:
        meta = c.get("metadata", {})
        time_str = meta.get("time", "")
        collection_label = c["collection"]

        if c["source"] == "ambient" and time_str:
            prefix = f"[ambient @ {time_str}]"
        elif c["source"] == "web":
            confidence = c.get("truth_confidence", 0.5)
            prefix = f"[web — {_confidence_label(confidence)}]"
        else:
            prefix = f"[{collection_label}]"

        chunk_lines.append(f"{prefix} {c['text']}")
    chunk_text = "\n\n".join(chunk_lines)

    # ── History block ─────────────────────────────────────────────────────────
    history_block = ""
    if history_summary:
        history_block = f"\n\nRecent conversation summary:\n{history_summary}"

    # ── Single LLM call ───────────────────────────────────────────────────────
    chunk_count = len(relevant)
    collection_list = ", ".join(sorted({c["collection"] for c in relevant}))
    temporal_note = (
        "\nNote: This is a temporal query — prioritize time-stamped ambient chunks "
        "and include specific times if available."
        if is_temporal else ""
    )
    web_note = (
        "\nNote: Web chunks are labelled with confidence — "
        "'well-established' = sources agree, 'contested' = sources disagree. "
        "Reflect this uncertainty naturally in your synthesis."
        if any(c["source"] == "web" for c in relevant) else ""
    )

    prompt = [
        {
            "role": "user",
            "content": (
                f"You have {chunk_count} knowledge chunks from these sources: {collection_list}."
                f"{history_block}{topics_hint}{temporal_note}{web_note}\n\n"
                f"READ ALL OF THESE CHUNKS carefully before writing anything:\n\n"
                f"{chunk_text}\n\n"
                f"Current query: {query}\n\n"
                f"Write a single cohesive paragraph combining all relevant information. "
                f"Do not just summarize the first chunk — weave together everything relevant. "
                f"Attribute naturally where perspectives differ. "
                f"For web chunks marked 'contested', reflect that uncertainty. "
                f"For ambient chunks, include time naturally if relevant. "
                f"If chunks contradict, note both perspectives. "
                f"No bullet points. No headers. Flowing prose only. "
                f"3-6 sentences maximum."
            )
        }
    ]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You produce concise, attributed context summaries in flowing prose. "
                "Never use bullet points or headers. Never invent information not present in the chunks. "
                "Reflect uncertainty honestly — don't present contested claims as settled facts."
            ),
            max_new_tokens=300,
            temperature=0.3,
        )
        print(f"[Synthesis] Result: {result[:200]}")
        return result.strip()
    except Exception as e:
        print(f"[Synthesis] LLM call failed: {e}")
        return "\n\n".join(
            f"[{c['collection']}] {c['text']}" for c in relevant[:4]
        )