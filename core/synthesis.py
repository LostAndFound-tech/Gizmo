"""
core/synthesis.py
Multi-collection RAG retrieval + combined overview/synthesis in one LLM call.

Changes from original:
  - ambient_log is always included in retrieval (lowest priority, separate label)
  - Temporal query detection: "earlier today", "this morning" etc. trigger
    time-filtered retrieval from ambient_log before standard semantic search
  - Topic cluster surfacing: synthesis includes related topics when available
  - Personality context: known traits/interests for current_host injected
    into synthesis prompt when available
  - DISTANCE_THRESHOLD reduced from 2.0 → 1.3 (2.0 was effectively no filter)
  - Extracted facts from memory_writer pulled in as a priority lane —
    speaker-filtered, so each headmate only sees their own facts
  - persona_prefix() used as system prompt so synthesis is always written
    through the correct relational lens
  - time_context_block() injected into user prompt for temporal grounding
"""

import re
from datetime import datetime
from typing import Optional
from core.rag import RAGStore, CHROMA_PERSIST_DIR
import chromadb

DISTANCE_THRESHOLD = 1.3
AMBIENT_COLLECTION = "ambient_log"

_TEMPORAL_PATTERNS = re.compile(
    r"\b(earlier today|this morning|this afternoon|this evening|"
    r"a (little |while )?(ago|earlier)|just now|recently|"
    r"before|last hour|few (minutes|hours) ago|"
    r"we were talking|i was (saying|talking|thinking)|"
    r"i mentioned|you heard|did (i|you) (say|mention|talk))\b",
    re.IGNORECASE,
)

_PERSONAL_QUERY_PATTERN = re.compile(
    r"\b(what (do i|does .* )(like|hate|enjoy|prefer|think|believe|care)|"
    r"(my|.+'s) (preference|taste|interest|opinion|feeling|view)|"
    r"(do i|does .*) (like|hate|enjoy|prefer)|"
    r"what (am i|is .* ) into|"
    r"tell me about (my|.*'s) (personality|interests|preferences|values))\b",
    re.IGNORECASE,
)


def _is_personal_query(query: str) -> bool:
    return bool(_PERSONAL_QUERY_PATTERN.search(query))


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
        return [c.name for c in client.list_collections()]
    except Exception:
        return ["main"]


def _extract_topics_from_chunks(chunks: list[dict]) -> list[str]:
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


def _get_personality_block(current_host: Optional[str]) -> str:
    if not current_host:
        return ""
    try:
        from voice.personality_tool import get_personality_context
        return get_personality_context(current_host)
    except Exception:
        return ""


def _get_facts_block(query: str, current_host: Optional[str]) -> str:
    if not current_host:
        return ""
    try:
        from memory.memory_writer import retrieve_facts
        facts = retrieve_facts(
            query=query,
            speaker=current_host,
            n_results=8,
            distance_threshold=DISTANCE_THRESHOLD,
        )
        if not facts:
            return ""
        lines = [f"[facts:{current_host}] {f['text']}" for f in facts]
        print(f"[Synthesis] {len(facts)} extracted fact(s) retrieved for '{current_host}'")
        return "\n\n".join(lines)
    except Exception as e:
        print(f"[Synthesis] Facts retrieval failed: {e}")
        return ""


async def retrieve_and_synthesize(
    query: str,
    current_host: Optional[str] = None,
    fronters: Optional[list] = None,
    history_summary: Optional[str] = None,
    n_per_collection: int = 8,
    llm=None,
) -> str:
    from core.persona import persona_prefix_multi
    from core.temporal import time_context_block

    all_known  = _get_all_collections()
    is_temporal = _is_temporal_query(query)

    if is_temporal:
        print(f"[Synthesis] Temporal query detected: '{query[:60]}'")

    # Priority ordering: host → fronters → everything else → ambient last
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

    non_ambient = [c for c in all_known if c not in priority and c != AMBIENT_COLLECTION]
    collections_to_search = priority + non_ambient
    if "main" not in collections_to_search:
        collections_to_search.append("main")

    # ── Priority lane: extracted facts for current speaker ────────────────────
    facts_block = _get_facts_block(query, current_host)

    # ── Standard semantic retrieval ───────────────────────────────────────────
    all_chunks = []
    for collection_name in collections_to_search:
        store = _get_collection_store(collection_name)
        if store is None:
            continue
        try:
            results = store.retrieve(query, n_results=n_per_collection)
            for r in results:
                if r.get("metadata", {}).get("type") == "personality_signal":
                    continue
                if r.get("metadata", {}).get("type") == "extracted_fact":
                    continue
                all_chunks.append({
                    "collection": collection_name,
                    "text":       r["text"],
                    "distance":   r["distance"],
                    "metadata":   r.get("metadata", {}),
                    "source":     "memory",
                })
        except Exception as e:
            print(f"[Synthesis] Error querying '{collection_name}': {e}")

    # ── Ambient retrieval ─────────────────────────────────────────────────────
    ambient_store = _get_collection_store(AMBIENT_COLLECTION)
    if ambient_store is not None:
        try:
            if is_temporal:
                ambient_results = ambient_store.retrieve_recent(
                    query=query, hours_back=8, n_results=6,
                )
                print(f"[Synthesis] Temporal ambient results: {len(ambient_results)}")
            else:
                ambient_results = ambient_store.retrieve(query, n_results=4)

            for r in ambient_results:
                meta       = r.get("metadata", {})
                time_label = meta.get("time", "")
                spkr       = meta.get("speaker", "")
                if spkr and time_label:
                    label = f"ambient @ {time_label}, {spkr}"
                elif time_label:
                    label = f"ambient @ {time_label}"
                else:
                    label = AMBIENT_COLLECTION
                all_chunks.append({
                    "collection": label,
                    "text":       r["text"],
                    "distance":   r["distance"],
                    "metadata":   meta,
                    "source":     "ambient",
                })
        except Exception as e:
            print(f"[Synthesis] Error querying ambient_log: {e}")

    if not all_chunks and not facts_block:
        return ""

    # ── Filter and rank ───────────────────────────────────────────────────────
    all_chunks.sort(key=lambda x: x["distance"])
    relevant = [c for c in all_chunks if c["distance"] < DISTANCE_THRESHOLD]

    if not relevant and not facts_block:
        print(f"[Synthesis] Nothing relevant found for: {query[:60]}")
        return ""

    relevant = relevant[:12]

    print(
        f"[Synthesis] Found {len(relevant)} relevant chunks across "
        f"{len({c['collection'] for c in relevant})} collections "
        f"({'temporal' if is_temporal else 'semantic'})"
    )

    # ── Related topics ────────────────────────────────────────────────────────
    related_topics = _extract_topics_from_chunks(relevant)
    topics_hint = ""
    if related_topics:
        topics_hint = f"\nRelated topics found: {', '.join(related_topics[:8])}"

    # ── Personality context — only for direct personal queries ────────────────
    personality_block = ""
    if current_host and _is_personal_query(query):
        personality_context = _get_personality_block(current_host)
        if personality_context:
            personality_block = f"\n\n[Personality & interests]\n{personality_context}"
            print(f"[Synthesis] Personality context injected for '{current_host}' (personal query)")

    # ── Build chunk text — facts first, then general memory ───────────────────
    chunk_lines = []
    if facts_block:
        chunk_lines.append(facts_block)
    for c in relevant:
        chunk_lines.append(f"[{c['collection']}] {c['text']}")
    chunk_text = "\n\n".join(chunk_lines)

    # ── History block ─────────────────────────────────────────────────────────
    history_block = ""
    if history_summary:
        history_block = f"\n\nRecent conversation summary:\n{history_summary}"

    # ── Persona + time ────────────────────────────────────────────────────────
    all_present = list({s for s in ([current_host] + (fronters or [])) if s})
    persona     = persona_prefix_multi(all_present, include_gizmo_seed=True)
    time_ctx    = time_context_block()

    chunk_count     = len(relevant) + (1 if facts_block else 0)
    collection_list = ", ".join(sorted({c["collection"] for c in relevant}))
    if facts_block:
        collection_list = f"facts:{current_host}, " + collection_list
    temporal_note = (
        "\nNote: This is a temporal query — prioritize time-stamped ambient chunks "
        "and include specific times if available."
        if is_temporal else ""
    )

    prompt = [{
        "role": "user",
        "content": (
            f"{time_ctx}\n\n"
            f"You have {chunk_count} knowledge chunks from these sources: {collection_list}."
            f"{history_block}{personality_block}{topics_hint}{temporal_note}\n\n"
            f"READ ALL OF THESE CHUNKS carefully before writing anything:\n\n"
            f"{chunk_text}\n\n"
            f"Current query: {query}\n\n"
            f"Now write a single cohesive paragraph that combines information from "
            f"ALL the chunks above that are relevant to the query. "
            f"Facts tagged with 'facts:{current_host or 'speaker'}' are the highest priority — "
            f"use them first. "
            f"Do not just summarize the first chunk — weave together everything relevant. "
            f"Attribute naturally where perspectives differ (e.g. 'Jonah mentioned...', 'Oren noted...'). "
            f"For ambient chunks, include speaker and time naturally if relevant "
            f"(e.g. 'Earlier this morning Alice mentioned...'). "
            f"If personality/interest context is present, use it to colour your response "
            f"naturally — don't list traits, just let them inform the answer. "
            f"If multiple chunks say similar things, merge them into one clear statement. "
            f"If chunks contradict, note both perspectives. "
            f"No bullet points. No headers. Flowing prose only. "
            f"3-6 sentences maximum."
        )
    }]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                f"{persona}\n\n"
                f"You produce concise, attributed context summaries in flowing prose. "
                f"Never use bullet points or headers. Never invent information not present in the chunks. "
                f"For ambient memory chunks, preserve time and speaker references naturally. "
                f"Use personality context to inform tone and relevance, not to list facts. "
                f"Facts labelled with the speaker's name are ground truth — always include them."
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
