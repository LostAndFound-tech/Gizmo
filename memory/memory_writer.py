"""
memory/memory_writer.py
Real-time fact extraction and immediate ChromaDB ingestion.

Called after every user message, before the response is generated.
Extracts subjects and all facts about them, tags each fact with the
speaker's identity, and writes to their named collection immediately —
no waiting for the archiver.

Write side:
  - Facts tagged with speaker name so reads can filter by it
  - Written to speaker's named collection AND "main"
  - persona_prefix(speaker) injected into extraction prompt so the
    LLM understands who it's reading about — their history, preferences,
    and how Gizmo relates to them
  - time_context_block() injected so temporal facts are grounded

Read side (used by synthesis.py and mind.py):
  - retrieve_facts(query, speaker) adds where={speaker: name} filter
  - Returns only that speaker's facts, ranked by relevance
  - Falls back to unfiltered main search if speaker collection is empty
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

FACTS_COLLECTION = "main"
FACT_TYPE        = "extracted_fact"
MIN_CONFIDENCE   = 0.6


# ── Extraction ────────────────────────────────────────────────────────────────

async def extract_and_store(
    message: str,
    speaker: str,
    session_id: str,
    llm,
    persona: Optional[str] = None,   # kept for backward compat; now built internally
) -> list[dict]:
    """
    Extract all subjects and facts from a user message and write them
    immediately to ChromaDB.

    Each fact is tagged with the speaker's name so retrieval can filter
    by speaker without touching anyone else's data.

    Returns list of stored fact dicts (for logging/debugging).
    """
    if not message.strip() or not speaker:
        return []

    facts = await _extract_facts(message, speaker, llm)
    if not facts:
        return []

    stored = []
    for fact in facts:
        _store_fact(fact, speaker, session_id)
        stored.append(fact)

    if stored:
        print(f"[MemWriter] Stored {len(stored)} fact(s) for '{speaker}'")

    return stored


async def _extract_facts(
    message: str,
    speaker: str,
    llm,
) -> list[dict]:
    """
    Ask the LLM to pull every subject + fact out of the message.
    Uses persona_prefix(speaker) so the extraction understands who this
    person is and what might be significant about what they're saying.
    Uses time_context_block() so temporal facts are grounded correctly.
    """
    from core.persona import persona_prefix
    from core.temporal import time_context_block

    persona  = persona_prefix(speaker, include_gizmo_seed=False)
    time_ctx = time_context_block()

    prompt = [{
        "role": "user",
        "content": (
            f"{time_ctx}\n\n"
            f"Speaker: {speaker.capitalize()}\n\n"
            f"Message:\n\"{message}\"\n\n"
            f"Extract every subject mentioned and all facts the speaker reveals about each one.\n"
            f"A fact is anything stated or strongly implied — preferences, relationships, "
            f"history, opinions, plans, feelings, experiences, or beliefs.\n\n"
            f"DO extract:\n"
            f"  - 'I hate mushrooms' → subject: mushrooms, fact: {speaker} dislikes mushrooms\n"
            f"  - 'My sister Sarah lives in Denver' → subject: Sarah, fact: {speaker}'s sister Sarah lives in Denver\n"
            f"  - 'I've been struggling with sleep' → subject: sleep, fact: {speaker} has been having trouble sleeping\n"
            f"  - 'We're thinking of moving' → subject: moving, fact: {speaker} is considering relocating\n\n"
            f"DO NOT extract:\n"
            f"  - Generic small talk with no factual content\n"
            f"  - Questions (unless they reveal something about the asker)\n"
            f"  - Things said about Gizmo\n\n"
            f"Respond with ONLY valid JSON, no markdown:\n"
            f'{{"facts": [\n'
            f'  {{\n'
            f'    "subject": "short label for what this fact is about",\n'
            f'    "fact": "one clear sentence stating the fact, attributed to {speaker}",\n'
            f'    "category": "preference|relationship|experience|plan|belief|emotion|other",\n'
            f'    "confidence": 0.0-1.0,\n'
            f'    "raw_snippet": "shortest quote from message that supports this fact"\n'
            f'  }}\n'
            f']}}\n\n'
            f'If nothing factual, respond with {{"facts": []}}'
        )
    }]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                f"{persona}\n\n"
                f"You extract structured facts from natural speech. "
                f"Be thorough — capture everything stated or strongly implied. "
                f"Use your knowledge of who {speaker.capitalize()} is to judge what's significant. "
                f"Write each fact as a clear attributable sentence. "
                f"Respond with valid JSON only. No markdown."
            ),
            max_new_tokens=600,
            temperature=0.1,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        facts = []
        for f in parsed.get("facts", []):
            confidence = float(f.get("confidence", 0.5))
            if confidence < MIN_CONFIDENCE:
                continue
            subject = f.get("subject", "").strip()[:80]
            fact    = f.get("fact", "").strip()
            if not subject or not fact:
                continue
            facts.append({
                "subject":     subject,
                "fact":        fact,
                "category":    f.get("category", "other").lower(),
                "confidence":  confidence,
                "raw_snippet": f.get("raw_snippet", "")[:200],
            })

        return facts

    except Exception as e:
        print(f"[MemWriter] Extraction failed: {e}")
        return []


# ── Storage ───────────────────────────────────────────────────────────────────

def _store_fact(fact: dict, speaker: str, session_id: str) -> None:
    """
    Write a single extracted fact to ChromaDB.
    Written to both the speaker's named collection and 'main'.
    Tagged with speaker so reads can filter precisely.
    """
    from core.rag import RAGStore

    now      = datetime.now()
    fact_id  = f"fact_{uuid.uuid4().hex[:12]}"
    doc_text = f"{speaker.capitalize()} — {fact['subject']}: {fact['fact']}"

    metadata = {
        "type":        FACT_TYPE,
        "subject":     fact["subject"],
        "category":    fact["category"],
        "speaker":     speaker.lower(),
        "confidence":  str(round(fact["confidence"], 3)),
        "date":        now.strftime("%Y-%m-%d"),
        "time":        now.strftime("%H:%M"),
        "session_id":  session_id,
        "raw_snippet": fact["raw_snippet"],
        "source":      "conversation",
    }

    try:
        speaker_store = RAGStore(collection_name=speaker.lower().strip())
        speaker_store.ingest_texts([doc_text], metadatas=[metadata], ids=[fact_id])
    except Exception as e:
        print(f"[MemWriter] Speaker collection write failed: {e}")

    try:
        main_store = RAGStore(collection_name=FACTS_COLLECTION)
        main_meta  = {**metadata, "collection": FACTS_COLLECTION}
        main_store.ingest_texts([doc_text], metadatas=[main_meta], ids=[f"{fact_id}_main"])
    except Exception as e:
        print(f"[MemWriter] Main collection write failed: {e}")


# ── Retrieval helper (used by synthesis.py and mind.py) ───────────────────────

def retrieve_facts(
    query: str,
    speaker: Optional[str] = None,
    n_results: int = 8,
    distance_threshold: float = 1.3,
) -> list[dict]:
    """
    Retrieve facts relevant to a query, optionally filtered to one speaker.

    speaker: if provided, only returns facts tagged with this name.
             Uses the speaker's named collection first (faster, cleaner),
             falls back to main with a where filter.
    """
    from core.rag import RAGStore, CHROMA_PERSIST_DIR
    import chromadb

    results = []

    if speaker:
        try:
            client   = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            existing = [c.name for c in client.list_collections()]
            col_name = speaker.lower().strip()

            if col_name in existing:
                store = RAGStore(collection_name=col_name)
                if store.count > 0:
                    raw = store.retrieve(
                        query,
                        n_results=min(n_results, store.count),
                        where={"type": {"$eq": FACT_TYPE}},
                    )
                    results = [r for r in raw if r["distance"] < distance_threshold]
        except Exception as e:
            print(f"[MemWriter] Speaker retrieve failed: {e}")

    if not results:
        try:
            store = RAGStore(collection_name=FACTS_COLLECTION)
            if store.count > 0:
                where = {"type": {"$eq": FACT_TYPE}}
                if speaker:
                    where = {"$and": [
                        {"type":    {"$eq": FACT_TYPE}},
                        {"speaker": {"$eq": speaker.lower()}},
                    ]}
                raw = store.retrieve(
                    query,
                    n_results=min(n_results, store.count),
                    where=where,
                )
                results = [r for r in raw if r["distance"] < distance_threshold]
        except Exception as e:
            print(f"[MemWriter] Main retrieve failed: {e}")

    return results
