"""
core/protocols.py
Dynamic protocol system — researches, builds, and runs coping/support protocols.

Flow:
  1. Distress detected by wellness.py → trigger_protocol() called
  2. Check RAG for existing protocol for this category
  3. If none, web-search for evidence-based strategies + distill into steps
  4. Store protocol to RAG (persists across reconnects)
  5. Agent delivers acknowledgment immediately, protocol steps arrive async
  6. Session tracks active protocol state — agent checks on every message
  7. Deflection is gently pushed back on using ambient/wellness context
  8. Protocol closes only when user confirms they're okay, or different fronter
     explicitly dismisses it (noted in log)

Protocol state schema in RAG (collection: 'protocols'):
  {
    "session_id":    str,
    "fronter":       str,       # who the protocol was opened for
    "category":      str,       # e.g. "executive_dysfunction"
    "steps":         str,       # JSON-encoded list of step strings
    "current_step":  int,       # 0-indexed
    "status":        "active" | "complete" | "dismissed",
    "opened_at":     ISO str,
    "closed_at":     ISO str | "",
    "closed_by":     str,
  }
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional

PROTOCOLS_COLLECTION = "protocols"

# How long before an active protocol auto-expires (hours)
PROTOCOL_TIMEOUT_HOURS = 4

# Wellness category → search query for protocol research
CATEGORY_QUERIES = {
    "executive_dysfunction": "executive dysfunction coping strategies evidence based practical steps",
    "dissociation":          "grounding techniques dissociation evidence based step by step",
    "anxiety":               "anxiety grounding techniques practical immediate relief evidence based",
    "hallucinations":        "managing hallucinations coping strategies practical support",
    "physical_symptoms":     "managing physical distress symptoms practical coping strategies",
    "overwhelm":             "overwhelm coping strategies practical immediate steps evidence based",
    "switching_difficulty":  "DID switching difficulty support strategies practical",
    "general_distress":      "general distress coping strategies immediate practical support",
}

# In-memory protocol state cache — keyed by session_id
# Mirrors RAG but avoids a lookup on every message
_active_protocols: dict[str, dict] = {}


# ── Protocol retrieval ────────────────────────────────────────────────────────

def get_active_protocol(session_id: str) -> Optional[dict]:
    """
    Return the active protocol for a session, or None.
    Checks in-memory cache first, falls back to RAG.
    Auto-expires stale protocols.
    """
    # Check memory cache
    proto = _active_protocols.get(session_id)
    if proto and proto.get("status") == "active":
        # Check timeout
        opened = proto.get("opened_at", "")
        if opened:
            try:
                age_hours = (datetime.now() - datetime.fromisoformat(opened)).total_seconds() / 3600
                if age_hours > PROTOCOL_TIMEOUT_HOURS:
                    print(f"[Protocols] Protocol expired for session {session_id[:8]}")
                    _close_protocol_in_memory(session_id, "expired", "system")
                    return None
            except ValueError:
                pass
        return proto

    # Fall back to RAG
    proto = _load_protocol_from_rag(session_id)
    if proto:
        _active_protocols[session_id] = proto
    return proto


def _load_protocol_from_rag(session_id: str) -> Optional[dict]:
    """Load active protocol from RAG for this session."""
    try:
        from core.rag import RAGStore
        import chromadb
        from core.rag import CHROMA_PERSIST_DIR

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [c.name for c in client.list_collections()]
        if PROTOCOLS_COLLECTION not in existing:
            return None

        store = RAGStore(collection_name=PROTOCOLS_COLLECTION)
        results = store.collection.get(
            where={
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"status": {"$eq": "active"}},
                ]
            }
        )

        if not results["ids"]:
            return None

        # Most recent active protocol
        meta = results["metadatas"][-1]
        proto_id = results["ids"][-1]

        # Check timeout
        opened = meta.get("opened_at", "")
        if opened:
            try:
                age_hours = (datetime.now() - datetime.fromisoformat(opened)).total_seconds() / 3600
                if age_hours > PROTOCOL_TIMEOUT_HOURS:
                    _mark_protocol_status(proto_id, "expired", "system")
                    return None
            except ValueError:
                pass

        return {
            "id": proto_id,
            "session_id": session_id,
            "fronter": meta.get("fronter", ""),
            "category": meta.get("category", ""),
            "steps": json.loads(meta.get("steps", "[]")),
            "current_step": int(meta.get("current_step", 0)),
            "status": meta.get("status", "active"),
            "opened_at": meta.get("opened_at", ""),
        }

    except Exception as e:
        print(f"[Protocols] RAG load failed: {e}")
        return None


# ── Protocol research ─────────────────────────────────────────────────────────

async def _fetch_cached_protocol_steps(category: str, llm) -> Optional[list[str]]:
    """
    Check if we already have researched steps for this category in RAG.
    Returns step list if found and recent (< 30 days), else None.
    """
    try:
        from core.rag import RAGStore
        store = RAGStore(collection_name=PROTOCOLS_COLLECTION)
        results = store.collection.get(
            where={
                "$and": [
                    {"category": {"$eq": category}},
                    {"type": {"$eq": "protocol_template"}},
                ]
            }
        )
        if results["ids"] and results["metadatas"]:
            meta = results["metadatas"][-1]
            steps_raw = meta.get("steps", "[]")
            steps = json.loads(steps_raw)
            if steps:
                print(f"[Protocols] Using cached protocol for '{category}'")
                return steps
    except Exception as e:
        print(f"[Protocols] Cache check failed: {e}")
    return None


async def _research_protocol(category: str, llm) -> list[str]:
    """
    Web-search for evidence-based strategies and distill into 4-6 actionable steps.
    Falls back to LLM knowledge if search unavailable.
    """
    query = CATEGORY_QUERIES.get(category, f"{category} coping strategies practical steps")
    search_context = ""

    # Try web search
    try:
        from core.llm import llm as shared_llm
        import httpx

        search_url = "https://api.anthropic.com/v1/messages"
        # Use the LLM's web search capability via a tool-enabled call
        resp = await shared_llm.generate(
            messages=[{
                "role": "user",
                "content": (
                    f"Search for: {query}\n\n"
                    f"Find 3-5 practical, evidence-based strategies. "
                    f"Summarize the key actionable points in plain language."
                )
            }],
            system_prompt=(
                "You are a research assistant. Summarize practical coping strategies "
                "from reputable sources. Be specific and actionable, not clinical."
            ),
            max_new_tokens=400,
            temperature=0.2,
        )
        search_context = resp.strip()
        print(f"[Protocols] Research complete for '{category}': {len(search_context)} chars")
    except Exception as e:
        print(f"[Protocols] Web search failed, using LLM knowledge: {e}")

    # Distill into protocol steps
    distill_prompt = [
        {
            "role": "user",
            "content": (
                f"Category: {category.replace('_', ' ')}\n\n"
                + (f"Research findings:\n{search_context}\n\n" if search_context else "")
                + "Based on this, write 4-6 practical protocol steps for someone "
                f"currently experiencing {category.replace('_', ' ')}. "
                "Each step should be:\n"
                "- Conversational, warm, not clinical\n"
                "- Immediately actionable (something they can do right now)\n"
                "- 1-2 sentences max\n"
                "- Written as if speaking directly to the person\n\n"
                "Respond with ONLY a JSON array of step strings, no markdown:\n"
                '["Step one text", "Step two text", ...]'
            )
        }
    ]

    try:
        raw = await llm.generate(
            distill_prompt,
            system_prompt=(
                "You distill coping strategies into warm, practical protocol steps. "
                "Respond only with a JSON array of strings."
            ),
            max_new_tokens=500,
            temperature=0.3,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        steps = json.loads(raw)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            print(f"[Protocols] Distilled {len(steps)} steps for '{category}'")
            return steps
    except Exception as e:
        print(f"[Protocols] Distillation failed: {e}")

    # Hard fallback
    return [
        "Let's take this one moment at a time. What does right now feel like?",
        "Try to find one small thing you can do — even just getting some water.",
        "You don't have to solve everything right now. What's the most immediate thing?",
        "I'm here. Take your time.",
    ]


async def _store_protocol_template(category: str, steps: list[str]) -> None:
    """Cache researched steps as a reusable template."""
    try:
        from core.rag import RAGStore
        store = RAGStore(collection_name=PROTOCOLS_COLLECTION)
        doc = f"Protocol template for {category}: " + " | ".join(steps)
        store.ingest_texts(
            [doc],
            metadatas=[{
                "type": "protocol_template",
                "category": category,
                "steps": json.dumps(steps),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }],
            ids=[f"template_{category}"],
        )
    except Exception as e:
        print(f"[Protocols] Template storage failed: {e}")


# ── Protocol lifecycle ────────────────────────────────────────────────────────

def _save_protocol_to_rag(proto: dict) -> str:
    """Persist a new active protocol to RAG. Returns its ID."""
    try:
        from core.rag import RAGStore
        proto_id = f"protocol_{uuid.uuid4().hex[:12]}"
        store = RAGStore(collection_name=PROTOCOLS_COLLECTION)
        doc = (
            f"Active protocol for {proto['fronter']} — {proto['category']} — "
            f"opened {proto['opened_at']}"
        )
        store.ingest_texts(
            [doc],
            metadatas=[{
                "type": "active_protocol",
                "session_id": proto["session_id"],
                "fronter": proto["fronter"],
                "category": proto["category"],
                "steps": json.dumps(proto["steps"]),
                "current_step": str(proto["current_step"]),
                "status": "active",
                "opened_at": proto["opened_at"],
                "closed_at": "",
                "closed_by": "",
            }],
            ids=[proto_id],
        )
        return proto_id
    except Exception as e:
        print(f"[Protocols] Save failed: {e}")
        return ""


def _mark_protocol_status(
    proto_id: str,
    status: str,
    closed_by: str,
) -> None:
    """Update protocol status in RAG."""
    try:
        from core.rag import RAGStore
        store = RAGStore(collection_name=PROTOCOLS_COLLECTION)
        result = store.collection.get(ids=[proto_id])
        if not result["ids"]:
            return
        meta = result["metadatas"][0]
        meta["status"] = status
        meta["closed_at"] = datetime.now().isoformat(timespec="seconds")
        meta["closed_by"] = closed_by
        store.collection.update(ids=[proto_id], metadatas=[meta])
        print(f"[Protocols] Protocol {proto_id} → {status} (by {closed_by})")
    except Exception as e:
        print(f"[Protocols] Status update failed: {e}")


def _advance_step_in_rag(proto_id: str, new_step: int) -> None:
    """Update current_step in RAG."""
    try:
        from core.rag import RAGStore
        store = RAGStore(collection_name=PROTOCOLS_COLLECTION)
        result = store.collection.get(ids=[proto_id])
        if not result["ids"]:
            return
        meta = result["metadatas"][0]
        meta["current_step"] = str(new_step)
        store.collection.update(ids=[proto_id], metadatas=[meta])
    except Exception as e:
        print(f"[Protocols] Step advance failed: {e}")


def _close_protocol_in_memory(session_id: str, status: str, closed_by: str) -> None:
    proto = _active_protocols.get(session_id)
    if proto:
        proto["status"] = status
        proto_id = proto.get("id", "")
        if proto_id:
            _mark_protocol_status(proto_id, status, closed_by)
        _active_protocols.pop(session_id, None)


# ── Public API ────────────────────────────────────────────────────────────────

async def trigger_protocol(
    session_id: str,
    category: str,
    fronter: str,
    llm,
    push_fn,  # async callable(message: str) — sends to client
) -> None:
    """
    Called when distress is detected. Runs async so the agent can
    yield its acknowledgment immediately while research happens.

    push_fn is the server's _push_to_all — used to deliver steps
    after research completes.
    """
    # Don't stack protocols — if one is already active, just continue it
    existing = get_active_protocol(session_id)
    if existing:
        print(f"[Protocols] Protocol already active for {session_id[:8]}, skipping trigger")
        return

    print(f"[Protocols] Triggering protocol for '{category}' — {fronter}")

    # Research phase (background)
    steps = await _fetch_cached_protocol_steps(category, llm)
    if not steps:
        steps = await _research_protocol(category, llm)
        await _store_protocol_template(category, steps)

    # Build and persist protocol state
    proto = {
        "session_id": session_id,
        "fronter": fronter,
        "category": category,
        "steps": steps,
        "current_step": 0,
        "status": "active",
        "opened_at": datetime.now().isoformat(timespec="seconds"),
    }
    proto_id = _save_protocol_to_rag(proto)
    proto["id"] = proto_id
    _active_protocols[session_id] = proto

    # Deliver first step
    step_text = steps[0] if steps else "I'm here. Take your time."
    await push_fn(f"Hey, still with me? {step_text}")
    print(f"[Protocols] Delivered step 0 for session {session_id[:8]}")


def advance_protocol(session_id: str) -> Optional[str]:
    """
    Move to the next step. Returns the next step text, or None if complete.
    Called by the agent when the user responds during an active protocol.
    """
    proto = get_active_protocol(session_id)
    if not proto:
        return None

    next_step = proto["current_step"] + 1
    steps = proto["steps"]

    if next_step >= len(steps):
        return None  # protocol complete — let agent handle close

    proto["current_step"] = next_step
    _active_protocols[session_id] = proto
    _advance_step_in_rag(proto.get("id", ""), next_step)

    return steps[next_step]


def close_protocol(
    session_id: str,
    closed_by: str,
    original_fronter: str,
) -> dict:
    """
    Close the active protocol. Returns info the agent can use to respond naturally.
    """
    proto = get_active_protocol(session_id)
    if not proto:
        return {"closed": False}

    different_fronter = (
        closed_by.lower().strip() != original_fronter.lower().strip()
        if closed_by and original_fronter else False
    )

    _close_protocol_in_memory(session_id, "complete", closed_by)

    return {
        "closed": True,
        "different_fronter": different_fronter,
        "original_fronter": original_fronter,
        "closed_by": closed_by,
        "steps_completed": proto.get("current_step", 0),
        "total_steps": len(proto.get("steps", [])),
    }


async def build_deflection_response(
    session_id: str,
    user_message: str,
    current_host: str,
    llm,
) -> str:
    """
    User is deflecting ("I'm fine", "it's okay", "I'll just lie here").
    Build a gentle pushback, cross-referencing ambient/wellness data
    only when clearly relevant.
    """
    proto = get_active_protocol(session_id)
    category = proto.get("category", "distress") if proto else "distress"

    # Pull relevant ambient context — wellness logs and today's ambient
    context_snippets = []
    try:
        from core.rag import RAGStore
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")

        # Recent wellness events
        wellness_store = RAGStore(collection_name="wellness")
        if wellness_store.count > 0:
            results = wellness_store.collection.get(
                where={"date": {"$eq": today}},
                limit=3,
            )
            for doc in (results.get("documents") or []):
                if doc:
                    context_snippets.append(f"[wellness] {doc[:200]}")

        # Today's ambient log — look for anything relevant (eating, activity, stress)
        ambient_store = RAGStore(collection_name="ambient_log")
        if ambient_store.count > 0:
            ambient_results = ambient_store.retrieve(
                query="eating food meal activity today",
                n_results=4,
            )
            for r in ambient_results:
                meta = r.get("metadata", {})
                if meta.get("date") == today:
                    context_snippets.append(f"[ambient {meta.get('time', '')}] {r['text'][:200]}")

    except Exception as e:
        print(f"[Protocols] Context fetch for deflection failed: {e}")

    context_block = ""
    if context_snippets:
        context_block = "\n".join(context_snippets)

    prompt = [
        {
            "role": "user",
            "content": (
                f"Someone experiencing {category.replace('_', ' ')} just said: \"{user_message}\"\n"
                f"This sounds like deflection — they're brushing off support.\n"
                + (f"\nContext from today:\n{context_block}\n" if context_block else "")
                + "\nWrite a single, warm, gentle response that:\n"
                "- Doesn't immediately accept the deflection\n"
                "- References specific context from today ONLY if it's clearly relevant "
                "  (e.g. if they haven't eaten, mention it; if it's unrelated, don't)\n"
                "- Isn't pushy or alarming — just present and caring\n"
                "- Ends with a soft question or observation, not a demand\n"
                "2-3 sentences max. Conversational tone."
            )
        }
    ]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You are a warm, present support figure. You notice when someone is "
                "brushing off help and respond with gentle care, not pressure. "
                "Use context only when it's clearly relevant."
            ),
            max_new_tokens=120,
            temperature=0.4,
        )
        return result.strip()
    except Exception as e:
        print(f"[Protocols] Deflection response failed: {e}")
        return "Are you sure? I'm happy to keep going if you need it."


def is_deflection(message: str) -> bool:
    """
    Quick check — does this message sound like the user brushing off support?
    Used to decide whether to push back vs advance the protocol.
    """
    import re
    patterns = re.compile(
        r"\b(i('m| am) (fine|ok|okay|good|alright)|it'?s (fine|okay|ok|alright)|"
        r"don'?t worry|i('ll| will) (just|be)|never ?mind|forget it|"
        r"i got it|i can handle|i('ll| will) manage|just (gonna|going to) (lie|sit|stay)|"
        r"it doesn'?t matter|doesn'?t matter)\b",
        re.IGNORECASE,
    )
    return bool(patterns.search(message))


def is_protocol_close(message: str) -> bool:
    """
    Does this message clearly indicate the person is done and okay?
    More definitive than deflection — "thanks, I feel better", "I'm good now", etc.
    """
    import re
    patterns = re.compile(
        r"\b(feel(ing)? better|all better|i'?m good now|that helped|"
        r"thank(s| you).{0,20}(helped|better|good)|"
        r"i'?m (okay|ok|alright) now|good now|much better|"
        r"can (close|stop|end) (the |this )?(protocol|check.?in)|"
        r"you can stop|we('re| are) (done|good)|that'?s enough)\b",
        re.IGNORECASE,
    )
    return bool(patterns.search(message))
