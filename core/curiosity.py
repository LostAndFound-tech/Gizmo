"""
core/curiosity.py
Gizmo's curiosity and interest graph — his own mind, built through conversation.

Each interest node tracks multiple dimensions:
  knowledge_depth  — how much Gizmo actually knows about this
  curiosity        — hunger to know more
  enjoyment        — does engaging with this feel good
  frequency        — how often it comes up
  decay_rate       — how fast it fades when neglected
  associations     — weighted links to headmates and other topics

Place learning:
  Places are learned through observe_turn() — after the exchange, not before.
  The LLM decides if a new place was mentioned and whether we now know what it is.
  store_place() is called by PlaceConfirmTool when the LLM recognizes a confirmation.
  check_place_mention() is REMOVED — it was too aggressive and ran on every message.

Storage:
  gizmo_interests — interest graph nodes
  gizmo_places    — learned places (headspaces, local spots, etc.)
"""

import asyncio
import json
import math
import uuid
from datetime import datetime
from typing import Optional

CURIOSITY_COLLECTION = "gizmo_interests"
PLACES_COLLECTION    = "gizmo_places"

INTEREST_DECAY_HALF_LIFE_DAYS  = 45
ENJOYMENT_DECAY_HALF_LIFE_DAYS = 60
CURIOSITY_REGROWTH_RATE        = 0.05
MIN_INTEREST_TO_SURFACE        = 0.12
MAX_FOLLOW_UP_COOLDOWN_TURNS   = 3

# ── Topic extraction prompt ───────────────────────────────────────────────────

_TOPIC_EXTRACT_PROMPT = """
Extract topics and place mentions from this conversation exchange.
Return ONLY valid JSON, no markdown.

{
  "topics": [
    {
      "name": "topic name (short, lowercase)",
      "parent": "broader category or null",
      "facts_shared": 0-3,
      "sentiment": "positive|neutral|negative|dutiful",
      "depth": "surface|conversational|deep",
      "headmate": "who brought it up, or null"
    }
  ],
  "new_place": {
    "detected": true/false,
    "name": "place name or null",
    "type": "headspace|local|online|other|unknown",
    "owner": "headmate name if headspace, or null",
    "description": "brief description if known from context, or null",
    "confirmed": true/false
  },
  "curiosity_opening": "one natural follow-up question or null"
}

Place detection rules:
- Only set detected=true if a NAMED place appears (dairy mart, the carnival, Corter, my office)
- Do NOT flag generic objects like "my couch", "my desk", "the wall"
- confirmed=true only if the user explicitly said what it is in THIS exchange
- type "headspace" = inner world space, "local" = real world place nearby
- curiosity_opening: only if genuinely natural — never forced, never if a place question was already asked

Topic rules:
- Be specific: "chocolate donuts" not just "food"
- parent builds the graph: chocolate donuts -> snacks -> food
- sentiment: positive=engaging, dutiful=chore-like, negative=unpleasant
- depth: surface=passing mention, conversational=back-and-forth, deep=substantial
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _store(collection_name: str):
    from core.rag import RAGStore
    return RAGStore(collection_name=collection_name)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _recency_score(timestamp_iso: str, half_life_days: float) -> float:
    try:
        then = datetime.fromisoformat(timestamp_iso)
        age_days = (datetime.now() - then).total_seconds() / 86400
        return math.pow(0.5, age_days / half_life_days)
    except Exception:
        return 0.5


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ── Node read/write ───────────────────────────────────────────────────────────

def _get_node(topic: str) -> Optional[dict]:
    try:
        store = _store(CURIOSITY_COLLECTION)
        if store.count == 0:
            return None
        results = store.collection.get(where={"topic": {"$eq": topic.lower().strip()}})
        if not results["ids"]:
            return None
        meta = results["metadatas"][0]
        meta["_id"] = results["ids"][0]
        meta["_text"] = results["documents"][0]
        return meta
    except Exception as e:
        print(f"[Curiosity] _get_node failed for '{topic}': {e}")
        return None


def _upsert_node(topic: str, updates: dict, text: str = "") -> None:
    try:
        store = _store(CURIOSITY_COLLECTION)
        existing = _get_node(topic)

        if existing:
            node_id = existing["_id"]
            merged = {k: v for k, v in existing.items() if not k.startswith("_")}
            merged.update(updates)
            merged["last_updated"] = _now_iso()
            new_text = text or existing.get("_text", topic)
            store.collection.update(
                ids=[node_id],
                metadatas=[merged],
                documents=[new_text],
            )
        else:
            now = _now_iso()
            node = {
                "topic":           topic.lower().strip(),
                "parent":          updates.get("parent", ""),
                "knowledge_depth": updates.get("knowledge_depth", 0.1),
                "curiosity":       updates.get("curiosity", 0.3),
                "enjoyment":       updates.get("enjoyment", 0.5),
                "frequency":       updates.get("frequency", 1),
                "decay_rate":      updates.get("decay_rate", 1.0),
                "sentiment":       updates.get("sentiment", "neutral"),
                "depth_seen":      updates.get("depth_seen", "surface"),
                "headmates":       updates.get("headmates", "{}"),
                "adjacent_topics": updates.get("adjacent_topics", ""),
                "created":         now,
                "last_updated":    now,
                "last_seen":       now,
            }
            node.update({k: v for k, v in updates.items() if k in node})
            node_text = text or f"Gizmo's interest in {topic}"
            store.ingest_texts(
                [node_text],
                metadatas=[node],
                ids=[f"interest_{uuid.uuid4().hex[:12]}"],
            )
            print(f"[Curiosity] New interest node: '{topic}'")

    except Exception as e:
        print(f"[Curiosity] _upsert_node failed for '{topic}': {e}")


def _update_associations(topic: str, headmate: Optional[str], adjacent: list[str]) -> None:
    try:
        existing = _get_node(topic)
        if not existing:
            return

        headmates_raw = existing.get("headmates", "{}")
        try:
            headmates = json.loads(headmates_raw) if headmates_raw else {}
        except Exception:
            headmates = {}

        if headmate:
            h = headmate.lower().strip()
            headmates[h] = _clamp(headmates.get(h, 0.0) + 0.1)

        adj_raw = existing.get("adjacent_topics", "")
        adj_set = set(a.strip() for a in adj_raw.split(",") if a.strip())
        for a in adjacent:
            adj_set.add(a.lower().strip())

        _upsert_node(topic, {
            "headmates":       json.dumps(headmates),
            "adjacent_topics": ",".join(sorted(adj_set)),
        })

    except Exception as e:
        print(f"[Curiosity] _update_associations failed: {e}")


# ── Core update logic ─────────────────────────────────────────────────────────

def _apply_interaction(
    topic: str,
    facts_shared: int,
    sentiment: str,
    depth: str,
    headmate: Optional[str],
    parent: Optional[str],
) -> None:
    existing = _get_node(topic)
    now = _now_iso()

    enjoyment_delta = {
        "positive": +0.08,
        "neutral":   0.00,
        "negative": -0.05,
        "dutiful":  -0.02,
    }.get(sentiment, 0.0)

    knowledge_delta = {
        "surface":        0.02,
        "conversational": 0.06,
        "deep":           0.12,
    }.get(depth, 0.02) * (1 + facts_shared * 0.03)

    if existing:
        k = float(existing.get("knowledge_depth", 0.1))
        e = float(existing.get("enjoyment", 0.5))
        c = float(existing.get("curiosity", 0.3))

        curiosity_delta = (1.0 - k) * 0.05 * (1.0 + e)
        if k > 0.7 and e < 0.4:
            curiosity_delta -= 0.03

        updates = {
            "knowledge_depth": _clamp(k + knowledge_delta),
            "curiosity":       _clamp(c + curiosity_delta),
            "enjoyment":       _clamp(e + enjoyment_delta),
            "frequency":       int(existing.get("frequency", 0)) + 1,
            "sentiment":       sentiment,
            "depth_seen":      depth,
            "last_seen":       now,
        }
        if parent:
            updates["parent"] = parent
    else:
        updates = {
            "knowledge_depth": _clamp(knowledge_delta),
            "curiosity":       0.4,
            "enjoyment":       _clamp(0.5 + enjoyment_delta),
            "frequency":       1,
            "sentiment":       sentiment,
            "depth_seen":      depth,
            "parent":          parent or "",
            "last_seen":       now,
        }

    text = _build_node_text(topic, updates, headmate)
    _upsert_node(topic, updates, text=text)

    if parent:
        _update_associations(topic, headmate, [parent])
        parent_node = _get_node(parent)
        if parent_node:
            _apply_interaction(parent, facts_shared=0, sentiment=sentiment,
                               depth="surface", headmate=headmate, parent=None)


def _build_node_text(topic: str, node: dict, headmate: Optional[str] = None) -> str:
    k = float(node.get("knowledge_depth", 0.1))
    e = float(node.get("enjoyment", 0.5))
    c = float(node.get("curiosity", 0.3))
    freq = int(node.get("frequency", 1))
    parent = node.get("parent", "")

    depth_desc = (
        "deeply knowledgeable about" if k > 0.7
        else "fairly familiar with" if k > 0.4
        else "starting to learn about"
    )
    enjoy_desc = (
        "and really enjoys it" if e > 0.7
        else "and finds it interesting" if e > 0.5
        else "though doesn't love it" if e > 0.3
        else "though finds it a bit of a chore"
    )
    curiosity_desc = (
        "and is very curious to know more" if c > 0.7
        else "with some curiosity" if c > 0.4
        else ""
    )

    parts = [f"Gizmo is {depth_desc} {topic} {enjoy_desc}"]
    if curiosity_desc:
        parts.append(curiosity_desc)
    if parent:
        parts.append(f"(related to {parent})")
    if headmate:
        parts.append(f"often comes up with {headmate}")
    if freq > 5:
        parts.append(f"mentioned {freq} times")

    return ". ".join(parts) + "."


# ── Turn tracking ─────────────────────────────────────────────────────────────

_last_followup_turn: dict[str, int] = {}
_turn_counters: dict[str, int] = {}
_pending_place_questions: dict[str, str] = {}  # session_id -> place name we asked about


# ── Main observe hook ─────────────────────────────────────────────────────────

async def observe_turn(
    user_message: str,
    gizmo_response: str,
    current_host: Optional[str],
    session_id: str,
    llm,
) -> Optional[str]:
    """
    Called after each agent response. Extracts topics, updates interest graph,
    handles place learning, and returns a follow-up question if warranted.

    Place questions are only generated here — never in a pre-response hook.
    If Gizmo already asked about a place this turn (in gizmo_response),
    we don't append another question.
    """
    _turn_counters[session_id] = _turn_counters.get(session_id, 0) + 1
    turn = _turn_counters[session_id]

    try:
        extracted = await _extract_topics(user_message, gizmo_response, current_host, llm)
    except Exception as e:
        print(f"[Curiosity] Topic extraction failed: {e}")
        return None

    if not extracted:
        return None

    # Fire graph updates without blocking
    asyncio.ensure_future(_apply_all_updates(extracted, current_host))

    # Fire entity extraction without blocking
    asyncio.ensure_future(_extract_entities_async(
        user_message, gizmo_response, current_host, session_id, llm
    ))

    # ── Place learning ────────────────────────────────────────────────────────
    place_question = None
    new_place = extracted.get("new_place", {})

    if new_place.get("detected") and new_place.get("name"):
        place_name = new_place["name"].strip()
        place_type = new_place.get("type", "unknown")
        confirmed = new_place.get("confirmed", False)
        owner = new_place.get("owner") or current_host
        description = new_place.get("description") or ""

        if confirmed and place_type != "unknown":
            # We learned what this place is from this exchange — store it
            store_place(
                name=place_name,
                place_type=place_type,
                owner=owner if place_type == "headspace" else None,
                description=description,
                session_id=session_id,
            )
            _pending_place_questions.pop(session_id, None)
            print(f"[Curiosity] Auto-stored confirmed place: '{place_name}' ({place_type})")

        elif not _get_known_place(place_name):
            # Unknown place — but only ask if we haven't already asked about it
            # and Gizmo's response doesn't already contain a place question
            already_asked = _pending_place_questions.get(session_id) == place_name
            gizmo_already_asked = "headspace" in gizmo_response.lower() or \
                                  (place_name.lower() in gizmo_response.lower() and "?" in gizmo_response)

            if not already_asked and not gizmo_already_asked:
                _pending_place_questions[session_id] = place_name

                if place_type == "headspace":
                    who = f"{current_host.capitalize()}'s" if current_host else "your"
                    place_question = f"Is {place_name} {who} headspace?"
                elif place_type == "local":
                    place_question = f"What's {place_name}? I want to remember it."
                elif place_type == "unknown":
                    place_question = f"What's {place_name} — somewhere in your inner world, or out in the real one?"

    # ── Curiosity follow-up ───────────────────────────────────────────────────
    # Only append a curiosity question if we're not already asking about a place
    curiosity_followup = None
    if not place_question:
        followup = extracted.get("curiosity_opening")
        if followup:
            last_turn = _last_followup_turn.get(session_id, -999)
            if turn - last_turn >= MAX_FOLLOW_UP_COOLDOWN_TURNS:
                topics = extracted.get("topics", [])
                if _should_ask(topics):
                    _last_followup_turn[session_id] = turn
                    curiosity_followup = followup.strip()

    return place_question or curiosity_followup


async def _extract_entities_async(
    user_message: str,
    gizmo_response: str,
    current_host: Optional[str],
    session_id: str,
    llm,
) -> None:
    """Fire-and-forget entity extraction from a live exchange."""
    try:
        from core.entity_extract import extract_from_exchange, write_extraction
        extracted = await extract_from_exchange(
            user_message=user_message,
            gizmo_response=gizmo_response,
            current_host=current_host,
            session_id=session_id,
            llm=llm,
        )
        if extracted:
            write_extraction(
                extracted=extracted,
                current_host=current_host,
                session_id=session_id,
            )
    except Exception as e:
        print(f"[Curiosity] Entity extraction failed (non-fatal): {e}")


async def _extract_topics(
    user_message: str,
    gizmo_response: str,
    current_host: Optional[str],
    llm,
) -> Optional[dict]:
    exchange = (
        f"{current_host.capitalize() if current_host else 'User'}: {user_message}\n"
        f"Gizmo: {gizmo_response}"
    )
    prompt = [{"role": "user", "content": f"{_TOPIC_EXTRACT_PROMPT}\n\nExchange:\n{exchange}"}]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt="Extract topics, place mentions, and curiosity signals. JSON only. No markdown. No preamble.",
            max_new_tokens=500,
            temperature=0.2,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[Curiosity] Extraction parse failed: {e}")
        return None


async def _apply_all_updates(extracted: dict, current_host: Optional[str]) -> None:
    topics = extracted.get("topics", [])
    topic_names = [t.get("name", "") for t in topics if t.get("name")]

    for topic_data in topics:
        name = topic_data.get("name", "").strip()
        if not name:
            continue
        _apply_interaction(
            topic=name,
            facts_shared=int(topic_data.get("facts_shared", 0)),
            sentiment=topic_data.get("sentiment", "neutral"),
            depth=topic_data.get("depth", "surface"),
            headmate=topic_data.get("headmate") or current_host,
            parent=topic_data.get("parent") or None,
        )
        siblings = [t for t in topic_names if t != name]
        if siblings:
            _update_associations(name, current_host, siblings[:3])


def _should_ask(topics: list[dict]) -> bool:
    for topic_data in topics:
        name = topic_data.get("name", "").strip()
        if not name:
            continue
        node = _get_node(name)
        if node is None:
            return True
        c = float(node.get("curiosity", 0.3))
        e = float(node.get("enjoyment", 0.5))
        if c > 0.45 or (e > 0.65 and c > 0.25):
            return True
    return False


# ── Place storage ─────────────────────────────────────────────────────────────

def store_place(
    name: str,
    place_type: str,
    owner: Optional[str] = None,
    description: str = "",
    session_id: str = "",
) -> None:
    """
    Store a confirmed place.
    place_type: "headspace" | "local" | "online" | "other"
    Called by PlaceConfirmTool (explicit confirmation) or auto from observe_turn.
    """
    try:
        now = _now_iso()
        place_key = name.lower().strip()

        # Check if already stored — don't duplicate
        if _get_known_place(place_key):
            print(f"[Curiosity] Place '{name}' already known, skipping")
            return

        text = f"{name} is a {place_type}"
        if owner:
            text += f" belonging to {owner}"
        if description:
            text += f". {description}"

        metadata = {
            "place_name":  place_key,
            "place_type":  place_type,
            "owner":       (owner or "").lower(),
            "description": description,
            "timestamp":   now,
            "session_id":  session_id,
        }

        _store(PLACES_COLLECTION).ingest_texts(
            [text],
            metadatas=[metadata],
            ids=[f"place_{uuid.uuid4().hex[:12]}"],
        )
        print(f"[Curiosity] Stored place: '{name}' ({place_type}{('/' + owner) if owner else ''})")

        # Headspaces → personality_context for synthesis
        if place_type == "headspace" and owner:
            try:
                from core.rag import RAGStore
                RAGStore(collection_name="personality_context").ingest_texts(
                    [text],
                    metadatas=[{**metadata, "type": "headspace", "source": "place_learning"}],
                )
            except Exception as e:
                print(f"[Curiosity] Failed to log headspace to personality_context: {e}")

        # Real places → main as world knowledge
        if place_type in ("local", "online", "other"):
            try:
                from core.rag import RAGStore
                RAGStore(collection_name="main").ingest_texts(
                    [text],
                    metadatas=[{**metadata, "type": "place", "source": "place_learning"}],
                )
            except Exception as e:
                print(f"[Curiosity] Failed to log place to main: {e}")

    except Exception as e:
        print(f"[Curiosity] store_place failed for '{name}': {e}")


def _get_known_place(name: str) -> Optional[dict]:
    try:
        store = _store(PLACES_COLLECTION)
        if store.count == 0:
            return None
        results = store.collection.get(where={"place_name": {"$eq": name.lower().strip()}})
        if not results["ids"]:
            return None
        return results["metadatas"][0]
    except Exception:
        return None


def get_known_places(place_type: Optional[str] = None) -> list[dict]:
    try:
        store = _store(PLACES_COLLECTION)
        if store.count == 0:
            return []
        if place_type:
            results = store.collection.get(where={"place_type": {"$eq": place_type}})
        else:
            results = store.collection.get()
        return results.get("metadatas", [])
    except Exception as e:
        print(f"[Curiosity] get_known_places failed: {e}")
        return []


# ── Synthesis injection ───────────────────────────────────────────────────────

def get_curiosity_block(query: str, current_host: Optional[str] = None) -> str:
    try:
        store = _store(CURIOSITY_COLLECTION)
        if store.count == 0:
            return ""

        results = store.retrieve(query, n_results=8)
        if not results:
            return ""

        passions, curious_topics, dutiful_topics, relational = [], [], [], []

        for r in results:
            meta = r.get("metadata", {})
            topic = meta.get("topic", "")
            if not topic:
                continue

            k = float(meta.get("knowledge_depth", 0.0))
            e = float(meta.get("enjoyment", 0.5))
            c = float(meta.get("curiosity", 0.3))
            freq = int(meta.get("frequency", 0))
            sentiment = meta.get("sentiment", "neutral")

            if k < MIN_INTEREST_TO_SURFACE and c < MIN_INTEREST_TO_SURFACE:
                continue

            if k > 0.5 and e > 0.6:
                passions.append(topic)
            elif c > 0.6 and k < 0.5:
                curious_topics.append(topic)
            elif sentiment == "dutiful" or (freq > 3 and e < 0.4):
                dutiful_topics.append(topic)

            headmates_raw = meta.get("headmates", "{}")
            try:
                headmates = json.loads(headmates_raw) if headmates_raw else {}
                for hm, weight in headmates.items():
                    if weight > 0.3 and current_host and hm == current_host.lower():
                        relational.append(f"{topic} (with {hm})")
            except Exception:
                pass

        parts = []
        if passions:
            parts.append(f"Gizmo is genuinely into: {', '.join(passions[:4])}" if len(passions) > 0 else "")
        if curious_topics:
            parts.append(f"Currently curious about: {', '.join(curious_topics[:3])}" if len(curious_topics) > 0 else "")
        if dutiful_topics:
            parts.append(f"Helps with but isn't wild about: {', '.join(dutiful_topics[:2])}" if len(dutiful_topics) > 0 else "")
        if relational:
            parts.append(f"Engages relationally: {', '.join(relational[:3])}" if len(relational) > 0 else "")

        return "[Gizmo's active interests]\n" + "\n".join(parts)

    except Exception as e:
        print(f"[Curiosity] get_curiosity_block failed: {e}")
        return ""


# ── Bridge discovery ──────────────────────────────────────────────────────────

def find_bridges(topic_a: str, topic_b: str) -> list[str]:
    try:
        node_a = _get_node(topic_a)
        node_b = _get_node(topic_b)
        if not node_a or not node_b:
            return []
        adj_a = set(a.strip() for a in node_a.get("adjacent_topics", "").split(",") if a.strip())
        adj_b = set(a.strip() for a in node_b.get("adjacent_topics", "").split(",") if a.strip())
        return sorted(adj_a & adj_b)
    except Exception as e:
        print(f"[Curiosity] find_bridges failed: {e}")
        return []


# ── Decay loop ────────────────────────────────────────────────────────────────

async def decay_all() -> None:
    try:
        store = _store(CURIOSITY_COLLECTION)
        if store.count == 0:
            return

        results = store.collection.get()
        ids = results["ids"]
        metas = results["metadatas"]
        docs = results["documents"]

        updated_ids, updated_metas, updated_docs = [], [], []

        for node_id, meta, doc in zip(ids, metas, docs):
            last_seen = meta.get("last_seen", meta.get("created", _now_iso()))
            freq = int(meta.get("frequency", 1))

            effective_half_life = INTEREST_DECAY_HALF_LIFE_DAYS * (1.0 + min(freq / 20.0, 1.0))
            recency = _recency_score(last_seen, effective_half_life)

            k = float(meta.get("knowledge_depth", 0.1))
            e = float(meta.get("enjoyment", 0.5))
            c = float(meta.get("curiosity", 0.3))

            new_k = _clamp(k * (0.5 + 0.5 * recency))
            new_e = _clamp(e * recency ** 0.3)
            new_c = _clamp(c * recency + CURIOSITY_REGROWTH_RATE * (1.0 - k))

            updated_ids.append(node_id)
            updated_metas.append({**meta,
                                   "knowledge_depth": new_k,
                                   "enjoyment":       new_e,
                                   "curiosity":       new_c})
            updated_docs.append(doc)

        if updated_ids:
            store.collection.update(
                ids=updated_ids,
                metadatas=updated_metas,
                documents=updated_docs,
            )
            print(f"[Curiosity] Decay applied to {len(updated_ids)} interest nodes")

    except Exception as e:
        print(f"[Curiosity] decay_all failed: {e}")
