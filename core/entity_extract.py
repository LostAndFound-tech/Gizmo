"""
core/entity_extract.py
LLM-powered entity extraction from conversation exchanges.

Called by the archiver after each session window is summarized,
and by observe_turn for live extraction during conversation.

Extracts:
  - Entities (headmates, places, objects, concepts)
  - Attributes (facts about entities)
  - Relations (edges between entities, typed)
  - Memories (events with emotional weights and entity refs)
  - Terms (system-specific vocabulary)

All extracted items are written to entity_store.py (SQLite).
ChromaDB documents get entity_uuid tags added to metadata.
"""

import json
from datetime import datetime
from typing import Optional

_EXTRACT_PROMPT = """
You are reading a conversation exchange and extracting structured knowledge.
Return ONLY valid JSON, no markdown, no preamble.

Extract everything that IS A THING — people, places, objects, concepts, events, terms.
Be thorough. "My couch" is a thing. "The encanto" is a thing. "Antsas" is a term.
Possessives ("my X", "Honey's X") signal ownership — extract both the owner and the thing.

{
  "entities": [
    {
      "name": "entity name",
      "type": "headmate|place|object|concept|event|unknown",
      "subtype": "headspace|local|internal_object|possession|etc or null",
      "owner": "owner name if possessed, or null",
      "description": "brief description from context, or null",
      "is_new": true/false
    }
  ],
  "attributes": [
    {
      "entity_name": "which entity",
      "entity_type": "type of that entity",
      "key": "attribute name",
      "value": "attribute value"
    }
  ],
  "relations": [
    {
      "from_name": "entity name",
      "from_type": "entity type",
      "to_name": "entity name",
      "to_type": "entity type",
      "relation_type": "has_space|owns|antsas|member_of|located_in|associated_with|etc",
      "is_system_term": true/false,
      "notes": "brief context or null"
    }
  ],
  "memories": [
    {
      "owner_name": "headmate this memory belongs to",
      "description": "what happened or was said, specific and factual",
      "tags": ["list", "of", "topic", "tags"],
      "entities_involved": ["list of entity names involved"],
      "emotions": {
        "emotion_name": 0.0-1.0
      },
      "significance": 0.0-1.0
    }
  ],
  "terms": [
    {
      "term": "the word or phrase",
      "definition": "what it means in this system",
      "origin": "who coined it or null",
      "example": "usage example or null"
    }
  ]
}

Rules:
- significance: 0.1=passing mention, 0.5=normal, 0.7+=emotionally significant, 0.9+=profound
- emotions: open vocabulary — deduce from tone and content. Multiple emotions allowed.
  Example: "spoke adoringly about the couch while stressed about deadlines"
  → {"joy": 0.6, "love": 0.7, "stress": 0.4}
- relations: always extract ownership ("Princess" owns_headspace "office")
  Always extract headmate membership if implied ("Honey" is_headmate)
- is_system_term: true only if the term is specific to this plural system (like "antsas")
- If something is mentioned in passing with no new info, still extract the entity, just low significance
- "my X" = the current speaker owns X. Use current_host to resolve who "my" refers to.

Current host (who "my/I/me" refers to): {current_host}
Exchange:
{exchange}
"""


async def extract_from_exchange(
    user_message: str,
    gizmo_response: str,
    current_host: Optional[str],
    session_id: str,
    llm,
) -> Optional[dict]:
    """
    Extract all entities, relations, memories, and terms from one exchange.
    Returns the raw extracted dict, or None on failure.
    """
    exchange = (
        f"{current_host.capitalize() if current_host else 'User'}: {user_message}\n"
        f"Gizmo: {gizmo_response}"
    )
    prompt_text = _EXTRACT_PROMPT.replace(
        "{current_host}", current_host or "unknown"
    ).replace(
        "{exchange}", exchange
    )

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt_text}],
            system_prompt=(
                "Extract structured knowledge from conversation. "
                "JSON only. No markdown. No preamble. Be thorough — extract everything that is a thing."
            ),
            max_new_tokens=1000,
            temperature=0.2,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[EntityExtract] Extraction failed: {e}")
        return None


async def extract_from_window(
    messages: list[dict],
    fronters: set,
    session_id: str,
    llm,
) -> Optional[dict]:
    """
    Extract from a multi-message archiver window.
    Used by archiver.py after session summarization.
    """
    # Build transcript
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content']}"
        for m in messages
        if m["role"] in ("user", "assistant")
    )
    if not transcript.strip():
        return None

    # Use first fronter as current_host for "my" resolution
    current_host = next(iter(fronters), None) if fronters else None

    exchange_block = f"[Multi-turn window — fronters: {', '.join(fronters) if fronters else 'unknown'}]\n{transcript}"

    prompt_text = _EXTRACT_PROMPT.replace(
        "{current_host}", current_host or "unknown"
    ).replace(
        "{exchange}", exchange_block
    )

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt_text}],
            system_prompt=(
                "Extract structured knowledge from conversation. "
                "JSON only. No markdown. No preamble. Be thorough."
            ),
            max_new_tokens=1500,
            temperature=0.2,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[EntityExtract] Window extraction failed: {e}")
        return None


def write_extraction(
    extracted: dict,
    current_host: Optional[str],
    session_id: str,
) -> dict:
    """
    Write all extracted entities, attributes, relations, memories, and terms
    to the entity store. Returns a summary of what was written.

    This is synchronous — call from async context with asyncio.to_thread if needed.
    """
    from core.entity_store import (
        upsert_entity, set_attribute, add_relation, add_memory,
        upsert_term, get_entity, get_term,
    )

    summary = {
        "entities": 0,
        "attributes": 0,
        "relations": 0,
        "memories": 0,
        "terms": 0,
    }

    now = datetime.now().isoformat(timespec="seconds")

    # ── Terms first — relation types may reference them ───────────────────────
    term_uuid_map: dict[str, str] = {}  # term string → UUID
    for t in extracted.get("terms", []):
        term = t.get("term", "").strip()
        definition = t.get("definition", "").strip()
        if not term or not definition:
            continue
        try:
            term_uuid = upsert_term(
                term=term,
                definition=definition,
                origin=t.get("origin"),
                example=t.get("example"),
            )
            term_uuid_map[term.lower()] = term_uuid
            summary["terms"] += 1
        except Exception as e:
            print(f"[EntityExtract] Term write failed for '{term}': {e}")

    # ── Entities ──────────────────────────────────────────────────────────────
    entity_uuid_map: dict[str, str] = {}  # "name|type" → UUID

    for ent in extracted.get("entities", []):
        name = ent.get("name", "").strip()
        etype = ent.get("type", "unknown").strip()
        if not name:
            continue

        owner_name = ent.get("owner")
        owner_uuid = None
        if owner_name:
            owner_key = f"{owner_name}|headmate"
            owner_uuid = entity_uuid_map.get(owner_key)
            if not owner_uuid:
                existing = get_entity(owner_name, "headmate")
                owner_uuid = existing["uuid"] if existing else None

        try:
            eid = upsert_entity(
                name=name,
                entity_type=etype,
                owner_uuid=owner_uuid,
                subtype=ent.get("subtype"),
                notes=ent.get("description"),
            )
            entity_uuid_map[f"{name}|{etype}"] = eid
            # Also index by name alone for fuzzy lookups
            entity_uuid_map[name.lower()] = eid
            summary["entities"] += 1
        except Exception as e:
            print(f"[EntityExtract] Entity write failed for '{name}': {e}")

    # ── Attributes ────────────────────────────────────────────────────────────
    for attr in extracted.get("attributes", []):
        entity_name = attr.get("entity_name", "").strip()
        entity_type = attr.get("entity_type", "").strip()
        key = attr.get("key", "").strip()
        value = str(attr.get("value", "")).strip()
        if not entity_name or not key or not value:
            continue

        eid = (
            entity_uuid_map.get(f"{entity_name}|{entity_type}")
            or entity_uuid_map.get(entity_name.lower())
        )
        if not eid:
            existing = get_entity(entity_name)
            eid = existing["uuid"] if existing else None
        if not eid:
            print(f"[EntityExtract] Attribute: no entity found for '{entity_name}'")
            continue

        try:
            set_attribute(eid, key, value)
            summary["attributes"] += 1
        except Exception as e:
            print(f"[EntityExtract] Attribute write failed: {e}")

    # ── Relations ─────────────────────────────────────────────────────────────
    for rel in extracted.get("relations", []):
        from_name = rel.get("from_name", "").strip()
        from_type = rel.get("from_type", "unknown").strip()
        to_name = rel.get("to_name", "").strip()
        to_type = rel.get("to_type", "unknown").strip()
        relation_type = rel.get("relation_type", "associated_with").strip()

        if not from_name or not to_name:
            continue

        from_uuid = (
            entity_uuid_map.get(f"{from_name}|{from_type}")
            or entity_uuid_map.get(from_name.lower())
        )
        to_uuid = (
            entity_uuid_map.get(f"{to_name}|{to_type}")
            or entity_uuid_map.get(to_name.lower())
        )

        # Create entities if they don't exist yet
        if not from_uuid:
            from_uuid = upsert_entity(from_name, from_type)
            entity_uuid_map[from_name.lower()] = from_uuid
        if not to_uuid:
            to_uuid = upsert_entity(to_name, to_type)
            entity_uuid_map[to_name.lower()] = to_uuid

        # Check if this is a system term
        term_uuid = None
        if rel.get("is_system_term"):
            term_entry = get_term(relation_type)
            term_uuid = term_entry["uuid"] if term_entry else term_uuid_map.get(relation_type.lower())

        try:
            from core.entity_store import add_relation as _add_relation
            _add_relation(
                from_uuid=from_uuid,
                to_uuid=to_uuid,
                relation_type=relation_type,
                relation_term_uuid=term_uuid,
                notes=rel.get("notes"),
            )
            summary["relations"] += 1
        except Exception as e:
            print(f"[EntityExtract] Relation write failed: {e}")

    # ── Memories ──────────────────────────────────────────────────────────────
    for mem in extracted.get("memories", []):
        owner_name = mem.get("owner_name", current_host or "").strip()
        description = mem.get("description", "").strip()
        if not owner_name or not description:
            continue

        owner_uuid = entity_uuid_map.get(owner_name.lower())
        if not owner_uuid:
            existing = get_entity(owner_name, "headmate")
            if existing:
                owner_uuid = existing["uuid"]
            else:
                owner_uuid = upsert_entity(owner_name, "headmate")
                entity_uuid_map[owner_name.lower()] = owner_uuid

        # Resolve involved entity UUIDs
        involved_uuids = []
        for ename in mem.get("entities_involved", []):
            eid = entity_uuid_map.get(ename.lower())
            if not eid:
                existing = get_entity(ename)
                eid = existing["uuid"] if existing else None
            if eid:
                involved_uuids.append(eid)

        emotions = mem.get("emotions", {})
        if not emotions:
            emotions = {"neutral": 0.5}

        try:
            add_memory(
                owner_uuid=owner_uuid,
                description=description,
                tags=mem.get("tags", []),
                entity_uuids=involved_uuids,
                emotions=emotions,
                significance=float(mem.get("significance", 0.5)),
                session_id=session_id,
                occurred_at=now,
            )
            summary["memories"] += 1
        except Exception as e:
            print(f"[EntityExtract] Memory write failed: {e}")

    print(
        f"[EntityExtract] Wrote: {summary['entities']} entities, "
        f"{summary['attributes']} attributes, {summary['relations']} relations, "
        f"{summary['memories']} memories, {summary['terms']} terms"
    )
    return summary


# ── Convenience: build context block for synthesis ───────────────────────────

def build_entity_context(name: str, entity_type: str = "headmate") -> str:
    """
    Build a compact context block about an entity for injection into synthesis.
    Returns empty string if entity not found.
    """
    from core.entity_store import get_entity_profile

    profile = get_entity_profile(name, entity_type)
    if not profile:
        return ""

    entity = profile["entity"]
    attrs = profile["attributes"]
    relations = profile["relations"]
    memories = profile["memories"]

    lines = [f"[{entity['type'].capitalize()}: {entity['name']}]"]

    if attrs:
        for key, val in list(attrs.items())[:6]:
            lines.append(f"  {key}: {val}")

    if relations:
        for rel in relations[:5]:
            direction = "→" if rel["direction"] == "to" else "←"
            lines.append(f"  {rel['relation_type']} {direction} {rel['other_name']} ({rel['other_type']})")

    if memories:
        lines.append("  Recent:")
        for mem in memories[:3]:
            emotion_str = ", ".join(
                f"{e}:{w:.1f}" for e, w in sorted(
                    mem["emotions"].items(), key=lambda x: -x[1]
                )[:2]
            )
            lines.append(f"    [{', '.join(mem['tags'][:3])}] {mem['description'][:80]} ({emotion_str})")

    return "\n".join(lines)
