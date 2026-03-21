"""
core/entity_query.py
Entity mention detection and full profile injection.

When a user asks about something Gizmo has a record of, pull the complete
entity profile from the store and inject it into the system prompt as a
[Known facts] block. The LLM answers from the record, not from recall.

Flow:
  detect_entity_mentions() — scans user message for known entity names
  build_entity_block()     — pulls full profiles for all detected entities
  
The full profile includes:
  - All attributes (description, location, significance, etc.)
  - All relations (owned_by, located_in, associated_with, etc.)
  - Recent memories involving this entity
  - Owner profile if entity is a possession

Called from agent.py before building the system prompt.
The returned block is injected as [Known facts] — separate from RAG synthesis.

Grounding rule injected with the block:
  "Answer questions about these entities ONLY from what appears in [Known facts].
   If a specific detail isn't in [Known facts], say you don't have that detail yet
   rather than guessing."
"""

import re
from typing import Optional

# Patterns that suggest the user is asking about a specific thing
_QUERY_PATTERNS = re.compile(
    r"\b(tell me about|describe|what is|what's|what are|who is|who's|"
    r"remind me|do you remember|what do you know about|"
    r"what color|what size|what shape|where is|where's|"
    r"how big|what kind|what type|detail|details about|"
    r"my |your |their |his |her )\b",
    re.IGNORECASE,
)


def detect_entity_mentions(
    user_message: str,
    current_host: Optional[str] = None,
) -> list[dict]:
    """
    Scan user message for known entity names.
    Returns list of matching entity dicts from the store.
    Only runs if message looks like a question about something specific.
    """
    # Quick pre-filter — only bother if this looks like a factual query
    if not _QUERY_PATTERNS.search(user_message):
        return []

    try:
        from core.entity_store import get_all_entities
        all_entities = get_all_entities()
        if not all_entities:
            return []

        msg_lower = user_message.lower()
        matches = []
        seen_uuids = set()

        for entity in all_entities:
            name = entity.get("name", "").lower().strip()
            if not name or len(name) < 3:
                continue

            if name in msg_lower and entity["uuid"] not in seen_uuids:
                matches.append(entity)
                seen_uuids.add(entity["uuid"])

        # Also check possessives — "my couch" when current_host is Princess
        # means look for entities owned by Princess
        if current_host and ("my " in msg_lower or "mine" in msg_lower):
            for entity in all_entities:
                if entity["uuid"] in seen_uuids:
                    continue
                owner_uuid = entity.get("owner_uuid")
                if not owner_uuid:
                    continue
                # Check if owner is current host
                try:
                    from core.entity_store import get_entity_by_uuid
                    owner = get_entity_by_uuid(owner_uuid)
                    if owner and owner["name"].lower() == current_host.lower():
                        # Check if something in the message matches this entity
                        name = entity.get("name", "").lower()
                        if name and name in msg_lower:
                            matches.append(entity)
                            seen_uuids.add(entity["uuid"])
                except Exception:
                    pass

        if matches:
            print(f"[EntityQuery] Detected entities: {[e['name'] for e in matches]}")

        return matches

    except Exception as e:
        print(f"[EntityQuery] Detection failed: {e}")
        return []


def build_full_profile(entity: dict) -> str:
    """
    Build a complete, verbose profile for a single entity.
    This is the FULL record — nothing is capped or summarized.
    The LLM reads this to answer specific questions about the entity.
    """
    try:
        from core.entity_store import get_entity_profile, get_entity_by_uuid

        profile = get_entity_profile(entity["name"], entity["type"])
        if not profile:
            return ""

        ent = profile["entity"]
        attrs = profile["attributes"]
        relations = profile["relations"]
        memories = profile["memories"]

        lines = [f"[{ent['type'].capitalize()}: {ent['name']}]"]

        # Owner info
        if ent.get("owner_uuid"):
            try:
                owner = get_entity_by_uuid(ent["owner_uuid"])
                if owner:
                    lines.append(f"  Owner: {owner['name']} ({owner['type']})")
            except Exception:
                pass

        if ent.get("subtype"):
            lines.append(f"  Subtype: {ent['subtype']}")

        if ent.get("notes"):
            lines.append(f"  Notes: {ent['notes']}")

        # All attributes — no cap
        if attrs:
            lines.append("  Attributes:")
            for key, val in attrs.items():
                lines.append(f"    {key}: {val}")

        # All relations — no cap
        if relations:
            lines.append("  Relations:")
            for rel in relations:
                direction = "→" if rel["direction"] == "to" else "←"
                notes = f" ({rel['notes']})" if rel.get("notes") else ""
                lines.append(
                    f"    {rel['relation_type']} {direction} "
                    f"{rel['other_name']} ({rel['other_type']}){notes}"
                )

        # All recent memories — no cap
        if memories:
            lines.append("  Memories involving this entity:")
            for mem in memories:
                emotion_str = ", ".join(
                    f"{e}: {w:.1f}"
                    for e, w in sorted(mem["emotions"].items(), key=lambda x: -x[1])
                )
                tags_str = ", ".join(mem["tags"]) if mem["tags"] else "untagged"
                lines.append(f"    [{tags_str}] {mem['description']}")
                lines.append(f"      Emotions: {emotion_str} | Significance: {mem['significance']:.1f}")
                lines.append(f"      When: {mem['occurred_at'][:16]}")

        return "\n".join(lines)

    except Exception as e:
        print(f"[EntityQuery] Profile build failed for '{entity['name']}': {e}")
        return ""


def build_entity_block(
    user_message: str,
    current_host: Optional[str] = None,
) -> str:
    """
    Main entry point. Detects entity mentions, pulls full profiles,
    returns a [Known facts] block for injection into the system prompt.
    Returns empty string if no known entities found.
    """
    entities = detect_entity_mentions(user_message, current_host)
    if not entities:
        return ""

    profiles = []
    for entity in entities:
        profile_text = build_full_profile(entity)
        if profile_text:
            profiles.append(profile_text)

    if not profiles:
        return ""

    block = "[Known facts — answer from these records, not from memory]\n"
    block += "\n\n".join(profiles)
    block += (
        "\n\nIMPORTANT: If asked about a specific detail that does not appear above, "
        "say you don't have that detail yet rather than guessing or inventing it."
    )

    print(f"[EntityQuery] Injecting {len(profiles)} entity profile(s) into system prompt")
    return block
