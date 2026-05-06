"""
core/observer.py
Background fact extractor. Fills in headmate, external, and pet files
from natural conversation — no explicit commands needed.

Triggered by Archivist.receive_outgoing() after each exchange.
Runs async, never blocks the response pipeline.

What it does:
  - Looks at the last user message + Gizmo's response
  - Identifies any active fronters from the session context
  - Asks the LLM: "what facts worth keeping were shared here?"
  - Writes those facts to the appropriate entity files

What counts as a fact worth keeping:
  - Personal attributes (age, species, occupation, relationship)
  - Preferences (likes, dislikes, recurring themes)
  - Important events or moments
  - Corrections to previous understanding
  - Anything Gizmo should remember next session

What doesn't count:
  - Chitchat with no lasting content ("lol", "that's funny")
  - Things already on file
  - Gizmo's own responses (those aren't facts about the person)

The extraction prompt is tight and cheap — small output, low temperature,
structured JSON. One call per exchange when there are active fronters.
Skips entirely if the exchange has no informational content.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_BASE_DIR        = Path(__file__).parent.parent
_PERSONALITY_DIR = _BASE_DIR / "personality"
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_EXTERNAL_DIR    = _PERSONALITY_DIR / "external"
_PETS_DIR        = _PERSONALITY_DIR / "pets"


# ── Minimum content threshold ─────────────────────────────────────────────────
# Don't bother extracting from very short or content-free exchanges

_MIN_WORDS = 8  # user message must have at least this many words
_SKIP_PATTERNS = re.compile(
    r"^(lol|haha|ok|okay|yep|yea|yeah|nope|nah|sure|cool|nice|"
    r"thanks|thank you|got it|sounds good|awesome|great|perfect|"
    r"wow|omg|oh|ah|hm|hmm|ikr|same|mood|fr|lmao|😂|😊|❤|💙)\W*$",
    re.IGNORECASE,
)


def _should_extract(user_message: str) -> bool:
    """Quick gate — skip extraction for content-free messages."""
    if len(user_message.split()) < _MIN_WORDS:
        if _SKIP_PATTERNS.match(user_message.strip()):
            return False
    return True


# ── File loaders / savers ─────────────────────────────────────────────────────

def _load_entity(name: str, entity_type: str) -> Optional[dict]:
    dirs = {
        "headmate": _HEADMATES_DIR,
        "external": _EXTERNAL_DIR,
        "pet":      _PETS_DIR,
    }
    d = dirs.get(entity_type)
    if not d:
        return None
    path = d / f"{name.lower()}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("Observer", f"failed to load {entity_type} file for {name}", exc=e)
        return None


def _save_entity(name: str, entity_type: str, data: dict) -> None:
    dirs = {
        "headmate": _HEADMATES_DIR,
        "external": _EXTERNAL_DIR,
        "pet":      _PETS_DIR,
    }
    d = dirs.get(entity_type)
    if not d:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name.lower()}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"#hds
        )
    except Exception as e:
        log_error("Observer", f"failed to save {entity_type} file for {name}", exc=e)


def _get_entity_type(name: str) -> Optional[str]:
    """Determine which directory a name lives in."""
    if (_HEADMATES_DIR / f"{name.lower()}.json").exists():
        return "headmate"
    if (_EXTERNAL_DIR / f"{name.lower()}.json").exists():
        return "external"
    if (_PETS_DIR / f"{name.lower()}.json").exists():
        return "pet"
    return None


def _existing_facts(data: dict, entity_type: str) -> set:
    """Collect all existing recorded facts to avoid duplicates."""
    facts = set()
    if entity_type == "headmate":
        for p in data.get("observed_patterns", []):
            if isinstance(p, dict):
                facts.add(p.get("pattern", "").lower())
            else:
                facts.add(str(p).lower())
        for m in data.get("moments_of_note", []):
            facts.add(str(m).lower()[:80])
        baseline = data.get("baseline", {})
        for k, v in baseline.items():
            if v and v != "unknown":
                facts.add(f"{k}:{v}".lower())
    else:
        for f in data.get("observed_facts", []):
            facts.add(str(f).lower()[:80])
        for m in data.get("moments_of_note", []):
            facts.add(str(m).lower()[:80])
    return facts


# ── LLM extraction ────────────────────────────────────────────────────────────

async def _extract_facts(
    user_message: str,
    gizmo_response: str,
    fronters: list[str],
    llm,
) -> dict:
    """
    Ask the LLM what facts worth keeping were shared in this exchange.
    Returns a dict: {name_lower: [fact_str, ...]}

    Only extracts facts *about* the fronters — not about Gizmo,
    not about abstract topics.
    """
    if not fronters:
        return {}

    fronter_list = ", ".join(f.title() for f in fronters)

    prompt = [{
        "role": "user",
        "content": (
            f"The following is a conversation exchange. "
            f"Active people: {fronter_list}.\n\n"
            f"User said: \"{user_message}\"\n"
            f"Gizmo responded: \"{gizmo_response[:300]}\"\n\n"
            f"Extract any facts worth remembering about the active people. "
            f"Facts include: personal attributes, preferences, relationships, "
            f"important events, corrections, species, age, occupation, etc.\n\n"
            f"Return ONLY a JSON object mapping lowercase names to arrays of short fact strings. "
            f"Each fact should be one concise sentence. "
            f"If there are no meaningful facts, return an empty object.\n\n"
            f"Example: "
            f'{{\"jess\": [\"goes by Jess, previously called Princess\"], '
            f'\"oren\": [\"84 years old\", \"werewolf\", \"astrophysicist\"]}}\n\n'
            f"Only include names from this list (lowercase): "
            f"{', '.join(f.lower() for f in fronters)}\n"
            f"No preamble. No explanation. JSON only."
        )
    }]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You extract memorable facts from conversations and return them as JSON. "
                "Be concise. Only include facts explicitly stated. Never invent."
            ),
            max_new_tokens=300,
            temperature=0.1,
        )

        if not result or not result.strip():
            return {}

        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        parsed = json.loads(clean)

        # Validate — only keep known fronters, string facts
        fronter_set = {f.lower() for f in fronters}
        validated = {}
        for name, facts in parsed.items():
            if name.lower() in fronter_set and isinstance(facts, list):
                clean_facts = [f for f in facts if isinstance(f, str) and f.strip()]
                if clean_facts:
                    validated[name.lower()] = clean_facts

        return validated

    except Exception as e:
        log_error("Observer", "fact extraction failed", exc=e)
        return {}


# ── File writer ───────────────────────────────────────────────────────────────

def _write_facts(name: str, facts: list[str]) -> int:
    """
    Write extracted facts to the appropriate entity file.
    Returns number of new facts written (skips duplicates).
    """
    from core.timezone import tz_now
    timestamp = tz_now().strftime("%Y-%m-%d %H:%M %Z")

    entity_type = _get_entity_type(name)
    if not entity_type:
        log_event("Observer", "ENTITY_NOT_FOUND",
            name=name,
            note="no file exists — skipping",
        )
        return 0

    data = _load_entity(name, entity_type)
    if data is None:
        return 0

    existing = _existing_facts(data, entity_type)
    written = 0

    for fact in facts:
        fact = fact.strip()
        if not fact:
            continue

        # Skip if too similar to existing facts
        fact_lower = fact.lower()
        if any(
            fact_lower[:40] in ex or ex[:40] in fact_lower
            for ex in existing
        ):
            continue

        # Write to appropriate field
        if entity_type == "headmate":
            # Try to detect if this is a baseline attribute
            baseline_keys = {
                "age": ["years old", "age"],
                "register": ["usually", "tends to", "baseline"],
                "verbosity": ["verbose", "concise", "chatty", "quiet"],
            }
            written_to_baseline = False
            for key, indicators in baseline_keys.items():
                if any(ind in fact_lower for ind in indicators):
                    # Don't overwrite if already set
                    current = data.get("baseline", {}).get(key, "unknown")
                    if current == "unknown":
                        data.setdefault("baseline", {})[key] = fact
                        written_to_baseline = True
                        break

            if not written_to_baseline:
                data.setdefault("moments_of_note", []).append(
                    f"[{timestamp}] {fact}"
                )

            data.setdefault("baseline", {})
            data["baseline"]["observations"] = (
                data["baseline"].get("observations", 0) + 1
            )

        else:  # external or pet
            data.setdefault("observed_facts", []).append(
                f"[{timestamp}] {fact}"
            )

        existing.add(fact_lower)
        written += 1

    if written > 0:
        _save_entity(name, entity_type, data)
        log_event("Observer", "FACTS_WRITTEN",
            name=name,
            type=entity_type,
            count=written,
            facts=facts[:3],  # log first 3 for readability
        )

    return written


# ── Main entry point ──────────────────────────────────────────────────────────

async def observe(
    user_message: str,
    gizmo_response: str,
    fronters: list[str],
    session_id: str,
    llm,
) -> None:
    """
    Main entry point. Called by Archivist.receive_outgoing() as a background task.
    Never raises — all errors are logged and swallowed.

    fronters: list of active fronter names (from host_tracker context)
    """
    if not fronters:
        return

    if not _should_extract(user_message):
        log_event("Observer", "SKIPPED",
            session=session_id[:8],
            reason="content-free message",
        )
        return

    t_start = time.monotonic()

    try:
        facts = await _extract_facts(user_message, gizmo_response, fronters, llm)

        if not facts:
            log_event("Observer", "NO_FACTS",
                session=session_id[:8],
                fronters=fronters,
                duration_ms=round((time.monotonic() - t_start) * 1000),
            )
            return

        total_written = 0
        for name, name_facts in facts.items():
            written = _write_facts(name, name_facts)
            total_written += written

        log_event("Observer", "COMPLETE",
            session=session_id[:8],
            fronters=fronters,
            facts_extracted=sum(len(f) for f in facts.values()),
            facts_written=total_written,
            duration_ms=round((time.monotonic() - t_start) * 1000),
        )

    except Exception as e:
        log_error("Observer", "observation failed", exc=e)
