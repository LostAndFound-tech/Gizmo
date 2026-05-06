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

import os

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
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
            json.dumps(data, indent=2), encoding="utf-8"
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

# ── Junk fact filter ──────────────────────────────────────────────────────────
# Facts that are context metadata, not real information worth keeping

_JUNK_PATTERNS = re.compile(
    r"\b(is (currently |now |the |a )?host(ing)?|is fronting|is co.?front|"
    r"is present|is (currently )?speaking|switched (in|out)|"
    r"has \d+ observation|has \d+ moment|observation count|"
    r"is active|is online|is (in|part of) (the |this )?session|"
    r"is (a |the )?headmate$|is (a |the )?system member$|"
    r"is (a |the )?(current |active )?fronter)\b",
    re.IGNORECASE,
)


def _is_junk_fact(fact: str) -> bool:
    """Return True if this fact is metadata noise, not real information."""
    if _JUNK_PATTERNS.search(fact):
        return True
    # Very short facts are usually noise
    if len(fact.split()) < 3:
        return True
    return False


async def _extract_facts(
    user_message: str,
    gizmo_response: str,
    fronters: list[str],
    llm,
) -> dict:
    """
    Ask the LLM what facts worth keeping were shared in this exchange.
    Returns a dict: {name_lower: [fact_str, ...]}

    Only extracts from user_message — Gizmo's response words aren't
    facts about the person.

    Extracts facts about ANYONE mentioned — active fronters AND
    third parties being talked about (e.g. "Kaylee is a preschool teacher"
    when Jess is speaking about Kaylee).

    Facts are attributed to the SUBJECT, not the speaker.
    """
    # Build full known entity list
    known_entities = set(f.lower() for f in fronters)
    try:
        if _HEADMATES_DIR.exists():
            known_entities.update(p.stem.lower() for p in _HEADMATES_DIR.glob("*.json")
                                  if p.stem != "count")
        if _EXTERNAL_DIR.exists():
            known_entities.update(p.stem.lower() for p in _EXTERNAL_DIR.glob("*.json"))
        if _PETS_DIR.exists():
            known_entities.update(p.stem.lower() for p in _PETS_DIR.glob("*.json"))
    except Exception:
        pass

    if not known_entities:
        return {}

    entity_list = ", ".join(sorted(known_entities))
    fronter_list = ", ".join(f.title() for f in fronters) if fronters else "unknown"

    prompt = [{
        "role": "user",
        "content": (
            f"Extract memorable facts from this message.\n"
            f"Speaker: {fronter_list}\n"
            f"Known people: {entity_list}\n\n"
            f"Message: \"{user_message}\"\n\n"
            f"Rules:\n"
            f"- Extract facts about the SUBJECT of each fact, not the speaker\n"
            f"  Example: 'Jess says Oren is a werewolf' → attribute to oren, not jess\n"
            f"- Only extract LASTING personal facts: species, age, occupation, "
            f"relationships, preferences, important personal details\n"
            f"- DO NOT extract: who is currently fronting/hosting, observation counts, "
            f"session metadata, or anything Gizmo said\n"
            f"- DO NOT extract facts that only describe the current moment\n"
            f"- Each fact should be a standalone sentence someone could read later\n\n"
            f"Return ONLY a JSON object: {{\"name\": [\"fact\", ...]}}\n"
            f"Only use names from: {entity_list}\n"
            f"If no lasting facts, return {{}}\n"
            f"No preamble. JSON only."
        )
    }]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You extract lasting personal facts from messages. "
                "Attribute facts to their subject, not the speaker. "
                "Never extract session metadata or who is currently fronting. "
                "JSON only."
            ),
            max_new_tokens=400,
            temperature=0.1,
        )

        if not result or not result.strip():
            return {}

        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        parsed = json.loads(clean)

        # Validate — keep any known entity, filter junk facts
        validated = {}
        for name, facts in parsed.items():
            if name.lower() in known_entities and isinstance(facts, list):
                clean_facts = [
                    f for f in facts
                    if isinstance(f, str) and f.strip() and not _is_junk_fact(f)
                ]
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
            # Only route to baseline if it's clearly a specific attribute
            # Age: must contain a number + "years old" or "year old"
            # Register/verbosity: only if it's genuinely about communication style
            written_to_baseline = False
            age_match = re.search(r'\b(\d+)\s+years?\s+old\b', fact_lower)
            if age_match:
                current = data.get("baseline", {}).get("age", "unknown")
                if current == "unknown":
                    data.setdefault("baseline", {})["age"] = fact
                    written_to_baseline = True

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
        # Clear stale cold-start note once real facts exist
        if data.get("note", "").startswith("Cold start"):
            data["note"] = f"Actively observed — {data['baseline'].get('observations', 0)} exchanges logged."
        _save_entity(name, entity_type, data)
        log_event("Observer", "FACTS_WRITTEN",
            name=name,
            type=entity_type,
            count=written,
            facts=facts[:3],
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