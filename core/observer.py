"""
core/observer.py
Background fact extractor. Fills in headmate, external, and pet files
from natural conversation — no explicit commands needed.

Triggered by Archivist.receive_outgoing() after each exchange.
Runs async, never blocks the response pipeline.

Extraction strategy: heuristic-first, LLM fallback.
  - Heuristic patterns catch 80%+ of facts instantly, zero LLM calls
  - LLM only fires if message has factual signal words AND heuristics found nothing
  - If LLM is down/slow, heuristics still work — memory keeps filling in

Heuristic patterns:
  "[Name] is [attribute]"           → fact about Name
  "[Name] is a/an [noun]"           → species/role/occupation
  "[Name] is [age] years old"       → age
  "[Name] works as/at [place]"      → occupation
  "[Name]'s [attribute] is [value]" → attribute
  "I'm [age] years old"             → speaker age
  "I work as/at [role]"             → speaker occupation
  "I'm a/an [noun]"                 → speaker species/role
  "I [like/love/hate] [thing]"      → speaker preference

Facts are attributed to their SUBJECT, not the speaker.
Only lasting personal facts are kept — no session metadata.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_EXTERNAL_DIR    = _PERSONALITY_DIR / "external"
_PETS_DIR        = _PERSONALITY_DIR / "pets"


# ── Content gate ──────────────────────────────────────────────────────────────

_SKIP_RE = re.compile(
    r"^(lol|haha|ok|okay|yep|yea|yeah|nope|nah|sure|cool|nice|"
    r"thanks|thank you|got it|sounds good|awesome|great|perfect|"
    r"wow|omg|oh|ah|hm+|ikr|same|mood|fr|lmao|😂|😊|❤|💙|"
    r"really\?|interesting|that'?s (cool|neat|awesome|great|wild)|"
    r"no way|what\?+|huh\?*)\W*$",
    re.IGNORECASE,
)

def _should_extract(message: str) -> bool:
    words = message.split()
    if len(words) < 4:
        return not bool(_SKIP_RE.match(message.strip()))
    return True


# ── Junk filter ───────────────────────────────────────────────────────────────

_JUNK_RE = re.compile(
    r"\b(is (currently |now |the |a )?host(ing)?|is fronting|is co.?front|"
    r"is present|is (currently )?speaking|switched (in|out)|"
    r"has \d+ observation|has \d+ moment|observation count|"
    r"is active|is online|is (in|part of) (the |this )?session|"
    r"is (a |the )?headmate$|is (a |the )?system member$|"
    r"is (a |the )?(current |active )?fronter)\b",
    re.IGNORECASE,
)

def _is_junk(fact: str) -> bool:
    if _JUNK_RE.search(fact):
        return True
    if len(fact.split()) < 3:
        return True
    return False


# ── Entity file helpers ───────────────────────────────────────────────────────

def _known_entities() -> set:
    entities = set()
    try:
        for d in [_HEADMATES_DIR, _EXTERNAL_DIR, _PETS_DIR]:
            if d.exists():
                entities.update(
                    p.stem.lower() for p in d.glob("*.json")
                    if p.stem not in ("count", "_schema")
                )
    except Exception:
        pass
    return entities


def _get_entity_type(name: str) -> Optional[str]:
    if (_HEADMATES_DIR / f"{name.lower()}.json").exists():
        return "headmate"
    if (_EXTERNAL_DIR / f"{name.lower()}.json").exists():
        return "external"
    if (_PETS_DIR / f"{name.lower()}.json").exists():
        return "pet"
    return None


def _load_entity(name: str, entity_type: str) -> Optional[dict]:
    dirs = {"headmate": _HEADMATES_DIR, "external": _EXTERNAL_DIR, "pet": _PETS_DIR}
    d = dirs.get(entity_type)
    if not d:
        return None
    try:
        return json.loads((d / f"{name.lower()}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("Observer", f"failed to load {entity_type} for {name}", exc=e)
        return None


def _save_entity(name: str, entity_type: str, data: dict) -> None:
    dirs = {"headmate": _HEADMATES_DIR, "external": _EXTERNAL_DIR, "pet": _PETS_DIR}
    d = dirs.get(entity_type)
    if not d:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name.lower()}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log_error("Observer", f"failed to save {entity_type} for {name}", exc=e)


def _existing_facts(data: dict, entity_type: str) -> set:
    facts = set()
    if entity_type == "headmate":
        for m in data.get("moments_of_note", []):
            facts.add(str(m).lower()[:80])
        baseline = data.get("baseline", {})
        for k, v in baseline.items():
            if v and v not in ("unknown", 0, 0.0):
                facts.add(f"{k}:{str(v)}".lower())
    else:
        for f in data.get("observed_facts", []):
            facts.add(str(f).lower()[:80])
        for m in data.get("moments_of_note", []):
            facts.add(str(m).lower()[:80])
    return facts


# ── Heuristic extraction ──────────────────────────────────────────────────────

# "[Name] is [something]" — third person
_IS_RE = re.compile(
    r'\b([A-Z][a-z]{1,20})\s+is\s+'
    r'(?!currently |now |still |just |also |the one)'
    r'((?:a |an |the )?[^.!?,\n]{4,80}?)(?:\.|,|!|\?|$|\n)',
    re.MULTILINE,
)

# "[Name]'s [attribute]"
_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][a-z]{1,20})'s\s+([a-z][^.!?,\n]{3,60})(?:\.|,|!|\?|$|\n)",
    re.MULTILINE,
)

# "[Name] works as/at/in/for [role]"
_WORK_RE = re.compile(
    r"\b([A-Z][a-z]{1,20})\s+(?:works?|worked)\s+(?:as|at|in|for)\s+"
    r"([^.!?,\n]{4,60})(?:\.|,|!|\?|$|\n)",
    re.IGNORECASE | re.MULTILINE,
)

# Speaker self-descriptions
_SELF_AGE_RE    = re.compile(r"\bI'?m\s+(\d+)\s+years?\s+old\b", re.IGNORECASE)
_SELF_WORK_RE   = re.compile(r"\bI\s+(?:work|worked)\s+(?:as|at|in|for)\s+([^.!?,\n]{4,60})(?:\.|,|$)", re.IGNORECASE)
_SELF_IS_RE     = re.compile(r"\bI'?m\s+(?:a|an)\s+([a-z][^.!?,\n]{3,50})(?:\.|,|!|\?|$)", re.IGNORECASE)
_SELF_PREF_RE   = re.compile(r"\bI\s+(like|love|hate|prefer|enjoy|adore|despise)\s+([^.!?,\n]{3,60})(?:\.|,|$)", re.IGNORECASE)
_SELF_FROM_RE   = re.compile(r"\bI'?(?:m from|come from|was born in|grew up in)\s+([^.!?,\n]{3,60})(?:\.|,|$)", re.IGNORECASE)

# Junk predicates — after "is" these aren't real facts
_JUNK_PRED_RE = re.compile(
    r"^(here|there|back|online|around|available|ready|gone|away|"
    r"fine|good|ok|okay|great|tired|busy|free|sure|right|wrong|that|this|"
    r"hosting|fronting|speaking|present|active|in (the )?front|"
    r"a headmate|an alter|part of|in the system)\b",
    re.IGNORECASE,
)

_NOT_NAMES = {
    "I", "Me", "My", "We", "Us", "You", "It", "He", "She", "They",
    "The", "A", "An", "And", "But", "Or", "So", "Ok", "Okay",
    "Hey", "Hi", "Hello", "Bye", "Yes", "No", "Not", "Now", "Just",
    "Gizmo", "Still", "Here", "There", "Back", "Out", "Also", "Like",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "Today", "Tomorrow", "Yesterday",
    "Baby", "Babe", "Gramma", "Grandma", "Grandpa", "Nana", "Papa",
    "Mama", "Momma", "Daddy", "Auntie", "Uncle", "Sis", "Bro",
    "Honey", "Sugar", "Sweetie", "Darling", "Love", "Hun",
    "Curious", "Excited", "Tired", "Happy", "Sad", "Good", "Fine",
    "Great", "Bad", "Bored", "Anxious", "Stressed", "Ready", "Sure",
    "Sorry", "Glad", "Worried", "Nervous", "Scared", "Lost", "Home",
    "Done", "Free", "Busy", "Late", "Early", "Hungry", "Sick", "Well",
    "Thinking", "Working", "Trying", "Going", "Coming", "Leaving",
    "Serious", "Kidding", "Joking", "Playing", "Watching", "Listening",
}


def _heuristic_extract(message: str, fronters: list[str], known: set) -> dict:
    """
    Extract facts using regex patterns. Zero LLM calls.
    Returns {name_lower: [fact_str, ...]}
    """
    facts: dict[str, list[str]] = {}
    speaker = fronters[0].lower() if fronters else None

    def add(name: str, fact: str) -> None:
        n = name.lower()
        if n not in known:
            return
        fact = fact.strip().rstrip(".,;!? ")
        if not fact or _is_junk(fact):
            return
        facts.setdefault(n, [])
        if fact.lower() not in [f.lower() for f in facts[n]]:
            facts[n].append(fact)

    # Third-person: "[Name] is [something]"
    for m in _IS_RE.finditer(message):
        name = m.group(1)
        predicate = m.group(2).strip().rstrip(".,;!? ")
        if name in _NOT_NAMES:
            continue
        if _JUNK_PRED_RE.match(predicate):
            continue
        if len(predicate.split()) >= 1:
            add(name, f"is {predicate}")

    # "[Name]'s [attribute]"
    for m in _POSSESSIVE_RE.finditer(message):
        name = m.group(1)
        attr = m.group(2).strip().rstrip(".,;!? ")
        if name in _NOT_NAMES:
            continue
        if len(attr.split()) >= 2:
            add(name, f"{name}'s {attr}")

    # "[Name] works as/at [role]"
    for m in _WORK_RE.finditer(message):
        name = m.group(1)
        role = m.group(2).strip().rstrip(".,;!? ")
        if name in _NOT_NAMES:
            continue
        add(name, f"works {role}")

    # Speaker self-descriptions
    if speaker and speaker in known:
        for m in _SELF_AGE_RE.finditer(message):
            add(speaker, f"is {m.group(1)} years old")

        for m in _SELF_WORK_RE.finditer(message):
            role = m.group(1).strip().rstrip(".,;!? ")
            add(speaker, f"works {role}")

        for m in _SELF_IS_RE.finditer(message):
            noun = m.group(1).strip().rstrip(".,;!? ")
            if not _JUNK_PRED_RE.match(noun):
                add(speaker, f"is a {noun}")

        for m in _SELF_PREF_RE.finditer(message):
            verb = m.group(1)
            thing = m.group(2).strip().rstrip(".,;!? ")
            add(speaker, f"{verb}s {thing}")

        for m in _SELF_FROM_RE.finditer(message):
            place = m.group(1).strip().rstrip(".,;!? ")
            add(speaker, f"is from {place}")

    return facts


# ── LLM fallback ──────────────────────────────────────────────────────────────

_FACTUAL_SIGNAL_RE = re.compile(
    r"\b(used to|originally|born|grew up|studied|graduated|married|"
    r"divorced|moved|lives? in|from|species|race|occupation|"
    r"profession|personality|ability|power|background|history|"
    r"experiment|military|alternate|dimension|universe|planet)\b",
    re.IGNORECASE,
)

def _worth_llm_fallback(message: str, heuristic_results: dict) -> bool:
    """Only call LLM if heuristics found nothing AND message has factual signals."""
    if heuristic_results:
        return False
    if len(message.split()) < 10:
        return False
    return bool(_FACTUAL_SIGNAL_RE.search(message))


async def _llm_extract(message: str, fronters: list[str], known: set, llm) -> dict:
    """LLM fallback. Only fires when heuristics miss real content."""
    entity_list = ", ".join(sorted(known))
    fronter_list = ", ".join(f.title() for f in fronters) if fronters else "unknown"

    prompt = [{
        "role": "user",
        "content": (
            f"Extract lasting personal facts from this message.\n"
            f"Speaker: {fronter_list} | Known people: {entity_list}\n\n"
            f"Message: \"{message}\"\n\n"
            f"- Attribute facts to their SUBJECT, not the speaker\n"
            f"- Only lasting facts: species, age, origin, occupation, background, abilities\n"
            f"- Skip: current fronting/hosting, session metadata\n"
            f"- One short sentence per fact\n\n"
            f"JSON only: {{\"name\": [\"fact\"]}} or {{}}\n"
            f"Names from: {entity_list}"
        )
    }]

    try:
        result = await llm.generate(
            prompt,
            system_prompt="Extract lasting personal facts as JSON. Attribute to subject not speaker. JSON only.",
            max_new_tokens=300,
            temperature=0.1,
        )

        if not result or not result.strip():
            return {}

        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        parsed = json.loads(clean)

        validated = {}
        for name, name_facts in parsed.items():
            if name.lower() in known and isinstance(name_facts, list):
                clean_facts = [
                    f for f in name_facts
                    if isinstance(f, str) and f.strip() and not _is_junk(f)
                ]
                if clean_facts:
                    validated[name.lower()] = clean_facts
        return validated

    except Exception as e:
        log_error("Observer", "LLM fallback failed", exc=e)
        return {}


# ── File writer ───────────────────────────────────────────────────────────────

def _write_facts(name: str, facts: list[str]) -> int:
    """Write extracted facts. Returns count written."""
    from core.timezone import tz_now
    timestamp = tz_now().strftime("%Y-%m-%d %H:%M %Z")

    entity_type = _get_entity_type(name)
    if not entity_type:
        return 0

    data = _load_entity(name, entity_type)
    if data is None:
        return 0

    existing = _existing_facts(data, entity_type)
    written = 0

    for fact in facts:
        fact = fact.strip()
        if not fact or _is_junk(fact):
            continue

        fact_lower = fact.lower()
        if any(fact_lower[:40] in ex or ex[:40] in fact_lower for ex in existing):
            continue

        if entity_type == "headmate":
            age_match = re.search(r'\b(\d+)\s+years?\s+old\b', fact_lower)
            if age_match:
                current = data.get("baseline", {}).get("age", "unknown")
                if current == "unknown":
                    data.setdefault("baseline", {})["age"] = fact
                    existing.add(fact_lower)
                    written += 1
                    continue

            data.setdefault("moments_of_note", []).append(f"[{timestamp}] {fact}")
            data.setdefault("baseline", {})
            data["baseline"]["observations"] = data["baseline"].get("observations", 0) + 1
        else:
            data.setdefault("observed_facts", []).append(f"[{timestamp}] {fact}")

        existing.add(fact_lower)
        written += 1

    if written > 0:
        if data.get("note", "").startswith("Cold start"):
            obs = data.get("baseline", {}).get("observations", written)
            data["note"] = f"Actively observed — {obs} exchanges logged."
        _save_entity(name, entity_type, data)
        log_event("Observer", "FACTS_WRITTEN",
            name=name,
            type=entity_type,
            count=written,
            facts=[f[:60] for f in facts[:3]],
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
    Main entry point. Called by Archivist.receive_outgoing().
    Never raises — errors logged and swallowed.
    """
    if not fronters:
        return

    if not _should_extract(user_message):
        return

    t_start = time.monotonic()

    try:
        known = _known_entities()
        if not known:
            return

        # Step 1: Heuristic extraction — always, zero cost
        facts = _heuristic_extract(user_message, fronters, known)
        source = "heuristic"

        # Step 2: LLM fallback — only if heuristics missed real content
        if _worth_llm_fallback(user_message, facts):
            llm_facts = await _llm_extract(user_message, fronters, known, llm)
            if llm_facts:
                for name, name_facts in llm_facts.items():
                    if name not in facts:
                        facts[name] = name_facts
                    else:
                        facts[name].extend(f for f in name_facts if f not in facts[name])
                source = "heuristic+llm"

        if not facts:
            log_event("Observer", "NO_FACTS",
                session=session_id[:8],
                fronters=fronters,
                source=source,
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
            source=source,
            facts_extracted=sum(len(f) for f in facts.values()),
            facts_written=total_written,
            duration_ms=round((time.monotonic() - t_start) * 1000),
        )

    except Exception as e:
        log_error("Observer", "observation failed", exc=e)