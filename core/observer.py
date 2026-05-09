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

Facts are written to:
  1. Entity JSON files (headmates/externals/pets) — human-readable backup
  2. memory ChromaDB collection — live queryable store, used by Mind

The JSON files are the backup. ChromaDB is the source of truth.
"""

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
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

_IS_RE = re.compile(
    r'\b([A-Z][a-z]{1,20})\s+is\s+'
    r'(?!currently |now |still |just |also |the one)'
    r'((?:a |an |the )?[^.!?,\n]{4,80}?)(?:\.|,|!|\?|$|\n)',
    re.MULTILINE,
)

_POSSESSIVE_RE = re.compile(
    r"\b([A-Z][a-z]{1,20})'s\s+([a-z][^.!?,\n]{3,60})(?:\.|,|!|\?|$|\n)",
    re.MULTILINE,
)

_WORK_RE = re.compile(
    r"\b([A-Z][a-z]{1,20})\s+(?:works?|worked)\s+(?:as|at|in|for)\s+"
    r"([^.!?,\n]{4,60})(?:\.|,|!|\?|$|\n)",
    re.IGNORECASE | re.MULTILINE,
)

_SELF_AGE_RE  = re.compile(r"\bI'?m\s+(\d+)\s+years?\s+old\b", re.IGNORECASE)
_SELF_WORK_RE = re.compile(r"\bI\s+(?:work|worked)\s+(?:as|at|in|for)\s+([^.!?,\n]{4,60})(?:\.|,|$)", re.IGNORECASE)
_SELF_IS_RE   = re.compile(r"\bI'?m\s+(?:a|an)\s+([\w][^.!?,\n]{2,60})(?:\.|,|!|\?|$)", re.IGNORECASE)
_SELF_PREF_RE = re.compile(r"\bI\s+(like|love|hate|prefer|enjoy|adore|despise)\s+([^.!?,\n]{3,60})(?:\.|,|$)", re.IGNORECASE)
_SELF_FROM_RE = re.compile(r"\bI'?(?:m from|come from|was born in|grew up in|'?m\s+from)\s+([^.!?,\n]{3,80})(?:\.|,|$)", re.IGNORECASE)

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


def _heuristic_extract(message: str, fronters: list, known: set) -> dict:
    facts: dict[str, list] = {}
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

    for m in _IS_RE.finditer(message):
        name = m.group(1)
        predicate = m.group(2).strip().rstrip(".,;!? ")
        if name in _NOT_NAMES or _JUNK_PRED_RE.match(predicate):
            continue
        add(name, f"is {predicate}")

    for m in _POSSESSIVE_RE.finditer(message):
        name = m.group(1)
        attr = m.group(2).strip().rstrip(".,;!? ")
        if name in _NOT_NAMES or len(attr.split()) < 2:
            continue
        add(name, f"{name}'s {attr}")

    for m in _WORK_RE.finditer(message):
        name = m.group(1)
        role = m.group(2).strip().rstrip(".,;!? ")
        if name not in _NOT_NAMES:
            add(name, f"works {role}")

    if speaker and speaker in known:
        for m in _SELF_AGE_RE.finditer(message):
            add(speaker, f"is {m.group(1)} years old")
        for m in _SELF_WORK_RE.finditer(message):
            add(speaker, f"works {m.group(1).strip().rstrip('.,;!? ')}")
        for m in _SELF_IS_RE.finditer(message):
            noun = m.group(1).strip().rstrip(".,;!? ")
            if not _JUNK_PRED_RE.match(noun):
                add(speaker, f"is a {noun}")
        for m in _SELF_PREF_RE.finditer(message):
            add(speaker, f"{m.group(1)}s {m.group(2).strip().rstrip('.,;!? ')}")
        for m in _SELF_FROM_RE.finditer(message):
            add(speaker, f"is from {m.group(1).strip().rstrip('.,;!? ')}")

    return facts


# ── LLM fallback ──────────────────────────────────────────────────────────────

_FACTUAL_SIGNAL_RE = re.compile(
    r"\b(used to|originally|born|grew up|studied|graduated|married|"
    r"divorced|moved|lives? in|from|species|race|occupation|"
    r"profession|personality|ability|power|background|history|"
    r"experiment|military|alternate|dimension|universe|planet|"
    r"dream|growing up|my goal|always wanted|hope to|wanted to be|"
    r"love kids|love children|passion|hobby|favorite|favourite)\b",
    re.IGNORECASE,
)

_RICH_MESSAGE_WORDS = 20


def _worth_llm_fallback(message: str, heuristic_results: dict) -> bool:
    words     = len(message.split())
    has_signal = bool(_FACTUAL_SIGNAL_RE.search(message))
    if not has_signal:
        return False
    if not heuristic_results and words >= 8:
        return True
    if words >= _RICH_MESSAGE_WORDS:
        return True
    return False


async def _llm_extract(message: str, fronters: list, known: set, llm) -> dict:
    entity_list     = ", ".join(sorted(known))
    primary_speaker = fronters[0].title() if fronters else "unknown"
    other_fronters  = [f.title() for f in fronters[1:]] if len(fronters) > 1 else []
    fronter_context = f"Primary speaker: {primary_speaker}"
    if other_fronters:
        fronter_context += f" (also present: {', '.join(other_fronters)})"

    prompt = [{
        "role": "user",
        "content": (
            f"Extract lasting personal facts from this message.\n"
            f"{fronter_context} | Known people: {entity_list}\n\n"
            f"Message: \"{message}\"\n\n"
            f"For each person with facts, write their name in [brackets] then bullet points.\n"
            f"Each bullet = one short standalone fact.\n\n"
            f"Rules:\n"
            f"- When the speaker says 'I' or 'my', attribute to {primary_speaker}\n"
            f"- Attribute to SUBJECT — if {primary_speaker} says 'Oren is a werewolf', write under [oren]\n"
            f"- Only lasting facts: age, species, occupation, origin, dreams, preferences, relationships\n"
            f"- Skip: who is fronting/hosting, session metadata, observation counts\n"
            f"- Only use names from: {entity_list}\n\n"
            f"If no lasting facts exist, write nothing."
        )
    }]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "Extract lasting personal facts as bullet points grouped by person name in [brackets]. "
                "One fact per bullet. No JSON. No preamble. Attribute to subject not speaker."
            ),
            max_new_tokens=400,
            temperature=0.1,
        )

        if not result or not result.strip():
            return {}

        facts: dict[str, list] = {}
        current_name = None

        for line in result.splitlines():
            line = line.strip()
            if not line:
                continue

            name_match = re.match(
                r'^\[([^\]]+)\]$|^\*\*([^*]+)\*\*$|^([a-z][a-z]{1,20}):$',
                line, re.IGNORECASE
            )
            if name_match:
                candidate = (
                    name_match.group(1) or
                    name_match.group(2) or
                    name_match.group(3) or ""
                ).strip().lower()
                if candidate in known:
                    current_name = candidate
                continue

            if current_name and (line.startswith("- ") or line.startswith("• ")):
                fact = line[2:].strip()
                if fact and not _is_junk(fact) and len(fact.split()) >= 2:
                    facts.setdefault(current_name, [])
                    if fact.lower() not in [f.lower() for f in facts[current_name]]:
                        facts[current_name].append(fact)

        log_event("Observer", "LLM_EXTRACT",
            names=list(facts.keys()),
            total=sum(len(f) for f in facts.values()),
        )
        return facts

    except Exception as e:
        log_error("Observer", "LLM fallback failed", exc=e)
        return {}


# ── Memory writer — writes to both JSON and ChromaDB ─────────────────────────

def _write_facts(name: str, facts: list) -> int:
    """
    Write extracted facts to entity JSON file AND memory ChromaDB collection.
    Returns count written.
    JSON = human-readable backup.
    ChromaDB = live queryable store used by Mind.
    """
    from core.timezone import tz_now
    timestamp  = tz_now().strftime("%Y-%m-%d %H:%M %Z")
    now_iso    = datetime.now().isoformat(timespec="seconds")

    entity_type = _get_entity_type(name)
    if not entity_type:
        return 0

    data = _load_entity(name, entity_type)
    if data is None:
        return 0

    existing = _existing_facts(data, entity_type)
    written  = 0
    new_facts = []

    for fact in facts:
        fact = fact.strip()
        if not fact or _is_junk(fact):
            continue

        fact_lower = fact.lower()
        if any(fact_lower[:40] in ex or ex[:40] in fact_lower for ex in existing):
            continue

        # Write to JSON
        if entity_type == "headmate":
            age_match = re.search(r'\b(\d+)\s+years?\s+old\b', fact_lower)
            if age_match:
                current = data.get("baseline", {}).get("age", "unknown")
                if current == "unknown":
                    data.setdefault("baseline", {})["age"] = fact
                    existing.add(fact_lower)
                    written += 1
                    new_facts.append(fact)
                    continue

            data.setdefault("moments_of_note", []).append(f"[{timestamp}] {fact}")
            data.setdefault("baseline", {})
            data["baseline"]["observations"] = data["baseline"].get("observations", 0) + 1
        else:
            data.setdefault("observed_facts", []).append(f"[{timestamp}] {fact}")

        existing.add(fact_lower)
        written += 1
        new_facts.append(fact)

    if written > 0:
        if data.get("note", "").startswith("Cold start"):
            obs = data.get("baseline", {}).get("observations", written)
            data["note"] = f"Actively observed — {obs} exchanges logged."
        _save_entity(name, entity_type, data)

        log_event("Observer", "FACTS_WRITTEN",
            name=name,
            type=entity_type,
            count=written,
            facts=[f[:60] for f in new_facts[:3]],
        )

        # Write to memory ChromaDB collection
        try:
            from tools.memory_tool import _get_collection, MEMORY_COLLECTION
            col = _get_collection(MEMORY_COLLECTION)

            for fact in new_facts:
                col.add(
                    documents=[fact],
                    metadatas=[{
                        "subject":     name.lower(),
                        "type":        "fact",
                        "entity_type": entity_type,
                        "written_at":  now_iso,
                        "source":      "observer",
                        "tags":        f"fact,{entity_type},{name.lower()}",
                    }],
                    ids=[f"obs_{uuid.uuid4().hex[:12]}"],
                )

            log_event("Observer", "MEMORY_WRITTEN",
                name=name,
                count=len(new_facts),
            )
        except Exception as e:
            log_error("Observer", f"memory collection write failed for {name}", exc=e)

    return written


# ── File cleaner ──────────────────────────────────────────────────────────────

def clean_junk_facts(name: str) -> int:
    entity_type = _get_entity_type(name)
    if not entity_type:
        return 0

    data = _load_entity(name, entity_type)
    if not data:
        return 0

    removed = 0

    if entity_type == "headmate":
        original = data.get("moments_of_note", [])
        cleaned  = [m for m in original if not _is_junk(
            re.sub(r'^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*', '', str(m))
        )]
        removed = len(original) - len(cleaned)
        data["moments_of_note"] = cleaned

        baseline = data.get("baseline", {})
        for key in list(baseline.keys()):
            val = str(baseline.get(key, ""))
            if key == "age" and not re.search(r'\d', val):
                baseline[key] = "unknown"
                removed += 1
    else:
        original = data.get("observed_facts", [])
        cleaned  = [f for f in original if not _is_junk(
            re.sub(r'^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*', '', str(f))
        )]
        removed = len(original) - len(cleaned)
        data["observed_facts"] = cleaned

    if removed > 0:
        _save_entity(name, entity_type, data)
        log_event("Observer", "JUNK_CLEANED", name=name, removed=removed)

    return removed


def clean_all_junk() -> dict:
    results = {}
    for d, etype in [(_HEADMATES_DIR, "headmate"), (_EXTERNAL_DIR, "external"), (_PETS_DIR, "pet")]:
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            if f.stem in ("count", "_schema"):
                continue
            removed = clean_junk_facts(f.stem)
            if removed > 0:
                results[f.stem] = removed
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

async def observe(
    user_message: str,
    gizmo_response: str,
    fronters: list,
    session_id: str,
    llm,
) -> None:
    """
    Main entry point. Called by Archivist.receive_outgoing().
    Never raises — errors logged and swallowed.
    """
    if not fronters or not _should_extract(user_message):
        return

    t_start = time.monotonic()

    try:
        known = _known_entities()
        if not known:
            return

        facts  = _heuristic_extract(user_message, fronters, known)
        source = "heuristic"

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
            total_written += _write_facts(name, name_facts)

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