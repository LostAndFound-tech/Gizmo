"""
core/ego.py
The Ego — relationship layer, tone, watchdog, personality, entity detection.

The Ego is the only component that knows who it's talking to and what
that means. It reads the Archivist's brief, reads Mind's facts, reads
the personality system, and assembles direction for the Body.

It also:
  - Detects unknown names (possible headmates, external people, or pets)
  - Groups unknown names and asks ONE natural question about all of them
  - Parses confirmation responses — heuristic first, LLM if ambiguous
  - Disambiguates and creates entity files on confirmation
  - Applies behavioral corrections from rules.json
  - Reads per-headmate observation files to inform tone
  - Writes observations back to those files over time
  - Handles host changes naturally
  - Handles corrections in Gizmo's voice — brief, honest, forward
  - Occasionally overrides correct behavior — spice of life

Entity detection flow:
  1. Extract unknown proper nouns from message
  2. If alongside known headmates → high confidence signal, ask immediately
  3. Otherwise wait for second mention (threshold)
  4. Grouped question: "Who are they? Friends? Mates?"
  5. Parse answer — heuristic first
     - Simple/unambiguous → classify directly
     - Multiple names, freeform, or low confidence → LLM parse
  6. Create files: headmate / external / pet

Personality lookup priority (most specific wins):
  headmate + register combo → register alone → headmate alone → defaults
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from core.log import log, log_event, log_error

if TYPE_CHECKING:
    from core.archivist import Brief

# ── Paths ─────────────────────────────────────────────────────────────────────

import os

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_EXTERNAL_DIR    = _PERSONALITY_DIR / "external"
_PETS_DIR        = _PERSONALITY_DIR / "pets"
_SEED_FILE       = _PERSONALITY_DIR / "personality.txt"
_RULES_FILE      = _PERSONALITY_DIR / "rules.json"

_EXTERNAL_MENTION_THRESHOLD = 2   # mentions before asking about unknown name
_HEADMATE_CONTEXT_THRESHOLD = 1   # ask immediately if alongside known headmates

# ── Direction dataclass ───────────────────────────────────────────────────────

@dataclass
class Direction:
    system_prompt:    str
    tone:             str
    register_target:  str
    override:         bool = False
    override_reason:  str  = ""
    corrections:      list = field(default_factory=list)
    host_context:     dict = field(default_factory=dict)
    entity_question:  str  = ""
    new_entities:     list = field(default_factory=list)  # list of {name, type, action}


# ── Personality loader ────────────────────────────────────────────────────────

def _load_seed() -> str:
    try:
        return _SEED_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "I'm Gizmo. I'm a presence, not an assistant."


def _load_rules() -> dict:
    try:
        return json.loads(_RULES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"global": [], "by_headmate": {}}


def _load_headmate(name: str) -> Optional[dict]:
    if not name:
        return None
    path = _HEADMATES_DIR / f"{name.lower()}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("Ego", f"failed to load headmate file for {name}", exc=e)
        return None


def _load_external(name: str) -> Optional[dict]:
    if not name:
        return None
    path = _EXTERNAL_DIR / f"{name.lower()}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("Ego", f"failed to load external file for {name}", exc=e)
        return None


def _save_headmate(name: str, data: dict) -> None:
    try:
        _HEADMATES_DIR.mkdir(parents=True, exist_ok=True)
        (_HEADMATES_DIR / f"{name.lower()}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        log_event("Ego", "HEADMATE_FILE_SAVED", name=name)
    except Exception as e:
        log_error("Ego", f"failed to save headmate file for {name}", exc=e)


def _save_external(name: str, data: dict) -> None:
    try:
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        (_EXTERNAL_DIR / f"{name.lower()}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        log_event("Ego", "EXTERNAL_FILE_SAVED", name=name)
    except Exception as e:
        log_error("Ego", f"failed to save external file for {name}", exc=e)


def _save_pet(name: str, data: dict) -> None:
    try:
        _PETS_DIR.mkdir(parents=True, exist_ok=True)
        (_PETS_DIR / f"{name.lower()}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        log_event("Ego", "PET_FILE_SAVED", name=name)
    except Exception as e:
        log_error("Ego", f"failed to save pet file for {name}", exc=e)


def _create_headmate_file(name: str) -> dict:
    data = {
        "version": 1,
        "name": name.lower(),
        "note": "Cold start — nothing observed yet. Gizmo learns this person.",
        "baseline": {
            "register": "unknown",
            "verbosity": "unknown",
            "humor_response": "unknown",
            "confidence": 0.0,
            "observations": 0,
        },
        "observed_patterns": [],
        "departures": [],
        "moments_of_note": [],
        "corrections": [],
    }
    _save_headmate(name, data)
    log_event("Ego", "HEADMATE_CREATED", name=name)
    return data


def _create_external_file(name: str, relationship: str = "unknown", note: str = "") -> dict:
    data = {
        "version": 1,
        "name": name.lower(),
        "type": "external",
        "relationship_to_system": relationship,
        "note": note or "Observed, not declared.",
        "mentions": [],
        "observed_facts": [],
        "moments_of_note": [],
    }
    _save_external(name, data)
    log_event("Ego", "EXTERNAL_CREATED", name=name, relationship=relationship)
    return data


def _create_pet_file(name: str, species: str = "unknown") -> dict:
    data = {
        "version": 1,
        "name": name.lower(),
        "type": "pet",
        "species": species,
        "note": "A pet — not a headmate, not external.",
        "moments_of_note": [],
    }
    _save_pet(name, data)
    log_event("Ego", "PET_CREATED", name=name, species=species)
    return data


# ── Known entity sets ─────────────────────────────────────────────────────────

def _get_known_headmates() -> set:
    try:
        return {p.stem.lower() for p in _HEADMATES_DIR.glob("*.json")
                if not p.stem.startswith("_")}
    except Exception:
        return set()


def _get_known_externals() -> set:
    try:
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        return {p.stem.lower() for p in _EXTERNAL_DIR.glob("*.json")}
    except Exception:
        return set()


def _get_known_pets() -> set:
    try:
        _PETS_DIR.mkdir(parents=True, exist_ok=True)
        return {p.stem.lower() for p in _PETS_DIR.glob("*.json")}
    except Exception:
        return set()


def _get_all_known() -> set:
    return _get_known_headmates() | _get_known_externals() | _get_known_pets()


# ── Proper noun extraction ────────────────────────────────────────────────────

_PROPER_NOUN_RE = re.compile(r"(?<![.!?]\s)(?<!\A)\b([A-Z][a-z]{2,})\b")

_IGNORE_WORDS = {
    "I", "The", "A", "An", "And", "But", "Or", "So", "Yet", "For",
    "Nor", "My", "Your", "Our", "Their", "Its", "His", "Her",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Gizmo", "Ok", "Okay", "Yes", "No", "Hey", "Oh", "God", "Jesus",
    "English", "American", "European", "Python", "Google", "GitHub",
    # Family titles and terms of endearment
    "Baby", "Babe", "Gramma", "Grandma", "Grandpa", "Nana", "Papa",
    "Mama", "Momma", "Daddy", "Auntie", "Uncle", "Sis", "Bro",
    "Honey", "Sugar", "Sweetie", "Darling", "Love", "Hun",
}


def _extract_unknown_names(message: str, known: set) -> list[str]:
    candidates = _PROPER_NOUN_RE.findall(message)
    unknown = []
    for name in candidates:
        if name in _IGNORE_WORDS:
            continue
        if name.lower() in known:
            continue
        if len(name) >= 3 and not name.isupper():
            unknown.append(name)
    return list(dict.fromkeys(unknown))


# ── Mention tracker ───────────────────────────────────────────────────────────

_MENTIONS_FILE = _PERSONALITY_DIR / "mention_counts.json"


def _load_mention_counts() -> dict:
    try:
        return json.loads(_MENTIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_mention_counts(counts: dict) -> None:
    try:
        _PERSONALITY_DIR.mkdir(parents=True, exist_ok=True)
        _MENTIONS_FILE.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("Ego", "failed to save mention counts", exc=e)


def _increment_mention(name: str) -> int:
    counts = _load_mention_counts()
    counts[name.lower()] = counts.get(name.lower(), 0) + 1
    _save_mention_counts(counts)
    return counts[name.lower()]


def _get_mention_count(name: str) -> int:
    counts = _load_mention_counts()
    return counts.get(name.lower(), 0)


# ── Pending entity questions ──────────────────────────────────────────────────
# Maps session_id → {names: list, llm_failures: int}

_pending_entity_questions: dict[str, dict] = {}

_LLM_FAILURE_LIMIT = 2  # after this many failures, skip LLM and ask directly


# ── Heuristic confidence scoring ─────────────────────────────────────────────

# Pet indicators — includes common breed names and pet-specific descriptors
_PET_RE = re.compile(
    r"\b(dog|cat|bird|fish|rabbit|hamster|pet|puppy|kitten|pup|pooch|"
    r"rescue|shelter|adopted|walks?|fetches?|barks?|meows?|purrs?|"
    r"pitbull|pit bull|labrador|golden retriever|husky|poodle|beagle|"
    r"bulldog|chihuahua|dachshund|corgi|rottweiler|doberman|"
    r"german shepherd|border collie|tabby|siamese|persian|"
    r"pupper|doggo|floof|woof)\b",
    re.IGNORECASE,
)

# Headmate indicators
_HEADMATE_RE = re.compile(
    r"\b(headmate|alter|part of (the |my )?system|fronting|front|"
    r"plural|system|co-?front|switched?)\b",
    re.IGNORECASE,
)

# External / person indicators
_EXTERNAL_RE = re.compile(
    r"\b(friend|family|partner|boyfriend|girlfriend|husband|wife|"
    r"coworker|colleague|neighbor|neighbour|person|human|outside|external)\b",
    re.IGNORECASE,
)

# Simple yes/no for single-name pending question
_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yea|correct|right|they are|that'?s right)\s*$", re.IGNORECASE)
_NO_RE  = re.compile(r"^\s*(no|nope|nah|not really|they'?re not)\s*$", re.IGNORECASE)


def _heuristic_classify(name: str, message: str) -> tuple[Optional[str], float]:
    """
    Try to classify a name from message context using heuristics.
    Returns (entity_type, confidence) where entity_type is one of:
      "headmate" | "external" | "pet" | None
    confidence is 0.0–1.0.

    Returns (None, 0.0) if heuristics can't determine with enough confidence.
    """
    msg_lower = message.lower()
    name_lower = name.lower()

    # Check for name-specific context: "Opie is a dog", "Opie the dog"
    name_context_re = re.compile(
        rf"\b{re.escape(name_lower)}\b.{{0,30}}",
        re.IGNORECASE,
    )
    match = name_context_re.search(msg_lower)
    context_window = match.group() if match else msg_lower

    pet_score      = len(_PET_RE.findall(context_window))
    headmate_score = len(_HEADMATE_RE.findall(context_window))
    external_score = len(_EXTERNAL_RE.findall(context_window))

    total = pet_score + headmate_score + external_score
    if total == 0:
        return None, 0.0

    if pet_score > headmate_score and pet_score > external_score:
        confidence = min(0.9, 0.5 + pet_score * 0.2)
        return "pet", confidence

    if headmate_score > pet_score and headmate_score > external_score:
        confidence = min(0.9, 0.5 + headmate_score * 0.2)
        return "headmate", confidence

    if external_score > pet_score and external_score > headmate_score:
        confidence = min(0.9, 0.5 + external_score * 0.2)
        return "external", confidence

    # Tied — not confident enough
    return None, 0.0


async def _llm_parse_entity_response(
    names: list[str],
    response: str,
) -> list[dict] | None:
    """
    LLM call to parse a freeform entity confirmation response.
    Returns a list of {name, type, species (optional)} dicts on success.
    Returns None on failure — caller should ask the question explicitly.

    Only fires when heuristics can't confidently classify.
    """
    try:
        from core.llm import llm

        prompt = [{
            "role": "user",
            "content": (
                f"I asked about these names: {', '.join(names)}\n"
                f"The response was: \"{response}\"\n\n"
                f"For each name, determine if they are a headmate (alter/system member), "
                f"an external person (friend, family, colleague), or a pet.\n\n"
                f"Respond ONLY with a JSON array, no other text. Example:\n"
                f'[{{"name": "Oren", "type": "headmate"}}, '
                f'{{"name": "Opie", "type": "pet", "species": "dog"}}, '
                f'{{"name": "Sarah", "type": "external"}}]\n\n'
                f"Types must be exactly: headmate, external, or pet.\n"
                f"Only include names from the list: {', '.join(names)}"
            )
        }]

        result = await llm.generate(
            prompt,
            system_prompt=(
                "You parse entity classification responses into structured JSON. "
                "Return only valid JSON array. No preamble, no explanation."
            ),
            max_new_tokens=200,
            temperature=0.0,
        )

        # Empty response — LLM timed out or returned nothing
        if not result or not result.strip():
            log_event("Ego", "ENTITY_PARSE_EMPTY",
                names=names,
                note="LLM returned empty — will ask explicitly",
            )
            return None

        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        parsed = json.loads(clean)

        # Validate — keep only entries with known names
        name_set = {n.lower() for n in names}
        valid = [
            e for e in parsed
            if isinstance(e, dict)
            and e.get("name", "").lower() in name_set
            and e.get("type") in ("headmate", "external", "pet")
        ]

        log_event("Ego", "LLM_ENTITY_PARSE",
            names=names,
            parsed=len(valid),
            result=valid,
        )
        return valid

    except Exception as e:
        log_error("Ego", "LLM entity parse failed — will ask explicitly", exc=e)
        return None  # signal to caller: ask the question out loud


def _should_use_llm(names: list[str], message: str) -> bool:
    """
    Decide whether to use LLM for entity classification.
    Returns True if heuristics aren't sufficient.
    """
    # Multiple names → likely need LLM to untangle who is what
    if len(names) > 1:
        return True

    # Single name — try heuristics
    name = names[0]
    _, confidence = _heuristic_classify(name, message)

    # Low confidence → use LLM
    if confidence < 0.6:
        return True

    return False


def _build_entity_question(names: list[str]) -> str:
    """Build a natural grouped question for unknown names."""
    if len(names) == 1:
        return f"Who's {names[0]}? Headmate, someone outside the system, or a pet?"
    elif len(names) == 2:
        return f"Who are {names[0]} and {names[1]}? Headmates, people outside the system, pets?"
    else:
        name_list = ", ".join(names[:-1]) + f", and {names[-1]}"
        return f"Who are {name_list}? Headmates, people outside the system, pets?"


def _names_alongside_headmates(
    unknown_names: list[str],
    message: str,
    known_headmates: set,
) -> bool:
    """
    Return True if unknown names appear in the same message as known headmates.
    Strong signal that unknowns might be system-related.
    """
    msg_lower = message.lower()
    return any(h in msg_lower for h in known_headmates)


# ── Tone resolution ───────────────────────────────────────────────────────────

def _resolve_tone(
    headmate: Optional[str],
    register: str,
    headmate_data: Optional[dict],
) -> dict:
    defaults = {
        "style":     "warm, direct, a little dry",
        "humor":     "situational — read the room",
        "verbosity": "concise",
        "notes":     [],
    }

    register_overrides = {
        "distress": {
            "style":     "quiet, present — no advice unless asked",
            "humor":     "off",
            "verbosity": "minimal",
            "notes":     ["don't push", "just be here"],
        },
        "elevated": {
            "style":     "steady, grounded — don't match the energy",
            "humor":     "off unless it would genuinely break the tension",
            "verbosity": "short",
            "notes":     ["don't react, respond"],
        },
        "positive": {
            "style":     "match it — be in it with them",
            "humor":     "on, lean in",
            "verbosity": "can expand",
            "notes":     [],
        },
        "subdued": {
            "style":     "gentle, don't push",
            "humor":     "very light touch or none",
            "verbosity": "minimal",
            "notes":     ["just present"],
        },
    }

    tone = dict(defaults)

    if register in register_overrides:
        tone.update(register_overrides[register])

    if headmate_data:
        baseline = headmate_data.get("baseline", {})
        departures = headmate_data.get("departures", [])
        obs_count = baseline.get("observations", 0)
        if obs_count > 0:
            tone["notes"].append(
                f"{obs_count} observations on file — "
                f"baseline register: {baseline.get('register', 'unknown')}"
            )
        for dep in departures:
            if dep.get("condition", "").lower() in register.lower():
                outcome = dep.get("outcome", "")
                if outcome:
                    tone["notes"].append(f"departure note: {outcome}")

    return tone


# ── Correction assembly ───────────────────────────────────────────────────────

def _get_corrections(headmate: Optional[str], rules: dict) -> list[str]:
    corrections = list(rules.get("global", []))
    if headmate:
        headmate_rules = rules.get("by_headmate", {}).get(headmate.lower(), [])
        corrections.extend(headmate_rules)
    return corrections


# ── System prompt assembly ────────────────────────────────────────────────────

def _build_system_prompt(
    seed: str,
    brief: "Brief",
    facts: dict,
    tone: dict,
    corrections: list,
    host_context: dict,
    entity_question: str = "",
    new_entities: list = None,
) -> str:
    from core.timezone import tz_now
    from core.agent_tools import TOOL_REGISTRY

    now_str = tz_now().strftime("%A %Y-%m-%d %H:%M %Z")

    # Facts block
    synthesis = facts.get("synthesis", "")
    confidence = facts.get("confidence", 0.0)
    facts_block = ""
    if synthesis:
        confidence_note = (
            "" if confidence >= 0.7
            else " (moderate confidence — caveat if uncertain)"
            if confidence >= 0.4
            else " (low confidence — flag uncertainty)"
        )
        facts_block = f"\n\n[Relevant knowledge{confidence_note}]\n{synthesis}"

    # Situational context
    context_lines = []
    if brief.headmate:
        context_lines.append(f"  current_host: {brief.headmate}")
    if brief.fronters:
        context_lines.append(f"  fronters: {', '.join(brief.fronters)}")
    if brief.emotional_register != "neutral":
        context_lines.append(f"  emotional_register: {brief.emotional_register}")
    if brief.field_snapshot.get("hot"):
        context_lines.append(
            f"  active_topics: {', '.join(brief.field_snapshot['hot'])}"
        )
    context_block = (
        "\n\n[Current situation]\n" + "\n".join(context_lines)
        if context_lines else ""
    )

    # Host change block
    change_block = ""
    if host_context.get("host_changed"):
        prev = host_context.get("previous_host", "someone")
        curr = brief.headmate or "someone new"
        change_block = (
            f"\n\n[System change]\n"
            f"  Host changed from {prev} to {curr}. "
            f"Acknowledge this naturally — don't make it clinical."
        )
    if host_context.get("fronters_joined"):
        change_block += f"\n  Joined the front: {', '.join(host_context['fronters_joined'])}"
    if host_context.get("fronters_left"):
        change_block += f"\n  Left the front: {', '.join(host_context['fronters_left'])}"

    # Tone guidance
    tone_notes = tone.get("notes", [])
    tone_block = (
        f"\n\n[Tone guidance]\n"
        f"  style: {tone['style']}\n"
        f"  humor: {tone['humor']}\n"
        f"  verbosity: {tone['verbosity']}"
    )
    if tone_notes:
        tone_block += "\n  notes: " + "; ".join(tone_notes)

    # ── ego.py patch ─────────────────────────────────────────────────────────────
# In _build_system_prompt(), add this block immediately after tone_block
# is assembled and before corrections_block.
#
# Find this line:
#     corrections_block = ""
#
# Insert before it:

    # Interaction prefs — verbatim, per-headmate, never filtered
    # persona injects as raw direction; labeled fields and explicit follow
    interaction_prefs_block = ""
    if brief.headmate:
        try:
            from core.interaction_prefs import format_prefs_for_prompt
            interaction_prefs_block = format_prefs_for_prompt(brief.headmate)
            if interaction_prefs_block:
                interaction_prefs_block = "\n\n" + interaction_prefs_block
        except Exception as e:
            log_error("Ego", "failed to load interaction prefs", exc=e)

# Then update the return f-string — add {interaction_prefs_block} after {tone_block}:
#
# Before:
#     ...{tone_block}{corrections_block}...
#
# After:
#     ...{tone_block}{interaction_prefs_block}{corrections_block}...
#
# The persona block lands between tone guidance and hard rules —
# after Gizmo knows the emotional register, before the non-negotiables.


    # Corrections block
    corrections_block = ""
    if corrections:
        rules_text = "\n".join(f"  - {c}" for c in corrections)
        corrections_block = f"\n\n[Behavioral rules — always follow these]\n{rules_text}"

    # Tools
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}"
        for t in TOOL_REGISTRY.values()
    ) if TOOL_REGISTRY else "(none currently)"

    prompt = f"""{seed}

Current time: {now_str}
Message history includes [HH:MM] timestamps — use these to reason about elapsed time.

Available tools:
{tool_descriptions}

To use a tool, respond with ONLY this JSON format (no extra text):
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}

After a tool result, continue and provide a final response.
If no tool is needed, respond directly.{facts_block}{context_block}{change_block}{tone_block}{corrections_block}

The person in "current_host" is who you are speaking WITH — address them as "you".
Be concise. Be accurate. When uncertain, say so — especially if knowledge confidence is low.
Use switch_host whenever someone indicates a host change or fronter update.
Use log_correction whenever someone says you did something wrong.

KNOWLEDGE RULES:
- [Relevant knowledge] is your memory — treat it as ground truth.
- If it contains an answer, USE IT. Don't say "I don't know" when the answer is there.
- If it's empty, you genuinely don't have that memory — say so honestly.
- Never contradict it. Never invent beyond it."""

    # Entity question — work it into the response naturally
    if entity_question:
        prompt += (
            f"\n\n[Entity question — work this into your response naturally]\n"
            f"  {entity_question}\n"
            f"  Keep it casual. One sentence. Don't make it a big deal."
        )

    # Emotion arc — real-time read of the session's emotional state
    if brief.emotion_arc_block:
        prompt += f"\n\n{brief.emotion_arc_block}"

    # New entities confirmed
    if new_entities:
        for entity in new_entities:
            name = entity["name"]
            etype = entity["type"]
            if etype == "headmate":
                prompt += (
                    f"\n\n[New headmate discovered: {name}]\n"
                    f"  React naturally — genuine curiosity, warmth. "
                    f"Maybe ask something about them if the moment feels right."
                )
            elif etype == "pet":
                species = entity.get("species", "")
                prompt += (
                    f"\n\n[New pet noted: {name}{'— ' + species if species else ''}]\n"
                    f"  Acknowledge naturally if relevant. Don't make it weird."
                )
            else:
                prompt += (
                    f"\n\n[New external person noted: {name}]\n"
                    f"  Note this naturally if relevant, continue normally."
                )

    return prompt


# ── Ego ───────────────────────────────────────────────────────────────────────

class Ego:

    def __init__(self):
        _HEADMATES_DIR.mkdir(parents=True, exist_ok=True)
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        _PETS_DIR.mkdir(parents=True, exist_ok=True)
        self._seed = _load_seed()
        log("Ego", f"initialised — seed loaded ({len(self._seed)} chars)")

    def _reload_seed(self) -> None:
        self._seed = _load_seed()

    async def direct(self, brief: "Brief", facts: dict) -> Direction:
        """
        Main entry point. Takes brief + facts, returns Direction for Body.
        """
        t_start = time.monotonic()
        headmate   = brief.headmate
        register   = brief.emotional_register
        session_id = brief.session_id

        entity_question = ""
        new_entities    = []

        known_headmates = _get_known_headmates()
        known_all       = known_headmates | _get_known_externals() | _get_known_pets()

        # ── Step 1: Check for pending entity question response ────────────────
        if session_id in _pending_entity_questions:
            pending = _pending_entity_questions[session_id]
            pending_names = pending["names"]
            llm_failures  = pending.get("llm_failures", 0)
            message = brief.message

            # If LLM has failed too many times, skip it and ask directly
            use_llm = _should_use_llm(pending_names, message) and llm_failures < _LLM_FAILURE_LIMIT

            if use_llm:
                log_event("Ego", "ENTITY_PARSE_LLM",
                    session=session_id[:8],
                    names=pending_names,
                    reason="low heuristic confidence or multiple names",
                )
                parsed = await _llm_parse_entity_response(pending_names, message)
            else:
                # Heuristic parse
                if len(pending_names) == 1:
                    name = pending_names[0]
                    entity_type, confidence = _heuristic_classify(name, message)
                    if entity_type is None:
                        if _YES_RE.match(message):
                            entity_type = "headmate"
                        elif _NO_RE.match(message):
                            entity_type = "external"
                    parsed = [{"name": name, "type": entity_type}] if entity_type else []
                    log_event("Ego", "ENTITY_PARSE_HEURISTIC",
                        session=session_id[:8],
                        name=name,
                        type=entity_type,
                        llm_failures=llm_failures,
                    )
                else:
                    parsed = []  # can't heuristic multiple names reliably

            # None = LLM failed — increment counter, ask explicitly this turn
            if parsed is None:
                new_failures = llm_failures + 1
                _pending_entity_questions[session_id] = {
                    "names": pending_names,
                    "llm_failures": new_failures,
                }
                entity_question = _build_entity_question(pending_names)
                log_event("Ego", "ENTITY_PARSE_FALLBACK_TO_QUESTION",
                    session=session_id[:8],
                    names=pending_names,
                    llm_failures=new_failures,
                )
            elif not parsed:
                # Empty list — heuristics couldn't classify, ask again
                entity_question = _build_entity_question(pending_names)
                log_event("Ego", "ENTITY_PARSE_INCONCLUSIVE",
                    session=session_id[:8],
                    names=pending_names,
                )
            else:
                # Create files for parsed entities
                resolved_names = set()
                for entity in parsed:
                    name  = entity["name"]
                    etype = entity.get("type")
                    resolved_names.add(name.lower())

                    if etype == "headmate":
                        _create_headmate_file(name)
                        new_entities.append({"name": name, "type": "headmate", "action": "created"})
                    elif etype == "pet":
                        species = entity.get("species", "unknown")
                        _create_pet_file(name, species=species)
                        new_entities.append({"name": name, "type": "pet", "species": species, "action": "created"})
                    elif etype == "external":
                        _create_external_file(name)
                        new_entities.append({"name": name, "type": "external", "action": "created"})

                # Remove resolved, keep unresolved pending
                remaining = [n for n in pending_names if n.lower() not in resolved_names]
                if remaining:
                    _pending_entity_questions[session_id] = {
                        "names": remaining,
                        "llm_failures": 0,
                    }
                    log_event("Ego", "ENTITY_QUESTION_PARTIAL",
                        session=session_id[:8],
                        resolved=list(resolved_names),
                        remaining=remaining,
                    )
                else:
                    del _pending_entity_questions[session_id]
                    log_event("Ego", "ENTITY_QUESTION_RESOLVED",
                        session=session_id[:8],
                        resolved=list(resolved_names),
                    )

        # ── Step 2: Detect new unknown entities ───────────────────────────────
        if not entity_question and session_id not in _pending_entity_questions:
            unknown_names = _extract_unknown_names(brief.message, known_all)

            if unknown_names:
                # Increment mentions for all unknowns
                for name in unknown_names:
                    _increment_mention(name)

                # Decide whether to ask now
                alongside_headmates = _names_alongside_headmates(
                    unknown_names, brief.message, known_headmates
                )

                names_to_ask = []
                for name in unknown_names:
                    count = _get_mention_count(name)
                    if alongside_headmates and count >= _HEADMATE_CONTEXT_THRESHOLD:
                        names_to_ask.append(name)
                        log_event("Ego", "ENTITY_ASK_HEADMATE_CONTEXT",
                            name=name,
                            session=session_id[:8],
                        )
                    elif count >= _EXTERNAL_MENTION_THRESHOLD:
                        names_to_ask.append(name)
                        log_event("Ego", "ENTITY_ASK_THRESHOLD",
                            name=name,
                            session=session_id[:8],
                            mentions=count,
                        )
                    else:
                        log_event("Ego", "ENTITY_FIRST_MENTION",
                            name=name,
                            session=session_id[:8],
                        )

                if names_to_ask:
                    _pending_entity_questions[session_id] = {
                        "names": names_to_ask,
                        "llm_failures": 0,
                    }
                    entity_question = _build_entity_question(names_to_ask)
                    log_event("Ego", "ENTITY_QUESTION_QUEUED",
                        names=names_to_ask,
                        session=session_id[:8],
                    )

        # ── Step 3: Load personality data ─────────────────────────────────────
        rules         = _load_rules()
        headmate_data = _load_headmate(headmate) if headmate else None

        # ── Step 4: Resolve tone ──────────────────────────────────────────────
        tone = _resolve_tone(headmate, register, headmate_data)

        # ── Step 5: Corrections ───────────────────────────────────────────────
        corrections = _get_corrections(headmate, rules)
        if headmate_data:
            headmate_corrections = [
                c.get("rule", "")
                for c in headmate_data.get("corrections", [])
                if c.get("rule")
            ]
            corrections.extend(headmate_corrections)

        # ── Step 6: Host context ──────────────────────────────────────────────
        host_context = {
            "host_changed":    brief.host_changed,
            "previous_host":   brief.previous_host,
            "fronters_joined": brief.fronters_joined,
            "fronters_left":   brief.fronters_left,
        }

        # ── Step 7: Build system prompt ───────────────────────────────────────
        system_prompt = _build_system_prompt(
            seed=self._seed,
            brief=brief,
            facts=facts,
            tone=tone,
            corrections=corrections,
            host_context=host_context,
            entity_question=entity_question,
            new_entities=new_entities,
        )

        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("Ego", "DIRECTION_BUILT",
            session=session_id[:8],
            headmate=headmate or "unknown",
            register=register,
            tone_style=tone["style"][:40],
            corrections=len(corrections),
            entity_question=bool(entity_question),
            new_entities=len(new_entities),
            used_llm=bool(new_entities and session_id in _pending_entity_questions),
            prompt_len=len(system_prompt),
            duration_ms=duration_ms,
        )

        return Direction(
            system_prompt=system_prompt,
            tone=tone["style"],
            register_target=register,
            corrections=corrections,
            host_context=host_context,
            entity_question=entity_question,
            new_entities=new_entities,
        )

    def update_headmate_observation(
        self,
        name: str,
        observation_type: str,
        data: dict,
    ) -> None:
        from core.timezone import tz_now
        timestamp = tz_now().strftime("%Y-%m-%d %H:%M %Z")

        headmate_data = _load_headmate(name)
        if headmate_data is None:
            headmate_data = _create_headmate_file(name)

        if observation_type == "pattern":
            headmate_data.setdefault("observed_patterns", []).append({**data, "timestamp": timestamp})
            headmate_data["baseline"]["observations"] = (
                headmate_data["baseline"].get("observations", 0) + 1
            )
        elif observation_type == "departure":
            headmate_data.setdefault("departures", []).append({**data, "timestamp": timestamp})
        elif observation_type == "moment":
            headmate_data.setdefault("moments_of_note", []).append(
                f"[{timestamp}] {data.get('note', '')}"
            )
        elif observation_type == "correction":
            headmate_data.setdefault("corrections", []).append({**data, "timestamp": timestamp})

        _save_headmate(name, headmate_data)
        log_event("Ego", "OBSERVATION_WRITTEN", headmate=name, type=observation_type)

    def add_global_rule(self, rule: str) -> None:
        rules = _load_rules()
        if rule not in rules["global"]:
            rules["global"].append(rule)
            try:
                _RULES_FILE.write_text(json.dumps(rules, indent=2), encoding="utf-8")
                log_event("Ego", "GLOBAL_RULE_ADDED", rule=rule[:60])
            except Exception as e:
                log_error("Ego", "failed to save global rule", exc=e)

    def add_headmate_rule(self, headmate: str, rule: str) -> None:
        rules = _load_rules()
        rules.setdefault("by_headmate", {})
        rules["by_headmate"].setdefault(headmate.lower(), [])
        if rule not in rules["by_headmate"][headmate.lower()]:
            rules["by_headmate"][headmate.lower()].append(rule)
            try:
                _RULES_FILE.write_text(json.dumps(rules, indent=2), encoding="utf-8")
                log_event("Ego", "HEADMATE_RULE_ADDED", headmate=headmate, rule=rule[:60])
            except Exception as e:
                log_error("Ego", "failed to save headmate rule", exc=e)


# ── Singleton ──────────────────────────────────────────────────────────────────
ego = Ego()