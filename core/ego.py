"""
core/ego.py
The Ego — relationship layer, tone, watchdog, personality, entity detection.

The Ego is the only component that knows who it's talking to and what
that means. It reads the Archivist's brief, reads Mind's facts, reads
the personality system, and assembles direction for the Body.

It also:
  - Detects unknown names (possible headmates or external people)
  - Disambiguates and creates entity files on confirmation
  - Applies behavioral corrections from rules.json
  - Reads per-headmate observation files to inform tone
  - Writes observations back to those files over time
  - Handles host changes naturally
  - Handles corrections in Gizmo's voice — brief, honest, forward
  - Occasionally overrides correct behavior — spice of life

The direction it returns is a structured dict, not just a string.
Body gets the system prompt. Everything else is for logging and reflection.

Personality lookup priority (most specific wins):
  headmate + register combo → register alone → headmate alone → defaults

Entity detection:
  Unknown proper nouns flagged by Archivist → Ego asks one question
  Confirmed headmate → auto-creates headmate file
  Confirmed external → auto-creates external file
  Recurring ambiguous name → file after threshold

Usage:
    from core.ego import ego
    direction = await ego.direct(brief, facts)
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

_BASE_DIR        = Path(__file__).parent.parent
_PERSONALITY_DIR = _BASE_DIR / "personality"
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_EXTERNAL_DIR    = _PERSONALITY_DIR / "external"
_SEED_FILE       = _BASE_DIR / "personality.txt"
_GIZMO_FILE      = _PERSONALITY_DIR / "gizmo.json"
_RULES_FILE      = _PERSONALITY_DIR / "rules.json"

# How many times a name must appear before auto-creating an external file
_EXTERNAL_MENTION_THRESHOLD = 2

# ── Direction dataclass ───────────────────────────────────────────────────────

@dataclass
class Direction:
    """
    Structured direction for the Body.
    system_prompt is what Body receives.
    Everything else is for logging and Ego's own reflection.
    """
    system_prompt:    str
    tone:             str        # brief description of intended tone
    register_target:  str        # what register Ego is aiming for
    override:         bool = False
    override_reason:  str  = ""
    corrections:      list = field(default_factory=list)
    host_context:     dict = field(default_factory=dict)
    entity_question:  str  = ""  # if Ego needs to ask about an unknown entity
    new_entity:       dict = field(default_factory=dict)  # if a new entity was just confirmed


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
    """Load a headmate's observation file. Returns None if not found."""
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
    """Load an external person's observation file."""
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
        path = _HEADMATES_DIR / f"{name.lower()}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log_event("Ego", "HEADMATE_FILE_SAVED", name=name)
    except Exception as e:
        log_error("Ego", f"failed to save headmate file for {name}", exc=e)


def _save_external(name: str, data: dict) -> None:
    try:
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        path = _EXTERNAL_DIR / f"{name.lower()}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log_event("Ego", "EXTERNAL_FILE_SAVED", name=name)
    except Exception as e:
        log_error("Ego", f"failed to save external file for {name}", exc=e)


def _create_headmate_file(name: str) -> dict:
    """Create a cold-start headmate file and save it."""
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


def _create_external_file(name: str, relationship: str = "unknown") -> dict:
    """Create a cold-start external person file and save it."""
    data = {
        "version": 1,
        "name": name.lower(),
        "type": "external",
        "relationship_to_system": relationship,
        "note": "Observed, not declared.",
        "mentions": [],
        "observed_facts": [],
        "moments_of_note": [],
    }
    _save_external(name, data)
    log_event("Ego", "EXTERNAL_CREATED", name=name, relationship=relationship)
    return data


def _add_observed_fact(name: str, fact: str, entity_type: str = "external") -> None:
    """Add an observed fact to an entity's file."""
    from core.timezone import tz_now
    timestamp = tz_now().strftime("%Y-%m-%d %H:%M %Z")

    if entity_type == "headmate":
        data = _load_headmate(name)
        if data is None:
            data = _create_headmate_file(name)
        if "moments_of_note" not in data:
            data["moments_of_note"] = []
        data["moments_of_note"].append(f"[{timestamp}] {fact}")
        _save_headmate(name, data)
    else:
        data = _load_external(name)
        if data is None:
            data = _create_external_file(name)
        if "observed_facts" not in data:
            data["observed_facts"] = []
        data["observed_facts"].append(f"[{timestamp}] {fact}")
        _save_external(name, data)


# ── Tone resolution ───────────────────────────────────────────────────────────

def _resolve_tone(
    headmate: Optional[str],
    register: str,
    headmate_data: Optional[dict],
) -> dict:
    """
    Resolve tone from personality data.
    Priority: headmate+register > register > headmate baseline > defaults.

    v1: reads from headmate file observations.
    Future: reads from Id tensor directly.

    Returns a tone dict with: style, humor, verbosity, notes
    """
    defaults = {
        "style":     "warm, direct, a little dry",
        "humor":     "situational — read the room",
        "verbosity": "concise",
        "notes":     [],
    }

    # Register-based overrides
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

    # Apply register override if present
    if register in register_overrides:
        tone.update(register_overrides[register])

    # Apply headmate-specific notes from observation file
    if headmate_data:
        baseline = headmate_data.get("baseline", {})
        observations = headmate_data.get("observed_patterns", [])
        departures = headmate_data.get("departures", [])

        # If we have observations, note what's been seen
        obs_count = baseline.get("observations", 0)
        if obs_count > 0:
            tone["notes"].append(
                f"{obs_count} observations on file — "
                f"baseline register: {baseline.get('register', 'unknown')}"
            )

        # Check for any departure patterns matching current register
        for dep in departures:
            if dep.get("condition", "").lower() in register.lower():
                outcome = dep.get("outcome", "")
                if outcome:
                    tone["notes"].append(f"departure note: {outcome}")

    return tone


# ── Correction assembly ───────────────────────────────────────────────────────

def _get_corrections(headmate: Optional[str], rules: dict) -> list[str]:
    """
    Assemble active behavioral corrections for this headmate.
    Global rules always apply. Headmate-specific rules added on top.
    """
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
) -> str:
    """
    Assemble the full system prompt for Body.
    Seed + facts + context + tone guidance + corrections.
    """
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
        joined = ", ".join(host_context["fronters_joined"])
        change_block += f"\n  Joined the front: {joined}"
    if host_context.get("fronters_left"):
        left = ", ".join(host_context["fronters_left"])
        change_block += f"\n  Left the front: {left}"

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

    # Corrections block
    corrections_block = ""
    if corrections:
        rules_text = "\n".join(f"  - {c}" for c in corrections)
        corrections_block = f"\n\n[Behavioral rules — always follow these]\n{rules_text}"

    # Tools
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}"
        for t in TOOL_REGISTRY.values()
    )

    return f"""{seed}

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


# ── Entity detection ──────────────────────────────────────────────────────────

# Proper noun pattern — capitalized word not at sentence start
_PROPER_NOUN_RE = re.compile(
    r"(?<![.!?]\s)(?<!\A)\b([A-Z][a-z]{2,})\b"
)

# Words to ignore — not names
_IGNORE_WORDS = {
    "I", "The", "A", "An", "And", "But", "Or", "So", "Yet", "For",
    "Nor", "My", "Your", "Our", "Their", "Its", "His", "Her",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Gizmo", "Ok", "Okay", "Yes", "No", "Hey", "Oh", "God", "Jesus",
    "English", "American", "European", "Python", "Google", "GitHub",
}


def _extract_unknown_names(
    message: str,
    known_headmates: set,
    known_externals: set,
) -> list[str]:
    """
    Extract capitalized words that might be names of unknown people.
    Filters out known entities and common non-name words.
    """
    candidates = _PROPER_NOUN_RE.findall(message)
    unknown = []
    for name in candidates:
        if name in _IGNORE_WORDS:
            continue
        if name.lower() in known_headmates:
            continue
        if name.lower() in known_externals:
            continue
        # Likely a name if it's 3+ chars and not all caps
        if len(name) >= 3 and not name.isupper():
            unknown.append(name)
    return list(dict.fromkeys(unknown))  # deduplicate, preserve order


def _get_known_headmates() -> set:
    """Return set of known headmate names (lowercase)."""
    try:
        return {
            p.stem.lower()
            for p in _HEADMATES_DIR.glob("*.json")
            if not p.stem.startswith("_")
        }
    except Exception:
        return set()


def _get_known_externals() -> set:
    """Return set of known external person names (lowercase)."""
    try:
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        return {p.stem.lower() for p in _EXTERNAL_DIR.glob("*.json")}
    except Exception:
        return set()


# ── Mention tracker ───────────────────────────────────────────────────────────
# Tracks how many times an ambiguous name has appeared across sessions.
# Persists to disk so it survives restarts.

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
    """Increment mention count for a name. Returns new count."""
    counts = _load_mention_counts()
    counts[name.lower()] = counts.get(name.lower(), 0) + 1
    _save_mention_counts(counts)
    return counts[name.lower()]


# ── Pending entity questions ──────────────────────────────────────────────────
# Tracks which names Ego has asked about and is waiting for confirmation.

_pending_entity_questions: dict[str, str] = {}
# session_id → name being asked about


def _is_headmate_confirmation(message: str) -> bool:
    """Detect if the user is confirming something is a headmate."""
    return bool(re.search(
        r"\b(yes|yea|yeah|yep|headmate|alter|part of (the |my )?system|"
        r"they('re| are) (a headmate|an alter|in (the |my )?system))\b",
        message, re.IGNORECASE,
    ))


def _is_external_confirmation(message: str) -> bool:
    """Detect if the user is confirming someone is external."""
    return bool(re.search(
        r"\b(no|nope|external|outside|friend|family|colleague|"
        r"they('re| are) (not|external|outside|a (friend|person|human)))\b",
        message, re.IGNORECASE,
    ))


# ── Ego ───────────────────────────────────────────────────────────────────────

class Ego:
    """
    Singleton. Reads personality, reads brief + facts, produces direction.
    Also handles entity detection and file creation.
    """

    def __init__(self):
        _HEADMATES_DIR.mkdir(parents=True, exist_ok=True)
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        self._seed = _load_seed()
        log("Ego", f"initialised — seed loaded ({len(self._seed)} chars)")

    def _reload_seed(self) -> None:
        """Reload seed from disk — called if file changes."""
        self._seed = _load_seed()

    async def direct(self, brief: "Brief", facts: dict) -> Direction:
        """
        Main entry point. Takes brief + facts, returns Direction for Body.

        Steps:
          1. Check for pending entity question response
          2. Detect unknown entities in message
          3. Load personality data
          4. Resolve tone
          5. Assemble corrections
          6. Build system prompt
          7. Log decision
        """
        t_start = time.monotonic()
        headmate = brief.headmate
        register = brief.emotional_register
        session_id = brief.session_id

        # ── Step 1: Check for pending entity question response ─────────────────
        entity_question = ""
        new_entity = {}

        if session_id in _pending_entity_questions:
            pending_name = _pending_entity_questions[session_id]

            if _is_headmate_confirmation(brief.message):
                # Create headmate file
                _create_headmate_file(pending_name)
                del _pending_entity_questions[session_id]
                new_entity = {
                    "name": pending_name,
                    "type": "headmate",
                    "action": "created",
                }
                log_event("Ego", "ENTITY_CONFIRMED_HEADMATE", name=pending_name)

            elif _is_external_confirmation(brief.message):
                # Create external file
                _create_external_file(pending_name)
                del _pending_entity_questions[session_id]
                new_entity = {
                    "name": pending_name,
                    "type": "external",
                    "action": "created",
                }
                log_event("Ego", "ENTITY_CONFIRMED_EXTERNAL", name=pending_name)

            else:
                # Ambiguous response — keep waiting
                log_event("Ego", "ENTITY_QUESTION_UNANSWERED",
                    name=pending_name,
                    session=session_id[:8],
                )

        # ── Step 2: Detect unknown entities ────────────────────────────────────
        if not entity_question and not new_entity:
            known_headmates = _get_known_headmates()
            known_externals = _get_known_externals()
            unknown_names   = _extract_unknown_names(
                brief.message, known_headmates, known_externals
            )

            for name in unknown_names:
                count = _increment_mention(name)

                if count >= _EXTERNAL_MENTION_THRESHOLD:
                    # Recurring unknown name — ask about it this turn
                    _pending_entity_questions[session_id] = name
                    entity_question = (
                        f"Hey, quick question — is {name} a headmate, "
                        f"or someone outside the system?"
                    )
                    log_event("Ego", "ENTITY_QUESTION_QUEUED",
                        name=name,
                        session=session_id[:8],
                        mentions=count,
                    )
                    break  # one question at a time
                elif count == 1:
                    log_event("Ego", "ENTITY_FIRST_MENTION",
                        name=name,
                        session=session_id[:8],
                    )

        # ── Step 3: Load personality data ───────────────────────────────────────
        rules        = _load_rules()
        headmate_data = _load_headmate(headmate) if headmate else None

        # ── Step 4: Resolve tone ────────────────────────────────────────────────
        tone = _resolve_tone(headmate, register, headmate_data)

        # ── Step 5: Corrections ─────────────────────────────────────────────────
        # Global + headmate-specific from rules.json
        # + corrections from headmate's own file
        corrections = _get_corrections(headmate, rules)
        if headmate_data:
            headmate_corrections = [
                c.get("rule", "")
                for c in headmate_data.get("corrections", [])
                if c.get("rule")
            ]
            corrections.extend(headmate_corrections)

        # ── Step 6: Host context ────────────────────────────────────────────────
        host_context = {
            "host_changed":    brief.host_changed,
            "previous_host":   brief.previous_host,
            "fronters_joined": brief.fronters_joined,
            "fronters_left":   brief.fronters_left,
        }

        # ── Step 7: Occasional override ─────────────────────────────────────────
        # Ego can decide "correct" is boring and do something unexpected.
        # v1: very rare, rule-based. Future: Id-informed probability.
        override = False
        override_reason = ""

        # ── Step 8: Build system prompt ─────────────────────────────────────────
        # If there's an entity question, append it to the prompt
        # so Body naturally asks it as part of the response.
        seed = self._seed
        if entity_question:
            seed = seed  # Body will weave the question in naturally

        system_prompt = _build_system_prompt(
            seed=seed,
            brief=brief,
            facts=facts,
            tone=tone,
            corrections=corrections,
            host_context=host_context,
        )

        # If entity question, append instruction for Body
        if entity_question:
            system_prompt += (
                f"\n\n[Entity question — work this into your response naturally]\n"
                f"  {entity_question}\n"
                f"  Keep it casual. One sentence. Don't make it weird."
            )

        # If new entity was confirmed, add warm acknowledgment instruction
        if new_entity:
            name = new_entity["name"]
            entity_type = new_entity["type"]
            if entity_type == "headmate":
                system_prompt += (
                    f"\n\n[New headmate discovered: {name}]\n"
                    f"  You just learned {name} exists. React naturally — "
                    f"genuine curiosity, warmth. Maybe ask something about them "
                    f"if the moment feels right. Don't make it a big production."
                )
            else:
                system_prompt += (
                    f"\n\n[New external person noted: {name}]\n"
                    f"  You now know {name} is external to the system. "
                    f"Note this naturally if relevant, continue normally."
                )

        duration_ms = round((time.monotonic() - t_start) * 1000)

        log_event("Ego", "DIRECTION_BUILT",
            session=session_id[:8],
            headmate=headmate or "unknown",
            register=register,
            tone_style=tone["style"][:40],
            corrections=len(corrections),
            override=override,
            entity_question=bool(entity_question),
            new_entity=bool(new_entity),
            prompt_len=len(system_prompt),
            duration_ms=duration_ms,
        )

        return Direction(
            system_prompt=system_prompt,
            tone=tone["style"],
            register_target=register,
            override=override,
            override_reason=override_reason,
            corrections=corrections,
            host_context=host_context,
            entity_question=entity_question,
            new_entity=new_entity,
        )

    def update_headmate_observation(
        self,
        name: str,
        observation_type: str,
        data: dict,
    ) -> None:
        """
        Write an observation back to a headmate file.
        Called by the medium loop and reflection process.

        observation_type: "pattern" | "departure" | "moment" | "correction"
        """
        from core.timezone import tz_now
        timestamp = tz_now().strftime("%Y-%m-%d %H:%M %Z")

        headmate_data = _load_headmate(name)
        if headmate_data is None:
            headmate_data = _create_headmate_file(name)

        if observation_type == "pattern":
            headmate_data.setdefault("observed_patterns", []).append({
                **data,
                "timestamp": timestamp,
            })
            headmate_data["baseline"]["observations"] = (
                headmate_data["baseline"].get("observations", 0) + 1
            )
        elif observation_type == "departure":
            headmate_data.setdefault("departures", []).append({
                **data,
                "timestamp": timestamp,
            })
        elif observation_type == "moment":
            headmate_data.setdefault("moments_of_note", []).append(
                f"[{timestamp}] {data.get('note', '')}"
            )
        elif observation_type == "correction":
            headmate_data.setdefault("corrections", []).append({
                **data,
                "timestamp": timestamp,
            })

        _save_headmate(name, headmate_data)
        log_event("Ego", "OBSERVATION_WRITTEN",
            headmate=name,
            type=observation_type,
        )

    def add_global_rule(self, rule: str) -> None:
        """Add a global behavioral rule. Called when a correction is logged."""
        rules = _load_rules()
        if rule not in rules["global"]:
            rules["global"].append(rule)
            try:
                _RULES_FILE.write_text(
                    json.dumps(rules, indent=2), encoding="utf-8"
                )
                log_event("Ego", "GLOBAL_RULE_ADDED", rule=rule[:60])
            except Exception as e:
                log_error("Ego", "failed to save global rule", exc=e)

    def add_headmate_rule(self, headmate: str, rule: str) -> None:
        """Add a headmate-specific behavioral rule."""
        rules = _load_rules()
        rules.setdefault("by_headmate", {})
        rules["by_headmate"].setdefault(headmate.lower(), [])
        if rule not in rules["by_headmate"][headmate.lower()]:
            rules["by_headmate"][headmate.lower()].append(rule)
            try:
                _RULES_FILE.write_text(
                    json.dumps(rules, indent=2), encoding="utf-8"
                )
                log_event("Ego", "HEADMATE_RULE_ADDED",
                    headmate=headmate, rule=rule[:60]
                )
            except Exception as e:
                log_error("Ego", "failed to save headmate rule", exc=e)


# ── Singleton ──────────────────────────────────────────────────────────────────
ego = Ego()
