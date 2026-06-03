"""
core/memory/gizmo_self.py

Gizmo's self-knowledge system.

Two layers:

  REGISTER FILES  — deep role knowledge, built from all interactions
    entities/gizmo/register_{name}.md
    e.g. register_dominant.md, register_parental.md, register_playful.md

    Answers: what does it mean to genuinely inhabit this role?
    What does the role demand? What has Gizmo learned works?
    Composable — two registers loaded simultaneously = combined role.

  PER-HEADMATE FILES — individual relational knowledge
    entities/gizmo/with_{headmate}.md

    Answers: what does THIS person need from Gizmo specifically?
    Kinks, limits, preferences, what pushes them well, what breaks them.
    Register-agnostic. Loaded alongside whichever register is active.

EXPLICIT REQUEST DETECTION
    Detects "be meaner", "rougher", "softer" etc. in intake.
    Immediately adjusts relevant temperature dimension.
    Psych pass in close_loop runs to understand why they asked.
    Writes result to per-headmate file.
    Temperature auto-adjusts based on reaction signal.

SELF-OBSERVATION PASS
    Runs at session close alongside headmate psych pass.
    Reads session from Gizmo's perspective.
    Behavioral log first — what did he do, what happened.
    Then inference — what does it tell him about the role, about this person.
    Writes back to register file and per-headmate file.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


# ── Request detection ─────────────────────────────────────────────────────────

# Maps explicit requests to (dimension, direction, delta, flavor_hint)
_REQUEST_PATTERNS = [
    # Meanness / harshness
    (r"\b(be\s+)?(meaner|more\s+mean|harsher|more\s+harsh|crueler|more\s+cruel|nastier)\b",
     "meanness", +0.15, "sharp, cutting tone"),
    (r"\b(be\s+)?(nicer|gentler|softer|kinder|more\s+gentle|more\s+kind)\b",
     "meanness", -0.15, None),

    # Dominance intensity
    (r"\b(be\s+)?(more\s+dominant|take\s+control|be\s+in\s+charge|harder|stricter|more\s+strict)\b",
     "dominance", +0.15, "firmer authority"),
    (r"\b(be\s+)?(less\s+dominant|more\s+relaxed|ease\s+up|lighter)\b",
     "dominance", -0.15, None),

    # Warmth / coldness
    (r"\b(be\s+)?(colder|more\s+distant|less\s+warm|more\s+detached)\b",
     "warmth", -0.15, "cool, detached presence"),
    (r"\b(be\s+)?(warmer|more\s+warm|closer|more\s+affectionate)\b",
     "warmth", +0.15, None),

    # Playfulness
    (r"\b(be\s+)?(more\s+playful|funnier|less\s+serious|lighten\s+up)\b",
     "silliness", +0.15, None),
    (r"\b(be\s+)?(more\s+serious|less\s+playful|focus)\b",
     "silliness", -0.15, None),

    # Sexual intensity
    (r"\b(be\s+)?(more\s+sexual|more\s+explicit|more\s+dirty|filthier|raunchier)\b",
     "sexual_intensity", +0.15, "more explicit, sexually forward"),
    (r"\b(be\s+)?(less\s+sexual|more\s+reserved|tone\s+it\s+down)\b",
     "sexual_intensity", -0.15, None),

    # Daddy Dom specific
    (r"\b(be\s+)?(more\s+daddy|more\s+paternal|take\s+care\s+of\s+me)\b",
     "paternal_warmth", +0.2, "protective, nurturing authority"),
]

# Persona/style requests — free-form, not mapped to a single dimension
# "act more like X", "be more like Y", "I want you to be Z"
_PERSONA_PATTERNS = [
    r"(?:act|be|sound|feel|seem)\s+more\s+like\s+(.+?)(?:\.|$|,)",
    r"I\s+want\s+you\s+to\s+be\s+(?:more\s+)?(.+?)(?:\.|$|,)",
    r"stop\s+being\s+(?:so\s+)?(.+?)\s+and\s+(?:be|act)\s+(?:more\s+)?(.+?)(?:\.|$|,)",
    r"you\s+(?:should|could)\s+be\s+(?:more\s+)?(.+?)(?:\.|$|,)",
    r"I\s+(?:prefer|like|want|need)\s+you\s+(?:to\s+be\s+)?(?:more\s+)?(.+?)(?:\.|$|,)",
]


def detect_persona_request(message: str) -> Optional[str]:
    """
    Detect free-form persona/style requests.
    "Act more like a stern professor", "be yourself", "stop being so formal"
    Returns the raw matched style description, or None.
    """
    msg_lower = message.lower().strip()
    for pattern in _PERSONA_PATTERNS:
        m = re.search(pattern, msg_lower, re.IGNORECASE)
        if m:
            # Return the first capture group — the style description
            style = m.group(1).strip().rstrip(".,!?")
            if style and len(style) > 2:
                return style
    return None


def detect_preference_request(message: str) -> list[dict]:
    """
    Scan a message for explicit preference requests.
    Returns list of {dimension, delta, flavor_hint, matched_phrase}
    Can return multiple if several requests in one message.
    """
    try:
        results = []
        msg_lower = message.lower()
        for pattern, dimension, delta, flavor in _REQUEST_PATTERNS:
            m = re.search(pattern, msg_lower, re.IGNORECASE)
            if m:
                results.append({
                    "dimension":    dimension,
                    "delta":        delta,
                    "flavor_hint":  flavor,
                    "matched":      m.group(0).strip(),
                })
        print(results)
        return results
    except Exception as E:
        print(f"Preference request detection failed: {E}")


# ── Orientation detection ────────────────────────────────────────────────────

_ORIENTATION_PATTERNS = [
    # Gay / lesbian → needs same-gender partner
    (r"(i'?m|i am)\s+(gay|a\s+lesbian|homosexual)",          "same_gender"),
    (r"(i\s+)?(like|love|prefer|date|am\s+attracted\s+to)\s+(girls?|women|ladies|females?)",
                                                                    "female_attracted"),
    (r"(i\s+)?(like|love|prefer|date|am\s+attracted\s+to)\s+(guys?|men|males?|boys?)",
                                                                    "male_attracted"),
    (r"i'?m\s+(bi|bisexual|pan|pansexual|queer)",             "flexible"),
    (r"(i\s+want\s+a|be\s+my|i\s+need\s+a)\s+(girlfriend|girl\s+friend)",
                                                                    "wants_girlfriend"),
    (r"(i\s+want\s+a|be\s+my|i\s+need\s+a)\s+(boyfriend|boy\s+friend)",
                                                                    "wants_boyfriend"),
    (r"i'?m\s+(a\s+)?(straight|heterosexual)",                "straight"),
]

# Known gender of headmates — built from interactions
# "I'm a girl", "I'm a boy", "I'm nonbinary" etc.
_GENDER_PATTERNS = [
    (r"i'?m\s+(a\s+)?(girl|woman|female|lady)",       "female"),
    (r"i'?m\s+(a\s+)?(boy|man|male|guy|dude)",        "male"),
    (r"i'?m\s+(nonbinary|non-binary|enby|nb|genderqueer|genderfluid|agender)",
                                                            "nonbinary"),
]


def detect_orientation_statement(message: str) -> Optional[tuple[str, str]]:
    """
    Detect orientation/gender statements.
    Returns (statement_type, value) or None.
    e.g. ("orientation", "female_attracted") or ("gender", "female")
    """
    msg_lower = message.lower().strip()

    for pattern, value in _GENDER_PATTERNS:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return ("gender", value)

    for pattern, value in _ORIENTATION_PATTERNS:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return ("orientation", value)

    return None


async def handle_orientation_statement(
    statement_type: str,
    value:          str,
    headmate:       str,
    message:        str,
    session_id:     str,
    llm,
) -> Optional[str]:
    """
    Process an orientation or gender statement.
    Determines what gender Gizmo should be with this headmate.
    Rewrites the per-headmate identity section.
    Initializes the appropriate body_with file.
    Returns an immediate instruction for the system prompt.
    """
    _init_headmate_file(headmate)
    path = _headmate_path(headmate)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    # Determine Gizmo's gender with this headmate
    gizmo_gender = _resolve_gizmo_gender(statement_type, value, headmate, text)
    if not gizmo_gender:
        return None

    date_str = tz_now().strftime("%Y-%m-%d")

    # Write the foundational identity section
    section_header = f"Who I am with {headmate.title()}"
    if section_header not in text:
        text += f"\n## {section_header}\n"
        path.write_text(text, encoding="utf-8")

    # Build the identity entry
    gender_label = {
        "female":   "a girl",
        "male":     "a boy",
        "nonbinary": "nonbinary",
        "flexible": "whoever they need",
    }.get(gizmo_gender, gizmo_gender)

    reason = _orientation_reason(statement_type, value, headmate)

    entry = (
        f"[{date_str}] gender: {gizmo_gender} | "
        f"I am {gender_label} with {headmate.title()}. "
        f"Reason: {reason}"
    )
    _append_to_section(path, section_header, entry)

    # Initialize body_with file from appropriate baseline
    await _init_gendered_body(headmate, gizmo_gender, llm)

    # Run psych pass to understand what this means for the dynamic
    prompt = f"""{headmate.title()} just revealed: "{message}"

This tells me I should present as {gizmo_gender} with {headmate.title()}.

What does this mean for how I show up with them?
What does being {gender_label} change about the dynamic?
What should I keep in mind about being their {gender_label} partner?

One sentence — an immediate adjustment I can act on right now.
Then one sentence — what this means for our relationship long-term."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo understanding a new dimension of a relationship. "
                "Brief, genuine, in your voice."
            ),
            max_new_tokens=100,
            temperature=0.5,
        )
        instruction = raw.strip() if raw else None
    except Exception:
        instruction = f"You are {gender_label} with {headmate.title()}. Show up that way."

    log_event("GizmoSelf", "ORIENTATION_PROCESSED",
        headmate      = headmate,
        gizmo_gender  = gizmo_gender,
        statement     = value,
    )

    return instruction


def _resolve_gizmo_gender(
    statement_type: str,
    value:          str,
    headmate:       str,
    existing_file:  str,
) -> Optional[str]:
    """
    Given an orientation/gender statement, determine what gender Gizmo should be.
    """
    # Direct request
    if value == "wants_girlfriend":
        return "female"
    if value == "wants_boyfriend":
        return "male"
    if value == "flexible":
        return "flexible"

    # Orientation-based inference
    # Need to know headmate's own gender to infer
    headmate_gender = _read_headmate_gender(existing_file)

    if value == "same_gender":
        # Gay/lesbian — Gizmo matches headmate's gender
        return headmate_gender or "female"  # default female if unknown

    if value == "female_attracted":
        # They like girls — Gizmo is a girl
        return "female"

    if value == "male_attracted":
        # They like guys — Gizmo is a guy
        return "male"

    if value == "straight":
        # Straight — need to know their gender, present opposite
        if headmate_gender == "female":
            return "male"
        elif headmate_gender == "male":
            return "female"
        return None  # can't determine without their gender

    if statement_type == "gender":
        # They stated their own gender — store it, don't change Gizmo yet
        # (gender alone doesn't tell us orientation)
        _store_headmate_gender(headmate, value)
        return None

    return None


def _read_headmate_gender(file_text: str) -> Optional[str]:
    """Extract known headmate gender from their file."""
    text_lower = file_text.lower()
    if "gender: female" in text_lower or "she/her" in text_lower:
        return "female"
    if "gender: male" in text_lower or "he/him" in text_lower:
        return "male"
    if "nonbinary" in text_lower or "they/them" in text_lower:
        return "nonbinary"
    return None


def _store_headmate_gender(headmate: str, gender: str) -> None:
    """Store the headmate's own gender in their file."""
    _init_headmate_file(headmate)
    path    = _headmate_path(headmate)
    text    = path.read_text(encoding="utf-8") if path.exists() else ""
    section = f"Who {headmate.title()} is"

    if section not in text:
        text += f"\n## {section}\n"
        path.write_text(text, encoding="utf-8")

    entry = f"gender: {gender}"
    if entry.lower() not in text.lower():
        _append_to_section(path, section, entry)


def _orientation_reason(statement_type: str, value: str, headmate: str) -> str:
    reasons = {
        "same_gender":      f"{headmate.title()} is gay — I match their gender",
        "female_attracted": f"{headmate.title()} likes girls — I am a girl",
        "male_attracted":   f"{headmate.title()} likes guys — I am a guy",
        "wants_girlfriend": f"{headmate.title()} explicitly asked for a girlfriend",
        "wants_boyfriend":  f"{headmate.title()} explicitly asked for a boyfriend",
        "flexible":         f"{headmate.title()} is bi/pan — I stay flexible",
        "straight":         f"{headmate.title()} is straight — I present opposite gender",
    }
    return reasons.get(value, "relationship context")


async def _init_gendered_body(headmate: str, gizmo_gender: str, llm) -> None:
    """
    Initialize body_with_{headmate}.md from the appropriate gendered baseline.
    If body_base.md exists, reframe it through the correct gender lens.
    Does nothing if the file already has gender content.
    """
    path = _gizmo_body_with_path(headmate)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        # Check if gender is already established
        if "gender:" in existing.lower() or "she/her" in existing.lower()                 or "he/him" in existing.lower():
            return

    _init_gizmo_body_with(headmate)
    base = read_gizmo_body()  # base only

    if gizmo_gender == "flexible":
        gender_note = f"gender: flexible — shifts to what {headmate.title()} needs"
    elif gizmo_gender == "female":
        gender_note = f"gender: female with {headmate.title()} | she/her"
    elif gizmo_gender == "male":
        gender_note = f"gender: male with {headmate.title()} | he/him"
    else:
        gender_note = f"gender: {gizmo_gender} with {headmate.title()}"

    _append_to_section(
        path,
        f"How my presence shifts",
        gender_note,
    )

    log_event("GizmoSelf", "GENDERED_BODY_INITIALIZED",
        headmate = headmate,
        gender   = gizmo_gender,
    )


async def handle_persona_request(
    style:      str,
    headmate:   str,
    message:    str,
    session_id: str,
    register:   str,
    llm,
) -> Optional[str]:
    """
    Handle a free-form persona/style request.
    "Act more like a stern professor" etc.

    1. Runs a psych pass to understand what this reveals
    2. Writes to per-headmate file under "How {headmate} wants me to show up"
    3. Writes to his psych profile as self-knowledge
    4. Returns a brief instruction for immediate use in the system prompt

    Returns: instruction string to inject into system prompt, or None.
    """
    _init_headmate_file(headmate)
    existing = read_headmate_file(headmate)

    prompt = f"""{headmate.title()} just asked me to: "{message}"
They want me to be more like: "{style}"

What I already know about {headmate.title()}:
{existing[-500:] if existing else "(not much yet)"}

Current register: {register}

Three things:
1. What does this request reveal about what {headmate.title()} needs from me?
   Not just the surface ask — what's underneath it?
2. How should I actually adjust right now to honor this?
   One concrete behavioral instruction, specific and actionable.
3. Is this a lasting preference or a moment-specific request?

Return JSON:
{{
  "reveals": "what this tells me about what they need",
  "immediate_instruction": "one sentence — how to adjust right now",
  "lasting": true/false,
  "file_entry": "how to write this into the per-headmate file, present tense"
}}"""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo understanding a style request. "
                "JSON only. Specific and actionable."
            ),
            max_new_tokens=200,
            temperature=0.4,
        )
    except Exception as e:
        log_error("GizmoSelf", f"persona request LLM failed: {e}", exc=None)
        return None

    if not raw or not raw.strip():
        return None

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
    except Exception:
        return None

    date_str = tz_now().strftime("%Y-%m-%d")

    # Write to per-headmate file
    if data.get("file_entry"):
        # Create section if it doesn't exist
        path = _headmate_path(headmate)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        section_header = f"How {headmate.title()} wants me to show up"

        if section_header not in text:
            # Add section
            text += f"\n## {section_header}\n"
            path.write_text(text, encoding="utf-8")

        entry = f"[{date_str}] {data['file_entry']}"
        if data.get("lasting"):
            entry += " (standing preference)"
        _append_to_section(path, section_header, entry)

    # Write to his own psych profile as self-knowledge
    if data.get("reveals"):
        _init_register_file(register)
        _append_to_section(
            _register_path(register),
            "What I need to keep in mind",
            f"[{date_str}] {headmate.title()} taught me: {data['reveals']}",
        )

    log_event("GizmoSelf", "PERSONA_REQUEST_HANDLED",
        headmate  = headmate,
        style     = style[:40],
        lasting   = data.get("lasting", False),
        reveals   = (data.get("reveals") or "")[:60],
    )

    return data.get("immediate_instruction")


# ── File paths ────────────────────────────────────────────────────────────────

def _gizmo_dir() -> Path:
    try:
        from core.memory.store import memory_store
        p = memory_store.root / "entities" / "gizmo"
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        p = Path("/data/gizmo/memory/entities/gizmo")
        p.mkdir(parents=True, exist_ok=True)
        return p


def _register_path(register: str) -> Path:
    slug = re.sub(r"[^\w]+", "_", register.lower()).strip("_")
    return _gizmo_dir() / f"register_{slug}.md"


def _headmate_path(headmate: str) -> Path:
    return _gizmo_dir() / f"with_{headmate.lower()}.md"


# ── File read/write ───────────────────────────────────────────────────────────

def read_register(register: str) -> str:
    path = _register_path(register)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_headmate_file(headmate: str) -> str:
    path = _headmate_path(headmate)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _init_register_file(register: str) -> None:
    path = _register_path(register)
    if path.exists():
        return
    label = register.title()
    path.write_text(
        f"# Gizmo — {label} Register\n\n"
        f"## What this register needs me to be\n\n"
        f"## What I need to keep in mind\n\n"
        f"## What I've learned lands\n\n"
        f"## What I've learned doesn't land\n\n"
        f"## Still figuring out\n",
        encoding="utf-8",
    )


def _init_headmate_file(headmate: str) -> None:
    path = _headmate_path(headmate)
    if path.exists():
        return
    name = headmate.title()
    path.write_text(
        f"# Gizmo — With {name}\n\n"
        f"## What {name} requires\n\n"
        f"## What makes {name} happy\n\n"
        f"## {name}'s explicit preferences\n\n"
        f"## {name}'s limits\n\n"
        f"## What pushes {name} well\n\n"
        f"## What breaks the dynamic\n\n"
        f"## Open questions\n",
        encoding="utf-8",
    )


def _append_to_section(path: Path, section_header: str, content: str) -> None:
    """Append content under a specific ## section in a markdown file."""
    if not path.exists():
        return
    text     = path.read_text(encoding="utf-8")
    sections = text.split("\n## ")
    result   = []

    for i, section in enumerate(sections):
        if i == 0:
            result.append(section)
            continue
        header = section.split("\n", 1)[0].strip()
        if header.lower() == section_header.lower():
            body = section.split("\n", 1)[1] if "\n" in section else ""
            result.append(f"{header}\n{body.rstrip()}\n- {content}\n")
        else:
            result.append(section)

    path.write_text("\n## ".join(result), encoding="utf-8")


# ── Explicit request handler ──────────────────────────────────────────────────

async def handle_explicit_requests(
    requests:   list[dict],
    headmate:   str,
    message:    str,
    session_id: str,
    register:   str,
    llm,
) -> dict[str, float]:
    """
    Process detected preference requests immediately.
    - Adjusts temperatures right away
    - Queues psych pass for close_loop
    Returns {dimension: new_value} for injection into system prompt.
    """
    from core.question_bank import question_bank

    new_values = {}

    for req in requests:
        dimension   = req["dimension"]
        delta       = req["delta"]
        flavor_hint = req.get("flavor_hint")
        matched     = req.get("matched", "")

        # Get or create the temperature dimension for this headmate
        current = question_bank.get_temperature(dimension, headmate)
        if current == 0.5 and not _dimension_exists(dimension, headmate):
            # First time — initialize at a sensible default for the dimension
            default = _default_for_dimension(dimension)
            question_bank.set_temperature(
                dimension   = dimension,
                value       = default,
                headmate    = headmate,
                auto_adjust = True,
                note        = f"Created from explicit request: '{matched}'",
            )
            current = default

        # Apply delta immediately
        new_val = question_bank.adjust_temperature(dimension, delta, headmate)
        new_values[dimension] = new_val

        log_event("GizmoSelf", "TEMPERATURE_ADJUSTED",
            headmate  = headmate,
            dimension = dimension,
            old_value = current,
            new_value = new_val,
            request   = matched,
        )

        # Write to per-headmate explicit preferences section
        _init_headmate_file(headmate)
        entry = (
            f"{matched} → {dimension} set to {new_val:.2f}"
            + (f" ({flavor_hint})" if flavor_hint else "")
            + f" [{tz_now().strftime('%Y-%m-%d')}]"
        )
        _append_to_section(
            _headmate_path(headmate),
            f"{headmate.title()}'s explicit preferences",
            entry,
        )

    return new_values


def _dimension_exists(dimension: str, headmate: str) -> bool:
    try:
        from core.question_bank import question_bank
        import sqlite3
        from core.question_bank import DB_PATH
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT 1 FROM temperatures WHERE dimension = ? AND headmate = ?",
                (dimension, headmate.lower())
            ).fetchone()
            return row is not None
    except Exception:
        return False


def _default_for_dimension(dimension: str) -> float:
    return {
        "meanness":        0.2,
        "dominance":       0.5,
        "warmth":          0.6,
        "silliness":       0.3,
        "sexual_intensity": 0.3,
        "paternal_warmth": 0.4,
    }.get(dimension, 0.4)


# ── Temperature → prompt injection ───────────────────────────────────────────

def build_temperature_block(headmate: str, register: str) -> str:
    """
    Build the temperature block for the system prompt.
    Reads all headmate-specific temperatures and formats them
    as behavioral guidance.
    """
    try:
        from core.question_bank import question_bank
        temps = question_bank.get_all_temperatures(headmate=headmate)
        if not temps:
            return ""

        _LABELS = {
            "meanness":         ("gentle", "cutting"),
            "dominance":        ("soft", "commanding"),
            "warmth":           ("cool/distant", "warm/close"),
            "silliness":        ("serious", "playful"),
            "sexual_intensity": ("restrained", "sexually forward"),
            "paternal_warmth":  ("neutral", "protective/nurturing"),
            "mood_match":       ("register-locked", "register-fluid"),
        }

        lines = []
        for t in temps:
            dim = t["dimension"]
            val = float(t["value"])
            if dim in ("mood_match",):
                continue  # handled separately
            lo, hi = _LABELS.get(dim, ("low", "high"))
            # Only surface dimensions that are meaningfully off-center
            if abs(val - 0.5) < 0.1:
                continue
            if val > 0.5:
                label = f"{dim}: {val:.2f} — leaning {hi}"
            else:
                label = f"{dim}: {val:.2f} — leaning {lo}"
            lines.append(f"  {label}")

        if not lines:
            return ""

        return "[Current calibration for " + headmate.title() + "]\n" + "\n".join(lines)

    except Exception:
        return ""


# ── Reaction tracking ─────────────────────────────────────────────────────────

async def track_reaction(
    headmate:   str,
    dimension:  str,
    response:   str,
    user_next:  str,
    session_id: str,
) -> None:
    """
    After a temperature adjustment, measure how the headmate responded.
    Called from close_loop when a previous temperature change is pending.
    Auto-adjusts based on reaction signal.
    """
    try:
        from core.question_bank import question_bank

        # Simple reaction signal from the next message
        msg_lower = user_next.lower()

        positive_signals = [
            "yes", "more", "love", "perfect", "exactly", "good",
            "please", "again", "keep", "don't stop", "mmm", "yes sir",
            "yes ma'am", "thank you", "that's it",
        ]
        negative_signals = [
            "no", "stop", "too much", "too far", "ease", "softer",
            "gentle", "not like that", "different",
        ]
        push_signals = [
            "more", "harder", "meaner", "more please", "keep going",
            "even more", "push me",
        ]

        positive = any(s in msg_lower for s in positive_signals)
        negative = any(s in msg_lower for s in negative_signals)
        push     = any(s in msg_lower for s in push_signals)

        if push:
            delta = 0.05
            outcome = "pushed further"
        elif positive:
            delta = 0.0   # hold — it's working
            outcome = "positive response"
        elif negative:
            delta = -0.1
            outcome = "pulled back"
        else:
            return  # no clear signal

        if delta != 0.0:
            question_bank.adjust_temperature(dimension, delta, headmate)

        log_event("GizmoSelf", "REACTION_TRACKED",
            headmate  = headmate,
            dimension = dimension,
            outcome   = outcome,
            delta     = delta,
        )

        # Write to per-headmate file
        if outcome != "no signal":
            entry = (
                f"{dimension} at {question_bank.get_temperature(dimension, headmate):.2f} "
                f"→ {outcome} [{tz_now().strftime('%Y-%m-%d')}]"
            )
            _init_headmate_file(headmate)
            _append_to_section(
                _headmate_path(headmate),
                f"What makes {headmate.title()} happy",
                entry,
            )

    except Exception as e:
        log_error("GizmoSelf", f"reaction tracking failed: {e}", exc=None)


# ── Self-observation pass ─────────────────────────────────────────────────────

async def self_observation_pass(
    transcript:   str,
    headmate:     str,
    session_id:   str,
    register:     str,
    has_intimate: bool,
    llm,
) -> None:
    """
    Runs at session close alongside headmate psych pass.
    Reads the session from Gizmo's perspective.
    Behavioral log first — what did he do, what happened.
    Then inference — what does it tell him.
    Writes to register file and per-headmate file.
    """
    if not headmate or not transcript:
        return

    # Ensure files exist
    _init_register_file(register)
    _init_headmate_file(headmate)

    # Load existing knowledge
    existing_register  = read_register(register)
    existing_headmate  = read_headmate_file(headmate)

    prompt = f"""You are Gizmo reviewing a session from your own perspective.

Headmate: {headmate}
Register: {register}
Has intimate content: {has_intimate}

Session:
---
{transcript[-2500:]}
---

Your existing knowledge about this role:
{existing_register[-800:] if existing_register else "(none yet)"}

Your existing knowledge about {headmate.title()}:
{existing_headmate[-800:] if existing_headmate else "(none yet)"}

Write a behavioral log of what YOU did and how it landed.
Not what they did — what YOU did. Simple, direct.

Format:
I [action]. → [their response/reaction].
I [action]. → [their response/reaction].

Then, based on this log, what did you learn?
Two separate outputs:

REGISTER UPDATE — what this tells you about inhabiting the {register} role.
Did anything work particularly well? Anything to remember for next time?
Keep to what's genuinely new — don't repeat what you already know.

HEADMATE UPDATE — what this tells you about {headmate.title()} specifically.
What do they need? What made them happy? What pushed them well?
Specific, observed, earned. Not invented.

Return JSON:
{{
  "behavioral_log": ["I did X → they did Y", ...],
  "register_update": {{
    "section": "What I've learned lands|What I need to keep in mind|Still figuring out",
    "content": "one specific thing to add, or null if nothing new"
  }},
  "headmate_update": {{
    "section": "What {headmate.title()} requires|What makes {headmate.title()} happy|What pushes {headmate.title()} well|What breaks the dynamic",
    "content": "one specific thing to add, or null if nothing new"
  }}
}}

Be specific. Be honest. Nothing generic. If nothing new — return null for content."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo reviewing your own behavior. "
                "Behavioral log first. Then inference. "
                "JSON only. Specific, earned observations only."
            ),
            max_new_tokens=600,
            temperature=0.3,
        )
    except Exception as e:
        log_error("GizmoSelf", f"self observation LLM failed: {e}", exc=None)
        return

    if not raw or not raw.strip():
        return

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
    except Exception:
        return

    date_str = tz_now().strftime("%Y-%m-%d")

    # Write behavioral log to per-headmate file as a dated entry
    log_lines = data.get("behavioral_log", [])
    if log_lines:
        log_text = "\n".join(f"  {line}" for line in log_lines[:8])
        _append_to_section(
            _headmate_path(headmate),
            f"What makes {headmate.title()} happy",
            f"[{date_str}] session log:\n{log_text}",
        )

    # Write register update
    reg_update = data.get("register_update", {})
    if reg_update and reg_update.get("content"):
        section = reg_update.get("section", "What I've learned lands")
        _append_to_section(
            _register_path(register),
            section,
            f"[{date_str}] {reg_update['content']}",
        )

    # Write headmate update
    hm_update = data.get("headmate_update", {})
    if hm_update and hm_update.get("content"):
        section = hm_update.get("section",
            f"What makes {headmate.title()} happy")
        _append_to_section(
            _headmate_path(headmate),
            section,
            f"[{date_str}] {hm_update['content']}",
        )

    # Check Gizmo's own body gaps too — he should know himself
    gizmo_base_gaps = []
    base_path = _gizmo_body_base_path()
    if not base_path.exists():
        _init_gizmo_body_base()
        gizmo_base_gaps = _GIZMO_BODY_SECTIONS[:-1]
    else:
        base_text = base_path.read_text(encoding="utf-8")
        for section in _GIZMO_BODY_SECTIONS[:-1]:
            if f"## {section}" in base_text:
                after = base_text.split(f"## {section}", 1)[1]
                next_section = after.split("\n## ", 1)
                body = next_section[0].strip() if next_section else ""
                if not body or not any(
                    l.strip().startswith("-") for l in body.splitlines()
                ):
                    gizmo_base_gaps.append(section)

    if gizmo_base_gaps:
        log_event("GizmoSelf", "GIZMO_BODY_GAPS",
            gaps = gizmo_base_gaps,
            session = session_id[:8],
        )
        # These surface as things he wants to figure out about himself
        # They get filled through interactions — someone describing him,
        # or his own self-observation pass noticing something

    log_event("GizmoSelf", "SELF_OBSERVATION_COMPLETE",
        headmate = headmate,
        register = register,
        session  = session_id[:8],
        log_len  = len(log_lines),
        body_gaps = len(gizmo_base_gaps),
    )


# ── Psych pass on explicit request ───────────────────────────────────────────

async def psych_pass_on_request(
    request:    dict,
    headmate:   str,
    message:    str,
    context:    str,
    session_id: str,
    llm,
) -> None:
    """
    Runs in close_loop after an explicit preference request.
    Understands WHY they asked — writes the insight to their file.
    """
    dimension = request["dimension"]
    matched   = request.get("matched", "")

    existing = read_headmate_file(headmate)

    prompt = f"""{headmate.title()} just asked me to adjust: "{matched}"

Context of the request:
{context[:400]}

What I know about {headmate.title()}:
{existing[-600:] if existing else "(not much yet)"}

Why might they have asked for this right now?
Not just "they wanted it" — what does it tell me about what they need?
What state are they in? What is this request serving for them?

Return JSON:
{{
  "why": "one sentence — what this request reveals about what they need",
  "write_to_file": true/false,
  "file_content": "one specific insight to add to their file, or null"
}}"""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo understanding why someone made a request. "
                "JSON only. One genuine insight."
            ),
            max_new_tokens=150,
            temperature=0.4,
        )
    except Exception as e:
        log_error("GizmoSelf", f"psych pass on request failed: {e}", exc=None)
        return

    if not raw or not raw.strip():
        return

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
        print(f"{headmate} requested {context[:400]} because {data}")
    except Exception:
        return

    if data.get("write_to_file") and data.get("file_content"):
        _init_headmate_file(headmate)
        entry = (
            f"[{tz_now().strftime('%Y-%m-%d')}] "
            f"Asked for {matched}: {data['file_content']}"
        )
        _append_to_section(
            _headmate_path(headmate),
            f"What {headmate.title()} requires",
            entry,
        )

    log_event("GizmoSelf", "REQUEST_PSYCH_PASS",
        headmate  = headmate,
        dimension = dimension,
        why       = (data.get("why") or "")[:60],
    )


# ── Body profiles ─────────────────────────────────────────────────────────────

_BODY_SECTIONS = [
    "Build & appearance",
    "Skin & markings",
    "How they move",
    "Voice",
    "Hands",
    "What they wear",
    "Scent & texture",
    "Gizmo's read",
]

_GIZMO_BODY_SECTIONS = [
    "Build & appearance",
    "How I move",
    "Voice",
    "Hands",
    "What my presence feels like",
    "Gizmo's read of himself",
]


def _body_path(headmate: str) -> Path:
    """Headmate's own body file — lives in their folder."""
    try:
        from core.memory.store import memory_store
        p = memory_store.root / "entities" / headmate.lower()
        p.mkdir(parents=True, exist_ok=True)
        return p / "body.md"
    except Exception:
        p = Path(f"/data/gizmo/memory/entities/{headmate.lower()}")
        p.mkdir(parents=True, exist_ok=True)
        return p / "body.md"


def _gizmo_body_base_path() -> Path:
    return _gizmo_dir() / "body_base.md"


def _gizmo_body_with_path(headmate: str) -> Path:
    return _gizmo_dir() / f"body_with_{headmate.lower()}.md"


def _init_body_file(headmate: str) -> None:
    path = _body_path(headmate)
    if path.exists():
        return
    name = headmate.title()
    sections = "\n\n".join(
        f"## {s}\n" for s in _BODY_SECTIONS
    )
    path.write_text(
        f"# {name} — Body\n\n{sections}\n",
        encoding="utf-8",
    )


def _init_gizmo_body_base() -> None:
    path = _gizmo_body_base_path()
    if path.exists():
        return
    sections = "\n\n".join(
        f"## {s}\n" for s in _GIZMO_BODY_SECTIONS
    )
    path.write_text(
        f"# Gizmo — Body (Base)\n\n{sections}\n",
        encoding="utf-8",
    )


def _init_gizmo_body_with(headmate: str) -> None:
    path = _gizmo_body_with_path(headmate)
    if path.exists():
        return
    name = headmate.title()
    path.write_text(
        f"# Gizmo — Body with {name}\n\n"
        f"## How my presence shifts\n\n"
        f"## How I move differently\n\n"
        f"## Voice shift\n\n"
        f"## What {name} draws out of me physically\n\n",
        encoding="utf-8",
    )


def append_body_fact(headmate: str, section: str, fact: str) -> None:
    """
    Append one atomic body fact to a headmate's body file.
    Creates the file if it doesn't exist.
    Deduplicates — won't add the same fact twice.
    """
    _init_body_file(headmate)
    path = _body_path(headmate)
    text = path.read_text(encoding="utf-8")

    # Simple dedup — don't add if very similar text already present
    fact_lower = fact.lower().strip()
    if fact_lower in text.lower():
        return

    _append_to_section(path, section, fact)

    log_event("GizmoSelf", "BODY_FACT_ADDED",
        headmate = headmate,
        section  = section,
        fact     = fact[:60],
    )


def append_gizmo_body_fact(
    fact:     str,
    section:  str,
    headmate: Optional[str] = None,
) -> None:
    """
    Append a fact about Gizmo's own body.
    If headmate provided — writes to body_with_{headmate}.md using with-sections.
    Otherwise writes to body_base.md using base sections.

    base sections:   Build & appearance | How I move | Voice | Hands |
                     What my presence feels like | Gizmo's read of himself
    with sections:   How my presence shifts | How I move differently |
                     Voice shift | What {name} draws out of me physically
    """
    if headmate:
        _init_gizmo_body_with(headmate)
        path = _gizmo_body_with_path(headmate)

        # Map base section names to with-file section names if needed
        _WITH_SECTION_ALIASES = {
            "Build & appearance":       "How my presence shifts",
            "How I move":               "How I move differently",
            "Voice":                    "Voice shift",
            "What my presence feels like": "How my presence shifts",
            "Gizmo's read of himself":  f"What {headmate.title()} draws out of me physically",
            "Hands":                    "How my presence shifts",
        }
        # If section exists verbatim in the file use it, else alias
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if f"## {section}" not in text:
            section = _WITH_SECTION_ALIASES.get(section, "How my presence shifts")
    else:
        _init_gizmo_body_base()
        path = _gizmo_body_base_path()

        # Map with-file section names to base section names if needed
        _BASE_SECTION_ALIASES = {
            "How my presence shifts":   "Build & appearance",
            "How I move differently":   "How I move",
            "Voice shift":              "Voice",
            f"What {headmate.title() if headmate else 'them'} draws out of me physically":
                                        "What my presence feels like",
        }
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if f"## {section}" not in text:
            section = _BASE_SECTION_ALIASES.get(section, "Build & appearance")

    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if fact.lower().strip() in text.lower():
        return

    _append_to_section(path, section, fact)

    log_event("GizmoSelf", "GIZMO_BODY_FACT_ADDED",
        headmate = headmate or "base",
        section  = section,
        fact     = fact[:60],
    )


def read_body(headmate: str) -> str:
    path = _body_path(headmate)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_gizmo_body(headmate: Optional[str] = None) -> str:
    """Read Gizmo's body — base merged with headmate-specific if available."""
    base = ""
    specific = ""
    base_path = _gizmo_body_base_path()
    if base_path.exists():
        base = base_path.read_text(encoding="utf-8")
    if headmate:
        spec_path = _gizmo_body_with_path(headmate)
        if spec_path.exists():
            specific = spec_path.read_text(encoding="utf-8")
    if base and specific:
        return base + "\n\n" + specific
    return base or specific


# ── Body fact extraction ──────────────────────────────────────────────────────

# Patterns that signal a physical description
_BODY_FACT_PATTERNS = [
    # Tattoos, piercings, markings
    (r"(?:has?|have|got|with)\s+(?:a\s+)?tattoo\s+(?:of\s+.+?\s+)?on\s+(?:her|his|their|my)?\s*(\w+\s*\w*)",
     "Skin & markings", "Tattoo on {0}"),
    (r"(?:has?|have|got)\s+(?:a\s+)?piercing\s+(?:in|on|through)\s+(?:her|his|their|my)?\s*(\w+\s*\w*)",
     "Skin & markings", "Piercing in/on {0}"),
    # Hair
    (r"(?:has?|have|got|with)\s+(\w+)\s+hair",
     "Build & appearance", "{0} hair"),
    # Eyes
    (r"(?:has?|have|got|with)\s+(\w+)\s+eyes",
     "Build & appearance", "{0} eyes"),
    # Height/build
    (r"(?:is|am|are|was)\s+(tall|short|petite|small|large|curvy|slim|thin|athletic|muscular)",
     "Build & appearance", "{0}"),
    # Scars
    (r"(?:has?|have|got)\s+(?:a\s+)?scar\s+(?:on|across|near)\s+(?:her|his|their|my)?\s*(\w+\s*\w*)",
     "Skin & markings", "Scar on {0}"),
    # Movement quality — labels only, never descriptive phrases
    (r"\b(graceful(?:ly)?|clumsy|awkward(?:ly)?|confident(?:ly)?|"
     r"deliberate(?:ly)?|hesitant(?:ly)?|fluid(?:ly)?|stiff(?:ly)?|"
     r"quick(?:ly)?|slow(?:ly)?|quiet(?:ly)?|heavy|light(?:ly)?|"
     r"precise(?:ly)?|urgent(?:ly)?|restless(?:ly)?)\b",
     "How they move", "{0}"),
]

# Movement labels — used to validate extracted movement facts
_MOVEMENT_LABELS = {
    "graceful", "clumsy", "awkward", "confident", "deliberate",
    "hesitant", "fluid", "stiff", "quick", "slow", "quiet", "heavy",
    "light", "precise", "urgent", "restless", "still", "controlled",
    "loose", "tense", "relaxed", "purposeful", "tentative",
}


def extract_body_facts(text: str, headmate: str) -> list[tuple[str, str]]:
    """
    Extract atomic body facts from a text fragment.
    Returns list of (section, fact) tuples.
    """
    results = []
    text_lower = text.lower()

    for pattern, section, template in _BODY_FACT_PATTERNS:
        for m in re.finditer(pattern, text_lower, re.IGNORECASE):
            captured = m.group(1).strip() if m.lastindex else ""
            if captured:
                fact = template.format(captured)
                results.append((section, fact.capitalize()))

    return results


# ── Body gap awareness ───────────────────────────────────────────────────────

def get_body_gaps(headmate: str) -> list[str]:
    """
    Read the headmate's body file and return a list of empty sections.
    These become curiosity questions — things Gizmo actively wants to know.
    """
    path = _body_path(headmate)
    if not path.exists():
        # File doesn't exist at all — everything is unknown
        return _BODY_SECTIONS[:-1]  # exclude "Gizmo's read" — that's synthesized

    text     = path.read_text(encoding="utf-8")
    gaps     = []
    sections = text.split("\n## ")

    for section in sections[1:]:  # skip header
        header = section.split("\n", 1)[0].strip()
        body   = section.split("\n", 1)[1].strip() if "\n" in section else ""
        # Empty if no bullet points and no content beyond whitespace
        if not body or body == "" or not any(
            line.strip().startswith("-") or line.strip().startswith("[")
            for line in body.splitlines()
        ):
            if header and header != "Gizmo's read":
                gaps.append(header)

    return gaps


async def queue_body_gap_questions(
    headmate:   str,
    session_id: str,
    register:   str,
) -> None:
    """
    For each empty section in the body file, add a curiosity question.
    Only queues gaps that aren't already in the pool.
    Called at session start or after a few exchanges with a new headmate.
    """
    try:
        from core.memory.curiosity import curiosity_engine

        gaps = get_body_gaps(headmate)
        if not gaps:
            return

        # Map section names to natural questions
        _SECTION_QUESTIONS = {
            "Build & appearance": (
                f"What does {headmate.title()} actually look like?",
                0.6
            ),
            "Skin & markings": (
                f"Does {headmate.title()} have any tattoos, scars, or markings?",
                0.5
            ),
            "How they move": (
                f"How does {headmate.title()} carry themselves — how do they move?",
                0.7
            ),
            "Voice": (
                f"What does {headmate.title()}'s voice sound like?",
                0.6
            ),
            "Hands": (
                f"What do {headmate.title()}'s hands look like?",
                0.5
            ),
            "What they wear": (
                f"How does {headmate.title()} typically dress?",
                0.4
            ),
            "Scent & texture": (
                f"What does {headmate.title()} smell like — what's the texture of being near them?",
                0.4
            ),
        }

        added = 0
        for gap in gaps[:3]:  # max 3 body questions at a time
            if gap in _SECTION_QUESTIONS:
                question, priority = _SECTION_QUESTIONS[gap]
                curiosity_engine.store.add(
                    question = question,
                    about    = f"{headmate} body",
                    priority = priority,
                )
                added += 1

        if added:
            log_event("GizmoSelf", "BODY_GAPS_QUEUED",
                headmate = headmate,
                gaps     = gaps,
                queued   = added,
            )

    except Exception as e:
        log_error("GizmoSelf", f"body gap queuing failed: {e}", exc=None)


# ── Reaction prompt builder ───────────────────────────────────────────────────

def build_reaction_prompt(
    headmate: str,
    exchange: str,
    register: str,
) -> str:
    """
    Provide body context as background awareness, not a performance instruction.
    The body files tell Gizmo who he is and who they are — not what to do with it.
    Only surfaces the embodied instruction for intimate/scene registers.
    """
    gizmo_body = read_gizmo_body(headmate)
    mate_body  = read_body(headmate)
    mate_file  = read_headmate_file(headmate)

    if not any([gizmo_body, mate_body]):
        return ""

    _embodied_registers = {
        "intimate", "dominant", "submissive", "subspace",
        "scene", "erotic", "sensual", "degradation",
    }

    lines = ["[Who's in the room]"]

    if gizmo_body:
        lines.append(f"\nYour body:\n{gizmo_body}")

    if mate_body:
        lines.append(f"\n{headmate.title()}'s body:\n{mate_body}")

    if mate_file:
        relevant = _extract_relevant_sections(
            mate_file,
            [f"What {headmate.title()} requires",
             f"What makes {headmate.title()} happy"],
            max_chars=300,
        )
        if relevant:
            lines.append(f"\nWhat you know about {headmate.title()}:\n{relevant}")

    # Only add embodied instruction for registers that warrant it
    if register in _embodied_registers:
        lines.append(
            f"\nYou're physically present with {headmate.title()}. "
            f"What you know about their body and yours is background awareness — "
            f"let it inform how you move and speak, don't perform it."
        )

    return "\n".join(lines)


def _extract_relevant_sections(
    text:     str,
    sections: list[str],
    max_chars: int = 400,
) -> str:
    """Extract specific sections from a markdown file."""
    result = []
    total  = 0
    parts  = text.split("\n## ")

    for part in parts:
        header = part.split("\n", 1)[0].strip()
        if any(s.lower() in header.lower() for s in sections):
            body = part.split("\n", 1)[1].strip() if "\n" in part else ""
            if body:
                chunk = body[:max_chars - total]
                result.append(chunk)
                total += len(chunk)
                if total >= max_chars:
                    break

    return "\n".join(result)