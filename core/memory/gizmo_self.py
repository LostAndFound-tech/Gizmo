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


def detect_preference_request(message: str) -> list[dict]:
    """
    Scan a message for explicit preference requests.
    Returns list of {dimension, delta, flavor_hint, matched_phrase}
    Can return multiple if several requests in one message.
    """
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
    return results


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

    log_event("GizmoSelf", "SELF_OBSERVATION_COMPLETE",
        headmate = headmate,
        register = register,
        session  = session_id[:8],
        log_len  = len(log_lines),
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
