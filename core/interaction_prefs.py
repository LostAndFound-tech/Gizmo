"""
core/interaction_prefs.py
Per-headmate interaction preferences — how each headmate wants Gizmo to engage.

These live inside the headmate's JSON file under "interaction_prefs", alongside
corrections, moments_of_note, and baseline. Ego reads them in _build_system_prompt()
and injects them into the tone block.

Not inferred. Set explicitly, stored verbatim, read back verbatim.
Survive personality rewrites. Never softened or synthesized away.

Schema (inside headmate JSON):
    "interaction_prefs": {
        "tone":     str | null,   -- how Gizmo should sound with this person
        "pacing":   str | null,   -- verbose vs terse, elaboration vs just the answer
        "checkins": str | null,   -- whether to proactively ask how they're doing
        "humor":    str | null,   -- what kind, how much
        "distress": str | null,   -- how to respond when this person seems distressed
        "persona":  str | null,   -- freeform mini-prompt: "When talking to X, do A, say B..."
                                     injected as raw instruction, not as a label
        "explicit": [str, ...]    -- verbatim freeform one-liners, accumulates
    }

Structured fields (tone/pacing/checkins/humor/distress/persona): upserted — latest wins.
Explicit: append-only — each statement is its own entry, nothing overwritten.

Injection order in system prompt:
  1. persona  — raw instruction block, no label, reads as direct direction
  2. tone/pacing/checkins/humor/distress — labeled key-value lines
  3. explicit — bullet list of verbatim instructions
"""

import json
from pathlib import Path
from typing import Optional
import os

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"

STRUCTURED_FIELDS = {"tone", "pacing", "checkins", "humor", "distress", "persona"}
FREEFORM_FIELD    = "explicit"
ALL_FIELDS        = STRUCTURED_FIELDS | {FREEFORM_FIELD}

FIELD_LABELS = {
    "tone":     "Tone",
    "pacing":   "Pacing",
    "checkins": "Check-ins",
    "humor":    "Humor",
    "distress": "When distressed",
    "persona":  "Persona",   # label used in view_interaction_prefs only
    "explicit": "Explicit instructions",
}

_EMPTY_PREFS = {
    "tone":     None,
    "pacing":   None,
    "checkins": None,
    "humor":    None,
    "distress": None,
    "persona":  None,
    "explicit": [],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _headmate_path(name: str) -> Path:
    return _HEADMATES_DIR / f"{name.lower()}.json"


def _load_headmate(name: str) -> Optional[dict]:
    path = _headmate_path(name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[Prefs] Failed to load headmate file for {name}: {e}")
        return None


def _save_headmate(name: str, data: dict) -> None:
    path = _headmate_path(name)
    try:
        _HEADMATES_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Prefs] Failed to save headmate file for {name}: {e}")


def _get_prefs_block(data: dict) -> dict:
    """Return the interaction_prefs block from headmate data, with defaults filled in."""
    prefs = data.get("interaction_prefs", {})
    result = dict(_EMPTY_PREFS)
    result.update({k: v for k, v in prefs.items() if k in ALL_FIELDS})
    if not isinstance(result["explicit"], list):
        result["explicit"] = []
    return result


# ── Write ─────────────────────────────────────────────────────────────────────

def set_pref(host: str, field: str, value: str) -> bool:
    """
    Set or update a preference for a headmate.
    Structured fields: upserted — latest wins.
    Explicit: appended — accumulates.
    Returns True on success.
    """
    field = field.lower().strip()
    if field not in ALL_FIELDS:
        raise ValueError(f"Unknown field '{field}'. Valid: {', '.join(sorted(ALL_FIELDS))}")

    data = _load_headmate(host)
    if data is None:
        print(f"[Prefs] No headmate file found for {host} — cannot set pref")
        return False

    prefs = _get_prefs_block(data)

    if field == FREEFORM_FIELD:
        if value not in prefs["explicit"]:
            prefs["explicit"].append(value)
    else:
        prefs[field] = value

    data["interaction_prefs"] = prefs
    _save_headmate(host, data)
    print(f"[Prefs] Set '{field}' for {host}: {value[:60]}")
    return True


def delete_pref(host: str, field: str, value: str = None) -> bool:
    """
    Delete a preference.
    For explicit: removes matching string if value provided, clears all if not.
    For structured fields (including persona): sets to None.
    Returns True on success.
    """
    field = field.lower().strip()
    data = _load_headmate(host)
    if data is None:
        return False

    prefs = _get_prefs_block(data)

    if field == FREEFORM_FIELD:
        if value:
            prefs["explicit"] = [e for e in prefs["explicit"] if e != value]
        else:
            prefs["explicit"] = []
    else:
        prefs[field] = None

    data["interaction_prefs"] = prefs
    _save_headmate(host, data)
    return True


# ── Read ──────────────────────────────────────────────────────────────────────

def get_prefs(host: str) -> Optional[dict]:
    """
    Return the interaction_prefs block for a headmate.
    Returns None if the headmate file doesn't exist.
    Returns an empty prefs dict if the file exists but has no prefs set.
    """
    data = _load_headmate(host)
    if data is None:
        return None
    return _get_prefs_block(data)


def has_prefs(host: str) -> bool:
    """Return True if this headmate has any prefs set."""
    prefs = get_prefs(host)
    if prefs is None:
        return False
    return any(prefs.get(f) for f in STRUCTURED_FIELDS) or bool(prefs.get("explicit"))


def format_prefs_for_prompt(host: str) -> str:
    """
    Format a headmate's prefs for injection into the system prompt.
    Returns empty string if no prefs are set.
    Called by ego.py's _build_system_prompt().

    Injection order:
      1. persona — raw, unlabeled, reads as direct direction to the LLM
      2. labeled fields (tone/pacing/checkins/humor/distress)
      3. explicit bullet list
    """
    prefs = get_prefs(host)
    if not prefs:
        return ""

    sections = []

    # Persona — raw instruction block, no wrapping label
    persona = prefs.get("persona")
    if persona:
        sections.append(persona.strip())

    # Labeled fields
    labeled_lines = []
    for field in ("tone", "pacing", "checkins", "humor", "distress"):
        val = prefs.get(field)
        if val:
            labeled_lines.append(f"  {FIELD_LABELS[field]}: {val}")
    if labeled_lines:
        sections.append(
            f"[How {host.title()} wants to be engaged]\n" + "\n".join(labeled_lines)
        )

    # Explicit instructions
    explicit = [e for e in prefs.get("explicit", []) if e]
    if explicit:
        lines = [f"  - {e}" for e in explicit]
        sections.append(
            f"[Explicit instructions for {host.title()} — follow verbatim]\n"
            + "\n".join(lines)
        )

    return "\n\n".join(sections)