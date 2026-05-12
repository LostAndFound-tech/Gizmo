"""
core/persona.py
Lightweight persona prefix utility.

Builds a system prompt prefix that puts Gizmo in the right relational
and emotional context for any given headmate. Used by every internal
LLM call that produces output Gizmo "owns" — archiver, overview,
synthesis, memory writer, conversation archive.

This is intentionally cheap: reads from the headmate JSON file on disk,
no LLM call, no ChromaDB query. Fast enough to call on every internal
prompt without adding latency.

Usage:
    from core.persona import persona_prefix

    system_prompt = persona_prefix(current_host) + "\\n\\n" + your_system_prompt

If speaker is None or the file doesn't exist, returns a sensible
Gizmo-as-himself default so callers never need to handle None.
"""

import json
import os
from pathlib import Path
from typing import Optional

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_SEED_FILE       = _PERSONALITY_DIR / "personality.txt"


def _load_gizmo_seed() -> str:
    """Load Gizmo's own personality seed. Cached after first read."""
    if _load_gizmo_seed._cache is not None:
        return _load_gizmo_seed._cache
    try:
        text = _SEED_FILE.read_text(encoding="utf-8").strip()
        _load_gizmo_seed._cache = text or "You are Gizmo, a persistent AI companion."
    except FileNotFoundError:
        _load_gizmo_seed._cache = "You are Gizmo, a persistent AI companion."
    return _load_gizmo_seed._cache

_load_gizmo_seed._cache = None


def _load_headmate(speaker: str) -> Optional[dict]:
    """Load headmate JSON. Returns None if not found."""
    try:
        path = _HEADMATES_DIR / f"{speaker.lower().strip()}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def persona_prefix(
    speaker: Optional[str],
    include_gizmo_seed: bool = True,
) -> str:
    """
    Build a system prompt prefix for a given speaker.

    Always includes:
      - Gizmo's own personality seed (who he is)
      - Who the speaker is and what Gizmo knows about them
      - How the speaker wants to be engaged (interaction_prefs)

    include_gizmo_seed: set False if the caller already includes the seed
                        to avoid doubling it.

    Returns a string ready to prepend to any system prompt.
    """
    parts = []

    # ── Gizmo's own identity ──────────────────────────────────────────────────
    if include_gizmo_seed:
        seed = _load_gizmo_seed()
        if seed:
            parts.append(seed)

    if not speaker:
        return "\n\n".join(parts) if parts else "You are Gizmo, a persistent AI companion."

    data = _load_headmate(speaker)
    if not data:
        # Speaker unknown — just note who we're talking to
        parts.append(f"You are currently speaking with {speaker.title()}.")
        return "\n\n".join(parts)

    name = data.get("name", speaker).title()

    # ── Who this person is ────────────────────────────────────────────────────
    about_lines = [f"You are speaking with {name}."]

    baseline = data.get("baseline", {})
    for k, v in baseline.items():
        if k == "observations":
            continue
        if v and v not in ("unknown", 0, 0.0, "", None):
            about_lines.append(f"  {k}: {v}")

    moments = data.get("moments_of_note", [])
    if moments:
        about_lines.append("  Things worth remembering:")
        for m in moments[-5:]:
            import re as _re
            clean = _re.sub(r'^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*', '', str(m))
            about_lines.append(f"    - {clean}")

    corrections = data.get("corrections", [])
    if corrections:
        about_lines.append("  Rules from this person:")
        for c in corrections[-3:]:
            rule = c.get("rule", str(c)) if isinstance(c, dict) else str(c)
            about_lines.append(f"    - {rule}")

    parts.append("\n".join(about_lines))

    # ── How they want to be engaged ───────────────────────────────────────────
    prefs = data.get("interaction_prefs", {})
    pref_lines = []

    persona_text = prefs.get("persona", "")
    if persona_text:
        pref_lines.append(persona_text)

    for field, label in (
        ("tone",     "Tone"),
        ("pacing",   "Pacing"),
        ("humor",    "Humor"),
        ("checkins", "Check-ins"),
        ("distress", "When distressed"),
    ):
        v = prefs.get(field)
        if v:
            pref_lines.append(f"{label}: {v}")

    explicit = [e for e in prefs.get("explicit", []) if e]
    if explicit:
        pref_lines.extend(explicit)

    if pref_lines:
        parts.append("[How " + name + " wants to be engaged]\n" + "\n".join(pref_lines))

    return "\n\n".join(parts)


def persona_prefix_multi(
    speakers: list[str],
    include_gizmo_seed: bool = True,
) -> str:
    """
    Build a prefix covering multiple fronters.
    Gizmo's seed appears once; each speaker gets their own block.
    Useful when multiple headmates are co-fronting.
    """
    parts = []

    if include_gizmo_seed:
        seed = _load_gizmo_seed()
        if seed:
            parts.append(seed)

    seen = set()
    for speaker in speakers:
        if not speaker or speaker.lower() in seen:
            continue
        seen.add(speaker.lower())
        # Get per-speaker block without re-including Gizmo seed
        block = persona_prefix(speaker, include_gizmo_seed=False)
        if block:
            parts.append(block)

    return "\n\n".join(parts) if parts else "You are Gizmo, a persistent AI companion."


def invalidate_cache() -> None:
    """Call when personality.txt is rewritten so the seed reloads."""
    _load_gizmo_seed._cache = None
