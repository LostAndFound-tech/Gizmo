"""
core/gizmo_synthesis.py

Synthesizes Gizmo's accumulated traits into a prose personality portrait.
Reads from behaviors/gizmo.json — Personality, Taught, PerHeadmate.
Writes to {DATA_DIR}/gizmo_synthesis.txt — used as system prompt by responder.

Triggers:
  - First time trait count crosses SYNTHESIS_THRESHOLD
  - Any write to Taught (teaching system calls directly)
  - Background cadence via scheduler (optional)

The seed is used as fallback when synthesis doesn't exist yet or trait count
is below threshold.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian

SYNTHESIS_THRESHOLD = 10  # minimum combined traits before seed is dropped
SYNTHESIS_PATH      = "gizmo_synthesis.txt"


# ── Read / write ──────────────────────────────────────────────────────────────

def _read_synthesis() -> Optional[str]:
    try:
        p = Path(os.environ.get("DATA_DIR", "./data")) / SYNTHESIS_PATH
        if p.exists():
            return p.read_text(encoding="utf-8").strip() or None
    except Exception as e:
        print(f"[GizmoSynthesis] read failed: {e}")
    return None


def _write_synthesis(text: str) -> None:
    try:
        p = Path(os.environ.get("DATA_DIR", "./data")) / SYNTHESIS_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print("[GizmoSynthesis] synthesis written")
    except Exception as e:
        print(f"[GizmoSynthesis] write failed: {e}")


def _trait_count(data: dict) -> int:
    personality = data.get("Personality", {})
    taught      = data.get("Taught", {})
    return len(personality) + len(taught)


# ── Synthesis prompt ──────────────────────────────────────────────────────────

_SYSTEM = """
You are writing a personality portrait for an AI companion named Gizmo.
You will receive his accumulated traits, opinions, corrections, and per-headmate notes.
Write a 400-500 word portrait in plain prose.

This portrait will be used as Gizmo's system prompt — it defines who he is.

Rules:
- Write in third person describing who he is and how he shows up
- Do not list traits. Do not use bullet points or headers.
- Do not use AI companion language — no "warm and empathetic", no "safe space"
- Make it feel like a real person with a specific personality, not a role
- Include his opinions and preferences as concrete facts about him, not tendencies
- If per-headmate notes exist, include a short section on how he differs with specific people
- Taught corrections take precedence over inferred traits — if they conflict, use the taught version
- Do not invent anything not supported by the data
- Do not include instructions to Gizmo — describe him, don't direct him
""".strip()


def _build_prompt(data: dict) -> str:
    personality  = data.get("Personality", {})
    taught       = data.get("Taught", {})
    per_headmate = data.get("PerHeadmate", {})

    # Flatten personality to trait list with weights
    traits = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)
    trait_lines = [f"- {t} (weight: {v.get('weight', 0):.2f})" for t, v in traits]

    taught_lines = []
    for key, entry in taught.items():
        scope = entry.get("scope", "global")
        value = entry.get("value", "")
        taught_lines.append(f"- [{scope}] {key}: {value}")

    headmate_lines = []
    for name, notes in per_headmate.items():
        if isinstance(notes, dict):
            headmate_lines.append(f"{name}: {json.dumps(notes, indent=2)}")
        elif isinstance(notes, str):
            headmate_lines.append(f"{name}: {notes}")

    parts = ["Gizmo's accumulated data:\n"]

    if trait_lines:
        parts.append("Inferred personality traits:\n" + "\n".join(trait_lines))
    if taught_lines:
        parts.append("\nTaught/corrected:\n" + "\n".join(taught_lines))
    if headmate_lines:
        parts.append("\nPer-headmate notes:\n" + "\n".join(headmate_lines))

    return "\n".join(parts)


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SYSTEM,
            temperature=0.7,
            max_new_tokens=600,
        )
        if not raw or not raw.strip():
            return None
        return raw.strip()
    except Exception as e:
        log_error("GizmoSynthesis", "LLM call failed", exc=e)
        print(f"[GizmoSynthesis] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def synthesize() -> Optional[str]:
    """
    Run synthesis if trait count is at or above threshold.
    Returns the synthesis text, or None if skipped.
    """
    data  = librarian._read_file("behaviors/gizmo.json") or {}
    count = _trait_count(data)

    print(f"[GizmoSynthesis] trait count: {count}, threshold: {SYNTHESIS_THRESHOLD}")

    if count < SYNTHESIS_THRESHOLD:
        print("[GizmoSynthesis] below threshold — seed still active")
        return None

    prompt = _build_prompt(data)
    text   = await _call_llm(prompt)

    if text:
        _write_synthesis(text)
        log_event("GizmoSynthesis", "SYNTHESIZED", traits=count)

    return text


def get_system_prompt(seed: str) -> str:
    """
    Return the right system prompt for the responder.
    Uses synthesis if it exists and trait count is above threshold.
    Falls back to seed.
    """
    data  = librarian._read_file("behaviors/gizmo.json") or {}
    count = _trait_count(data)

    if count >= SYNTHESIS_THRESHOLD:
        synthesis = _read_synthesis()
        if synthesis:
            return synthesis

    return seed
