"""
core/rp_preferences.py

Tracks roleplay preferences per person.

Confirmed preferences: explicitly stated or inferred 3+ times without negative signal.
Inferred preferences:  observed in scene beats, not yet confirmed.
Sub-preferences:       how they like specific kinks (shibari vs cuffs, hard vs soft).
Limits:                hard (never) and soft (check first).
gizmo_can_initiate:    Gizmo can reach for these unprompted.

File: {DATA_DIR}/rp_preferences/{name}.json

Weight lifecycle:
  First seen (inferred)  → weight 0.3, use_count 1
  Each subsequent use    → weight += 0.15, use_count++
  use_count >= 3         → moved to confirmed, added to gizmo_can_initiate
  Negative signal        → removed from gizmo_can_initiate, moved to soft limits
  Explicit no            → hard limit, removed everywhere

Sub-preferences track context (scene register when used) and accumulate weight
independently within their parent kink.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Constants ─────────────────────────────────────────────────────────────────

INFER_TO_CONFIRM_COUNT = 3
INFER_BASE_WEIGHT      = 0.3
INFER_WEIGHT_INCREMENT = 0.15


# ── File I/O ──────────────────────────────────────────────────────────────────

def _path(name: str) -> str:
    return f"rp_preferences/{name.lower()}.json"


def _read(name: str) -> dict:
    return librarian._read_file(_path(name)) or {
        "confirmed":          {},
        "inferred":           {},
        "limits":             {"hard": [], "soft": []},
        "gizmo_can_initiate": [],
    }


def _write(name: str, data: dict) -> None:
    librarian._write_json(_path(name), data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _promote_to_confirmed(name: str, kink: str, data: dict) -> None:
    """Move a kink from inferred to confirmed and add to initiate list."""
    entry = data["inferred"].pop(kink, {})
    entry["confirmed_at"] = _now()
    data["confirmed"][kink] = entry
    if kink not in data["gizmo_can_initiate"]:
        data["gizmo_can_initiate"].append(kink)
    print(f"[RPPrefs] '{kink}' confirmed for {name}, added to initiate list")
    log_event("RPPrefs", "KINK_CONFIRMED", name=name, kink=kink)


def _update_sub_preference(
    kink_entry:    dict,
    sub:           str,
    context:       list[str],
) -> None:
    """Update sub-preference weight and context within a kink entry."""
    if "sub_preferences" not in kink_entry:
        kink_entry["sub_preferences"] = {}

    subs = kink_entry["sub_preferences"]
    if sub not in subs:
        subs[sub] = {"weight": 0.3, "use_count": 1, "context": context}
    else:
        subs[sub]["use_count"] = subs[sub].get("use_count", 0) + 1
        subs[sub]["weight"]    = min(1.0, subs[sub].get("weight", 0.3) + 0.1)
        # Merge context tags
        existing = set(subs[sub].get("context", []))
        subs[sub]["context"] = list(existing | set(context))


# ── Public API ────────────────────────────────────────────────────────────────

def get_preferences(name: str) -> dict:
    """Return full preference file for a person."""
    return _read(name)


def get_initiatable(name: str) -> dict:
    """
    Return kinks Gizmo can reach for unprompted, with their sub-preferences.
    Used during scene generation to inform what Gizmo can confidently deploy.
    """
    data     = _read(name)
    initiate = data.get("gizmo_can_initiate", [])
    result   = {}
    for kink in initiate:
        entry = data["confirmed"].get(kink) or data["inferred"].get(kink)
        if entry:
            result[kink] = {
                "weight":          entry.get("weight", 0.5),
                "sub_preferences": entry.get("sub_preferences", {}),
            }
    return result


def get_limits(name: str) -> dict:
    """Return hard and soft limits."""
    return _read(name).get("limits", {"hard": [], "soft": []})


def record_use(
    name:      str,
    kink:      str,
    sub:       Optional[str]       = None,
    context:   Optional[list[str]] = None,
    evidence:  str                 = "",
) -> None:
    """
    Record one observed use of a kink in a scene.
    Updates weight, use_count, sub-preferences.
    Promotes to confirmed after INFER_TO_CONFIRM_COUNT uses.
    """
    data    = _read(name)
    context = context or []

    # Check limits — don't re-add anything that's been limited
    limits = data.get("limits", {"hard": [], "soft": []})
    if kink in limits.get("hard", []):
        print(f"[RPPrefs] '{kink}' is a hard limit for {name} — skipping")
        return

    bucket = "confirmed" if kink in data["confirmed"] else "inferred"
    target = data[bucket]

    if kink not in target:
        target[kink] = {
            "weight":          INFER_BASE_WEIGHT,
            "use_count":       1,
            "sub_preferences": {},
            "first_seen":      _now(),
            "last_seen":       _now(),
            "evidence":        evidence,
        }
    else:
        target[kink]["use_count"] = target[kink].get("use_count", 0) + 1
        target[kink]["weight"]    = min(
            1.0,
            target[kink].get("weight", INFER_BASE_WEIGHT) + INFER_WEIGHT_INCREMENT
        )
        target[kink]["last_seen"] = _now()
        if evidence:
            target[kink]["evidence"] = evidence

    # Update sub-preference if provided
    if sub:
        _update_sub_preference(target[kink], sub, context)

    # Promote if threshold reached and still inferred
    if bucket == "inferred" and target[kink]["use_count"] >= INFER_TO_CONFIRM_COUNT:
        _promote_to_confirmed(name, kink, data)

    _write(name, data)
    log_event("RPPrefs", "USE_RECORDED", name=name, kink=kink, sub=sub or "", bucket=bucket)


def record_explicit(name: str, kink: str, sub: Optional[str] = None) -> None:
    """
    User explicitly confirmed a preference.
    Moves directly to confirmed and initiate list regardless of use_count.
    """
    data = _read(name)

    # Remove from limits if it was soft-limited
    data["limits"]["soft"] = [l for l in data["limits"].get("soft", []) if l != kink]

    entry = data["inferred"].pop(kink, data["confirmed"].get(kink, {
        "weight":          1.0,
        "use_count":       1,
        "sub_preferences": {},
        "first_seen":      _now(),
    }))
    entry["weight"]       = max(entry.get("weight", 0.0), 0.9)
    entry["last_seen"]    = _now()
    entry["confirmed_at"] = _now()

    if sub:
        _update_sub_preference(entry, sub, [])

    data["confirmed"][kink] = entry
    if kink not in data["gizmo_can_initiate"]:
        data["gizmo_can_initiate"].append(kink)

    _write(name, data)
    log_event("RPPrefs", "EXPLICIT_CONFIRM", name=name, kink=kink)


def record_negative(name: str, kink: str, hard: bool = False) -> None:
    """
    Negative signal on a kink.
    hard=False → soft limit, removed from initiate list
    hard=True  → hard limit, removed from everywhere
    """
    data = _read(name)

    # Remove from initiate list
    data["gizmo_can_initiate"] = [
        k for k in data["gizmo_can_initiate"] if k != kink
    ]

    if hard:
        # Remove from confirmed and inferred entirely
        data["confirmed"].pop(kink, None)
        data["inferred"].pop(kink, None)
        if kink not in data["limits"]["hard"]:
            data["limits"]["hard"].append(kink)
        print(f"[RPPrefs] '{kink}' → hard limit for {name}")
        log_event("RPPrefs", "HARD_LIMIT", name=name, kink=kink)
    else:
        if kink not in data["limits"]["soft"]:
            data["limits"]["soft"].append(kink)
        print(f"[RPPrefs] '{kink}' → soft limit for {name}")
        log_event("RPPrefs", "SOFT_LIMIT", name=name, kink=kink)

    _write(name, data)


def infer_from_beats(name: str, beats: list[dict], register: str = "") -> None:
    """
    Post-scene preference inference pass.
    Reads thought logs from beats, extracts kink/sub references,
    updates preference file accordingly.

    Called after de-escalation detection closes the scene.
    """
    if not beats:
        return

    # Collect all thoughts from beats
    thoughts = [b.get("thought", "") for b in beats if b.get("thought")]
    if not thoughts:
        return

    # Run async inference in a fire-and-forget task
    import asyncio
    asyncio.create_task(_async_infer(name, thoughts, beats, register))


async def _async_infer(
    name:     str,
    thoughts: list[str],
    beats:    list[dict],
    register: str,
) -> None:
    """
    LLM pass to extract kink/sub-preference signals from scene thoughts.
    """
    _INFER_SYSTEM = """
You extract roleplay preference signals from a Game Master's internal thought log.
The thoughts are from an AI narrating a scene — they note what creative choices were made and why.

Return ONLY valid JSON. No markdown. No explanation.

{
  "signals": [
    {
      "kink": "bondage",
      "sub": "shibari",
      "signal_type": "use",
      "evidence": "GM chose shibari specifically, noted the person responded with escalating intensity"
    },
    {
      "kink": "praise",
      "sub": null,
      "signal_type": "use",
      "evidence": "GM deployed praise, scene escalated immediately after"
    }
  ],
  "negative_signals": [
    {
      "kink": "humiliation",
      "hard": false,
      "evidence": "scene de-escalated immediately after humiliation beat, person disengaged"
    }
  ]
}

signal_type is always "use" for now.
Only include signals where there is genuine evidence in the thoughts.
If thoughts contain no preference signals, return {"signals": [], "negative_signals": []}.
""".strip()

    try:
        from core.llm import llm
        import re

        prompt = (
            f"Scene register: {register}\n\n"
            f"GM thought log:\n" +
            "\n---\n".join(thoughts)
        )

        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_INFER_SYSTEM,
            temperature=0.0,
            max_new_tokens=1000,
        )

        if not raw or not raw.strip():
            return

        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)

        context = [register] if register else []

        for signal in result.get("signals", []):
            kink = signal.get("kink")
            if kink:
                record_use(
                    name=name,
                    kink=kink,
                    sub=signal.get("sub"),
                    context=context,
                    evidence=signal.get("evidence", ""),
                )

        for neg in result.get("negative_signals", []):
            kink = neg.get("kink")
            if kink:
                record_negative(name=name, kink=kink, hard=neg.get("hard", False))

        print(f"[RPPrefs] inference complete for {name}: "
              f"{len(result.get('signals', []))} signals, "
              f"{len(result.get('negative_signals', []))} negative")

    except Exception as e:
        log_error("RPPrefs", "inference pass failed", exc=e)
