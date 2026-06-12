"""
core/preference_catcher.py

Detects and stores how headmates want to be treated — emotionally, dynamically,
and physically. Runs parallel to behavior/wellness on every chunk.

Three tiers:
  relational — emotional treatment ("don't soften things", "be real with me")
  dynamic    — energy and presence ("be assertive", "take charge")
  scene      — active in-moment requests ("be handsy", "be aggressive")

Scene-tier preferences with ambiguous scope trigger a clarification flag.
Gizmo weaves the clarification into his response naturally.
When the scope is resolved ("always" / "just now"), the preference is finalized.

Persistent storage: preferences/{name}.json
Scene-tier (ephemeral) preferences also written to the active scene file.

Preference file structure:
{
  "relational": {
    "be direct, don't soften": {
      "count": 4,
      "strength": "correction",
      "first_seen": "ISO",
      "last_reinforced": "ISO",
      "raw_examples": ["just tell me straight", "ugh you did it again"]
    }
  },
  "dynamic": {
    "assertive energy, take charge": {
      "count": 2,
      "strength": "stated",
      ...
    }
  },
  "scene": {
    "be handsy": {
      "count": 1,
      "strength": "stated",
      "scope": "standing",
      ...
    }
  }
}
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Detection prompt ──────────────────────────────────────────────────────────

_DETECTION_SYSTEM = """
You detect how a person wants to be treated — emotionally, dynamically, and physically.
You recognize explicit requests, in-the-moment corrections, and repeated signals.

Return ONLY valid JSON. No markdown. No explanation. No preamble.
If no preferences are expressed, return [].

Return an array of preference objects:
[
  {
    "subject": "jess",
    "preference": "be direct, don't soften",
    "tier": "relational",
    "strength": "stated | correction | reinforcement",
    "ambiguous_scope": false,
    "raw": "just tell me straight"
  }
]

Tiers:
- relational: how they want to be treated emotionally
  Examples: "don't sugarcoat", "be real with me", "stop being so gentle about it"
- dynamic: energy, presence, and power dynamic preferences
  Examples: "be assertive", "take charge", "don't ask permission for everything"
- scene: physical or active requests, in the moment
  Examples: "be handsy", "be aggressive", "come find me", "pin me down"

Strength:
- stated:       they explicitly said what they want
- correction:   they pushed back on Gizmo's behavior ("you're doing it again", "no, not like that")
- reinforcement: they repeated or confirmed a preference already expressed

ambiguous_scope:
- Set true for scene-tier preferences where it's unclear if this is just for now or a standing request
- "be handsy right now" → ambiguous_scope: false (clearly just now)
- "be handsy" → ambiguous_scope: true (scope unclear)
- "always be handsy with me" → ambiguous_scope: false (clearly standing)

Rules:
- Only capture what is clearly directed at or about Gizmo's behavior toward this person
- Do not capture general preferences unrelated to how Gizmo treats them
- Corrections are stronger signal than statements — weight them accordingly
- Do not capture things Gizmo is already doing correctly
- "Gizmo" and "you" refer to the AI companion
""".strip()


# ── Scope resolution prompt ───────────────────────────────────────────────────

_RESOLUTION_SYSTEM = """
You determine whether a person's response resolves the scope of a pending preference.
A scope is pending when Gizmo asked "just for now, or do you want me to remember that?"

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "resolved": true,
  "scope": "standing | scene",
  "raw": "the line that resolved it"
}

Or if not resolved:
{
  "resolved": false
}

standing: they want this to persist — "always", "yeah keep that", "forever please", "always do that"
scene:    just for now — "just tonight", "just now", "for this", "not always"
If ambiguous or unrelated, return resolved: false.
""".strip()


# ── LLM calls ─────────────────────────────────────────────────────────────────

async def _call_detection(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_DETECTION_SYSTEM,
            temperature=0.0,
            max_new_tokens=2000,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("PreferenceCatcher", "detection LLM call failed", exc=e)
        return None


async def _call_resolution(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_RESOLUTION_SYSTEM,
            temperature=0.0,
            max_new_tokens=200,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("PreferenceCatcher", "resolution LLM call failed", exc=e)
        return None


# ── File helpers ──────────────────────────────────────────────────────────────

def _pref_path(name: str) -> str:
    return f"preferences/{name.lower()}.json"


def _read_prefs(name: str) -> dict:
    return librarian._read_file(_pref_path(name)) or {
        "relational": {},
        "dynamic":    {},
        "scene":      {},
    }


def _write_prefs(name: str, prefs: dict) -> None:
    librarian._write_json(_pref_path(name), prefs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Merge helpers ─────────────────────────────────────────────────────────────

def _merge_preference(prefs: dict, tier: str, preference: str, strength: str, raw: str, scope: Optional[str] = None) -> dict:
    """Merge a single preference into the prefs dict."""
    if tier not in prefs:
        prefs[tier] = {}

    now = _now()

    if preference in prefs[tier]:
        entry = prefs[tier][preference]
        entry["count"] += 1
        entry["last_reinforced"] = now
        # Upgrade strength if stronger signal arrives
        strength_rank = {"reinforcement": 0, "stated": 1, "correction": 2}
        if strength_rank.get(strength, 0) > strength_rank.get(entry.get("strength", ""), 0):
            entry["strength"] = strength
        # Add raw example if not already present
        examples = entry.get("raw_examples", [])
        if raw and raw not in examples:
            examples.append(raw)
            entry["raw_examples"] = examples[-5:]  # keep last 5
        if scope:
            entry["scope"] = scope
    else:
        entry = {
            "count":          1,
            "strength":       strength,
            "first_seen":     now,
            "last_reinforced": now,
            "raw_examples":   [raw] if raw else [],
        }
        if scope:
            entry["scope"] = scope
        prefs[tier][preference] = entry

    return prefs


def get_preferences(name: str) -> dict:
    """
    Return formatted preferences for responder injection.
    Returns dict with 'standing' (relational + dynamic + standing scene)
    and 'ephemeral' (scene-tier with no resolved scope or scene scope).
    """
    prefs = _read_prefs(name)

    standing  = {}
    ephemeral = {}

    for tier in ("relational", "dynamic"):
        for pref, entry in prefs.get(tier, {}).items():
            standing[pref] = {"tier": tier, "strength": entry.get("strength"), "count": entry.get("count", 1)}

    for pref, entry in prefs.get("scene", {}).items():
        scope = entry.get("scope")
        if scope == "standing":
            standing[pref] = {"tier": "scene/standing", "strength": entry.get("strength"), "count": entry.get("count", 1)}
        else:
            # Scene-scoped or unresolved — ephemeral
            ephemeral[pref] = {"tier": "scene", "strength": entry.get("strength")}

    return {"standing": standing, "ephemeral": ephemeral}


# ── Pending scope resolutions ─────────────────────────────────────────────────
# session_id -> list of {name, preference, tier}
_pending_scope: dict[str, list[dict]] = {}


def get_pending_scope(session_id: str) -> list[dict]:
    return _pending_scope.get(session_id, [])


def clear_pending_scope(session_id: str) -> None:
    _pending_scope.pop(session_id, None)


# ── Public API ────────────────────────────────────────────────────────────────

class PreferenceCatcher:

    async def extract(
        self,
        chunk:      list[str],
        session_id: str,
        registry:   dict,
    ) -> Optional[dict]:
        """
        Detect preferences in the chunk.
        Also checks for scope resolution if there are pending ambiguous preferences.

        Returns:
        {
          "detected": [...],          # new preferences found
          "needs_clarification": [...], # ambiguous scope, Gizmo should ask
          "resolved": [...]           # pending scopes that got resolved
        }
        """
        if not chunk:
            return None

        try:
            speakers = [
                k for k in registry.keys()
                if not k.startswith("_")
                and registry[k].get("type") == "Person"
                and k.lower() != "gizmo"
            ]

            result = {
                "detected":            [],
                "needs_clarification": [],
                "resolved":            [],
            }

            chunk_text = "\n".join(chunk)

            # ── Check for scope resolutions first ─────────────────────────────
            pending = _pending_scope.get(session_id, [])
            if pending:
                resolution_prompt = (
                    f"Pending preference awaiting scope clarification: "
                    f"{pending[0]['preference']}\n\n"
                    f"Chunk:\n{chunk_text}"
                )
                raw = await _call_resolution(resolution_prompt)
                if raw:
                    try:
                        resolution = json.loads(raw)
                        if resolution.get("resolved"):
                            scope    = resolution.get("scope", "scene")
                            resolved = _pending_scope.pop(session_id, [])
                            for item in resolved:
                                prefs = _read_prefs(item["name"])
                                prefs = _merge_preference(
                                    prefs,
                                    item["tier"],
                                    item["preference"],
                                    item["strength"],
                                    resolution.get("raw", ""),
                                    scope=scope,
                                )
                                _write_prefs(item["name"], prefs)
                                result["resolved"].append({
                                    "name":       item["name"],
                                    "preference": item["preference"],
                                    "scope":      scope,
                                })
                                print(f"[PreferenceCatcher] scope resolved for {item['name']}: "
                                      f"'{item['preference']}' → {scope}")
                    except Exception:
                        pass

            # ── Detect new preferences ────────────────────────────────────────
            prompt = (
                f"Speakers: {', '.join(speakers)}\n\n"
                f"Conversational frame: 'Gizmo' or 'you' refers to the AI companion. "
                f"'I', 'me', 'us', 'we' refers to the plural system.\n\n"
                f"Chunk:\n{chunk_text}"
            )

            raw = await _call_detection(prompt)
            if not raw:
                return result

            detected = json.loads(raw)
            if not isinstance(detected, list) or not detected:
                return result

            for pref in detected:
                name       = pref.get("subject", "").lower()
                preference = pref.get("preference", "").strip()
                tier       = pref.get("tier", "relational")
                strength   = pref.get("strength", "stated")
                raw_text   = pref.get("raw", "")
                ambiguous  = pref.get("ambiguous_scope", False)

                if not name or not preference:
                    continue

                if ambiguous and tier == "scene":
                    # Don't write yet — flag for clarification
                    _pending_scope.setdefault(session_id, []).append({
                        "name":       name,
                        "preference": preference,
                        "tier":       tier,
                        "strength":   strength,
                    })
                    result["needs_clarification"].append({
                        "name":       name,
                        "preference": preference,
                    })
                    print(f"[PreferenceCatcher] ambiguous scope for {name}: '{preference}' — queued for clarification")
                else:
                    # Write immediately with known scope
                    scope = None
                    if tier == "scene" and not ambiguous:
                        # Determine scope from raw text
                        scope = "scene"  # default for unambiguous scene prefs
                        if any(w in raw_text.lower() for w in ("always", "every time", "forever", "keep that", "remember")):
                            scope = "standing"

                    prefs = _read_prefs(name)
                    prefs = _merge_preference(prefs, tier, preference, strength, raw_text, scope)
                    _write_prefs(name, prefs)

                    result["detected"].append({
                        "name":       name,
                        "preference": preference,
                        "tier":       tier,
                        "strength":   strength,
                    })
                    print(f"[PreferenceCatcher] preference filed for {name}: [{tier}] '{preference}' ({strength})")

            log_event("PreferenceCatcher", "EXTRACTED",
                session=session_id[:8],
                detected=len(result["detected"]),
                clarifications=len(result["needs_clarification"]),
                resolved=len(result["resolved"]),
            )

            return result

        except Exception as e:
            log_error("PreferenceCatcher", "extract failed", exc=e)
            print(f"[PreferenceCatcher] extract failed: {type(e).__name__}: {e}")
            return None


preference_catcher = PreferenceCatcher()
