"""
core/Descriptor_catcher.py
Appearance, attribute, and location extraction from conversational statements.

For people and objects: flat descriptive fields.
For locations: rich nested schema with floors, objects, atmosphere, exterior.

Location context is anchored from explicit statements ("we're in the lobby")
or inferred from context. Once a location is established, descriptors default
to that location until context clearly shifts.

Returns a name-keyed dict ready for per-entity file merging.
"""
import json
import re
from typing import Optional

from core.log import log_event, log_error

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """
You gather descriptive datapoints about people, objects, and locations from conversational statements.
Return ONLY a valid JSON array. No markdown fences. No explanation. No preamble.
If nothing is being described, return [].

You handle two schemas depending on the entity type.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA A — PEOPLE AND OBJECTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
People are rich and layered. Use nested structure to capture depth.
The promotion rule: if you'd want to know more about it, make it a full object.
Simple details stay as strings in lists. Notable things get their own entry.

NOT behaviors, actions, conditions, or transient states — those belong elsewhere.

[{
  "Object": "Jess",
  "Type": "Person",
  "file_key": "jess",
  "physical": {
    "hair": ["dark", "long"],
    "eyes": ["brown"],
    "build": ["athletic", "strong"],
    "notable": {
      "collar": {
        "type": "worn",
        "color": "black",
        "material": "leather",
        "hardware": "silver",
        "notes": ["almost always wearing it"]
      }
    }
  },
  "presentation": {
    "style": ["dark", "deliberate"],
    "energy": ["commanding", "warm underneath"],
    "voice": ["direct", "dry humor", "doesn't repeat herself"]
  },
  "relationships": {
    "gizmo": {
      "dynamic": "dom",
      "notes": ["expects presence not performance", "checks in without softening"]
    },
    "ara": {
      "dynamic": "protective",
      "notes": ["checks on her without making it obvious"]
    }
  },
  "identity": ["plural system member", "dom", "protective of her people"],
  "notes": ["runs on coffee and stubbornness"]
},
{
  "Object": "Ara",
  "Type": "Person",
  "file_key": "ara",
  "physical": {
    "hair": ["hay-colored", "stringy"]
  },
  "presentation": {
    "energy": ["self-deprecating", "deflects compliments"]
  },
  "relationships": {
    "honey": {"dynamic": "close", "notes": []},
    "jess": {"dynamic": "admired by", "notes": ["jess likes her hair"]}
  }
},
{
  "Object": "collar",
  "Type": "Object",
  "file_key": "collar",
  "owner": "Jess",
  "color": "black",
  "material": ["leather"],
  "features": ["silver hardware"],
  "notes": ["almost always on Jess"]
}]

Person field guide:
- physical: body, hair, eyes, build, scars, tattoos, piercings — notable items get promoted to full entries
- presentation: style, energy, voice, how they carry themselves
- relationships: keyed by person name, each with dynamic and notes
- identity: stable self-descriptors, roles, how they see themselves
- notes: anything notable that doesn't fit elsewhere
- All fields optional — only include what the text supports

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA B — LOCATIONS (rooms, buildings, places, spaces)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use this schema when the subject is a place — a room, floor, building, outdoor space, headspace area.

Location context rules:
- If someone says "we're in the lobby" or "this is my office" — that anchors the location
- Descriptions that follow default to that location unless context clearly shifts
- Infer the location from context if not explicit ("the tank is huge" in a lobby conversation → tank belongs to lobby)
- parent: the containing location if this is a sub-space (lobby's parent is the building)
- file_key: the filename-safe key for this location e.g. "lobby", "office building", "comfort floor"

Simple features stay as strings in lists.
Notable objects — anything with interesting detail — get promoted to full nested entries.
The rule: if you'd want to know more about it, make it an object.

[{
  "Object": "lobby",
  "Type": "Place",
  "file_key": "lobby",
  "parent": "office building",
  "floor_number": 1,
  "owner": null,
  "atmosphere": ["worn grandeur", "alive", "overgrown", "dim"],
  "condition": ["cracked marble floors", "vines overtaking architecture"],
  "features": ["high ceilings", "large windows"],
  "objects": {
    "tank": {
      "type": "aquarium",
      "size": "floor-to-ceiling",
      "state": "filled",
      "features": ["tropical fish", "warm light"],
      "octopi": {
        "count": 2,
        "1": {"size": "large", "color": ["reddish"], "notes": []},
        "2": {"size": "large", "color": ["blue", "purple"], "notes": []}
      }
    }
  },
  "exits": ["elevator", "stairwell"],
  "exterior": null
},
{
  "Object": "office building",
  "Type": "Place",
  "file_key": "office building",
  "parent": null,
  "owner": "the speaker",
  "condition": ["cracked marble", "vines overtaking architecture", "seen better days"],
  "atmosphere": ["imposing", "alive", "decaying grandeur"],
  "floors": {
    "1": {"name": "lobby", "file_key": "lobby"},
    "51": {"name": "sex dungeon"},
    "52": {"name": "rage room"},
    "53": {"name": "comfort floor", "features": ["soft", "safe"]},
    "54": {"name": "office", "owner": "the speaker"}
  },
  "exterior": {
    "immediate": ["streets", "empty", "broken cars", "nature retaking", "vines on buildings"],
    "atmosphere": ["abandoned", "post-collapse", "quiet"],
    "objects_of_note": {}
  }
}]

Location rules:
- Only include fields supported by the text — never invent
- Simple things stay as strings in lists
- Notable things (named, detailed, or interesting) become nested objects
- floor_number: include if mentioned or inferable
- file_key: lowercase, spaces allowed, matches how you'd name the file
- parent: the containing space if this is a sub-space
- exterior: only for buildings/structures, null otherwise
- objects_of_note in exterior: same promotion rule — notable things get full entries
- If a location is mentioned but barely described, still capture what's there with minimal fields

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Never use null, Unknown, or empty lists — omit fields not supported by the text
- One entry per entity
- If two speakers describe the same thing differently, include both in the same list
- Do NOT include Actions, Behaviors, Conditions (for people), or transient states
- Preferences and wishes about oneself are not descriptors
""".strip()

_CONVERSATIONAL_FRAME = (
    "Conversational frame: 'Gizmo' or 'you' refers to the AI companion, not a system member. "
    "'I', 'me', 'us', 'we' refers to the plural system. "
    "Lines prefixed 'Gizmo:' are AI responses — do not extract descriptors about Gizmo as if he were a system member."
)

def _build_prompt(
    user_message:    str,
    thread:          str,
    current_location: Optional[str] = None,
    known_entities:  Optional[dict] = None,
) -> str:
    parts = [_CONVERSATIONAL_FRAME]
    if current_location:
        parts.append(
            f"Current established location: {current_location} — "
            f"descriptors default to this location unless context shifts."
        )
    if known_entities:
        parts.append(
            "What you already know about entities in this conversation — "
            "expand on these if the text supports it, don't restate what's already there:\n"
            + json.dumps(known_entities, indent=2)
        )
    parts.append(f"The original message:\n{user_message}")
    parts.append(f"Thread summary:\n{thread}")
    return "\n\n".join(parts)


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm

        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SYSTEM,
            temperature=0.0,
            max_new_tokens=8000,
        )

        if not raw or not raw.strip():
            log_event("DescriptorCatcher", "EMPTY_RESPONSE")
            return None

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return clean

    except Exception as e:
        log_error("DescriptorCatcher", "LLM call failed", exc=e)
        print(f"[DescriptorCatcher] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── Reshape ───────────────────────────────────────────────────────────────────

def _reshape(raw_list: list) -> dict:
    """
    Convert [{Object: "Ember", Type: "Person", ...}, ...]
    into     {"Ember": {"Type": "Person", ...}, ...}

    For locations, uses file_key if present for the dict key.
    """
    result = {}
    for entry in raw_list:
        name = entry.get("file_key") or entry.get("Object")
        if not name:
            continue
        data = {k: v for k, v in entry.items() if k not in ("Object", "file_key")}
        result[name] = data
    return result


def _detect_location_anchor(raw_list: list) -> Optional[str]:
    """
    If any entry in the result is a Place, return its file_key or name.
    Used to track the current location context across chunks.
    """
    for entry in raw_list:
        if entry.get("Type") == "Place":
            return entry.get("file_key") or entry.get("Object")
    return None


# ── Public API ────────────────────────────────────────────────────────────────

class DescriptorCatcher:

    def __init__(self):
        self._current_location: Optional[str] = None

    async def extract(
        self,
        user_message: str,
        thread:       str,
        subject:      str,
        session_file: str,
    ) -> Optional[dict]:
        print("[DescriptorCatcher] extract called")
        if not user_message.strip():
            return None
        try:
            # Read existing descriptor files for known entities so the LLM
            # can expand on them rather than restating or duplicating
            known_entities = {}
            import core.librarian as _lib
            import os, re as _re

            # Always include the current location if we have one
            if self._current_location:
                loc_data = _lib._read_file(f"descriptors/{self._current_location.lower()}.json")
                if loc_data:
                    known_entities[self._current_location] = loc_data

            # Include the primary subject
            if subject and subject.lower() != "unknown":
                subj_data = _lib._read_file(f"descriptors/{subject.lower()}.json")
                if subj_data:
                    known_entities[subject] = subj_data

            prompt  = _build_prompt(
                user_message,
                thread,
                self._current_location,
                known_entities if known_entities else None,
            )
            raw_str = await _call_llm(prompt)

            if not raw_str:
                log_event("DescriptorCatcher", "NO_CONTEXT_EXTRACTED",
                    subject=subject,
                    session=session_file,
                )
                return None

            raw_list = json.loads(raw_str)
            if not isinstance(raw_list, list) or len(raw_list) == 0:
                return None

            # Update location anchor if a place was described
            new_anchor = _detect_location_anchor(raw_list)
            if new_anchor:
                self._current_location = new_anchor
                print(f"[DescriptorCatcher] location anchor: {new_anchor}")

            return _reshape(raw_list)

        except Exception as e:
            log_error("DescriptorCatcher", "extract failed", exc=e)
            print(f"[DescriptorCatcher] extract failed: {type(e).__name__}: {e}")
            return None

    def reset_location(self) -> None:
        """Call when a session ends or location context should clear."""
        self._current_location = None


descriptor_catcher = DescriptorCatcher()
