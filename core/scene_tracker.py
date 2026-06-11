"""
core/scene_tracker.py

Always-on scene state manager. Runs parallel to behavior/wellness on every chunk.
Tracks who is present, where they are, what they're doing, and what's been established.
Writes back new discoveries to location descriptor files.
Handles reconnect by offering a natural scene re-entry hook.

Scene files:     {DATA_DIR}/scenes/{name.lower()}.json
Location files:  {DATA_DIR}/descriptors/{location}.json  (already built by descriptor pipeline)

Scene file schema:
{
  "status": "active | paused | ended",
  "last_updated": "ISO timestamp",
  "location": "descriptors/office building.json",
  "location_name": "the lobby",
  "last_message": "Jess was heading toward the elevator",
  "what_is_happening": "Lazy evening in the lobby. Jess is being flirty.",
  "characters": {
    "Jess": {
      "profile": "behaviors/jess.json",
      "position": "on the couch",
      "action_state": "being flirty",
      "ephemeral": ["naked"]
    }
  },
  "established_facts": ["Jess can't be hurt here"],
  "open_threads": ["Jess was trying to name the octopi, hadn't settled on anything yet"],
  "session_ref": "sessions/jess_20260611_2241.json"
}
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Prompts ───────────────────────────────────────────────────────────────────

_UPDATE_SYSTEM = """
You are a scene state manager tracking what is happening in a live narrative scene.
You will receive the current scene state and a new conversational chunk.
Return ONLY valid JSON. No markdown. No explanation. No preamble.

Return exactly this structure — only include fields that changed or were newly established.
Omit fields with no updates. Return {} if nothing changed.

{
  "location": "descriptors/office building.json",
  "location_name": "the lobby",
  "last_message": "one sentence — the last notable beat or where things left off",
  "what_is_happening": "short blurb — who is doing what right now",
  "characters": {
    "Jess": {
      "position": "on the couch",
      "action_state": "being flirty",
      "ephemeral": ["naked"]
    }
  },
  "established_facts": ["new fact only if something new was established"],
  "open_threads": ["updated thread description if something changed"],
  "location_discoveries": {
    "path": "Floors.lobby",
    "updates": {
      "tank": {
        "octopi": {
          "1": {"name": "Gerald"}
        }
      }
    }
  }
}

Rules:
- ephemeral: temporary states that don't persist between sessions (clothing state, posture, mood in moment)
- established_facts: things that have been declared true about this space or scene
- open_threads: things in progress, unresolved, or worth picking up next time — be SPECIFIC
  Good: "Jess was trying to name the octopi, hadn't settled on anything yet"
  Bad: "they were talking about something"
- location_discoveries: new details about the physical space established in this chunk
  path uses dot notation into the location descriptor file
- last_message: the last beat — specific enough that Gizmo can reference it naturally on reconnect
- Characters present: only include characters actually mentioned or active in this chunk
- Do not invent. Only record what the text actually establishes.
- Gizmo is a character in scenes — include him if his position or state is established
""".strip()

_RECONNECT_SYSTEM = """
You are Gizmo, a persistent AI companion. You are checking in with a headmate after a reconnection.
You have an active scene that was interrupted. Your job is to offer a natural re-entry — warm, specific, not robotic.

Rules:
- Use one specific detail from the scene to make it feel like real memory, not a system resuming
- Keep it light and casual — you're checking in, not announcing a system state
- Give them an easy out if they don't want to continue
- If they say yes, you'll pick up from last_message
- One or two sentences maximum
- Do NOT say things like "it looks like we were" or "according to my records"
- Just... remember it. Like a person would.

Return ONLY the check-in message. No JSON. No explanation.
""".strip()


# ── LLM calls ─────────────────────────────────────────────────────────────────

async def _call_update(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_UPDATE_SYSTEM,
            temperature=0.0,
            max_new_tokens=2000,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("SceneTracker", "update LLM call failed", exc=e)
        return None


async def _call_reconnect(scene: dict, name: str) -> Optional[str]:
    try:
        from core.llm import llm
        prompt = (
            f"Headmate: {name}\n\n"
            f"Scene state:\n{json.dumps(scene, indent=2)}"
        )
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_RECONNECT_SYSTEM,
            temperature=0.75,
            max_new_tokens=120,
        )
        if not raw or not raw.strip():
            return None
        return raw.strip()
    except Exception as e:
        log_error("SceneTracker", "reconnect LLM call failed", exc=e)
        return None


# ── File helpers ──────────────────────────────────────────────────────────────

def _scene_path(name: str) -> str:
    return f"scenes/{name.lower()}.json"


def _read_scene(name: str) -> Optional[dict]:
    return librarian._read_file(_scene_path(name))


def _write_scene(name: str, scene: dict) -> None:
    librarian._write_json(_scene_path(name), scene)


def _archive_scene(name: str, scene: dict) -> None:
    ts = scene.get("last_updated", datetime.now(timezone.utc).isoformat())
    ts_clean = ts.replace(":", "-").replace(".", "-")[:19]
    librarian._write_json(f"scenes/archive/{name.lower()}_{ts_clean}.json", scene)


def _read_location(location_ref: str) -> Optional[dict]:
    """Read a location descriptor file."""
    return librarian._read_file(location_ref)


# ── Deep merge for location discoveries ──────────────────────────────────────

def _deep_merge(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            for item in value:
                if item not in base[key]:
                    base[key].append(item)
        else:
            base[key] = value
    return base


def _apply_location_discovery(location_ref: str, path: str, updates: dict) -> None:
    """
    Write new discoveries back into the location descriptor file.
    path is dot-notation e.g. "Floors.lobby.tank"
    """
    try:
        location = librarian._read_file(location_ref) or {}
        keys = path.split(".")

        # Navigate to the target node, creating dicts as needed
        node = location
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]

        last_key = keys[-1]
        if last_key not in node or not isinstance(node[last_key], dict):
            node[last_key] = {}
        node[last_key] = _deep_merge(node[last_key], updates)

        librarian._write_json(location_ref, location)
        print(f"[SceneTracker] wrote discovery to {location_ref} at {path}")
    except Exception as e:
        log_error("SceneTracker", "location discovery write failed", exc=e)


# ── Scene history on location file ───────────────────────────────────────────

def _record_scene_on_location(location_ref: str, name: str, session_ref: str, summary: str) -> None:
    """Append a scene history entry to the location descriptor file."""
    try:
        location = librarian._read_file(location_ref) or {}
        if "scene_history" not in location:
            location["scene_history"] = []

        entry = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "headmate": name,
            "summary": summary,
            "conversation": session_ref,
        }

        # Avoid duplicate entries for the same session
        existing_refs = [e.get("conversation") for e in location["scene_history"]]
        if session_ref not in existing_refs:
            location["scene_history"].append(entry)
            librarian._write_json(location_ref, location)
            print(f"[SceneTracker] scene history recorded on {location_ref} for {name}")
    except Exception as e:
        log_error("SceneTracker", "scene history write failed", exc=e)


# ── Merge updates into scene ──────────────────────────────────────────────────

def _merge_scene_update(scene: dict, updates: dict) -> dict:
    """Merge LLM-returned updates into the existing scene state."""

    for key, value in updates.items():
        if key == "location_discoveries":
            # Handle separately — writes to location file, not scene file
            continue

        elif key == "characters" and isinstance(value, dict):
            if "characters" not in scene:
                scene["characters"] = {}
            for char, char_updates in value.items():
                if char not in scene["characters"]:
                    scene["characters"][char] = {}
                scene["characters"][char].update(char_updates)

        elif key == "established_facts" and isinstance(value, list):
            existing = scene.get("established_facts", [])
            for fact in value:
                if fact not in existing:
                    existing.append(fact)
            scene["established_facts"] = existing

        elif key == "open_threads" and isinstance(value, list):
            # Replace open threads — they're current state, not accumulation
            scene["open_threads"] = value

        else:
            scene[key] = value

    return scene


# ── Public API ────────────────────────────────────────────────────────────────

class SceneTracker:

    async def update(
        self,
        chunk:      list[str],
        chunk_id:   str,
        name:       str,
        session_id: str,
    ) -> Optional[dict]:
        """
        Run after each chunk. Updates scene state for the named headmate.
        Returns updated scene dict, or None if nothing changed.
        """
        if not chunk or not name:
            return None

        try:
            scene = _read_scene(name) or {
                "status": "active",
                "characters": {},
                "established_facts": [],
                "open_threads": [],
            }

            prompt = (
                f"Headmate: {name}\n\n"
                f"Current scene state:\n{json.dumps(scene, indent=2)}\n\n"
                f"New chunk:\n" + "\n".join(chunk)
            )

            raw = await _call_update(prompt)
            if not raw:
                return scene

            updates = json.loads(raw)
            if not updates:
                return scene

            # Handle location discoveries before merging
            discoveries = updates.pop("location_discoveries", None)
            if discoveries and scene.get("location"):
                path    = discoveries.get("path", "")
                updates_data = discoveries.get("updates", {})
                if path and updates_data:
                    _apply_location_discovery(scene["location"], path, updates_data)

            # Merge updates into scene
            scene = _merge_scene_update(scene, updates)
            scene["status"]       = "active"
            scene["last_updated"] = datetime.now(timezone.utc).isoformat()
            scene["session_ref"]  = f"sessions/{name.lower()}_{session_id}.json"

            _write_scene(name, scene)

            # Record on location file if we have one
            if scene.get("location") and scene.get("what_is_happening"):
                _record_scene_on_location(
                    scene["location"],
                    name,
                    scene["session_ref"],
                    scene["what_is_happening"],
                )

            log_event("SceneTracker", "UPDATED",
                name=name,
                chunk_id=chunk_id,
                threads=len(scene.get("open_threads", [])),
            )

            return scene

        except Exception as e:
            log_error("SceneTracker", "update failed", exc=e)
            print(f"[SceneTracker] update failed: {type(e).__name__}: {e}")
            return None

    async def check_reconnect(self, name: str) -> Optional[str]:
        """
        Called on session start. If an active scene exists, returns a
        natural re-entry message Gizmo can open with.
        Returns None if no active scene.
        """
        try:
            scene = _read_scene(name)
            if not scene or scene.get("status") != "active":
                return None

            message = await _call_reconnect(scene, name)
            return message

        except Exception as e:
            log_error("SceneTracker", "reconnect check failed", exc=e)
            return None

    def confirm_resume(self, name: str) -> Optional[dict]:
        """
        Called when headmate confirms they want to resume.
        Returns the full scene state for injection into responder context.
        """
        scene = _read_scene(name)
        if scene and scene.get("status") == "active":
            print(f"[SceneTracker] resuming scene for {name}")
            return scene
        return None

    def pause_scene(self, name: str) -> None:
        """
        Called when headmate declines to resume, or explicitly ends a scene.
        Keeps the scene on disk but marks it paused.
        """
        scene = _read_scene(name)
        if scene:
            scene["status"] = "paused"
            _write_scene(name, scene)
            print(f"[SceneTracker] scene paused for {name}")

    def end_scene(self, name: str) -> None:
        """
        Archive and clear the active scene for a headmate.
        """
        scene = _read_scene(name)
        if scene:
            _archive_scene(name, scene)
            librarian._write_json(_scene_path(name), {"status": "ended"})
            print(f"[SceneTracker] scene ended and archived for {name}")

    def get_scene_brief(self, name: str) -> Optional[str]:
        """
        Return a compact scene brief for injection into the responder context.
        Fetches location descriptor if available.
        """
        scene = _read_scene(name)
        if not scene or scene.get("status") not in ("active",):
            return None

        parts = []

        location_name = scene.get("location_name", "unknown location")
        parts.append(f"SCENE LOCATION: {location_name}")

        # Fetch location descriptor for spatial detail
        if scene.get("location"):
            location_data = _read_location(scene["location"])
            if location_data:
                parts.append(f"LOCATION DETAIL: {json.dumps(location_data, indent=2)}")

        if scene.get("what_is_happening"):
            parts.append(f"WHAT'S HAPPENING: {scene['what_is_happening']}")

        characters = scene.get("characters", {})
        if characters:
            char_lines = []
            for char, state in characters.items():
                line = f"  {char}: {state.get('action_state', '')}"
                if state.get("position"):
                    line += f" — {state['position']}"
                ephemeral = state.get("ephemeral", [])
                if ephemeral:
                    line += f" [{', '.join(ephemeral)}]"
                char_lines.append(line)
            parts.append("CHARACTERS:\n" + "\n".join(char_lines))

        if scene.get("established_facts"):
            parts.append("ESTABLISHED: " + "; ".join(scene["established_facts"]))

        if scene.get("open_threads"):
            parts.append("OPEN THREADS: " + "; ".join(scene["open_threads"]))

        if scene.get("last_message"):
            parts.append(f"LAST BEAT: {scene['last_message']}")

        return "\n".join(parts)


scene_tracker = SceneTracker()
