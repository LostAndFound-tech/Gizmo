"""
core/memory/message.py

The Message object. Everything the pipeline needs in one place.

Instead of passing loose strings around, the pipeline works from a
Message that carries:
  - Full scene context (location, props, atmosphere, characters)
  - Who said what, who's present, who's directing
  - Last two exchanges as compact summaries
  - Classification metadata

Built in server.py from raw input.
Consumed by agent.py instead of raw strings.
Scene state lives in SessionContext and gets written back to place docs.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from core.log import log_event, log_error


# ── Scene character ───────────────────────────────────────────────────────────

@dataclass
class SceneCharacter:
    name:        str
    role:        str   = "neutral"    # dominant|submissive|neutral|observer
    present:     bool  = True
    last_said:   str   = ""
    disposition: str   = ""           # current state in scene


# ── Scene ─────────────────────────────────────────────────────────────────────

@dataclass
class Scene:
    """
    Live scene state. Extracted from conversation, updated as scene progresses.
    Persists in SessionContext between messages.
    Writes back to place docs when new things are established.
    """
    # Where
    location:     Optional[str]  = None   # place name e.g. "jess's room"
    location_doc: Optional[str]  = None   # loaded place doc content
    interior:     bool           = False  # internal world scene

    # What's present
    props:        list[str]      = field(default_factory=list)
    atmosphere:   str            = ""     # lighting, smell, temperature, feel

    # Who
    characters:   list           = field(default_factory=list)  # SceneCharacter list

    # State
    scene_status: str            = "establishing"  # establishing|active|peak|winding_down|closed
    scene_type:   str            = "conversation"  # intimate|roleplay|conversation|task|exploration
    last_action:  Optional[str]  = None
    active_instructions: list[str] = field(default_factory=list)

    # Arc
    narrative:       str   = ""
    established_at:  float = field(default_factory=time.time)
    last_updated:    float = field(default_factory=time.time)

    # What's new this message — for writing back to place docs
    new_props:       list[str] = field(default_factory=list)
    new_atmosphere:  str       = ""

    def get_character(self, name: str) -> Optional[SceneCharacter]:
        for c in self.characters:
            if c.name.lower() == name.lower():
                return c
        return None

    def upsert_character(
        self,
        name:        str,
        role:        str = "neutral",
        last_said:   str = "",
        disposition: str = "",
        present:     bool = True,
    ) -> SceneCharacter:
        c = self.get_character(name)
        if c:
            if role != "neutral":
                c.role = role
            if last_said:
                c.last_said = last_said[:200]
            if disposition:
                c.disposition = disposition
            c.present = present
        else:
            c = SceneCharacter(
                name        = name,
                role        = role,
                last_said   = last_said[:200],
                disposition = disposition,
                present     = present,
            )
            self.characters.append(c)
        return c

    def to_prompt_block(self) -> str:
        if not self.location and not self.characters and not self.props:
            return ""

        lines = ["[Scene]"]

        if self.location:
            loc_line = f"  Location: {self.location}"
            if self.interior:
                loc_line += " (interior)"
            lines.append(loc_line)

        if self.atmosphere:
            lines.append(f"  Atmosphere: {self.atmosphere}")

        if self.props:
            lines.append(f"  Props: {', '.join(self.props[:8])}")

        if self.characters:
            lines.append("  Characters:")
            for c in self.characters:
                if not c.present:
                    continue
                role_str = f" [{c.role}]" if c.role != "neutral" else ""
                disp_str = f" — {c.disposition}" if c.disposition else ""
                said_str = f' / last: "{c.last_said[:80]}"' if c.last_said else ""
                lines.append(f"    {c.name}{role_str}{disp_str}{said_str}")

        if self.active_instructions:
            lines.append("  Active instructions:")
            for inst in self.active_instructions[:4]:
                lines.append(f"    - {inst}")

        if self.last_action:
            lines.append(f"  Last action: {self.last_action}")

        lines.append(f"  Status: {self.scene_status} | type: {self.scene_type}")

        return "\n".join(lines)


# ── Message summary ───────────────────────────────────────────────────────────

@dataclass
class MessageSummary:
    """Compact record of one exchange."""
    user:      str
    gizmo:     str
    speakers:  list[str]
    register:  str
    timestamp: float
    narrative: str = ""   # one sentence summary


# ── Message ───────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """
    Full message object. Everything the pipeline needs.
    Built in server.py, consumed by agent.py.
    """
    # Core content
    raw:       str                    # original input verbatim
    assembled: str                    # scene text if multi-part, else raw
    parts:     list[dict]             # parsed parts

    # Identity
    primary_speaker:  Optional[str]   # who's driving
    speakers:         list[str]       # everyone who said something
    fronters:         list[str]       # everyone present
    directed_at:      Optional[str]   # who it's addressed to

    # Classification
    register:    str
    topics:      list[str]
    is_multi:    bool
    word_count:  int
    has_intimate: bool = False

    # Time
    timestamp:   float = field(default_factory=time.time)
    session_id:  str   = ""

    # Conversation context — populated from SessionContext
    previous:    Optional[MessageSummary] = None
    before_that: Optional[MessageSummary] = None

    # Scene — populated from SessionContext
    scene:       Optional[Scene] = None

    def to_llm_string(self) -> str:
        """
        The string passed as the user message to the LLM.
        Assembled scene text — full, nothing dropped.
        """
        return self.assembled

    def context_block(self) -> str:
        """
        Render conversation + scene context for the system prompt.
        """
        lines = []

        # Scene
        if self.scene:
            block = self.scene.to_prompt_block()
            if block:
                lines.append(block)

        # Last two exchanges
        if self.previous:
            p = self.previous
            speakers_str = ", ".join(p.speakers) if p.speakers else "User"
            lines.append(f"[Previous exchange — {speakers_str}]")
            lines.append(f"  {speakers_str}: {p.user[:200]}")
            if p.gizmo:
                lines.append(f"  Gizmo: {p.gizmo[:200]}")
            if p.narrative:
                lines.append(f"  → {p.narrative}")

        if self.before_that:
            b = self.before_that
            lines.append(f"[Before that]")
            lines.append(f"  {b.user[:150]}")

        return "\n".join(lines) if lines else ""


# ── Scene extractor ───────────────────────────────────────────────────────────

class SceneExtractor:
    """
    Extracts and updates scene state from conversation.
    Runs every 3rd message alongside narrative update.
    Writes new props/atmosphere back to place docs.
    """

    async def extract(
        self,
        assembled:   str,
        parts:       list[dict],
        current_scene: Optional[Scene],
        headmate:    Optional[str],
        session_id:  str,
        llm,
    ) -> Scene:
        """
        Extract or update scene state from the current message.
        Returns updated Scene.
        """
        scene = current_scene or Scene()

        # Build current scene summary for context
        current_summary = ""
        if scene.location:
            current_summary = (
                f"Current location: {scene.location}\n"
                f"Props: {', '.join(scene.props) or 'none'}\n"
                f"Atmosphere: {scene.atmosphere or 'unknown'}\n"
                f"Characters: {', '.join(c.name for c in scene.characters) or 'none'}\n"
                f"Status: {scene.scene_status}"
            )

        prompt = f"""Analyze this message for scene information.

{f'Current scene:{chr(10)}{current_summary}' if current_summary else 'No established scene yet.'}

Message:
{assembled}

Extract scene information. Return ONE JSON object:
{{
  "location": "place name if mentioned or established, else null",
  "interior": true/false (is this the internal world?),
  "atmosphere": "lighting, smell, feel — only if mentioned, else null",
  "new_props": ["objects newly established as present"],
  "characters": [
    {{
      "name": "character name",
      "role": "dominant|submissive|neutral|observer",
      "present": true/false,
      "last_said": "their most recent line if any",
      "disposition": "current state — e.g. pinned, aroused, laughing"
    }}
  ],
  "last_action": "most recent physical action described, else null",
  "new_instructions": ["new instructions/tasks established in this message"],
  "fulfilled_instructions": ["instructions that were just completed"],
  "scene_status": "establishing|active|peak|winding_down|closed",
  "scene_type": "intimate|roleplay|conversation|task|exploration",
  "scene_shift": true/false
}}

Only include what's actually in the message. Null for anything not mentioned.
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You extract scene state from conversation for an AI system. "
                    "JSON only. Only what's explicitly in the message. No invention."
                ),
                max_new_tokens=500,
                temperature=0.1,
            )
        except Exception as e:
            log_error("SceneExtractor", f"extraction failed: {e}", exc=None)
            return scene

        if not raw or not raw.strip():
            return scene

        # Parse
        data = None
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(clean)
        except Exception:
            for line in clean.splitlines():
                if line.strip().startswith("{"):
                    try:
                        data = json.loads(line.strip())
                        break
                    except Exception:
                        continue

        if not data:
            return scene

        # Update location
        if data.get("location") and not scene.location:
            scene.location = data["location"]
            scene.interior = bool(data.get("interior", False))
            # Load place doc
            scene.location_doc = await _load_place_doc(
                data["location"], scene.interior
            )

        # Update atmosphere
        if data.get("atmosphere"):
            scene.atmosphere = data["atmosphere"]
            scene.new_atmosphere = data["atmosphere"]

        # Add new props
        new_props = data.get("new_props") or []
        for prop in new_props:
            if prop and prop not in scene.props:
                scene.props.append(prop)
                scene.new_props.append(prop)

        # Update characters
        for char_data in (data.get("characters") or []):
            name = char_data.get("name", "").strip()
            if not name:
                continue
            scene.upsert_character(
                name        = name,
                role        = char_data.get("role", "neutral"),
                last_said   = char_data.get("last_said", ""),
                disposition = char_data.get("disposition", ""),
                present     = bool(char_data.get("present", True)),
            )

        # Update instructions
        for inst in (data.get("new_instructions") or []):
            if inst and inst not in scene.active_instructions:
                scene.active_instructions.append(inst)

        # Fulfill completed instructions
        for fulfilled in (data.get("fulfilled_instructions") or []):
            scene.active_instructions = [
                i for i in scene.active_instructions
                if fulfilled.lower() not in i.lower()
            ]

        # Update state
        if data.get("last_action"):
            scene.last_action = data["last_action"]
        if data.get("scene_status"):
            scene.scene_status = data["scene_status"]
        if data.get("scene_type"):
            scene.scene_type = data["scene_type"]

        scene.last_updated = time.time()

        log_event("SceneExtractor", "SCENE_UPDATED",
            session      = session_id[:8],
            location     = scene.location or "unknown",
            characters   = len(scene.characters),
            props        = len(scene.props),
            status       = scene.scene_status,
        )

        # Write new props/atmosphere back to place doc
        if scene.location and (scene.new_props or scene.new_atmosphere):
            asyncio.ensure_future(
                _update_place_doc(scene, session_id)
            )
            scene.new_props      = []
            scene.new_atmosphere = ""

        return scene


async def _load_place_doc(name: str, interior: bool) -> Optional[str]:
    """Load a place doc from the memory store."""
    try:
        from core.memory.store import memory_store
        return (
            memory_store.read_place(name, interior=interior)
            or memory_store.read_place(name, interior=not interior)
        )
    except Exception:
        return None


async def _update_place_doc(scene: Scene, session_id: str) -> None:
    """
    Write new scene discoveries back to the place doc.
    New props, atmosphere changes, things established during a scene.
    """
    try:
        from core.memory.store import memory_store

        if not scene.location:
            return

        additions = []

        if scene.new_props:
            additions.append(
                f"Props established during scene ({_fmt_date()}): "
                + ", ".join(scene.new_props)
            )
        if scene.new_atmosphere:
            additions.append(
                f"Atmosphere noted ({_fmt_date()}): {scene.new_atmosphere}"
            )

        if not additions:
            return

        update_text = "\n".join(additions)
        updated = memory_store.update_place(
            scene.location,
            update_text,
            interior=scene.interior,
        )

        if not updated:
            # Place doc doesn't exist yet — create it
            content = f"Established during session {session_id[:8]}.\n\n"
            if scene.atmosphere:
                content += f"Atmosphere: {scene.atmosphere}\n"
            if scene.props:
                content += f"Props: {', '.join(scene.props)}\n"
            memory_store.write_place(
                name     = scene.location,
                content  = content,
                interior = scene.interior,
                keywords = scene.location.lower(),
            )

        log_event("SceneExtractor", "PLACE_UPDATED",
            location = scene.location,
            session  = session_id[:8],
        )

    except Exception as e:
        log_error("SceneExtractor", f"place doc update failed: {e}", exc=None)


def _fmt_date() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Singleton ─────────────────────────────────────────────────────────────────

scene_extractor = SceneExtractor()
