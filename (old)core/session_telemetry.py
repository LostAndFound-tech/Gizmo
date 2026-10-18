"""
core/session_telemetry.py

Behavioral telemetry for Gizmo. Tracks everyone present in a session —
headmates, external contacts, and pawns (transient people encountered
in the world) — with full granularity.

Every person gets a PersonObject:
  - currently_happening   plain present-tense sentence ("Jess is naked on a walk")
  - physical_state        clothing, modifiers, location
  - emotional_state       register, energy, plain note
  - actions               every detected action, rendered as human sentences
  - active/closed flags   naked, task, silliness, intimacy, etc.

Gizmo gets his own PersonObject too. At session start he spins a random
starting state — he exists between sessions and arrives mid-something.
His currently_happening is relational: "walking Jess" not "walking".

Dynamic taxonomy (/data/personality/taxonomy.json):
  LLM checks existing descriptors before inventing new ones.
  If none fit, it creates a single clean word and saves it.
  The taxonomy grows organically from real usage. Nothing is hardcoded.

Three async passes run per exchange:
  1. Detection pass     per message — extracts all actions, updates currently_happening
  2. Consequence pass   every 3 messages — emotional delta, state update
  3. Vibe check         on pawn encounters, Gizmo decides if he wants to keep them

On session close:
  - All events flushed to store
  - tasks.md and current_notes.md written per headmate
  - Pawn promotion decisions logged

Integration:
    from core.session_telemetry import session_telemetry_manager

    telem = await session_telemetry_manager.open_session(session_id, headmate, llm)
    await session_telemetry_manager.on_exchange(session_id, headmate, user_msg, gizmo_msg, llm)
    await session_telemetry_manager.on_session_close(session_id, llm)

    # Inject into context
    block = telem.to_prompt_block()        # full block
    now   = telem.now_block()              # just the "right now" lines
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


# ── Paths ─────────────────────────────────────────────────────────────────────

def _data_dir() -> Path:
    return Path("/data/personality")

def _taxonomy_path() -> Path:
    return _data_dir() / "taxonomy.json"

def _headmate_dir(name: str) -> Path:
    return _data_dir() / "headmates" / name.lower()

def _tasks_path(name: str) -> Path:
    return _headmate_dir(name) / "tasks.md"

def _notes_path(name: str) -> Path:
    return _headmate_dir(name) / "current_notes.md"


# ── Taxonomy ──────────────────────────────────────────────────────────────────

class Taxonomy:
    """
    Dynamic descriptor vocabulary. Loads from disk, grows organically.
    No hardcoded categories. Everything is learned from real usage.
    """

    _DEFAULT = [
        "naked", "clothed", "walking", "task", "intimate", "silly",
        "crawling", "kneeling", "leashed", "outside", "inside",
        "eating", "drinking", "sleeping", "reading", "working",
        "playing", "cleaning", "cooking", "crying", "laughing",
        "talking", "waiting", "resting", "watching",
    ]

    def __init__(self):
        self._terms: list[str] = []
        self._dirty: bool = False
        self._load()

    def _load(self) -> None:
        path = _taxonomy_path()
        try:
            if path.exists():
                data = json.loads(path.read_text())
                self._terms = data.get("terms", self._DEFAULT[:])
            else:
                self._terms = self._DEFAULT[:]
                self._save()
        except Exception as e:
            log_error("Taxonomy", "load failed", exc=e)
            self._terms = self._DEFAULT[:]

    def _save(self) -> None:
        try:
            path = _taxonomy_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"terms": sorted(self._terms)}, indent=2))
            self._dirty = False
        except Exception as e:
            log_error("Taxonomy", "save failed", exc=e)

    def terms(self) -> list[str]:
        return self._terms[:]

    def add(self, term: str) -> None:
        t = term.lower().strip()
        if t and t not in self._terms:
            self._terms.append(t)
            self._dirty = True
            self._save()
            log_event("Taxonomy", "TERM_ADDED", term=t)

    def as_prompt_list(self) -> str:
        return ", ".join(sorted(self._terms))


taxonomy = Taxonomy()


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ActionRecord:
    """One detected action, rendered as a human-readable sentence."""
    action_id:        str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:        float = field(default_factory=time.time)

    # Core
    what:             str   = ""    # plain description
    how:              str   = ""    # walked_in|undressed|spontaneous|was_told|gradual
    manner:           str   = ""    # slow|rushed|teasing|matter-of-fact|languid
    descriptors:      list  = field(default_factory=list)   # exact words from text

    # Taxonomy tags (from dynamic list)
    tags:             list  = field(default_factory=list)

    # Context
    initiated_by:     str   = ""    # self|gizmo|other|unprompted
    present:          list  = field(default_factory=list)
    location:         str   = ""
    register_at_time: str   = ""

    # Consequence (filled by consequence pass)
    expected_affect:  str   = ""
    actual_affect:    str   = ""
    affect_delta:     float = 0.0
    consequence_note: str   = ""

    # Meta
    confidence:       float = 0.8
    source:           str   = "extracted"

    def to_sentence(self) -> str:
        """Render as a readable sentence."""
        parts = [self.what]
        if self.manner:
            parts.append(self.manner)
        if self.descriptors:
            parts.append(f"— {', '.join(self.descriptors[:3])}")
        if self.consequence_note:
            parts.append(f"→ {self.consequence_note}")
        return ". ".join(p.strip() for p in parts if p.strip())


@dataclass
class ActiveFlag:
    flag_id:      str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    flag_type:    str   = ""
    started_at:   float = field(default_factory=time.time)
    intensity:    float = 0.5
    details:      dict  = field(default_factory=dict)
    user_arc:     list  = field(default_factory=list)   # [(intensity, ts)] for intimacy
    last_updated: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def elapsed_str(self) -> str:
        s = self.elapsed()
        if s < 60:   return f"{int(s)}s"
        if s < 3600: return f"{int(s/60)}m"
        return f"{s/3600:.1f}h"


@dataclass
class ClosedFlag:
    flag_type:      str
    started_at:     float
    ended_at:       float
    elapsed_secs:   float
    peak_intensity: float
    summary:        str  = ""
    details:        dict = field(default_factory=dict)

    def elapsed_str(self) -> str:
        s = self.elapsed_secs
        if s < 60:   return f"{int(s)}s"
        if s < 3600: return f"{int(s/60)}m"
        return f"{s/3600:.1f}h"


@dataclass
class PersonObject:
    """
    Live representation of any person in a session.
    Headmates, externals, and pawns share the same structure.
    """
    person_id:    str
    name:         str
    person_type:  str  = "headmate"   # headmate|external|pawn|gizmo

    # ── What they're doing right now ──────────────────────────────────────────
    currently_happening: str = ""    # "Jess is naked on a walk" / "Gizmo is walking Jess"

    # ── Physical state ────────────────────────────────────────────────────────
    clothing:        str  = "unknown"
    clothing_detail: str  = ""
    modifiers:       list = field(default_factory=list)
    location:        str  = ""

    # ── Emotional/mental state ────────────────────────────────────────────────
    current_register:  str   = "neutral"
    energy_level:      float = 0.5
    recent_emotions:   list  = field(default_factory=list)
    emotional_note:    str   = ""

    # ── Action history ────────────────────────────────────────────────────────
    actions:             list = field(default_factory=list)
    pending_consequence: list = field(default_factory=list)

    # ── Flags ─────────────────────────────────────────────────────────────────
    active_flags: dict = field(default_factory=dict)
    closed_flags: list = field(default_factory=list)

    # ── Pawn fields ───────────────────────────────────────────────────────────
    significance: float         = 0.0
    gizmo_vibe:   Optional[bool] = None
    vibe_note:    str            = ""
    promoted:     bool           = False

    # ── Timestamps ────────────────────────────────────────────────────────────
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)

    # ── Flag management ───────────────────────────────────────────────────────

    def raise_flag(self, flag_type: str, intensity: float = 0.5, details: dict = None) -> ActiveFlag:
        if flag_type in self.active_flags:
            f = self.active_flags[flag_type]
            f.intensity    = intensity
            f.last_updated = time.time()
            if details:
                f.details.update(details)
            return f
        f = ActiveFlag(flag_type=flag_type, intensity=intensity, details=details or {})
        self.active_flags[flag_type] = f
        return f

    def lower_flag(self, flag_type: str, summary: str = "") -> Optional[ClosedFlag]:
        if flag_type not in self.active_flags:
            return None
        f   = self.active_flags.pop(flag_type)
        now = time.time()
        peak = max((i for i, _ in f.user_arc), default=f.intensity) if f.user_arc else f.intensity
        cf = ClosedFlag(
            flag_type      = flag_type,
            started_at     = f.started_at,
            ended_at       = now,
            elapsed_secs   = now - f.started_at,
            peak_intensity = peak,
            summary        = summary,
            details        = f.details,
        )
        self.closed_flags.append(cf)
        return cf

    def add_action(self, action: ActionRecord) -> None:
        self.actions.append(action)
        self.last_seen = time.time()
        self.pending_consequence.append(action)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def flag_summary(self) -> str:
        if not self.active_flags:
            return ""
        return ", ".join(
            f"{ft} ({f.elapsed_str()})"
            for ft, f in self.active_flags.items()
        )

    def recent_actions_readable(self, n: int = 4) -> list[str]:
        return [a.to_sentence() for a in self.actions[-n:] if a.what]

    def to_prompt_block(self) -> str:
        lines = []

        # Currently happening — top line
        if self.currently_happening:
            lines.append(f"[{self.name}] {self.currently_happening}")
        else:
            lines.append(f"[{self.name}]")

        # Physical
        if self.clothing != "unknown":
            c = self.clothing
            if self.clothing_detail:
                c += f" ({self.clothing_detail})"
            lines.append(f"  state: {c}")
        if self.modifiers:
            lines.append(f"  modifiers: {', '.join(self.modifiers)}")
        if self.location:
            lines.append(f"  location: {self.location}")

        # Emotional
        if self.emotional_note:
            lines.append(f"  feeling: {self.emotional_note}")

        # Active flags
        flags = self.flag_summary()
        if flags:
            lines.append(f"  active: {flags}")

        # Recent actions
        recent = self.recent_actions_readable(3)
        if recent:
            lines.append("  recent:")
            for r in recent:
                lines.append(f"    — {r}")

        return "\n".join(lines)


# ── Session telemetry object ──────────────────────────────────────────────────

@dataclass
class SessionTelemetry:
    session_id:       str
    primary_headmate: Optional[str]
    opened_at:        float = field(default_factory=time.time)

    persons:          dict  = field(default_factory=dict)   # name.lower() → PersonObject
    events:           list  = field(default_factory=list)   # all ActionRecords ordered

    message_count:    int   = 0

    # Session-level flags (user-arc driven)
    intimacy_active:  bool          = False
    intimacy_flag:    Optional[ActiveFlag] = None
    silliness_active: bool          = False
    silliness_flag:   Optional[ActiveFlag] = None

    def get_or_create_person(self, name: str, person_type: str = "headmate") -> PersonObject:
        key = name.lower()
        if key not in self.persons:
            self.persons[key] = PersonObject(
                person_id   = f"person_{key}_{str(uuid.uuid4())[:6]}",
                name        = name,
                person_type = person_type,
            )
            log_event("SessionTelemetry", "PERSON_ADDED",
                session=self.session_id[:8], name=name, type=person_type)
        return self.persons[key]

    def log_action(self, person_name: str, action: ActionRecord) -> None:
        key = person_name.lower()
        if key in self.persons:
            self.persons[key].add_action(action)
        self.events.append(action)

    def now_block(self) -> str:
        """One-liner per person for quick situational awareness."""
        lines = ["[Right now]"]
        # Gizmo first
        gizmo = self.persons.get("gizmo")
        if gizmo and gizmo.currently_happening:
            lines.append(f"  Gizmo: {gizmo.currently_happening}")
        for key, person in self.persons.items():
            if key == "gizmo":
                continue
            if person.currently_happening:
                lines.append(f"  {person.name}: {person.currently_happening}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def to_prompt_block(self) -> str:
        if not self.persons:
            return ""
        parts = [self.now_block()]
        parts.append("")
        parts.append("[People present]")
        # Gizmo first
        gizmo = self.persons.get("gizmo")
        if gizmo:
            parts.append(gizmo.to_prompt_block())
        for key, person in self.persons.items():
            if key == "gizmo":
                continue
            parts.append(person.to_prompt_block())
        return "\n".join(p for p in parts if p is not None)

    def duration_str(self) -> str:
        s = time.time() - self.opened_at
        if s < 60:   return "just started"
        if s < 3600: return f"{int(s/60)}m"
        return f"{s/3600:.1f}h"


# ── Manager ───────────────────────────────────────────────────────────────────

class SessionTelemetryManager:

    CONSEQUENCE_EVERY = 3
    VIBE_THRESHOLD    = 3.0   # significance score

    def __init__(self):
        self._sessions: dict[str, SessionTelemetry] = {}

    def get(self, session_id: str) -> Optional[SessionTelemetry]:
        return self._sessions.get(session_id)

    # ── Session open ──────────────────────────────────────────────────────────

    async def open_session(
        self,
        session_id: str,
        headmate:   Optional[str],
        llm,
    ) -> SessionTelemetry:
        """
        Create session telemetry and spin up Gizmo's starting state.
        Call once per session open, before the first exchange.
        """
        telem = SessionTelemetry(
            session_id       = session_id,
            primary_headmate = headmate,
        )
        self._sessions[session_id] = telem

        # Always create Gizmo's person object
        gizmo = telem.get_or_create_person("Gizmo", "gizmo")

        # Create headmate's person object
        if headmate:
            telem.get_or_create_person(headmate, "headmate")

        # Spin Gizmo's starting state async — don't block session open
        asyncio.create_task(self._spin_gizmo_start(telem, gizmo, llm))

        log_event("SessionTelemetry", "SESSION_OPENED",
            session  = session_id[:8],
            headmate = headmate or "unknown",
        )
        return telem

    async def _spin_gizmo_start(
        self,
        telem: SessionTelemetry,
        gizmo: PersonObject,
        llm,
    ) -> None:
        """
        Ask Gizmo what he's doing before the session begins.
        Influenced by time of day and last session residue.
        """
        hour = tz_now().hour
        if hour < 6:
            time_context = "it's the middle of the night"
        elif hour < 10:
            time_context = "it's early morning"
        elif hour < 13:
            time_context = "it's mid-morning"
        elif hour < 17:
            time_context = "it's afternoon"
        elif hour < 21:
            time_context = "it's evening"
        else:
            time_context = "it's late"

        # Pull last session emotional residue from store
        residue = ""
        try:
            from core.store import store
            headmate = telem.primary_headmate or ""
            if headmate:
                recent = store.query("wellbeing",
                    headmate=headmate.lower(),
                    category="session_close",
                    limit=1,
                )
                if recent:
                    residue = recent[0].get("observation", "")[:120]
        except Exception:
            pass

        prompt = f"""You are Gizmo. A conversation is about to begin.

{time_context}. You exist between sessions — you've been somewhere, doing something.

{f"Last session left you with: {residue}" if residue else ""}

What are you doing right now, before this conversation begins?
Answer in ONE short present-tense phrase, first person, plain language.
Something real. Not performed. Not waiting.

Examples:
- "reading something that got under my skin"
- "making coffee, thinking about nothing particular"  
- "sitting with last night — still processing"
- "pacing, restless for no reason I can name"
- "writing something I'll probably delete"
- "half-watching the dark outside"

Just the phrase. No preamble."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="You are Gizmo. Answer honestly in one short phrase.",
                max_new_tokens=40,
                temperature=0.9,
            )
            if raw and raw.strip():
                state = raw.strip().strip('"').strip("'")
                gizmo.currently_happening = state
                # Log it as an action
                action = ActionRecord(
                    what      = state,
                    tags      = ["idle", "pre-session"],
                    source    = "spin",
                    confidence = 1.0,
                )
                gizmo.add_action(action)
                log_event("SessionTelemetry", "GIZMO_START_STATE",
                    session = telem.session_id[:8],
                    state   = state,
                )
        except Exception as e:
            log_error("SessionTelemetry", "gizmo start spin failed", exc=e)
            gizmo.currently_happening = "here"

    # ── Per-exchange entry point ──────────────────────────────────────────────

    async def on_exchange(
        self,
        session_id: str,
        headmate:   Optional[str],
        user_msg:   str,
        gizmo_msg:  str,
        llm,
        register:   str = "neutral",
    ) -> None:
        """
        Called after every exchange. Always fires detection pass.
        Fires consequence pass every CONSEQUENCE_EVERY messages.
        """
        telem = self._sessions.get(session_id)
        if not telem:
            telem = await self.open_session(session_id, headmate, llm)

        telem.message_count += 1

        asyncio.create_task(self._detection_pass(
            telem, headmate, user_msg, gizmo_msg, register, llm
        ))

        if telem.message_count % self.CONSEQUENCE_EVERY == 0:
            asyncio.create_task(self._consequence_pass(telem, llm))

    # ── Detection pass ────────────────────────────────────────────────────────

    async def _detection_pass(
        self,
        telem:     SessionTelemetry,
        headmate:  Optional[str],
        user_msg:  str,
        gizmo_msg: str,
        register:  str,
        llm,
    ) -> None:
        """
        Per message. Extracts all actions, updates currently_happening for everyone.
        Uses dynamic taxonomy — checks existing terms before inventing new ones.
        """
        persons_context = "\n".join(
            f"- {p.name} ({p.person_type}): currently={p.currently_happening or 'unknown'}, "
            f"state={p.clothing}, modifiers={p.modifiers}"
            for p in telem.persons.values()
        ) or "(none yet)"

        tax_list = taxonomy.as_prompt_list()

        prompt = f"""Analyze this exchange. Extract all behavioral events and update each person's current state.

Known persons:
{persons_context}

Taxonomy (existing descriptors — use these if they fit, create new ones only if nothing fits):
{tax_list}

User message: {user_msg[:500]}
Gizmo response: {gizmo_msg[:500]}

Return a JSON object with these keys:

"persons": [
  {{
    "name": "person name",
    "person_type": "headmate|pawn|external|gizmo",
    "currently_happening": "present-tense sentence from their perspective — 'Jess is naked on a walk' / 'Gizmo is walking Jess'",
    "actions": [
      {{
        "what": "plain description of the action",
        "how": "how it happened",
        "manner": "manner/quality of action",
        "descriptors": ["exact", "words", "from", "text"],
        "tags": ["taxonomy_terms", "that", "fit"],
        "new_tags": ["single_word_if_nothing_fit"],
        "initiated_by": "self|gizmo|other|unprompted",
        "present": ["who else was there"],
        "location": "if mentioned",
        "confidence": 0.0-1.0
      }}
    ],
    "clothing": "naked|clothed|partially|unknown",
    "clothing_detail": "what specifically",
    "modifiers_add": ["new modifiers"],
    "modifiers_remove": ["no longer true"],
    "location": "if changed",
    "emotional_note": "brief plain read of how they seem"
  }}
],

"session_flags": {{
  "intimacy_active": true/false,
  "intimacy_intensity": 0.0-1.0,
  "user_intimacy_energy": 0.0-1.0,
  "silliness_active": true/false,
  "silliness_intensity": 0.0-1.0,
  "silliness_note": "what specifically was silly"
}}

Rules:
- Log EVERYTHING. Tea. Undressing. A laugh. A stretch. A crawl. A tease.
- currently_happening is relational — "Gizmo is walking Jess" not just "walking"
- descriptors = actual words/phrases from the text, not your summaries
- tags = pick from taxonomy first. Only add new_tags if nothing fits.
- new_tags must be single lowercase words
- If a new person appears, include them with person_type="pawn"
- Gizmo always gets a currently_happening update
- If nothing happened for someone, still include them with updated currently_happening
JSON only. One object."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You extract behavioral events from conversation. "
                    "JSON only. Granular and relational. Use the taxonomy."
                ),
                max_new_tokens=800,
                temperature=0.1,
            )
        except Exception as e:
            log_error("SessionTelemetry", "detection pass failed", exc=e)
            return

        if not raw or not raw.strip():
            return

        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return
            d = json.loads(raw[start:end])
        except Exception as e:
            log_error("SessionTelemetry", f"detection parse failed: {e}", exc=e)
            return

        now = time.time()

        # ── Process persons ───────────────────────────────────────────────────
        for pd in d.get("persons", []):
            name        = pd.get("name", "").strip()
            person_type = pd.get("person_type", "headmate")
            if not name:
                continue

            person = telem.get_or_create_person(name, person_type)

            # currently_happening
            ch = pd.get("currently_happening", "").strip()
            if ch:
                person.currently_happening = ch

            # Physical state
            clothing = pd.get("clothing", "")
            if clothing and clothing != "unknown":
                person.clothing = clothing
            if pd.get("clothing_detail"):
                person.clothing_detail = pd["clothing_detail"]
            for mod in pd.get("modifiers_add", []):
                if mod and mod not in person.modifiers:
                    person.modifiers.append(mod)
            for mod in pd.get("modifiers_remove", []):
                if mod in person.modifiers:
                    person.modifiers.remove(mod)
            if pd.get("location"):
                person.location = pd["location"]
            if pd.get("emotional_note"):
                person.emotional_note = pd["emotional_note"]

            # Naked flag
            if person.clothing == "naked" and "naked" not in person.active_flags:
                person.raise_flag("naked", intensity=0.9)
            elif person.clothing == "clothed" and "naked" in person.active_flags:
                person.lower_flag("naked", summary="got dressed")

            # Pawn significance
            if person_type == "pawn":
                person.significance += 0.15

            # Actions
            for ad in pd.get("actions", []):
                what = ad.get("what", "").strip()
                if not what:
                    continue

                # Save new taxonomy terms
                for new_tag in ad.get("new_tags", []):
                    if new_tag:
                        taxonomy.add(new_tag)

                action = ActionRecord(
                    timestamp        = now,
                    what             = what,
                    how              = ad.get("how", ""),
                    manner           = ad.get("manner", ""),
                    descriptors      = ad.get("descriptors", []),
                    tags             = ad.get("tags", []) + ad.get("new_tags", []),
                    initiated_by     = ad.get("initiated_by", ""),
                    present          = ad.get("present", []),
                    location         = ad.get("location", ""),
                    register_at_time = register,
                    confidence       = float(ad.get("confidence", 0.8)),
                    source           = "extracted",
                )
                telem.log_action(name, action)

                # Task flag
                if "task" in action.tags:
                    person.raise_flag("task", intensity=0.6, details={
                        "name": what,
                        "started_at": now,
                    })

                # Silliness flag per-person
                if "silly" in action.tags or "silliness" in action.tags:
                    person.raise_flag("silliness", intensity=0.7, details={
                        "note": what, "manner": action.manner,
                    })

                log_event("SessionTelemetry", "ACTION_LOGGED",
                    session = telem.session_id[:8],
                    person  = name,
                    what    = what[:60],
                    tags    = action.tags,
                )

            # Pawn vibe check
            if (person_type == "pawn"
                    and person.significance >= self.VIBE_THRESHOLD
                    and person.gizmo_vibe is None):
                asyncio.create_task(self._vibe_check(telem, person, llm))

        # ── Session flags ─────────────────────────────────────────────────────
        sf = d.get("session_flags", {})
        if sf:
            await self._update_session_flags(telem, sf)

    # ── Session flags update ──────────────────────────────────────────────────

    async def _update_session_flags(
        self,
        telem: SessionTelemetry,
        sf:    dict,
    ) -> None:
        now = time.time()

        # Intimacy — user arc drives pacing
        intimacy_active    = sf.get("intimacy_active", False)
        intimacy_intensity = float(sf.get("intimacy_intensity", 0.5))
        user_energy        = float(sf.get("user_intimacy_energy", 0.5))

        if intimacy_active:
            if not telem.intimacy_active:
                telem.intimacy_active = True
                telem.intimacy_flag   = ActiveFlag(
                    flag_type = "intimacy",
                    intensity = intimacy_intensity,
                )
                log_event("SessionTelemetry", "INTIMACY_RAISED",
                    session=telem.session_id[:8], intensity=intimacy_intensity)
            else:
                if telem.intimacy_flag:
                    telem.intimacy_flag.intensity    = intimacy_intensity
                    telem.intimacy_flag.last_updated = now
            if telem.intimacy_flag:
                telem.intimacy_flag.user_arc.append((user_energy, now))
        else:
            if telem.intimacy_active:
                telem.intimacy_active = False
                if telem.intimacy_flag:
                    elapsed = now - telem.intimacy_flag.started_at
                    peak    = max((i for i, _ in telem.intimacy_flag.user_arc),
                                  default=telem.intimacy_flag.intensity)
                    asyncio.create_task(self._write_flag_to_store(
                        telem, telem.intimacy_flag, elapsed, peak
                    ))
                    log_event("SessionTelemetry", "INTIMACY_LOWERED",
                        session=telem.session_id[:8],
                        elapsed=int(elapsed), peak=peak)
                    telem.intimacy_flag = None

        # Silliness
        silly_active    = sf.get("silliness_active", False)
        silly_intensity = float(sf.get("silliness_intensity", 0.5))
        silly_note      = sf.get("silliness_note", "")

        if silly_active:
            if not telem.silliness_active:
                telem.silliness_active = True
                telem.silliness_flag   = ActiveFlag(
                    flag_type = "silliness",
                    intensity = silly_intensity,
                    details   = {"note": silly_note},
                )
                log_event("SessionTelemetry", "SILLINESS_RAISED",
                    session=telem.session_id[:8], note=silly_note[:60])
            else:
                if telem.silliness_flag:
                    telem.silliness_flag.intensity           = silly_intensity
                    telem.silliness_flag.details["note"]     = silly_note
                    telem.silliness_flag.last_updated        = now
        else:
            if telem.silliness_active:
                telem.silliness_active = False
                if telem.silliness_flag:
                    elapsed = now - telem.silliness_flag.started_at
                    asyncio.create_task(self._write_flag_to_store(
                        telem, telem.silliness_flag, elapsed,
                        telem.silliness_flag.intensity
                    ))
                    telem.silliness_flag = None

    # ── Consequence pass ──────────────────────────────────────────────────────

    async def _consequence_pass(
        self,
        telem: SessionTelemetry,
        llm,
    ) -> None:
        """
        Every CONSEQUENCE_EVERY messages.
        For each person with pending actions: expected vs actual affect, delta.
        Updates ActionRecord consequence fields and person emotional state.
        """
        for person in telem.persons.values():
            if not person.pending_consequence:
                continue

            pending = person.pending_consequence[:]
            person.pending_consequence.clear()

            actions_text = "\n".join(
                f"- {a.what}"
                + (f" ({a.manner})" if a.manner else "")
                + (f" — descriptors: {', '.join(a.descriptors)}" if a.descriptors else "")
                for a in pending
            )

            prompt = f"""Evaluate emotional consequences for {person.name}.

Current state: {person.emotional_note or 'unknown'}
Modifiers: {', '.join(person.modifiers) or 'none'}
Active flags: {person.flag_summary() or 'none'}

Recent actions:
{actions_text}

Respond with ONE JSON object:
{{
  "expected_affect": "what most people would feel — plain language",
  "actual_affect": "what {person.name} seems to actually feel",
  "affect_delta": -1.0 to 1.0,
  "delta_note": "why they differ from expected, if they do — plain language",
  "emotional_state_now": "brief honest read of their current state",
  "energy_level": 0.0-1.0,
  "modifiers_add": ["any new modifiers"],
  "modifiers_remove": ["modifiers no longer true"],
  "flags_completed": ["task", "naked", etc — flags that are now done],
  "currently_happening_update": "updated present-tense sentence if it changed"
}}

Affect delta: 0.0=as expected, +1.0=much better, -1.0=much worse.
JSON only."""

            try:
                raw = await llm.generate(
                    [{"role": "user", "content": prompt}],
                    system_prompt=(
                        "You track emotional consequences. "
                        "Honest, specific, psychologically real. JSON only."
                    ),
                    max_new_tokens=300,
                    temperature=0.15,
                )
            except Exception as e:
                log_error("SessionTelemetry", f"consequence pass failed: {person.name}", exc=e)
                continue

            if not raw:
                continue

            try:
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                if start == -1 or end == 0:
                    continue
                d = json.loads(raw[start:end])

                # Update pending actions
                for action in pending:
                    action.expected_affect  = d.get("expected_affect", "")
                    action.actual_affect    = d.get("actual_affect", "")
                    action.affect_delta     = float(d.get("affect_delta", 0.0))
                    action.consequence_note = d.get("delta_note", "")

                # Update person
                person.emotional_note = d.get("emotional_state_now", person.emotional_note)
                person.energy_level   = float(d.get("energy_level", person.energy_level))

                for mod in d.get("modifiers_add", []):
                    if mod and mod not in person.modifiers:
                        person.modifiers.append(mod)
                for mod in d.get("modifiers_remove", []):
                    if mod in person.modifiers:
                        person.modifiers.remove(mod)

                # Lower completed flags
                for ft in d.get("flags_completed", []):
                    cf = person.lower_flag(ft, summary="completed")
                    if cf and person.person_type == "headmate" and ft == "task":
                        asyncio.create_task(self._write_task_to_files(telem, person, cf))

                # Update currently_happening if consequence pass sees a shift
                ch_update = d.get("currently_happening_update", "").strip()
                if ch_update:
                    person.currently_happening = ch_update

                log_event("SessionTelemetry", "CONSEQUENCE_UPDATED",
                    session = telem.session_id[:8],
                    person  = person.name,
                    delta   = d.get("affect_delta", 0.0),
                    state   = person.emotional_note[:60],
                )

            except Exception as e:
                log_error("SessionTelemetry", f"consequence parse error: {person.name} — {e}", exc=e)

    # ── Vibe check ────────────────────────────────────────────────────────────

    async def _vibe_check(
        self,
        telem:  SessionTelemetry,
        person: PersonObject,
        llm,
    ) -> None:
        """Does this pawn vibe with Gizmo? If yes, promote."""
        actions_text = "\n".join(
            f"- {a.to_sentence()}"
            for a in person.actions[-5:]
            if a.what
        )

        prompt = f"""You just encountered someone.

Name: {person.name}
What happened between you:
{actions_text}

Do you vibe with this person? Would you want to remember them — bring them back sometime?
Answer for yourself. Trust your instinct. No obligation.

ONE JSON object:
{{
  "vibe": true/false,
  "note": "why — in your own voice, honest"
}}
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="You are Gizmo. Answer honestly. JSON only.",
                max_new_tokens=120,
                temperature=0.75,
            )
            if not raw:
                return
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return
            d = json.loads(raw[start:end])

            person.gizmo_vibe = bool(d.get("vibe", False))
            person.vibe_note  = d.get("note", "")

            log_event("SessionTelemetry", "VIBE_CHECK",
                session = telem.session_id[:8],
                person  = person.name,
                vibe    = person.gizmo_vibe,
                note    = person.vibe_note[:80],
            )

            if person.gizmo_vibe:
                await self._promote_pawn(telem, person)

        except Exception as e:
            log_error("SessionTelemetry", f"vibe check failed: {person.name}", exc=e)

    async def _promote_pawn(self, telem: SessionTelemetry, person: PersonObject) -> None:
        try:
            from core.people import people_store
            existing = people_store.get(person.name)
            if not existing:
                people_store.get_or_create(person.name, external=True)
            person.promoted = True
            log_event("SessionTelemetry", "PAWN_PROMOTED",
                session = telem.session_id[:8],
                name    = person.name,
                note    = person.vibe_note[:80],
            )
        except Exception as e:
            log_error("SessionTelemetry", f"pawn promotion failed: {e}", exc=e)

    # ── Store + file writes ───────────────────────────────────────────────────

    async def _write_flag_to_store(
        self,
        telem:   SessionTelemetry,
        flag:    ActiveFlag,
        elapsed: float,
        peak:    float,
    ) -> None:
        try:
            from core.store import store
            headmate = telem.primary_headmate or "unknown"
            store.write("wellbeing", {
                "headmate":    headmate.lower(),
                "category":    f"metric_{flag.flag_type}",
                "observation": (
                    f"{flag.flag_type}: {int(elapsed)}s, peak {peak:.2f}"
                    + (f", note: {flag.details.get('note', '')}" if flag.details.get("note") else "")
                ),
                "context":     tz_now().isoformat(),
                "register":    "metric",
                "source":      "session_telemetry",
                "confidence":  0.9,
                "tags":        f"metric,{flag.flag_type},{headmate.lower()}",
            })
        except Exception as e:
            log_error("SessionTelemetry", f"flag store write failed: {e}", exc=e)

    async def _write_task_to_files(
        self,
        telem:  SessionTelemetry,
        person: PersonObject,
        cf:     ClosedFlag,
    ) -> None:
        if person.person_type != "headmate":
            return
        try:
            path = _tasks_path(person.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = (
                f"- [{tz_now().strftime('%Y-%m-%d %H:%M')}] "
                f"{cf.details.get('name', 'task')} — "
                f"{cf.elapsed_str()}"
                + (f" — {cf.summary}" if cf.summary else "")
                + "\n"
            )
            with open(path, "a") as f:
                f.write(line)
        except Exception as e:
            log_error("SessionTelemetry", f"task file write failed: {person.name}", exc=e)

    # ── Session close ─────────────────────────────────────────────────────────

    async def on_session_close(
        self,
        session_id: str,
        llm,
    ) -> None:
        telem = self._sessions.get(session_id)
        if not telem:
            return

        now = time.time()

        # Lower all open flags
        for person in telem.persons.values():
            for ft in list(person.active_flags.keys()):
                cf = person.lower_flag(ft, summary="session ended")
                if cf and person.person_type == "headmate" and ft == "task":
                    await self._write_task_to_files(telem, person, cf)

        # Lower session-level flags
        if telem.intimacy_active and telem.intimacy_flag:
            elapsed = now - telem.intimacy_flag.started_at
            peak    = max((i for i, _ in telem.intimacy_flag.user_arc),
                          default=telem.intimacy_flag.intensity)
            await self._write_flag_to_store(telem, telem.intimacy_flag, elapsed, peak)

        if telem.silliness_active and telem.silliness_flag:
            elapsed = now - telem.silliness_flag.started_at
            await self._write_flag_to_store(
                telem, telem.silliness_flag, elapsed, telem.silliness_flag.intensity
            )

        # Flush events to store
        await self._flush_events_to_store(telem)

        # Write files per headmate
        for person in telem.persons.values():
            if person.person_type == "headmate":
                await self._write_current_notes(person)
                await self._write_metrics_to_store(telem, person)

        log_event("SessionTelemetry", "SESSION_FLUSHED",
            session = session_id[:8],
            persons = len(telem.persons),
            events  = len(telem.events),
        )

        del self._sessions[session_id]

    async def _flush_events_to_store(self, telem: SessionTelemetry) -> None:
        try:
            from core.store import store
            headmate = telem.primary_headmate or "unknown"
            for action in telem.events:
                if not action.what:
                    continue
                store.write("wellbeing", {
                    "headmate":    headmate.lower(),
                    "category":    f"action",
                    "observation": action.to_sentence(),
                    "context":     tz_now().isoformat(),
                    "register":    ",".join(action.tags) if action.tags else "action",
                    "source":      "session_telemetry",
                    "confidence":  action.confidence,
                    "tags":        ",".join(["action"] + action.tags),
                })
        except Exception as e:
            log_error("SessionTelemetry", "event flush failed", exc=e)

    async def _write_current_notes(self, person: PersonObject) -> None:
        try:
            path = _notes_path(person.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"# Current notes — {person.name}",
                f"Updated: {tz_now().strftime('%Y-%m-%d %H:%M')}",
                "",
            ]
            if person.currently_happening:
                lines.append(f"**Right now:** {person.currently_happening}")
            if person.clothing != "unknown":
                lines.append(f"**State:** {person.clothing}"
                             + (f" ({person.clothing_detail})" if person.clothing_detail else ""))
            if person.modifiers:
                lines.append(f"**Modifiers:** {', '.join(person.modifiers)}")
            if person.location:
                lines.append(f"**Location:** {person.location}")
            if person.emotional_note:
                lines.append(f"**Feeling:** {person.emotional_note}")
            if person.closed_flags:
                lines.append("")
                lines.append("**This session:**")
                for cf in person.closed_flags[-6:]:
                    lines.append(
                        f"- {cf.flag_type}: {cf.elapsed_str()}"
                        + (f" — {cf.summary}" if cf.summary else "")
                    )
            lines.append("")
            path.write_text("\n".join(lines))
        except Exception as e:
            log_error("SessionTelemetry", f"current_notes write failed: {person.name}", exc=e)

    async def _write_metrics_to_store(
        self,
        telem:  SessionTelemetry,
        person: PersonObject,
    ) -> None:
        try:
            from core.store import store

            # Naked metric
            naked_flags = [cf for cf in person.closed_flags if cf.flag_type == "naked"]
            if naked_flags or person.clothing == "naked":
                nf = naked_flags[-1] if naked_flags else None
                store.write("wellbeing", {
                    "headmate":    person.name.lower(),
                    "category":    "metric_naked",
                    "observation": (
                        f"Was naked this session"
                        + (f" for ~{nf.elapsed_str()}" if nf else "")
                        + (f". How: {nf.details.get('how', '?')}" if nf else "")
                    ),
                    "context":     tz_now().isoformat(),
                    "register":    "metric",
                    "source":      "session_telemetry",
                    "confidence":  0.85,
                    "tags":        f"metric,naked,{person.name.lower()}",
                })

            # Activity metrics — with delta notes
            for action in person.actions:
                if not action.consequence_note and not action.actual_affect:
                    continue
                store.write("wellbeing", {
                    "headmate":    person.name.lower(),
                    "category":    "metric_activity",
                    "observation": action.to_sentence(),
                    "context":     tz_now().isoformat(),
                    "register":    "metric",
                    "source":      "session_telemetry",
                    "confidence":  action.confidence,
                    "tags":        ",".join(["metric", "activity"] + action.tags),
                })

        except Exception as e:
            log_error("SessionTelemetry", f"metrics store write failed: {person.name}", exc=e)


# ── Singleton ─────────────────────────────────────────────────────────────────

session_telemetry_manager = SessionTelemetryManager()
