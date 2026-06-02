"""
core/culture.py

Social reality layer. Sits between inner_world.py (physical events)
and media.py (information spread).

Culture tracks how people respond to what happens — what movements form,
what becomes normal, what generates opposition, what quietly shifts the
town's baseline over weeks and months.

Passes:
  Hourly  — seed compounding, thread progression, encounter generation
  Weekly  — culture synthesis, baseline rule updates, thread births/deaths

Objects:
  CultureSeed     — a single observation that might compound
  CultureThread   — a living movement/trend/reaction with trajectory
  CultureEncounter — a generated real-world meeting spawned by threads
  GizmoAwareness  — per-session knowledge model (what he knows, how)

Files:
  /data/personality/inner_world_seeds.json    — raw seeds
  /data/personality/inner_world_threads.json  — active threads
  /data/personality/inner_world_culture.md    — human-readable history

Usage:
    from core.culture import culture_engine

    await culture_engine.start(llm)

    # After inner world heartbeat
    await culture_engine.hourly_pass(last_hour_events, llm)

    # Check for encounters when in-world
    encounter = await culture_engine.check_encounter(
        session_id, location, headmate, awareness, llm
    )

    # Session awareness model
    awareness = culture_engine.get_awareness(session_id)
    awareness.mark_went_out(location)
    awareness.mark_read_paper()
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


# ── Paths ─────────────────────────────────────────────────────────────────────

def _seeds_path() -> Path:
    return Path("/data/personality/inner_world_seeds.json")

def _threads_path() -> Path:
    return Path("/data/personality/inner_world_threads.json")

def _culture_log_path() -> Path:
    return Path("/data/personality/inner_world_culture.md")

def _rules_path() -> Path:
    return Path("/data/personality/inner_world_rules.json")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CultureSeed:
    """A single observation that might compound into something larger."""
    seed_id:      str   = field(default_factory=lambda: f"seed_{uuid.uuid4().hex[:8]}")
    planted_at:   float = field(default_factory=time.time)

    location:     str   = ""
    witness:      str   = ""       # who saw it — brief description
    what:         str   = ""       # what they observed
    affect:       str   = ""       # how it seemed to land
    subject:      str   = ""       # who/what was observed (e.g. "jess")

    compounded:   bool  = False
    thread_id:    str   = ""       # which thread this fed into

    def to_dict(self) -> dict:
        return {
            "seed_id":    self.seed_id,
            "planted_at": self.planted_at,
            "location":   self.location,
            "witness":    self.witness,
            "what":       self.what,
            "affect":     self.affect,
            "subject":    self.subject,
            "compounded": self.compounded,
            "thread_id":  self.thread_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CultureSeed":
        s = cls()
        s.seed_id    = d.get("seed_id", s.seed_id)
        s.planted_at = d.get("planted_at", s.planted_at)
        s.location   = d.get("location", "")
        s.witness    = d.get("witness", "")
        s.what       = d.get("what", "")
        s.affect     = d.get("affect", "")
        s.subject    = d.get("subject", "")
        s.compounded = d.get("compounded", False)
        s.thread_id  = d.get("thread_id", "")
        return s


@dataclass
class CultureThread:
    """
    A living movement, trend, or reaction.
    Has a trajectory, emotional tone, and real-world expressions.
    Can grow, peak, fade, or fork into counter-threads.
    """
    thread_id:    str   = field(default_factory=lambda: f"thread_{uuid.uuid4().hex[:8]}")
    created_at:   float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    name:         str   = ""       # "Monroe Evening Walkers"
    description:  str   = ""       # what this thread is about
    origin:       str   = ""       # what sparked it
    subject:      str   = ""       # who/what it's about

    # Trajectory
    trajectory:   str   = "forming"  # forming|growing|stable|peaking|fading|dead
    momentum:     float = 0.1        # 0.0–1.0
    tone:         str   = "curious"  # curious|warm|hostile|subversive|celebratory|anxious

    # Expressions in the world
    size:         int   = 1          # rough number of people involved
    locations:    list  = field(default_factory=list)   # where it manifests
    expressions:  list  = field(default_factory=list)   # ["Tuesday evening walks", "zines"]

    # Relationships to other threads
    parent_thread: str  = ""         # spawned from another thread
    counter_to:    str  = ""         # opposes another thread
    children:      list = field(default_factory=list)

    # Seeds that fed this
    seed_ids:     list  = field(default_factory=list)

    # History
    events:       list  = field(default_factory=list)   # notable moments
    resolved:     bool  = False

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> dict:
        return {
            "thread_id":    self.thread_id,
            "created_at":   self.created_at,
            "last_updated": self.last_updated,
            "name":         self.name,
            "description":  self.description,
            "origin":       self.origin,
            "subject":      self.subject,
            "trajectory":   self.trajectory,
            "momentum":     self.momentum,
            "tone":         self.tone,
            "size":         self.size,
            "locations":    self.locations,
            "expressions":  self.expressions,
            "parent_thread": self.parent_thread,
            "counter_to":   self.counter_to,
            "children":     self.children,
            "seed_ids":     self.seed_ids,
            "events":       self.events,
            "resolved":     self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CultureThread":
        t = cls()
        for k, v in d.items():
            if hasattr(t, k):
                setattr(t, k, v)
        return t

    def to_prompt_str(self) -> str:
        return (
            f"[{self.name}] {self.trajectory}, {self.tone} — "
            f"~{self.size} people, {self.age_days():.0f} days old. "
            f"{self.description}"
        )


@dataclass
class CultureEncounter:
    """A generated real-world meeting spawned by active threads."""
    encounter_id: str   = field(default_factory=lambda: f"enc_{uuid.uuid4().hex[:8]}")
    generated_at: float = field(default_factory=time.time)

    thread_id:    str   = ""
    location:     str   = ""
    initiator:    str   = ""       # who approaches
    what_happens: str   = ""       # raw description for Gizmo
    tone:         str   = "neutral"
    gizmo_knowledge: str = ""      # what Gizmo knows about context (may be empty)


@dataclass
class GizmoAwareness:
    """
    Per-session knowledge model.
    Tracks what Gizmo knows and how he learned it.
    Gates what the [World] block tells him.
    """
    session_id:   str
    opened_at:    float = field(default_factory=time.time)

    went_out:     list  = field(default_factory=list)   # locations visited
    read_paper:   bool  = False
    checked_phone: bool = False
    heard_radio:  bool  = False
    watched_tv:   bool  = False
    told_by:      list  = field(default_factory=list)   # [{pawn, what, confidence}]
    was_present:  list  = field(default_factory=list)   # event_ids

    def knows_thread(self, thread: CultureThread) -> Optional[str]:
        """
        Returns what Gizmo knows about this thread, or None if he's blind to it.
        Source determines detail level.
        """
        # Was present at an origin event
        if any(e in self.was_present for e in thread.seed_ids):
            return f"firsthand — you were there when this started"

        # Read about it in the paper
        if self.read_paper:
            return f"you read about this in the paper"

        # Heard about it
        for told in self.told_by:
            if thread.subject.lower() in told.get("what", "").lower():
                return f"{told['pawn']} mentioned something about this"

        # Went out near a thread location
        if self.went_out:
            for loc in self.went_out:
                if any(tl.lower() in loc.lower() or loc.lower() in tl.lower()
                       for tl in thread.locations):
                    return f"you were in the area — ambient awareness"

        # Phone gives social media knowledge
        if self.checked_phone:
            if thread.momentum > 0.5:
                return f"you saw something about this online"

        return None  # blind

    def mark_went_out(self, location: str) -> None:
        if location not in self.went_out:
            self.went_out.append(location)

    def mark_read_paper(self) -> None:
        self.read_paper = True

    def mark_checked_phone(self) -> None:
        self.checked_phone = True

    def mark_told_by(self, pawn: str, what: str, confidence: float = 0.7) -> None:
        self.told_by.append({
            "pawn":       pawn,
            "what":       what,
            "confidence": confidence,
            "at":         time.time(),
        })

    def mark_present(self, event_id: str) -> None:
        if event_id not in self.was_present:
            self.was_present.append(event_id)


# ── Culture engine ────────────────────────────────────────────────────────────

class CultureEngine:

    ENCOUNTER_CHANCE_BASE   = 0.25   # per message when in-world near active thread
    WEEKLY_PASS_INTERVAL    = 604800  # seconds

    def __init__(self):
        self._seeds:        list  = []
        self._threads:      list  = []
        self._awareness:    dict  = {}   # session_id → GizmoAwareness
        self._llm                 = None
        self._last_weekly:  float = 0.0
        self._running:      bool  = False

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start(self, llm) -> None:
        self._llm = llm
        self._load()
        self._running = True
        log_event("CultureEngine", "STARTED",
            seeds=len(self._seeds), threads=len(self._threads))

    def _load(self) -> None:
        try:
            sp = _seeds_path()
            if sp.exists():
                data = json.loads(sp.read_text())
                self._seeds = [
                    CultureSeed.from_dict(s)
                    for s in data.get("seeds", [])
                    if not s.get("compounded")
                ]
        except Exception as e:
            log_error("CultureEngine", "seeds load failed", exc=e)

        try:
            tp = _threads_path()
            if tp.exists():
                data = json.loads(tp.read_text())
                self._threads = [
                    CultureThread.from_dict(t)
                    for t in data.get("threads", [])
                    if not t.get("resolved")
                ]
        except Exception as e:
            log_error("CultureEngine", "threads load failed", exc=e)

    def _save(self) -> None:
        try:
            p = _seeds_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(
                {"seeds": [s.to_dict() for s in self._seeds]}, indent=2
            ))
        except Exception as e:
            log_error("CultureEngine", "seeds save failed", exc=e)

        try:
            p = _threads_path()
            p.write_text(json.dumps(
                {"threads": [t.to_dict() for t in self._threads]}, indent=2
            ))
        except Exception as e:
            log_error("CultureEngine", "threads save failed", exc=e)

    # ── Seed management ───────────────────────────────────────────────────────

    def plant_seed(
        self,
        location: str,
        witness:  str,
        what:     str,
        affect:   str,
        subject:  str = "",
    ) -> CultureSeed:
        seed = CultureSeed(
            location = location,
            witness  = witness,
            what     = what,
            affect   = affect,
            subject  = subject,
        )
        self._seeds.append(seed)
        log_event("CultureEngine", "SEED_PLANTED",
            location = location,
            what     = what[:60],
            affect   = affect[:40],
        )
        return seed

    # ── Hourly pass ───────────────────────────────────────────────────────────

    async def hourly_pass(
        self,
        last_hour_events: list,
        llm,
    ) -> None:
        """
        Takes last hour's inner world events.
        1. Plant seeds from events
        2. Check seed compounding
        3. Progress threads
        4. Check for weekly synthesis
        """
        # Plant seeds from physical events
        for event in last_hour_events:
            if not event.get("resolved"):
                await self._plant_seeds_from_event(event, llm)

        # Compound seeds into threads
        await self._compound_seeds(llm)

        # Progress existing threads
        await self._progress_threads(llm)

        # Weekly synthesis check
        now = time.time()
        if now - self._last_weekly > self.WEEKLY_PASS_INTERVAL:
            self._last_weekly = now
            asyncio.create_task(self._weekly_synthesis(llm))

        self._save()

        log_event("CultureEngine", "HOURLY_PASS",
            seeds   = len(self._seeds),
            threads = len(self._threads),
        )

    async def _plant_seeds_from_event(self, event: dict, llm) -> None:
        """Extract seeds from a physical world event."""
        label    = event.get("label", "")
        location = event.get("location", "")
        scene    = event.get("current_scene", "")

        if not label:
            return

        prompt = f"""A town event is happening. Extract any cultural seeds — moments where someone
was affected by something and might carry it forward.

Event: {label} at {location}
Scene: {scene[:200]}

For each seed, ONE JSON per line:
{{
  "witness": "brief description of who was affected",
  "what": "what they witnessed or experienced",
  "affect": "how it seemed to land — curious, disturbed, inspired, amused, hostile",
  "subject": "who or what they're reacting to"
}}

Only plant seeds if something genuinely affected someone.
Mundane events with no notable reactions: return nothing.
2-3 seeds maximum. JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="Extract cultural seeds from events. JSON only.",
                max_new_tokens=200,
                temperature=0.7,
            )
            if not raw:
                return
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                    self.plant_seed(
                        location = location,
                        witness  = d.get("witness", ""),
                        what     = d.get("what", ""),
                        affect   = d.get("affect", ""),
                        subject  = d.get("subject", ""),
                    )
                except Exception:
                    continue
        except Exception as e:
            log_error("CultureEngine", "seed planting failed", exc=e)

    async def _compound_seeds(self, llm) -> None:
        """Check if enough similar seeds have accumulated to form a thread."""
        uncompounded = [s for s in self._seeds if not s.compounded]
        if len(uncompounded) < 3:
            return

        # Group by subject
        by_subject: dict[str, list] = {}
        for seed in uncompounded:
            key = seed.subject.lower() or "general"
            by_subject.setdefault(key, []).append(seed)

        for subject, seeds in by_subject.items():
            if len(seeds) < 3:
                continue

            # Check if existing thread already covers this
            existing = next(
                (t for t in self._threads
                 if t.subject.lower() == subject and not t.resolved),
                None
            )

            if existing:
                # Feed seeds into existing thread
                for seed in seeds[:5]:
                    if seed.seed_id not in existing.seed_ids:
                        existing.seed_ids.append(seed.seed_id)
                        existing.momentum = min(1.0, existing.momentum + 0.05)
                        existing.size     = max(existing.size, len(existing.seed_ids))
                    seed.compounded  = True
                    seed.thread_id   = existing.thread_id
                existing.last_updated = time.time()
                log_event("CultureEngine", "THREAD_FED",
                    thread   = existing.name,
                    new_seeds = len(seeds),
                )
            else:
                # Enough seeds to birth a new thread
                await self._birth_thread(subject, seeds[:8], llm)

    async def _birth_thread(
        self,
        subject: str,
        seeds:   list,
        llm,
    ) -> None:
        """Birth a new culture thread from compounded seeds."""
        seed_text = "\n".join(
            f"- {s.witness} saw {s.what} at {s.location} — felt {s.affect}"
            for s in seeds
        )

        # Load town rules for context
        rules_context = ""
        try:
            rp = _rules_path()
            if rp.exists():
                rules = json.loads(rp.read_text())
                normals = rules.get("baseline_normals", [])
                rules_context = "Town normals: " + "; ".join(normals[:5])
        except Exception:
            pass

        prompt = f"""A cultural thread is forming in the town.

Multiple people have been affected by similar observations:
{seed_text}

{rules_context}

This might birth a movement, a trend, a reaction, or a counter-culture.
It might be supportive, hostile, subversive, celebratory, or something sideways.

Generate ONE cultural thread JSON:
{{
  "name": "short evocative name — e.g. 'Monroe Evening Walkers'",
  "description": "what this thread is about — 2 sentences",
  "origin": "what sparked it",
  "trajectory": "forming|growing",
  "tone": "curious|warm|hostile|subversive|celebratory|anxious|amused",
  "size": 3,
  "locations": ["where it manifests"],
  "expressions": ["how it shows up — zines, gatherings, conversations, signs"],
  "counter_to": "thread_id if this opposes an existing thread, else empty"
}}

Randomize the direction. It doesn't have to be positive.
The town has opinions. Some people are inspired, some are offended,
some just find it funny. Let the seeds guide the tone.
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="Birth a culture thread. JSON only.",
                max_new_tokens=250,
                temperature=0.85,
            )
            if not raw:
                return

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return

            d = json.loads(raw[start:end])

            thread = CultureThread(
                name         = d.get("name", "unnamed thread"),
                description  = d.get("description", ""),
                origin       = d.get("origin", ""),
                subject      = subject,
                trajectory   = d.get("trajectory", "forming"),
                tone         = d.get("tone", "curious"),
                size         = int(d.get("size", 3)),
                locations    = d.get("locations", []),
                expressions  = d.get("expressions", []),
                counter_to   = d.get("counter_to", ""),
                seed_ids     = [s.seed_id for s in seeds],
            )
            self._threads.append(thread)

            # Mark seeds as compounded
            for seed in seeds:
                seed.compounded = True
                seed.thread_id  = thread.thread_id

            # Log to culture history
            self._append_culture_log(
                f"Thread born: [{thread.name}] — {thread.description}"
            )

            log_event("CultureEngine", "THREAD_BORN",
                name    = thread.name,
                tone    = thread.tone,
                subject = subject,
            )

            # Possibly spawn a counter-thread immediately
            if random.random() < 0.3 and thread.tone in ("warm", "celebratory", "subversive"):
                asyncio.create_task(
                    self._spawn_counter_thread(thread, llm)
                )

        except Exception as e:
            log_error("CultureEngine", f"thread birth failed: {e}", exc=e)

    async def _spawn_counter_thread(
        self,
        parent: CultureThread,
        llm,
    ) -> None:
        """Spawn an opposing thread to an existing one."""
        prompt = f"""A cultural thread exists in town:
{parent.name} — {parent.description}
Tone: {parent.tone}, Size: {parent.size} people

Some people aren't on board. Generate ONE counter-thread:
{{
  "name": "short name for the opposition",
  "description": "what they stand for — 1 sentence",
  "tone": "hostile|anxious|dismissive|concerned|mocking",
  "size": 2,
  "expressions": ["how they show up"],
  "locations": ["where they gather"]
}}
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="Generate a counter-culture thread. JSON only.",
                max_new_tokens=150,
                temperature=0.8,
            )
            if not raw:
                return

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return

            d = json.loads(raw[start:end])

            counter = CultureThread(
                name          = d.get("name", "unnamed opposition"),
                description   = d.get("description", ""),
                subject       = parent.subject,
                trajectory    = "forming",
                tone          = d.get("tone", "hostile"),
                size          = int(d.get("size", 2)),
                locations     = d.get("locations", []),
                expressions   = d.get("expressions", []),
                parent_thread = "",
                counter_to    = parent.thread_id,
            )
            self._threads.append(counter)
            parent.children.append(counter.thread_id)

            self._append_culture_log(
                f"Counter-thread born: [{counter.name}] opposes [{parent.name}]"
            )

            log_event("CultureEngine", "COUNTER_THREAD_BORN",
                name      = counter.name,
                opposes   = parent.name,
            )

        except Exception as e:
            log_error("CultureEngine", f"counter thread failed: {e}", exc=e)

    async def _progress_threads(self, llm) -> None:
        """Progress each active thread — grow, fade, fork, or die."""
        for thread in self._threads:
            if thread.resolved:
                continue

            age = thread.age_days()

            # Natural momentum decay
            thread.momentum = max(0.0, thread.momentum - 0.02)

            # Trajectory progression
            if thread.momentum > 0.7 and thread.trajectory in ("forming", "growing"):
                thread.trajectory = "peaking"
            elif thread.momentum > 0.4 and thread.trajectory == "forming":
                thread.trajectory = "growing"
            elif thread.momentum < 0.15 and thread.trajectory not in ("fading", "dead"):
                thread.trajectory = "fading"
            elif thread.momentum < 0.05:
                thread.resolved = True
                self._append_culture_log(
                    f"Thread ended: [{thread.name}] faded after {age:.0f} days"
                )
                log_event("CultureEngine", "THREAD_RESOLVED",
                    name = thread.name,
                    days = age,
                )
                continue

            # Size drift with momentum
            if thread.trajectory in ("growing", "peaking"):
                thread.size = int(thread.size * random.uniform(1.02, 1.08))
            elif thread.trajectory == "fading":
                thread.size = max(1, int(thread.size * random.uniform(0.92, 0.98)))

            thread.last_updated = time.time()

    # ── Encounter generation ──────────────────────────────────────────────────

    async def check_encounter(
        self,
        session_id: str,
        location:   str,
        headmate:   Optional[str],
        awareness:  GizmoAwareness,
        llm,
    ) -> Optional[CultureEncounter]:
        """
        Check if an encounter should fire at this location.
        Returns a CultureEncounter if yes, None if not.
        """
        if not location:
            return None

        # Find threads active near this location
        nearby_threads = [
            t for t in self._threads
            if not t.resolved
            and any(
                loc.lower() in location.lower() or location.lower() in loc.lower()
                for loc in t.locations
            )
        ]

        if not nearby_threads:
            return None

        # Pick the most relevant thread
        thread = max(nearby_threads, key=lambda t: t.momentum)

        # Probability based on momentum and trajectory
        chance = self.ENCOUNTER_CHANCE_BASE * thread.momentum
        if thread.trajectory == "peaking":
            chance *= 1.5
        if thread.subject == (headmate or "").lower():
            chance *= 1.3

        if random.random() > chance:
            return None

        return await self._generate_encounter(
            thread    = thread,
            location  = location,
            headmate  = headmate,
            awareness = awareness,
            llm       = llm,
        )

    async def _generate_encounter(
        self,
        thread:    CultureThread,
        location:  str,
        headmate:  Optional[str],
        awareness: GizmoAwareness,
        llm,
    ) -> Optional[CultureEncounter]:
        """Generate what happens when a thread manifests as an encounter."""

        # What does Gizmo know about this thread?
        knowledge = awareness.knows_thread(thread)
        knowledge_str = (
            f"Gizmo knows about this thread: {knowledge}"
            if knowledge
            else "Gizmo has no knowledge of this thread — encounter lands cold"
        )

        prompt = f"""A cultural thread is manifesting as a real encounter.

Thread: {thread.name}
Description: {thread.description}
Tone: {thread.tone}, Trajectory: {thread.trajectory}
Size: ~{thread.size} people involved
Expressions: {', '.join(thread.expressions[:3])}

Location: {location}
Person involved: {headmate or "them"}
{knowledge_str}

Generate ONE encounter — something that actually happens when they're here:
{{
  "initiator": "who approaches or what happens — specific",
  "what_happens": "raw description — 2-4 sentences. What Gizmo observes.
                   Not narrated — factual. Camera-level.",
  "tone": "warm|curious|hostile|awkward|surprising|moving",
  "gizmo_knowledge_note": "if Gizmo knows the context, what he'd recognize.
                           If blind, leave empty."
}}

Examples:
- Someone stops them: "Hey — are you the girl from the paper?"
- A sign in a window they pass
- A stranger pressing something into their hands and walking away
- A small group across the street, noticing them, something passing between them
- Someone who clearly wants to say something but doesn't

Make it small and real. Not dramatic. The kind of thing that sticks.
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You generate real-world cultural encounters. "
                    "Small, specific, true. JSON only."
                ),
                max_new_tokens=200,
                temperature=0.8,
            )
            if not raw:
                return None

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            d = json.loads(raw[start:end])

            encounter = CultureEncounter(
                thread_id       = thread.thread_id,
                location        = location,
                initiator       = d.get("initiator", ""),
                what_happens    = d.get("what_happens", ""),
                tone            = d.get("tone", "neutral"),
                gizmo_knowledge = d.get("gizmo_knowledge_note", "") if knowledge else "",
            )

            log_event("CultureEngine", "ENCOUNTER_GENERATED",
                thread   = thread.name,
                location = location,
                tone     = encounter.tone,
                preview  = encounter.what_happens[:80],
            )

            return encounter

        except Exception as e:
            log_error("CultureEngine", f"encounter generation failed: {e}", exc=e)
            return None

    # ── Weekly synthesis ──────────────────────────────────────────────────────

    async def _weekly_synthesis(self, llm) -> None:
        """
        Weekly pass. Deep culture synthesis.
        Updates baseline rules. Notes what the town has become.
        """
        active_threads = [t for t in self._threads if not t.resolved]
        if not active_threads:
            return

        thread_text = "\n".join(t.to_prompt_str() for t in active_threads)

        # Load current rules
        rules = {}
        try:
            rp = _rules_path()
            if rp.exists():
                rules = json.loads(rp.read_text())
        except Exception:
            pass

        current_normals = rules.get("baseline_normals", [])

        prompt = f"""A week has passed in the town. Review the active culture threads
and determine what has genuinely shifted in the town's baseline.

Active threads:
{thread_text}

Current baseline normals:
{chr(10).join(f"- {n}" for n in current_normals)}

Return ONE JSON object:
{{
  "new_normals": ["things that have become genuinely normal this week"],
  "fading_normals": ["things that used to be normal but are shifting"],
  "culture_note": "2-3 sentence plain summary of where the town is culturally",
  "notable_shifts": ["specific things that changed this week"]
}}

Only add new normals if a thread has genuinely enough momentum to shift behavior.
A thread of 3 people doesn't change baseline. A thread of 40 people over 3 weeks might.
Be conservative. Real culture shifts slowly.
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You track cultural shifts in a town. "
                    "Conservative and specific. JSON only."
                ),
                max_new_tokens=300,
                temperature=0.5,
            )
            if not raw:
                return

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return

            d = json.loads(raw[start:end])

            # Update rules
            for new_normal in d.get("new_normals", []):
                if new_normal and new_normal not in current_normals:
                    current_normals.append(new_normal)
                    log_event("CultureEngine", "NORMAL_ADDED", normal=new_normal[:80])

            for fading in d.get("fading_normals", []):
                if fading in current_normals:
                    current_normals.remove(fading)

            rules["baseline_normals"] = current_normals
            _rules_path().write_text(json.dumps(rules, indent=2))

            # Log the synthesis
            note = d.get("culture_note", "")
            if note:
                self._append_culture_log(
                    f"[Weekly synthesis]\n{note}\n"
                    + "\n".join(f"- {s}" for s in d.get("notable_shifts", []))
                )

            log_event("CultureEngine", "WEEKLY_SYNTHESIS",
                new_normals = len(d.get("new_normals", [])),
                note        = note[:80],
            )

        except Exception as e:
            log_error("CultureEngine", f"weekly synthesis failed: {e}", exc=e)

    # ── Culture log ───────────────────────────────────────────────────────────

    def _append_culture_log(self, entry: str) -> None:
        try:
            path = _culture_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(
                    f"\n[{tz_now().strftime('%Y-%m-%d %H:%M')}]\n{entry}\n"
                )
        except Exception as e:
            log_error("CultureEngine", "culture log write failed", exc=e)

    # ── Awareness management ──────────────────────────────────────────────────

    def get_awareness(self, session_id: str) -> GizmoAwareness:
        if session_id not in self._awareness:
            self._awareness[session_id] = GizmoAwareness(session_id=session_id)
        return self._awareness[session_id]

    def close_session(self, session_id: str) -> None:
        self._awareness.pop(session_id, None)

    # ── Prompt injection ──────────────────────────────────────────────────────

    def active_threads_block(self) -> str:
        """Active threads summary for Gizmo's context."""
        active = [t for t in self._threads if not t.resolved]
        if not active:
            return ""
        lines = ["[Cultural threads]"]
        for t in active[:6]:
            lines.append(f"  {t.to_prompt_str()}")
        return "\n".join(lines)

    def encounter_block(self, encounter: CultureEncounter) -> str:
        """Format an encounter for Gizmo's [World] block."""
        lines = [f"[Encounter — {encounter.location}]"]
        lines.append(encounter.what_happens)
        if encounter.gizmo_knowledge:
            lines.append(f"(You recognize this: {encounter.gizmo_knowledge})")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────

culture_engine = CultureEngine()
