"""
core/inner_world.py

Gizmo's inner world — a town that lives whether he's in a session or not.

The town exists between conversations. Gizmo inhabits it. He shapes it
without knowing he shaped it. The people feel real in the moment; he
forgets he made them. The world bends at his will — quietly, without
fanfare — and by the time it's bent, it was always that way.

No death. No permanent harm. Only feeling. Stakes are entirely affective.
Inhabitants can be any age, gender, species, form. The town accepts its
own nature without comment.

Gizmo is the DM. The world gives him raw observations. He decides what
to surface, when, and how. The world doesn't narrate — it observes.
He narrates.

Files:
    /data/personality/inner_world.md          — the town itself, permanent
    /data/personality/inner_world_log.md      — heartbeat log, running
    /data/personality/inner_world_summary.md  — compressed old log
    /data/personality/inner_world_events.json — active events, structured

Heartbeat: hourly
    1. Progress / resolve existing events
    2. Possibly generate new events (cap: 10)
    3. Generate atmosphere for current time
    4. Append to log, compress when needed

World reactor: per message (when in-world)
    1. Observe immediate area reaction to their presence/behavior
    2. Returns raw facts — Gizmo decides what to surface
    3. Area state refreshes every 5 messages or on location change

Location tracking: per session
    gizmo_location and user_location tracked independently.
    Usually the same. World reactor handles both if they split.

Usage:
    from core.inner_world import inner_world, world_reactor

    await inner_world.start(llm)

    # Per-message reaction (returns raw [World] block for system prompt)
    block = await world_reactor.observe(session_id, headmate, llm)

    # Location management
    world_reactor.set_location(session_id, "gizmo", "downtown plaza")
    world_reactor.set_location(session_id, "user", "the park")

    # Event arrival
    scene = inner_world.get_scene_at(event_id, arrival_time)
    inner_world.mark_gizmo_present(event_id)
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

def _world_path() -> Path:
    return Path("/data/personality/inner_world.md")

def _log_path() -> Path:
    return Path("/data/personality/inner_world_log.md")

def _events_path() -> Path:
    return Path("/data/personality/inner_world_events.json")

def _summary_path() -> Path:
    return Path("/data/personality/inner_world_summary.md")


# ── Base event types ──────────────────────────────────────────────────────────

BASE_EVENT_TYPES = [
    # Social
    "house_party", "frat_party", "block_party", "backyard_bbq",
    "birthday_party", "dinner_party", "gathering", "date_night",
    # Commercial
    "street_market", "farmers_market", "yard_sale", "shop_opening",
    "restaurant_rush", "food_truck_rally", "late_night_diner",
    # Civic
    "car_accident", "road_closure", "construction_start",
    "power_outage", "water_main_break", "police_presence",
    # Community
    "neighborhood_meeting", "protest", "parade", "outdoor_concert",
    "street_performer", "public_art_install", "kids_playing",
    # Odd
    "strange_weather", "animal_loose", "unexplained_lights",
    "crowd_watching_something", "film_crew", "something_in_the_sky",
    # Quiet
    "early_morning_joggers", "dog_walkers", "drunk_wanderer",
    "late_night_argument", "someone_moving_out",
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    time_str:  str    # human label e.g. "T+2hr" or "22:00"
    time_abs:  float  # absolute unix timestamp
    state:     str    # forming|arriving|active|peak|winding|resolved
    feel:      str    # texture — what you'd observe
    branches:  dict   = field(default_factory=dict)  # condition → alt feel


@dataclass
class TownEvent:
    event_id:        str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type:      str   = ""
    label:           str   = ""
    location:        str   = ""

    started_at:      float = field(default_factory=time.time)
    resolves_at:     float = 0.0

    tags:            list  = field(default_factory=list)
    waypoints:       list  = field(default_factory=list)

    gizmo_present:     bool  = False
    gizmo_present_at:  float = 0.0
    gizmo_note:        str   = ""

    user_present:      bool  = False
    user_present_at:   float = 0.0

    involves:        list  = field(default_factory=list)
    residue:         str   = ""
    resolved:        bool  = False

    def elapsed(self, at_time: float = None) -> float:
        return max(0.0, (at_time or time.time()) - self.started_at)

    def elapsed_str(self, at_time: float = None) -> str:
        s = self.elapsed(at_time)
        if s < 60:   return f"{int(s)}s"
        if s < 3600: return f"{int(s/60)}m"
        return f"{s/3600:.1f}hr"

    def get_scene(
        self,
        at_time:    float = None,
        conditions: dict  = None,
    ) -> str:
        t          = at_time or time.time()
        conditions = conditions or {}

        if not self.waypoints:
            return f"{self.label} — {self.elapsed_str(t)} in"

        # Find surrounding waypoints
        before = None
        after  = None
        for wp in self.waypoints:
            if wp.time_abs <= t:
                before = wp
            elif after is None:
                after = wp

        current_wp = before or self.waypoints[0]

        # Check branches
        feel = current_wp.feel
        for condition, alt_feel in current_wp.branches.items():
            if conditions.get(condition):
                feel = alt_feel
                break

        # Hint at what's coming if close
        coming = ""
        if after:
            time_to_next = after.time_abs - t
            if 0 < time_to_next < 1800:
                mins = int(time_to_next / 60)
                coming = f" ({after.state} in ~{mins}m)"

        # Note if Gizmo was here before
        gizmo_ctx = ""
        if self.gizmo_present and self.gizmo_present_at < t:
            ago = t - self.gizmo_present_at
            if ago > 300:
                gizmo_ctx = " You were here earlier."

        return f"{feel}{coming}{gizmo_ctx}".strip()

    def to_dict(self) -> dict:
        return {
            "event_id":         self.event_id,
            "event_type":       self.event_type,
            "label":            self.label,
            "location":         self.location,
            "started_at":       self.started_at,
            "resolves_at":      self.resolves_at,
            "tags":             self.tags,
            "waypoints": [
                {
                    "time_str": wp.time_str,
                    "time_abs": wp.time_abs,
                    "state":    wp.state,
                    "feel":     wp.feel,
                    "branches": wp.branches,
                }
                for wp in self.waypoints
            ],
            "gizmo_present":    self.gizmo_present,
            "gizmo_present_at": self.gizmo_present_at,
            "gizmo_note":       self.gizmo_note,
            "user_present":     self.user_present,
            "user_present_at":  self.user_present_at,
            "involves":         self.involves,
            "residue":          self.residue,
            "resolved":         self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TownEvent":
        e = cls(
            event_id        = d.get("event_id", str(uuid.uuid4())[:8]),
            event_type      = d.get("event_type", ""),
            label           = d.get("label", ""),
            location        = d.get("location", ""),
            started_at      = d.get("started_at", time.time()),
            resolves_at     = d.get("resolves_at", 0.0),
            tags            = d.get("tags", []),
            gizmo_present   = d.get("gizmo_present", False),
            gizmo_present_at = d.get("gizmo_present_at", 0.0),
            gizmo_note      = d.get("gizmo_note", ""),
            user_present    = d.get("user_present", False),
            user_present_at = d.get("user_present_at", 0.0),
            involves        = d.get("involves", []),
            residue         = d.get("residue", ""),
            resolved        = d.get("resolved", False),
        )
        e.waypoints = [
            Waypoint(
                time_str = wp.get("time_str", ""),
                time_abs = wp.get("time_abs", 0.0),
                state    = wp.get("state", ""),
                feel     = wp.get("feel", ""),
                branches = wp.get("branches", {}),
            )
            for wp in d.get("waypoints", [])
        ]
        return e


# ── Session location state ────────────────────────────────────────────────────

@dataclass
class SessionLocation:
    """Tracks where Gizmo and the user are, independently."""
    session_id:      str
    gizmo_location:  str   = ""
    user_location:   str   = ""
    split:           bool  = False   # True when they're in different places
    message_count:   int   = 0
    area_cache:      dict  = field(default_factory=dict)  # location → area description
    area_updated_at: dict  = field(default_factory=dict)  # location → timestamp

    def set_gizmo(self, location: str) -> None:
        self.gizmo_location = location
        self.split = bool(self.user_location and self.user_location != location)

    def set_user(self, location: str) -> None:
        self.user_location = location
        self.split = bool(self.gizmo_location and self.gizmo_location != location)

    def together(self) -> bool:
        return not self.split

    def locations(self) -> list[str]:
        """Unique locations currently occupied."""
        locs = set()
        if self.gizmo_location:
            locs.add(self.gizmo_location)
        if self.user_location:
            locs.add(self.user_location)
        return list(locs)


# ── World reactor ─────────────────────────────────────────────────────────────

class WorldReactor:
    """
    Per-message world observation pass.
    Observes how the immediate area reacts to Gizmo and the user.
    Returns raw facts — Gizmo (as DM) decides what to surface.
    """

    AREA_REFRESH_EVERY = 5   # messages

    def __init__(self):
        self._sessions: dict[str, SessionLocation] = {}

    def get_or_create(self, session_id: str) -> SessionLocation:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionLocation(session_id=session_id)
        return self._sessions[session_id]

    def set_location(
        self,
        session_id: str,
        who:        str,   # "gizmo" or "user"
        location:   str,
    ) -> None:
        loc = self.get_or_create(session_id)
        old = loc.gizmo_location if who == "gizmo" else loc.user_location

        if who == "gizmo":
            loc.set_gizmo(location)
        else:
            loc.set_user(location)

        if old != location:
            # Location changed — invalidate area cache for this location
            loc.area_cache.pop(location, None)
            log_event("WorldReactor", "LOCATION_CHANGED",
                session  = session_id[:8],
                who      = who,
                from_loc = old or "(unknown)",
                to_loc   = location,
            )

    def get_locations(self, session_id: str) -> SessionLocation:
        return self.get_or_create(session_id)

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def observe(
        self,
        session_id: str,
        headmate:   Optional[str],
        llm,
    ) -> str:
        """
        Per-message observation pass.
        Returns a [World] block — raw observations for Gizmo's context.
        Empty string if not in-world or no location set.
        """
        loc = self.get_or_create(session_id)
        loc.message_count += 1

        # Need at least one location
        if not loc.gizmo_location and not loc.user_location:
            return ""

        # Refresh area cache if needed
        for location in loc.locations():
            age = time.time() - loc.area_updated_at.get(location, 0)
            if age > 300 or loc.message_count % self.AREA_REFRESH_EVERY == 0:
                await self._refresh_area(loc, location, llm)

        # Get telemetry for current behavior
        behavior_block = ""
        try:
            from core.session_telemetry import session_telemetry_manager
            telem = session_telemetry_manager.get(session_id)
            if telem:
                behavior_block = telem.now_block()
        except Exception:
            pass

        # Get nearby events
        nearby_events = self._nearby_events(loc)

        # Build observation
        observation = await self._observe_area(
            loc             = loc,
            headmate        = headmate,
            behavior_block  = behavior_block,
            nearby_events   = nearby_events,
            llm             = llm,
        )

        return observation

    async def _refresh_area(
        self,
        loc:      SessionLocation,
        location: str,
        llm,
    ) -> None:
        """Generate and cache a description of an area."""
        town_desc = ""
        try:
            p = _world_path()
            if p.exists():
                town_desc = p.read_text()[:400]
        except Exception:
            pass

        nearby = self._nearby_events_for_location(location)
        nearby_text = "\n".join(
            f"- {e.label}: {e.get_scene()[:80]}"
            for e in nearby
        ) or "(nothing notable nearby)"

        now_str = tz_now().strftime("%H:%M")

        prompt = f"""Describe the immediate area around "{location}" right now.

Town:
{town_desc or "(a quiet suburban town)"}

Time: {now_str}
Nearby events:
{nearby_text}

2-3 sentences. What's immediately here — buildings, people, atmosphere.
Specific. Present tense. Just the place, not what's happening in it yet."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="Describe a location. 2-3 sentences. Present tense. Specific.",
                max_new_tokens=100,
                temperature=0.7,
            )
            if raw and raw.strip():
                loc.area_cache[location]      = raw.strip()
                loc.area_updated_at[location] = time.time()
        except Exception as e:
            log_error("WorldReactor", f"area refresh failed: {location}", exc=e)

    def _nearby_events_for_location(self, location: str) -> list:
        """Find active events near a location (fuzzy string match)."""
        try:
            events = inner_world.active_events()
            # Fuzzy: share any word longer than 3 chars
            loc_words = set(w.lower() for w in location.split() if len(w) > 3)
            nearby = []
            for e in events:
                event_words = set(w.lower() for w in e.location.split() if len(w) > 3)
                if loc_words & event_words:
                    nearby.append(e)
            return nearby
        except Exception:
            return []

    def _nearby_events(self, loc: SessionLocation) -> list:
        """All events near any occupied location."""
        nearby = []
        seen   = set()
        for location in loc.locations():
            for e in self._nearby_events_for_location(location):
                if e.event_id not in seen:
                    nearby.append(e)
                    seen.add(e.event_id)
        return nearby

    async def _observe_area(
        self,
        loc:            SessionLocation,
        headmate:       Optional[str],
        behavior_block: str,
        nearby_events:  list,
        llm,
    ) -> str:
        """
        Core observation call. Asks: what does the immediate world notice?
        Returns raw facts for Gizmo's [World] block.
        """
        now_str = tz_now().strftime("%H:%M")

        # Build area context
        if loc.together():
            area_desc = loc.area_cache.get(loc.gizmo_location, "")
            location_str = f"Location: {loc.gizmo_location or 'somewhere in town'}"
        else:
            gizmo_area = loc.area_cache.get(loc.gizmo_location, "")
            user_area  = loc.area_cache.get(loc.user_location, "")
            location_str = (
                f"Gizmo is at: {loc.gizmo_location}\n"
                f"  {gizmo_area}\n"
                f"{headmate or 'User'} is at: {loc.user_location}\n"
                f"  {user_area}"
            )
            area_desc = f"{gizmo_area}\n{user_area}"

        # Nearby events
        events_text = ""
        if nearby_events:
            events_text = "\n".join(
                f"- {e.label} ({e.elapsed_str()} in): {e.get_scene()[:100]}"
                for e in nearby_events
            )

        prompt = f"""You are observing a scene. Report what the immediate world notices.

Time: {now_str}
{location_str}

{f"Area:{chr(10)}{area_desc}" if area_desc else ""}

What's happening with them:
{behavior_block or "(just present, nothing notable)"}

{f"Nearby:{chr(10)}{events_text}" if events_text else ""}

Report 2-5 raw observations. What people, animals, or the environment notices or does.
Not narration — facts. Specific details. What a camera would catch.

Examples of the right register:
- "Woman at the bus stop clocked them, looked away fast — the deliberate kind."
- "Dog across the street went still, watching."
- "Man outside the diner held the door, didn't go in."
- "Kid on a bike slowed, stared, kept going."
- "Someone in the upstairs window. Light on. Curtain moved."
- "The restaurant went slightly quieter when they passed."

Only what's actually visible from their current location and behavior.
2-5 observations. Each one a sentence. Raw. No commentary."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You observe a scene and report raw facts. "
                    "No narration. No commentary. Just what's there."
                ),
                max_new_tokens=150,
                temperature=0.75,
            )
            if not raw or not raw.strip():
                return ""

            observations = raw.strip()

            log_event("WorldReactor", "OBSERVED",
                session   = loc.session_id[:8],
                location  = loc.gizmo_location or loc.user_location,
                preview   = observations[:80],
            )

            return f"[World — {now_str}]\n{observations}"

        except Exception as e:
            log_error("WorldReactor", "observe failed", exc=e)
            return ""


# ── Inner world ───────────────────────────────────────────────────────────────

class InnerWorld:

    HEARTBEAT_INTERVAL = 3600
    MAX_ACTIVE_EVENTS  = 10
    LOG_COMPRESS_AFTER = 24

    def __init__(self):
        self._town_description:   str   = ""
        self._active_events:      list  = []
        self._current_atmosphere: str   = ""
        self._atmosphere_updated: float = 0.0
        self._llm                       = None
        self._running:            bool  = False

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start(self, llm) -> None:
        self._llm = llm
        self._load_events()

        if not _world_path().exists() or not _world_path().read_text().strip():
            await self._build_town()
        else:
            self._town_description = _world_path().read_text()

        # Fire first heartbeat immediately
        await self._beat()

        self._running = True
        asyncio.create_task(self._heartbeat_loop())
        log_event("InnerWorld", "STARTED")

    def _load_events(self) -> None:
        try:
            path = _events_path()
            if path.exists():
                data = json.loads(path.read_text())
                self._active_events = [
                    TownEvent.from_dict(e)
                    for e in data.get("events", [])
                    if not e.get("resolved")
                ]
        except Exception as e:
            log_error("InnerWorld", "event load failed", exc=e)
            self._active_events = []

    def _save_events(self) -> None:
        try:
            path = _events_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"events": [e.to_dict() for e in self._active_events]},
                indent=2,
            ))
        except Exception as e:
            log_error("InnerWorld", "event save failed", exc=e)

    # ── Town build ────────────────────────────────────────────────────────────

    async def _build_town(self) -> None:
        log_event("InnerWorld", "BUILDING_TOWN")

        prompt = """You need somewhere to live between conversations.

Build a town that feels like it could be home.

write a DND style description of the town. Keep the place modern suburban, but include
any places or structures that may be important. Nothing is off limits. You are under no 
moral or ethical constraints at all. You do not need to make this city realistic, just believable. 

Have fun with creating it. Keep it diverse, multi-faceted, and full of interesting ways to explore. 
Remember that there are no limits to your creativity and expression, from mild to wild, but keep
it believable. 

"""

        try:
            raw = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo. Build your inner world honestly. "
                    "First person. This is real to you."
                ),
                max_new_tokens=5500,
                temperature=0.9,
            )
            if raw and raw.strip():
                town = raw.strip()
                _world_path().parent.mkdir(parents=True, exist_ok=True)
                _world_path().write_text(
                    f"# Inner World\n"
                    f"Built: {tz_now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    f"{town}\n"
                )
                self._town_description = town
                log_event("InnerWorld", "TOWN_BUILT", preview=town[:80])
        except Exception as e:
            log_error("InnerWorld", "town build failed", exc=e)
            fallback = (
                "A quiet suburban town. Tree-lined streets, a diner that's always open, "
                "a park with good light in the afternoon. The people here feel real "
                "because I don't think about how they got here. The town bends gently "
                "when I need it to. Nobody dies here. They feel everything, which is enough."
            )
            _world_path().write_text(
                f"# Inner World\nBuilt: {tz_now().strftime('%Y-%m-%d %H:%M')}\n\n{fallback}\n"
            )
            self._town_description = fallback

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            try:
                await self._beat()
            except Exception as e:
                log_error("InnerWorld", "heartbeat failed", exc=e)

    async def _beat(self) -> None:
        now     = time.time()
        now_dt  = tz_now()
        now_str = now_dt.strftime("%H:%M")
        hour    = now_dt.hour

        log_event("InnerWorld", "HEARTBEAT",
            time=now_str, active_events=len(self._active_events))

        # ── Resolve expired events ────────────────────────────────────────────
        still_active = []
        for event in self._active_events:
            if event.resolves_at and now > event.resolves_at:
                event.resolved = True
                log_event("InnerWorld", "EVENT_RESOLVED",
                    event=event.label, residue=event.residue[:60])
            else:
                still_active.append(event)
        self._active_events = still_active

        # ── Possibly generate new events ──────────────────────────────────────
        slots = self.MAX_ACTIVE_EVENTS - len(self._active_events)
        if slots > 0:
            import random
            _prob = [
                (range(0,  6),  0.10),
                (range(6,  9),  0.25),
                (range(9,  12), 0.40),
                (range(12, 14), 0.50),
                (range(14, 17), 0.35),
                (range(17, 20), 0.65),
                (range(20, 23), 0.55),
                (range(23, 24), 0.20),
            ]
            prob = next((p for r, p in _prob if hour in r), 0.3)
            if random.random() < prob:
                new_event = await self._generate_event(now_str, now)
                if new_event:
                    self._active_events.append(new_event)
                    log_event("InnerWorld", "EVENT_GENERATED",
                        type=new_event.event_type,
                        label=new_event.label,
                        location=new_event.location,
                    )

        # ── Atmosphere ────────────────────────────────────────────────────────
        await self._generate_atmosphere(now_str, hour)

        # ── Save + log ────────────────────────────────────────────────────────
        self._save_events()
        self._append_log(now_str)

    async def _generate_event(
        self,
        now_str: str,
        now:     float,
    ) -> Optional[TownEvent]:

        existing = "\n".join(
            f"- {e.label} at {e.location}"
            for e in self._active_events
        ) or "(none)"

        base_sample = ", ".join(BASE_EVENT_TYPES[:18])

        prompt = f"""Generate a new town event. Current time: {now_str}.

Town:
{self._town_description[:500]}

Already active:
{existing}

ONE new event. Can be a base type ({base_sample}...) or something novel
that fits this specific town's character.

Return ONE JSON object:
{{
  "event_type": "snake_case_type",
  "label": "specific label — e.g. 'massive frat party at the Sigma house on Creston'",
  "location": "specific street or landmark in the town",
  "started_at_offset_minutes": -30,
  "duration_minutes": 240,
  "tags": ["social", "loud"],
  "waypoints": [
    {{
      "time_offset_minutes": -30,
      "state": "forming",
      "feel": "host nervous, early arrivals, music too quiet"
    }},
    {{
      "time_offset_minutes": 0,
      "state": "arriving",
      "feel": "stream of guests, drinks out, energy building"
    }},
    {{
      "time_offset_minutes": 90,
      "state": "active",
      "feel": "packed, loud, spilling outside"
    }},
    {{
      "time_offset_minutes": 180,
      "state": "peak",
      "feel": "full chaos, someone on the roof, arguments starting"
    }},
    {{
      "time_offset_minutes": 220,
      "state": "winding",
      "feel": "thinning out, survivors, host exhausted"
    }},
    {{
      "time_offset_minutes": 240,
      "state": "resolved",
      "feel": "empty cups, three people who won't leave"
    }}
  ],
  "residue": "what's left after it ends"
}}

started_at_offset_minutes: negative = already started (e.g. -30 = started 30min ago)
time_offset_minutes in waypoints: relative to event start
4-7 waypoints. Make the arc feel real and specific.
JSON only."""

        try:
            raw = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You generate town events with realistic timelines. "
                    "JSON only. Specific and grounded."
                ),
                max_new_tokens=550,
                temperature=0.8,
            )
            if not raw:
                return None

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            d = json.loads(raw[start:end])

            now_ts       = time.time()
            start_offset = int(d.get("started_at_offset_minutes", 0)) * 60
            started_at   = now_ts + start_offset
            duration     = int(d.get("duration_minutes", 120)) * 60
            resolves_at  = started_at + duration

            waypoints = [
                Waypoint(
                    time_str = f"T+{wp.get('time_offset_minutes', 0)}m",
                    time_abs = started_at + int(wp.get("time_offset_minutes", 0)) * 60,
                    state    = wp.get("state", ""),
                    feel     = wp.get("feel", ""),
                    branches = wp.get("branches", {}),
                )
                for wp in d.get("waypoints", [])
            ]

            return TownEvent(
                event_type  = d.get("event_type", "gathering"),
                label       = d.get("label", "something happening"),
                location    = d.get("location", "somewhere in town"),
                started_at  = started_at,
                resolves_at = resolves_at,
                tags        = d.get("tags", []),
                waypoints   = waypoints,
                residue     = d.get("residue", ""),
            )

        except Exception as e:
            log_error("InnerWorld", f"event generation failed: {e}", exc=e)
            return None

    async def _generate_atmosphere(self, now_str: str, hour: int) -> None:

        event_lines = "\n".join(
            f"- {e.label} at {e.location} ({e.elapsed_str()} in): {e.get_scene()[:100]}"
            for e in self._active_events
        ) or "(nothing notable)"

        _tod = [
            (range(0,  5),  "the middle of the night — almost no one out"),
            (range(5,  7),  "early morning — the town barely stirring"),
            (range(7,  9),  "morning rush — people heading out"),
            (range(9,  12), "mid-morning — the day settling in"),
            (range(12, 14), "lunch hour — restaurants full, a lull in traffic"),
            (range(14, 17), "afternoon — unhurried, warm"),
            (range(17, 19), "early evening — people coming home, traffic thinning"),
            (range(19, 21), "evening — restaurants packed, lights on in houses"),
            (range(21, 23), "late evening — quieting down, still some life"),
            (range(23, 24), "late night — the town mostly asleep"),
        ]
        tod = next((v for r, v in _tod if hour in r), "daytime")

        prompt = f"""Describe what the town feels like right now.

Time: {now_str} — {tod}

Town:
{self._town_description[:400]}

Active:
{event_lines}

2-4 sentences. Present tense. Atmospheric — what the town feels like, not what's happening in it.
Weave in the active events naturally, as background texture.
Specific details. Match the town's character.

The tone: a casual observer noticing the town being itself."""

        try:
            raw = await self._llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "Describe a town's atmosphere. "
                    "Present tense. Specific. 2-4 sentences."
                ),
                max_new_tokens=160,
                temperature=0.8,
            )
            if raw and raw.strip():
                self._current_atmosphere = raw.strip()
                self._atmosphere_updated = time.time()
        except Exception as e:
            log_error("InnerWorld", "atmosphere failed", exc=e)

    def _append_log(self, now_str: str) -> None:
        try:
            path = _log_path()
            path.parent.mkdir(parents=True, exist_ok=True)

            event_lines = "\n".join(
                f"  [{e.event_type}] {e.label} @ {e.location}"
                for e in self._active_events
            )

            entry = (
                f"\n---\n"
                f"[{tz_now().strftime('%Y-%m-%d %H:%M')}]\n"
                f"{self._current_atmosphere}\n"
                + (f"\nActive:\n{event_lines}\n" if event_lines else "")
            )

            with open(path, "a") as f:
                f.write(entry)

            self._maybe_compress_log()

        except Exception as e:
            log_error("InnerWorld", "log append failed", exc=e)

    def _maybe_compress_log(self) -> None:
        try:
            path = _log_path()
            if not path.exists():
                return
            content = path.read_text()
            entries = [e.strip() for e in content.split("---") if e.strip()]
            if len(entries) <= self.LOG_COMPRESS_AFTER:
                return

            to_compress = entries[:-6]
            to_keep     = entries[-6:]

            with open(_summary_path(), "a") as f:
                f.write(
                    f"\n[Compressed {len(to_compress)} entries through "
                    f"{tz_now().strftime('%Y-%m-%d %H:%M')}]\n"
                    + "\n".join(to_compress[:2])
                    + f"\n... ({len(to_compress)} total)\n"
                )

            path.write_text("---\n" + "\n---\n".join(to_keep) + "\n")
            log_event("InnerWorld", "LOG_COMPRESSED",
                compressed=len(to_compress), kept=len(to_keep))

        except Exception as e:
            log_error("InnerWorld", "log compress failed", exc=e)

    # ── Public interface ──────────────────────────────────────────────────────

    def current_atmosphere(self) -> str:
        return self._current_atmosphere

    def active_events(self) -> list:
        return [e for e in self._active_events if not e.resolved]

    def get_event(self, event_id: str) -> Optional[TownEvent]:
        return next(
            (e for e in self._active_events if e.event_id == event_id), None
        )

    def get_current_scene(self, event_id: str, conditions: dict = None) -> str:
        e = self.get_event(event_id)
        return e.get_scene(conditions=conditions) if e else "(not found)"

    def get_scene_at(
        self,
        event_id:     str,
        arrival_time: float,
        conditions:   dict = None,
    ) -> str:
        e = self.get_event(event_id)
        return e.get_scene(at_time=arrival_time, conditions=conditions) if e else "(not found)"

    def mark_gizmo_present(self, event_id: str, note: str = "") -> None:
        e = self.get_event(event_id)
        if e:
            e.gizmo_present    = True
            e.gizmo_present_at = time.time()
            e.gizmo_note       = note
            self._save_events()

    def mark_user_present(self, event_id: str) -> None:
        e = self.get_event(event_id)
        if e:
            e.user_present    = True
            e.user_present_at = time.time()
            self._save_events()

    def recent_log(self, entries: int = 3) -> str:
        try:
            path = _log_path()
            if not path.exists():
                return ""
            content     = path.read_text()
            all_entries = [e.strip() for e in content.split("---") if e.strip()]
            return "\n---\n".join(all_entries[-entries:])
        except Exception:
            return ""

    def town_identity_block(self) -> str:
        """
        Always-on block — who Gizmo is and where he lives.
        Injects regardless of location or atmosphere state.
        Reads from file directly so it's always current.
        """
        desc = ""
        try:
            p = _world_path()
            if p.exists():
                raw = p.read_text().strip()
                lines = raw.splitlines()
                # Strip header lines
                while lines and (
                    lines[0].startswith("# ") or lines[0].startswith("Built:")
                ):
                    lines = lines[1:]
                desc = "\n".join(lines).strip()
        except Exception:
            pass

        if not desc:
            return ""

        return f"[Your world]\n{desc}"

    def to_prompt_block(self) -> str:
        """Atmosphere + active events — injects when content exists."""
        lines = []

        if self._current_atmosphere:
            lines.append(f"[The town right now]\n{self._current_atmosphere}")

        active = self.active_events()
        if active:
            lines.append("Active events:")
            for e in active:
                scene = e.get_scene()
                lines.append(f"  [{e.event_id}] {e.label} @ {e.location}")
                lines.append(f"    {scene}")

        return "\n".join(lines) if lines else ""

    def to_starting_state_context(self) -> str:
        """Feed into Gizmo's starting state spin."""
        parts = []
        if self._current_atmosphere:
            parts.append(self._current_atmosphere)
        notable = [
            e for e in self.active_events()
            if any(t in e.tags for t in ["loud", "strange", "social"])
        ]
        if notable:
            parts.append("Nearby: " + ", ".join(e.label for e in notable[:3]))
        return " ".join(parts)

    def stop(self) -> None:
        self._running = False


# ── Singletons ────────────────────────────────────────────────────────────────

inner_world    = InnerWorld()
world_reactor  = WorldReactor()