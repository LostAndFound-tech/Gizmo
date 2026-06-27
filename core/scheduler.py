"""
core/scheduler.py

Persistent life scheduler for Gizmo.

Builds and maintains a per-person schedule from casual conversation —
picking up time references, event labels, requirements, and item locations
as they emerge naturally.

Store:
  headmates/system/schedule.json          -- system-level schedule (default)
  headmates/{name}/schedule.json          -- headmate-specific schedule

Brief injection:
  _build_schedule_brief(name) -> str
  Returns the CURRENT TIME + TODAY'S SCHEDULE + TODAY'S REQUIREMENTS block
  for insertion at the top of every Gizmo brief.

Extraction:
  scheduler.extract(chunk, speaker, registry, session_id)
  Runs in the chunk_processor parallel gather.
  Detects schedule/requirement signals and writes to the appropriate file.

Lookup:
  scheduler.lookup(name, date_str, query) -> str
  Answers questions about past days from the daily_log.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now
import core.librarian as librarian


# ── Schema helpers ────────────────────────────────────────────────────────────

def _empty_schedule() -> dict:
    return {
        "routines":    {},   # day_name -> [event, ...]
        "requirements": {},  # key -> requirement def
        "one_offs":    [],   # [{date, time, label, requirements, completed}]
        "daily_log":   {},   # date_str -> {req_key -> status}
    }


def _empty_requirement(label: str, before: str = "", resets: str = "daily") -> dict:
    return {
        "label":  label,
        "before": before,
        "resets": resets,
        "items":  {},
    }


# ── File helpers ──────────────────────────────────────────────────────────────

def _schedule_path(name: str) -> str:
    """
    System schedule if name is 'system', headmate schedule otherwise.
    """
    if name == "system":
        return librarian.system_path("schedule.json")
    return librarian.headmate_path(name, "schedule.json")


def _read_schedule(name: str) -> dict:
    return librarian._read_file(_schedule_path(name)) or _empty_schedule()


def _write_schedule(name: str, data: dict) -> None:
    librarian._write_json(_schedule_path(name), data)


# ── Brief builder ─────────────────────────────────────────────────────────────

def _fmt_time(t: str) -> str:
    """Convert 24h 'HH:MM' to '12:00 PM' style."""
    try:
        h, m = map(int, t.split(":"))
        suffix = "AM" if h < 12 else "PM"
        h12    = h % 12 or 12
        return f"{h12:2d}:{m:02d} {suffix}"
    except Exception:
        return t


def _minutes_until(t: str, now: datetime) -> int:
    """Minutes from now until HH:MM today. Negative = past."""
    try:
        h, m   = map(int, t.split(":"))
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return int((target - now).total_seconds() / 60)
    except Exception:
        return 0


def _req_line(key: str, req: dict, log_today: dict) -> str:
    """Format one requirement line with sub-items if present."""
    label  = req.get("label", key)
    status = log_today.get(key, {})

    if req.get("items"):
        # Has sub-items
        parts = []
        items_log = status.get("items", {})
        for item_key, item_def in req["items"].items():
            item_status = items_log.get(item_key, {})
            done        = item_status.get("completed", False)
            loc         = item_def.get("known_location", "") or item_status.get("location", "")
            mark        = "yes" if done else "no"
            part        = f"{item_def.get('label', item_key)} ({mark})"
            if not done and loc:
                part += f" [location: '{loc}']"
            parts.append(part)
        obj = req.get("object", "")
        obj_note = f" [object: '{obj}']" if obj else ""
        completed = status.get("completed", False)
        wearing   = status.get("wearing", False)
        if wearing:
            state = "(Wearing)"
        elif completed:
            state = "(Completed)"
        else:
            state = ""
        return f"  {label}{obj_note} {state}\n    " + ", ".join(parts)

    else:
        # Simple requirement
        completed = status.get("completed", False)
        wearing   = status.get("wearing", False)
        obj       = req.get("object", "")
        obj_note  = f" [object: '{obj}']" if obj else ""
        if wearing:
            state = "(Wearing)"
        elif completed:
            state = "(Completed)"
        else:
            state = ""
        return f"  {label}{obj_note} {state}"


def build_schedule_brief(fronter: str = "system") -> str:
    """
    Build the TIME + SCHEDULE + REQUIREMENTS block for the top of every brief.
    Pulls from system schedule always, plus headmate schedule if fronter is specified.
    """
    now      = tz_now()
    today    = now.strftime("%A").lower()        # e.g. "tuesday"
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%I:%M %p %Z").lstrip("0")

    lines = []
    lines.append(f"CURRENT TIME: {now.strftime('%A, %B %d, %Y')} — {time_str}")

    # Collect events from system schedule (and headmate if different)
    schedules_to_check = ["system"]
    if fronter and fronter not in ("system", "unknown"):
        schedules_to_check.append(fronter)

    all_events   = []
    all_req_defs = {}
    log_today    = {}

    for sched_name in schedules_to_check:
        sched     = _read_schedule(sched_name)
        routines  = sched.get("routines", {})
        day_events = routines.get(today, [])
        all_events.extend(day_events)

        # One-offs for today
        for one_off in sched.get("one_offs", []):
            if one_off.get("date") == date_str and not one_off.get("completed"):
                all_events.append(one_off)

        # Merge requirements
        all_req_defs.update(sched.get("requirements", {}))

        # Merge today's log
        log_today.update(sched.get("daily_log", {}).get(date_str, {}))

    # Sort events by time
    all_events.sort(key=lambda e: e.get("time", "99:99"))

    # Schedule block
    if all_events:
        lines.append("\nTODAY'S SCHEDULE:")
        for event in all_events:
            t       = event.get("time", "")
            label   = event.get("label", "")
            mins    = _minutes_until(t, now) if t else None
            ft      = _fmt_time(t) if t else ""

            if mins is not None and mins < 0:
                tag = "[past]"
            elif mins is not None and mins <= 60:
                tag = f"[in {mins}m]"
            elif mins is not None:
                h = mins // 60
                m = mins % 60
                tag = f"[in {h}h {m}m]" if m else f"[in {h}h]"
            else:
                tag = ""

            lines.append(f"  {ft} — {label} {tag}".rstrip())
    else:
        lines.append("\nNo scheduled events today.")

    # Requirements block — only show reqs tied to upcoming or recent events
    # Find the next upcoming event that has requirements
    upcoming_reqs = set()
    next_event_label = ""
    for event in all_events:
        t    = event.get("time", "99:99")
        mins = _minutes_until(t, now)
        reqs = event.get("requirements", [])
        if reqs and mins > -30:  # within 30 min past or any future
            upcoming_reqs.update(reqs)
            if not next_event_label and mins > 0:
                next_event_label = event.get("label", "")

    if upcoming_reqs and all_req_defs:
        header = f"\nTODAY'S REQUIREMENTS"
        if next_event_label:
            header += f" (before {next_event_label})"
        header += ":"
        lines.append(header)

        for req_key in upcoming_reqs:
            req = all_req_defs.get(req_key)
            if req:
                lines.append(_req_line(req_key, req, log_today))

    # Incomplete past requirements (carry-forward for the day)
    incomplete_past = []
    for req_key, req in all_req_defs.items():
        if req_key in upcoming_reqs:
            continue
        status = log_today.get(req_key, {})
        if not status.get("completed", False):
            incomplete_past.append(req_key)

    if incomplete_past:
        lines.append("\nINCOMPLETE TODAY:")
        for req_key in incomplete_past:
            req = all_req_defs.get(req_key, {})
            lines.append(_req_line(req_key, req, log_today))

    print(f"THE SCHEDULING BRIEF AT THIS TIME:\n\n".join(lines))
    return "\n".join(lines)


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """
You are Gizmo's schedule intake pass. Extract schedule-relevant information from
a conversational exchange — time references, recurring events, requirements,
item locations, and completion signals.

Return ONLY valid JSON. No markdown. No explanation. No preamble.
If nothing schedule-relevant is present, return {"events": [], "completions": [], "locations": []}.

{
  "events": [
    {
      "owner":    "system | {headmate_name}",
      "type":     "routine | one_off",
      "day":      "monday | tuesday | ... | daily | weekdays | weekends | null",
      "date":     "YYYY-MM-DD or null",
      "time":     "HH:MM (24h) or null",
      "label":    "short event label",
      "requirements": ["dressed", "bag_packed"],
      "confidence": "stated | implied"
    }
  ],
  "requirements": [
    {
      "owner":  "system | {headmate_name}",
      "key":    "snake_case_key",
      "label":  "Human readable label",
      "before": "event label this must happen before",
      "resets": "daily | never",
      "items":  [
        {
          "key":            "snake_case",
          "label":          "Item label",
          "known_location": "where it lives, if mentioned"
        }
      ]
    }
  ],
  "completions": [
    {
      "owner":   "system | {headmate_name}",
      "req_key": "snake_case_key",
      "item_key": "snake_case or null if top-level",
      "completed": true,
      "wearing":   false,
      "location":  "current location if mentioned"
    }
  ],
  "locations": [
    {
      "owner":    "system | {headmate_name}",
      "req_key":  "snake_case_key",
      "item_key": "snake_case",
      "location": "where the item is right now"
    }
  ]
}

OWNER RULES:
- If a specific headmate is named ("Kaylee has to go to work"), owner = "kaylee"
- If first person ("I have to go to work", "we need to leave"), owner = "system"
- Default to "system" when unclear

EVENT DETECTION:
- "I have to go to work at noon" -> routine event, time 12:00
- "I catch the 121 bus at 12:22 on weekdays" -> routine, weekdays, 12:22
- "I have a doctor appointment Thursday at 2" -> one_off, date=thursday's date, 14:00
- "mini-journal session at 6" -> routine, 18:00

REQUIREMENT DETECTION:
- "I need to get dressed before work" -> requirement "dressed", before "work"
- "I need my vape, keys, phone, and cigarettes" -> requirement "pockets_packed" with items
- "gotta pack my bag (notebook, pens, pencils)" -> requirement "bag_packed" with items

COMPLETION DETECTION:
- "I'm dressed" / "got dressed" -> completion of "dressed"
- "grabbed my vape" -> completion of vape item in pockets
- "wearing my work shoes" -> completion of shoes, wearing=true
- "I found my keys, they were by the door" -> completion + location

LOCATION DETECTION:
- "my keys are always by the door" -> known_location for keys
- "pens are on the art table" -> known_location for pens

TIME FORMAT: always 24h HH:MM
DAY FORMAT: full lowercase day name, or "daily", "weekdays", "weekends"
""".strip()


async def _call_llm(prompt: str) -> Optional[dict]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_EXTRACT_SYSTEM,
            temperature=0.0,
            max_new_tokens=2000,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("Scheduler", "LLM call failed", exc=e)
        return None


# ── Write helpers ─────────────────────────────────────────────────────────────

def _merge_event(sched: dict, event: dict) -> None:
    """Merge an extracted event into the schedule."""
    event_type = event.get("type", "routine")
    day        = event.get("day")
    label      = event.get("label", "")
    time_      = event.get("time")
    reqs       = event.get("requirements", [])

    if event_type == "one_off":
        date = event.get("date")
        if date:
            existing = [e for e in sched["one_offs"] if e.get("date") == date and e.get("label") == label]
            if not existing:
                sched["one_offs"].append({
                    "date":         date,
                    "time":         time_,
                    "label":        label,
                    "requirements": reqs,
                    "completed":    False,
                    "confidence":   event.get("confidence", "stated"),
                })
    else:
        if not day:
            return
        days = []
        if day == "daily":
            days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        elif day == "weekdays":
            days = ["monday","tuesday","wednesday","thursday","friday"]
        elif day == "weekends":
            days = ["saturday","sunday"]
        else:
            days = [day]

        for d in days:
            if d not in sched["routines"]:
                sched["routines"][d] = []
            existing = [e for e in sched["routines"][d] if e.get("label") == label]
            if existing:
                # Update time if more specific
                if time_:
                    existing[0]["time"] = time_
                if reqs:
                    existing[0]["requirements"] = list(set(existing[0].get("requirements", []) + reqs))
            else:
                sched["routines"][d].append({
                    "time":         time_ or "",
                    "label":        label,
                    "requirements": reqs,
                    "confidence":   event.get("confidence", "stated"),
                })
                sched["routines"][d].sort(key=lambda e: e.get("time", "99:99"))


def _merge_requirement(sched: dict, req: dict) -> None:
    """Merge an extracted requirement definition into the schedule."""
    key = req.get("key")
    if not key:
        return
    existing = sched["requirements"].get(key, _empty_requirement(req.get("label", key)))
    existing["label"]  = req.get("label", existing["label"])
    existing["before"] = req.get("before", existing["before"])
    existing["resets"] = req.get("resets", existing["resets"])

    for item in req.get("items", []):
        ikey = item.get("key")
        if ikey:
            existing["items"][ikey] = {
                "label":          item.get("label", ikey),
                "known_location": item.get("known_location", ""),
            }

    sched["requirements"][key] = existing


def _apply_completion(sched: dict, comp: dict, date_str: str) -> None:
    """Mark a requirement or item as completed in the daily log."""
    req_key  = comp.get("req_key")
    item_key = comp.get("item_key")
    if not req_key:
        return

    if date_str not in sched["daily_log"]:
        sched["daily_log"][date_str] = {}

    log = sched["daily_log"][date_str]

    if item_key:
        if req_key not in log:
            log[req_key] = {"completed": False, "items": {}}
        if "items" not in log[req_key]:
            log[req_key]["items"] = {}
        log[req_key]["items"][item_key] = {
            "completed": comp.get("completed", True),
        }
        if comp.get("location"):
            log[req_key]["items"][item_key]["location"] = comp["location"]
        # Update known_location on the requirement definition
        if req_key in sched["requirements"]:
            req_def = sched["requirements"][req_key]
            if item_key in req_def.get("items", {}):
                if comp.get("location"):
                    req_def["items"][item_key]["known_location"] = comp["location"]
        # Check if all items done
        req_def  = sched["requirements"].get(req_key, {})
        all_keys = set(req_def.get("items", {}).keys())
        done_keys = {k for k, v in log[req_key].get("items", {}).items() if v.get("completed")}
        log[req_key]["completed"] = (all_keys == done_keys) if all_keys else False
    else:
        log[req_key] = {
            "completed": comp.get("completed", True),
            "wearing":   comp.get("wearing", False),
        }
        if comp.get("location"):
            log[req_key]["location"] = comp["location"]


def _apply_location(sched: dict, loc: dict) -> None:
    """Update known location of an item on the requirement definition."""
    req_key  = loc.get("req_key")
    item_key = loc.get("item_key")
    location = loc.get("location", "")
    if not req_key or not item_key or not location:
        return
    req_def = sched["requirements"].get(req_key, {})
    if item_key in req_def.get("items", {}):
        req_def["items"][item_key]["known_location"] = location


# ── Past day lookup ───────────────────────────────────────────────────────────

def lookup_day(name: str, date_str: str) -> str:
    """
    Return a human-readable summary of a past day's schedule completion.
    Used when someone asks "did I brush my teeth yesterday?"
    """
    sched     = _read_schedule(name)
    log       = sched.get("daily_log", {}).get(date_str, {})
    req_defs  = sched.get("requirements", {})

    if not log:
        return f"No schedule data recorded for {date_str}."

    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        day_label = d.strftime("%A, %B %d")
    except Exception:
        day_label = date_str

    lines = [f"Schedule log for {day_label}:"]
    for req_key, status in log.items():
        req_def  = req_defs.get(req_key, {})
        label    = req_def.get("label", req_key)
        completed = status.get("completed", False)
        wearing   = status.get("wearing", False)

        if wearing:
            state = "worn"
        elif completed:
            state = "done"
        else:
            state = "not done"

        line = f"  {label}: {state}"

        items = status.get("items", {})
        if items:
            item_parts = []
            for ikey, istatus in items.items():
                idef  = req_def.get("items", {}).get(ikey, {})
                ilabel = idef.get("label", ikey)
                idone  = istatus.get("completed", False)
                iloc   = istatus.get("location", idef.get("known_location", ""))
                part   = f"{ilabel}: {'yes' if idone else 'no'}"
                if not idone and iloc:
                    part += f" [{iloc}]"
                item_parts.append(part)
            line += " — " + ", ".join(item_parts)

        lines.append(line)

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

class Scheduler:

    async def extract(
        self,
        chunk:      list[str],
        speaker:    str,
        registry:   dict,
        session_id: str = "",
    ) -> Optional[dict]:
        """
        Extract schedule signals from a chunk and write to the appropriate schedule files.
        Runs in the chunk_processor parallel gather.
        """
        if not chunk:
            return None

        text = "\n".join(chunk)
        now  = tz_now()

        known_headmates = [
            k for k in registry
            if not k.startswith("_")
            and registry[k].get("type") == "Person"
            and k.lower() != "gizmo"
        ]

        prompt = (
            f"Speaker: {speaker}\n"
            f"Known headmates: {', '.join(known_headmates)}\n"
            f"Current date: {now.strftime('%A, %Y-%m-%d')}\n"
            f"Current time: {now.strftime('%H:%M')}\n\n"
            f"Exchange:\n{text}"
        )

        try:
            result = await _call_llm(prompt)
            if not result:
                return None

            date_str = now.strftime("%Y-%m-%d")
            written  = {"events": [], "completions": [], "locations": []}

            # Process events
            for event in result.get("events", []):
                owner = event.get("owner", "system")
                sched = _read_schedule(owner)
                _merge_event(sched, event)
                _write_schedule(owner, sched)
                written["events"].append(event.get("label", ""))
                print(f"[Scheduler] event -> {owner}: {event.get('label')} ({event.get('day') or event.get('date')} {event.get('time')})")

            # Process requirement definitions
            for req in result.get("requirements", []):
                owner = req.get("owner", "system")
                sched = _read_schedule(owner)
                _merge_requirement(sched, req)
                _write_schedule(owner, sched)
                print(f"[Scheduler] requirement -> {owner}: {req.get('key')}")

            # Process completions
            for comp in result.get("completions", []):
                owner = comp.get("owner", "system")
                sched = _read_schedule(owner)
                _apply_completion(sched, comp, date_str)
                _write_schedule(owner, sched)
                written["completions"].append(comp.get("req_key", ""))
                print(f"[Scheduler] completion -> {owner}: {comp.get('req_key')} / {comp.get('item_key')}")

            # Process location updates
            for loc in result.get("locations", []):
                owner = loc.get("owner", "system")
                sched = _read_schedule(owner)
                _apply_location(sched, loc)
                _write_schedule(owner, sched)
                written["locations"].append(loc.get("item_key", ""))
                print(f"[Scheduler] location -> {owner}: {loc.get('item_key')} @ {loc.get('location')}")

            if any(written.values()):
                log_event("Scheduler", "EXTRACTED",
                    session=session_id[:8],
                    events=len(written["events"]),
                    completions=len(written["completions"]),
                )
            print(f"SCHEDULER WRITING:{written}")
            return written if any(written.values()) else None

        except Exception as e:
            log_error("Scheduler", "extract failed", exc=e)
            print(f"[Scheduler] extract failed: {type(e).__name__}: {e}")
            return None


scheduler = Scheduler()
