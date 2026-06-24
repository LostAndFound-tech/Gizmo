"""
core/intent_classifier.py

Lightweight intent classifier. Runs on every incoming message before the
pipeline fires. Routes structured queries to the right data source and
injects a DIRECT ANSWER block into the brief so Gizmo can respond
from fact rather than inference.

Intents:
  schedule_lookup     -- "what time is my thing tomorrow", "when do I work Friday"
  schedule_update     -- "actually the bus is at 12:30 on Tuesdays"
  past_lookup         -- "did I eat this morning", "what did I do yesterday"
  location_query      -- "where are my keys", "where did I put my vape"
  requirement_check   -- "have I packed my bag", "am I ready for work"
  requirement_complete -- "I grabbed my keys", "I'm dressed", "found my phone"
  context_reset       -- "that was from before", "ignore that", "new topic"
  mode_switch         -- "journal mode", "let's do a scene"
  wellness_report     -- "run wellness report", "run report for jess"
  none                -- normal conversation, no special routing needed

For structured intents, resolve() returns:
  {
    "intent":       "schedule_lookup",
    "confidence":   0.92,
    "params":       {"date": "tomorrow", "query": "what time is my thing"},
    "direct_answer": "Tomorrow is Wednesday. Scheduled: 12:00 PM leaves for work..."
  }

direct_answer is injected into the brief as:
  DIRECT ANSWER (schedule_lookup):
  [answer text]

Gizmo then delivers this through his own voice — the answer is the data,
the response is still his.
"""

import json
import re
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now
import core.librarian as librarian


# ── Classifier prompt ─────────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM = """
You are a fast intent classifier for an AI companion named Gizmo.
Classify the user's message into one of the known intents.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "intent":     "intent_name",
  "confidence": 0.95,
  "params":     {}
}

KNOWN INTENTS:

schedule_lookup
  User is asking about upcoming events, times, or their schedule.
  params: {"date": "today|tomorrow|monday|...", "query": "what they asked"}
  Examples:
    "what time is my thing tomorrow"
    "when do I work this week"
    "do I have anything today"
    "what's on my schedule"
    "what time does my shift start"

schedule_update
  User is correcting or adding to their schedule.
  params: {"raw": "the full correction or addition"}
  Examples:
    "actually the bus is at 12:30 on Tuesdays"
    "I moved my appointment to Thursday"
    "add therapy every Wednesday at 3"

past_lookup
  User is asking about something that happened or was done on a past day.
  params: {"date": "yesterday|monday|2026-06-23|...", "query": "what they asked"}
  Examples:
    "did I eat this morning"
    "did I brush my teeth yesterday"
    "what did I do last Tuesday"
    "did I take my meds today"

location_query
  User is asking where something is.
  params: {"item": "the thing they're looking for"}
  Examples:
    "where are my keys"
    "where did I put my vape"
    "where's my phone"
    "have you seen my lighter"

requirement_check
  User is asking whether they've completed preparation steps.
  params: {"query": "what they asked about"}
  Examples:
    "am I ready for work"
    "have I packed my bag"
    "did I get everything"
    "what do I still need to do before I leave"

requirement_complete
  User is reporting a completion — they did something, found something, packed something.
  params: {"raw": "what they said they completed"}
  Examples:
    "I grabbed my keys"
    "I'm dressed"
    "found my phone, it was on the counter"
    "bag is packed"
    "got my vape"

context_reset
  User wants to drop the current conversational context and start fresh.
  params: {}
  Examples:
    "that was from before"
    "ignore what I said earlier"
    "different topic"
    "new scene"
    "starting fresh"
    "forget that"
    "that was yesterday"
    "we were talking about something else"
    "never mind that"

mode_switch
  User wants to switch Gizmo's mode.
  params: {"mode": "journal|brainstorm|roleplay|passive|chat|aftercare"}
  Examples:
    "journal mode"
    "let's do a scene"
    "brainstorm mode"
    "I need aftercare"

wellness_report
  User wants to run a wellness synthesis report.
  params: {"name": "person name or null for all"}
  Examples:
    "run wellness report"
    "run report for jess"
    "wellness synthesis"

none
  Normal conversation. No special routing needed.
  params: {}

CONFIDENCE:
- Use 0.95+ when the intent is unmistakable
- Use 0.7-0.94 when fairly clear but could be normal chat
- Use below 0.7 only if genuinely ambiguous — prefer "none" when unsure

Default to "none" when in doubt. False positives on structured intents
are worse than missed ones.
""".strip()


async def _call_llm(message: str, context: str = "") -> Optional[dict]:
    try:
        from core.llm import llm
        prompt = f"Message: {message}"
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_CLASSIFIER_SYSTEM,
            temperature=0.0,
            max_new_tokens=120,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("IntentClassifier", "classify failed", exc=e)
        return None


# ── Date resolution ───────────────────────────────────────────────────────────

_DAY_NAMES = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

def _resolve_date(date_str: str) -> str:
    """
    Convert natural date references to YYYY-MM-DD.
    Returns today's date as fallback.
    """
    now  = tz_now()
    today = now.date()

    s = date_str.lower().strip()

    if s in ("today", "this morning", "tonight", "this evening", "now"):
        return today.strftime("%Y-%m-%d")

    if s in ("yesterday", "last night", "this morning"):
        from datetime import timedelta
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    if s in ("tomorrow", "tomorrow morning", "tomorrow night"):
        from datetime import timedelta
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    # Day name — find next occurrence
    for i, day in enumerate(_DAY_NAMES):
        if day in s:
            current_dow = today.weekday()  # 0=Monday
            target_dow  = i
            days_ahead  = (target_dow - current_dow) % 7
            if days_ahead == 0 and "last" in s:
                days_ahead = 7
            if days_ahead == 0:
                # "this monday" when it IS monday = today
                pass
            from datetime import timedelta
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Try direct date parse
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d", "%B %d"):
        try:
            from datetime import datetime
            d = datetime.strptime(date_str, fmt)
            if d.year == 1900:
                d = d.replace(year=today.year)
            return d.strftime("%Y-%m-%d")
        except Exception:
            pass

    return today.strftime("%Y-%m-%d")


# ── Answer builders ───────────────────────────────────────────────────────────

def _answer_schedule_lookup(params: dict, fronter: str) -> str:
    from core.scheduler import _read_schedule, _fmt_time, _minutes_until
    date_str  = _resolve_date(params.get("date", "today"))
    now       = tz_now()

    try:
        from datetime import datetime
        d        = datetime.strptime(date_str, "%Y-%m-%d")
        day_name = d.strftime("%A").lower()
        day_label = d.strftime("%A, %B %d")
    except Exception:
        day_name  = now.strftime("%A").lower()
        day_label = now.strftime("%A, %B %d")

    is_today = date_str == now.strftime("%Y-%m-%d")

    # Pull from system + headmate schedule
    all_events = []
    for sched_name in (["system"] + ([fronter] if fronter not in ("system","unknown") else [])):
        sched = _read_schedule(sched_name)
        all_events.extend(sched.get("routines", {}).get(day_name, []))
        if is_today:
            for one_off in sched.get("one_offs", []):
                if one_off.get("date") == date_str and not one_off.get("completed"):
                    all_events.append(one_off)

    all_events.sort(key=lambda e: e.get("time", "99:99"))

    if not all_events:
        return f"Nothing scheduled for {day_label}."

    lines = [f"{day_label}:"]
    for event in all_events:
        t     = event.get("time", "")
        label = event.get("label", "")
        ft    = _fmt_time(t) if t else ""
        if is_today and t:
            mins = _minutes_until(t, now)
            if mins < 0:
                tag = "[past]"
            elif mins <= 60:
                tag = f"[in {mins}m]"
            else:
                h = mins // 60
                m = mins % 60
                tag = f"[in {h}h {m}m]" if m else f"[in {h}h]"
            lines.append(f"  {ft} — {label} {tag}")
        else:
            lines.append(f"  {ft} — {label}".rstrip())

    return "\n".join(lines)


def _answer_past_lookup(params: dict, fronter: str) -> str:
    from core.scheduler import lookup_day
    date_str = _resolve_date(params.get("date", "yesterday"))
    result   = lookup_day("system", date_str)
    if fronter and fronter not in ("system", "unknown"):
        hm_result = lookup_day(fronter, date_str)
        if "No schedule data" not in hm_result:
            result = result + "\n" + hm_result
    return result


def _answer_location_query(params: dict, fronter: str) -> str:
    """Search known_locations across all requirement definitions."""
    item_query = params.get("item", "").lower()
    if not item_query:
        return "I'm not sure what you're looking for."

    from core.scheduler import _read_schedule
    found = []

    for sched_name in (["system"] + ([fronter] if fronter not in ("system","unknown") else [])):
        sched    = _read_schedule(sched_name)
        req_defs = sched.get("requirements", {})
        for req_key, req_def in req_defs.items():
            for item_key, item_def in req_def.get("items", {}).items():
                label = item_def.get("label", item_key).lower()
                if item_query in label or label in item_query:
                    loc = item_def.get("known_location", "")
                    if loc:
                        found.append(f"{item_def.get('label', item_key)}: {loc}")

    if found:
        return "\n".join(found)
    return f"No known location for {params.get('item', 'that')}."


def _answer_requirement_check(params: dict, fronter: str) -> str:
    """Check today's requirement completion status."""
    from core.scheduler import _read_schedule, _req_line
    now      = tz_now()
    date_str = now.strftime("%Y-%m-%d")

    lines    = []
    for sched_name in (["system"] + ([fronter] if fronter not in ("system","unknown") else [])):
        sched     = _read_schedule(sched_name)
        req_defs  = sched.get("requirements", {})
        log_today = sched.get("daily_log", {}).get(date_str, {})
        for req_key, req_def in req_defs.items():
            lines.append(_req_line(req_key, req_def, log_today))

    if lines:
        return "Current requirement status:\n" + "\n".join(lines)
    return "No requirements tracked yet."


# ── Main classifier ───────────────────────────────────────────────────────────

class IntentClassifier:

    async def classify(
        self,
        message:  str,
        fronter:  str = "system",
        context:  str = "",
    ) -> dict:
        """
        Classify a message and resolve any structured query into a direct answer.

        Returns:
        {
          "intent":        "schedule_lookup",
          "confidence":    0.92,
          "params":        {...},
          "direct_answer": "Tomorrow is Wednesday. 12:00 PM — leaves for work..."
        }

        direct_answer is empty string for intents that don't need data lookup,
        or for "none".
        """
        result = await _call_llm(message, context)

        if not result:
            return {"intent": "none", "confidence": 1.0, "params": {}, "direct_answer": ""}

        intent     = result.get("intent", "none")
        confidence = result.get("confidence", 0.0)
        params     = result.get("params", {})

        # Low confidence — treat as none
        if confidence < 0.7:
            intent = "none"

        direct_answer = ""

        try:
            if intent == "schedule_lookup":
                direct_answer = _answer_schedule_lookup(params, fronter)

            elif intent == "past_lookup":
                direct_answer = _answer_past_lookup(params, fronter)

            elif intent == "location_query":
                direct_answer = _answer_location_query(params, fronter)

            elif intent == "requirement_check":
                direct_answer = _answer_requirement_check(params, fronter)

        except Exception as e:
            log_error("IntentClassifier", f"answer builder failed for {intent}", exc=e)
            direct_answer = ""

        log_event("IntentClassifier", "CLASSIFIED",
            intent=intent,
            confidence=f"{confidence:.2f}",
        )

        return {
            "intent":        intent,
            "confidence":    confidence,
            "params":        params,
            "direct_answer": direct_answer,
        }


intent_classifier = IntentClassifier()
