"""
core/psych_processor.py
Daily psychological batch processor.

Runs once per day per headmate who had activity.
Scheduled via APScheduler — fires at low-traffic hours (default 3am local).
Never blocks the main agent flow.

Pipeline stages:

  Stage 1 — Data gathering
    Pull current session events (first/last timestamp, bounds)
    Pull last 4 conversations for this headmate
    Sequential chunk scan (adjacent pairs) for similarities
    Random cross-scan (non-adjacent pairs) for recurring patterns
    Tag similarities with references
    Extract temporal and spatial context

  Stage 2 — Reasoning pass (DeepSeek V4 Flash)
    Part 1: Contextual summary — life context, exterior world
    Part 2: Clinical notes — observed behavior, named, referenced, non-interpretive
    Part 3: The story — first-person witness narrative, names named, gaps named

  Stage 3 — Behavioral analysis (DeepSeek V4 Flash)
    Baseline comparison against psych profile
    Aberration assessment — how much, what direction, what it suggests
    Or normalcy confirmation — what it confirms, whether stability is itself notable
    What to watch for next
    All referenced against specific prior sessions

  Output:
    Daily observation appended to headmate psych profile
    Noteworthy flags appended inline
    LoRA training data record written to training queue

Psych profile location:
    /data/headmates/{name}/psych_profile.md

LoRA training queue:
    /data/lora_queue/{name}/{YYYY-MM-DD}.jsonl

Usage:
    from core.psych_processor import psych_processor

    # Called by scheduler
    await psych_processor.run_for(headmate="jess")

    # Called at server boot to register with APScheduler
    psych_processor.schedule(scheduler)
"""

import json
import os
import random
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR          = Path("/data")
HEADMATES_DIR     = DATA_DIR / "headmates"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
LORA_QUEUE_DIR    = DATA_DIR / "lora_queue"
ACTION_LOG_DB     = DATA_DIR / "action_log.db"
PEOPLE_DB         = DATA_DIR / "people.db"

PROCESSOR_MODEL   = "deepseek/deepseek-v4-flash"
MAX_TOKENS        = 2000
TEMPERATURE       = 0.3   # low — this is analysis not generation

# Similarity scan config
SEQUENTIAL_CHUNKS = 2     # scan adjacent pairs
RANDOM_CROSS_SCANS = 2    # random non-adjacent pairs

# LoRA confidence threshold — only include high-confidence entries
LORA_MIN_CONFIDENCE = 0.7


# ── Path helpers ──────────────────────────────────────────────────────────────

def _psych_profile_path(headmate: str) -> Path:
    path = HEADMATES_DIR / headmate.lower() / "psych_profile.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _lora_queue_path(headmate: str) -> Path:
    date_str = tz_now().strftime("%Y-%m-%d")
    path = LORA_QUEUE_DIR / headmate.lower() / f"{date_str}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _actions_db_path() -> Path:
    return ACTION_LOG_DB


# ── Stage 1: Data gathering ───────────────────────────────────────────────────

def _get_session_events(
    headmate: str,
    session_file: str,
) -> list[dict]:
    """Pull all action log events for a specific session."""
    try:
        with sqlite3.connect(_actions_db_path()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM action_log "
                "WHERE subject = ? AND session_file = ? "
                "ORDER BY sequence",
                (headmate.lower(), session_file)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        log_error("PsychProcessor", "get_session_events failed", exc=e)
        return []


def _get_recent_conversations(headmate: str, limit: int = 4) -> list[dict]:
    """
    Pull the last N conversation summaries for a headmate from the archive index.
    Returns list of summary dicts ordered newest first.
    """
    results = []
    today = tz_now().date()

    # Scan back up to 30 days
    for days_back in range(0, 30):
        if len(results) >= limit:
            break

        date = today - timedelta(days=days_back)
        date_str = date.strftime("%Y-%m-%d")
        index_path = CONVERSATIONS_DIR / date_str / "index.json"

        if not index_path.exists():
            continue

        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for session in reversed(index.get("sessions", [])):
                if headmate.lower() in [h.lower() for h in session.get("hosts", [])]:
                    results.append({
                        "date":         date_str,
                        "session_id":   session.get("session_id"),
                        "filename":     session.get("filename"),
                        "summary":      session.get("summary", ""),
                        "mood":         session.get("mood", "unknown"),
                        "notable":      session.get("notable", []),
                        "opened_at":    session.get("opened_at", ""),
                        "closed_at":    session.get("closed_at", ""),
                        "topics":       session.get("topics", []),
                    })
                    if len(results) >= limit:
                        break
        except Exception as e:
            log_error("PsychProcessor", f"failed to read index {date_str}", exc=e)

    return results


def _get_session_action_summary(headmate: str, session_file: str) -> str:
    """
    Get a compact text summary of action log events for a session.
    Used as input for similarity scanning.
    """
    events = _get_session_events(headmate, session_file)
    if not events:
        return ""

    lines = []
    for ev in events:
        line = f"[{ev.get('timestamp', '')[:16]}] {ev.get('subject', '')} — {ev.get('action', '')}"
        if ev.get("intent"):
            line += f" (intent: {ev['intent']})"
        if ev.get("verbatim"):
            line += f' | said: "{ev["verbatim"]}"'
        lines.append(line)

    return "\n".join(lines)


def _scan_pair_for_similarities(
    summary_a: str,
    summary_b: str,
    label_a:   str,
    label_b:   str,
) -> list[dict]:
    """
    Heuristic similarity scan between two session summaries.
    Looks for recurring topics, actions, emotional registers.
    Returns list of similarity tags with source references.
    Fast, no LLM — keyword and pattern matching.
    """
    similarities = []

    # Extract topic words from both summaries
    words_a = set(summary_a.lower().split())
    words_b = set(summary_b.lower().split())
    shared  = words_a & words_b

    # Filter to meaningful words (longer than 4 chars, not stopwords)
    _STOP = {
        "that", "this", "with", "from", "they", "them", "their",
        "have", "been", "were", "said", "what", "when", "then",
        "just", "like", "into", "your", "will", "also", "more",
        "about", "which", "after", "before", "would", "could",
        "should", "gizmo",
    }
    meaningful = {w for w in shared if len(w) > 4 and w not in _STOP}

    if meaningful:
        similarities.append({
            "type":       "shared_topics",
            "tags":       list(meaningful)[:8],
            "source_a":   label_a,
            "source_b":   label_b,
            "confidence": min(0.9, 0.4 + len(meaningful) * 0.05),
        })

    # Check for recurring emotional register words
    _REGISTER_WORDS = {
        "tired", "exhausted", "happy", "anxious", "stressed",
        "frustrated", "calm", "distressed", "playful", "subdued",
        "energetic", "sad", "angry", "relieved", "excited",
    }
    shared_register = meaningful & _REGISTER_WORDS
    if shared_register:
        similarities.append({
            "type":       "recurring_register",
            "tags":       list(shared_register),
            "source_a":   label_a,
            "source_b":   label_b,
            "confidence": 0.7,
        })

    return similarities


def _gather_stage1(headmate: str, session_file: str) -> dict:
    """
    Execute stage 1 data gathering.
    Returns structured data bundle for stage 2.
    """
    # Current session events
    events = _get_session_events(headmate, session_file)
    first_event = events[0]  if events else None
    last_event  = events[-1] if events else None

    first_ts = first_event.get("timestamp") if first_event else None
    last_ts  = last_event.get("timestamp")  if last_event  else None

    # Current session action summary for scanning
    current_summary = _get_session_action_summary(headmate, session_file)

    # Last 4 conversations
    recent_convs = _get_recent_conversations(headmate, limit=4)

    # Build per-conversation summaries for scanning
    conv_summaries = []
    for conv in recent_convs:
        text = conv.get("summary", "")
        if not text:
            # Try to load action events for that session
            text = _get_session_action_summary(
                headmate, conv.get("filename", "")
            )
        conv_summaries.append({
            "label":   f"{conv['date']} {conv.get('opened_at', '')[:10]}",
            "date":    conv["date"],
            "text":    text,
            "mood":    conv.get("mood", "unknown"),
            "notable": conv.get("notable", []),
            "topics":  conv.get("topics", []),
        })

    # Sequential chunk scanning — adjacent pairs
    sequential_similarities = []
    all_summaries = [{"label": "current", "text": current_summary}] + conv_summaries

    for i in range(len(all_summaries) - 1):
        a = all_summaries[i]
        b = all_summaries[i + 1]
        if a["text"] and b["text"]:
            sims = _scan_pair_for_similarities(
                a["text"], b["text"], a["label"], b["label"]
            )
            sequential_similarities.extend(sims)

    # Random cross-scan — non-adjacent pairs
    random_similarities = []
    if len(all_summaries) >= 3:
        indices = list(range(len(all_summaries)))
        for _ in range(RANDOM_CROSS_SCANS):
            i, j = random.sample(indices, 2)
            if abs(i - j) > 1:  # non-adjacent only
                a = all_summaries[i]
                b = all_summaries[j]
                if a["text"] and b["text"]:
                    sims = _scan_pair_for_similarities(
                        a["text"], b["text"], a["label"], b["label"]
                    )
                    random_similarities.extend(sims)

    # Temporal context
    time_of_day = None
    day_of_week = None
    if first_ts:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(first_ts)
            hour = dt.hour
            if 5  <= hour < 12: time_of_day = "morning"
            elif 12 <= hour < 17: time_of_day = "afternoon"
            elif 17 <= hour < 21: time_of_day = "evening"
            else:                  time_of_day = "night"
            day_of_week = dt.strftime("%A")
        except Exception:
            pass

    return {
        "headmate":               headmate,
        "session_file":           session_file,
        "events":                 events,
        "first_timestamp":        first_ts,
        "last_timestamp":         last_ts,
        "event_count":            len(events),
        "current_summary":        current_summary,
        "recent_conversations":   recent_convs,
        "conv_summaries":         conv_summaries,
        "sequential_similarities": sequential_similarities,
        "random_similarities":    random_similarities,
        "time_of_day":            time_of_day,
        "day_of_week":            day_of_week,
    }


# ── Stage 2: Reasoning pass ───────────────────────────────────────────────────

def _build_stage2_prompt(data: dict, psych_profile: str) -> str:
    """Build the stage 2 prompt for the reasoning model."""

    # Format events for the prompt
    events_text = "\n".join([
        f"  [{e.get('timestamp', '')[:16]}] "
        f"{e.get('subject', '').title()} → {e.get('action', '')} "
        f"(intent: {e.get('intent') or 'unknown'})"
        + (f' | "{e.get("verbatim")}"' if e.get("verbatim") else "")
        for e in data["events"][:40]  # cap at 40 events for token budget
    ])

    # Format recent conversation summaries
    recent_text = ""
    for conv in data["conv_summaries"][:4]:
        recent_text += (
            f"\n  [{conv['date']}] mood={conv['mood']}\n"
            f"  {conv['text'][:300]}\n"
        )

    # Format similarity tags
    all_sims = data["sequential_similarities"] + data["random_similarities"]
    sims_text = ""
    if all_sims:
        for sim in all_sims[:6]:
            sims_text += (
                f"\n  {sim['type']}: {', '.join(sim['tags'][:5])} "
                f"(between {sim['source_a']} and {sim['source_b']})"
            )

    profile_excerpt = psych_profile[-2000:] if len(psych_profile) > 2000 else psych_profile

    return f"""You are analyzing a day of interactions for {data['headmate'].title()}.

CURRENT SESSION EVENTS ({data['event_count']} total):
Session file: {data['session_file']}
Time: {data['day_of_week']} {data['time_of_day']}
First event: {data['first_timestamp']}
Last event:  {data['last_timestamp']}

{events_text}

RECENT CONVERSATION HISTORY (last 4):
{recent_text}

PATTERN SIMILARITIES DETECTED:
{sims_text if sims_text else "  None detected."}

CURRENT PSYCH PROFILE:
{profile_excerpt if profile_excerpt else "  No profile yet — first observation."}

---

Produce three parts. Be thorough. Name names. Never invent. Mark gaps as gaps.

PART 1 — CONTEXTUAL SUMMARY
What was happening in {data['headmate'].title()}'s life today? Not what was said —
what was the exterior context. Work, home, commute, shopping, social situation.
What do the timing, topics, and events suggest about where they were and what
their day looked like? Reference specific events and timestamps.

PART 2 — CLINICAL NOTES
Observed behavioral notes from today. Precise, referenced, non-interpretive.
Name everyone involved. Note times. Flag anything that stands out.
Do not interpret yet — just observe accurately.

PART 3 — THE STORY AS I UNDERSTOOD IT
First person. What did you witness today? Assemble the fragments honestly.
Name everyone you can name. Mark what you couldn't hear clearly.
Mark what you're inferring vs what you directly observed.
Write it the way a careful witness would recount their day.
"""


def _build_stage3_prompt(
    data:          dict,
    stage2_output: str,
    psych_profile: str,
) -> str:
    """Build the stage 3 behavioral analysis prompt."""

    profile_excerpt = psych_profile[-2000:] if len(psych_profile) > 2000 else psych_profile

    return f"""You are now analyzing {data['headmate'].title()}'s behavior today
against their established psychological profile.

TODAY'S OBSERVATIONS:
{stage2_output}

PSYCH PROFILE:
{profile_excerpt if profile_excerpt else "No profile yet — this is the first observation."}

PATTERN SIMILARITIES FROM HISTORY:
{json.dumps(data['sequential_similarities'] + data['random_similarities'], indent=2)[:800]}

---

Produce a behavioral analysis. Be clinical, precise, and referenced.
Cite specific prior sessions when comparing to baseline.

BASELINE COMPARISON
Is today's behavior within normal range for {data['headmate'].title()}?
Reference the profile specifically — what does or doesn't match.

IF ABNORMAL:
- How significant is the deviation? (slight drift / notable / significant break)
- In what direction? More of what, less of what, or something new?
- What does this suggest? Give weighted hypotheses, not conclusions.
- What specifically should be watched for next?

IF NORMAL:
- What does today confirm in the profile?
- Is the consistency itself notable given any known stressors or context?
- What does their current state suggest about how they're doing?

WHAT TO WATCH FOR NEXT
Always end here. One to three specific things worth observing in coming sessions.
Referenced and reasoned — not generic.
"""


async def _call_reasoning_model(prompt: str) -> str:
    """Call DeepSeek V4 Flash for stage 2 and 3 reasoning passes."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=60,
        )

        response = await client.chat.completions.create(
            model=PROCESSOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        raw = response.choices[0].message.content or ""
        if not raw.strip():
            log_event("PsychProcessor", "EMPTY_RESPONSE", model=PROCESSOR_MODEL)
            return ""

        return raw.strip()

    except Exception as e:
        log_error("PsychProcessor", "reasoning model call failed", exc=e)
        return ""


# ── Psych profile read/write ──────────────────────────────────────────────────

def _read_psych_profile(headmate: str) -> str:
    """Read existing psych profile or return empty string."""
    path = _psych_profile_path(headmate)
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception as e:
        log_error("PsychProcessor", "failed to read psych profile", exc=e)
        return ""


def _append_daily_observation(
    headmate:       str,
    date_str:       str,
    stage2_output:  str,
    stage3_output:  str,
    data:           dict,
) -> None:
    """
    Append today's daily observation to the headmate's psych profile.
    Newest at top — prepended after the static header.
    """
    path = _psych_profile_path(headmate)

    observation = f"""
---

## {date_str} — Daily Observation
**Session:** {data['session_file']}
**Time:** {data['day_of_week']} {data['time_of_day']}
**Events logged:** {data['event_count']}
**First:** {data['first_timestamp']}
**Last:** {data['last_timestamp']}

{stage2_output}

### Behavioral Analysis

{stage3_output}

"""

    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""

        # Find insertion point — after header, before previous observations
        # Header ends at first "---" separator
        if "---" in existing:
            header_end = existing.index("---")
            header = existing[:header_end + 3]
            body   = existing[header_end + 3:]
            new_content = header + observation + body
        else:
            # No header yet — observation becomes the file
            new_content = f"# {headmate.title()} — Psychological Profile\n\n" + observation

        path.write_text(new_content, encoding="utf-8")

        log_event("PsychProcessor", "PROFILE_UPDATED",
            headmate=headmate,
            date=date_str,
            events=data["event_count"],
        )

    except Exception as e:
        log_error("PsychProcessor", "failed to append daily observation", exc=e)


# ── LoRA training queue ───────────────────────────────────────────────────────

def _write_lora_record(
    headmate:      str,
    data:          dict,
    stage2_output: str,
    stage3_output: str,
    confidence:    float,
) -> None:
    """
    Write a curated training record to the LoRA queue if confidence is high enough.
    Format: {input, output, metadata} — one record per JSONL line.
    """
    if confidence < LORA_MIN_CONFIDENCE:
        return

    try:
        record = {
            "input": {
                "headmate":      headmate,
                "session_file":  data["session_file"],
                "event_count":   data["event_count"],
                "time_of_day":   data["time_of_day"],
                "day_of_week":   data["day_of_week"],
                "events_sample": [
                    {
                        "action": e.get("action"),
                        "intent": e.get("intent"),
                        "action_type": e.get("action_type"),
                    }
                    for e in data["events"][:20]
                ],
                "similarities":  (
                    data["sequential_similarities"] +
                    data["random_similarities"]
                )[:5],
            },
            "output": {
                "stage2": stage2_output,
                "stage3": stage3_output,
            },
            "metadata": {
                "date":       tz_now().strftime("%Y-%m-%d"),
                "timestamp":  tz_now().isoformat(),
                "confidence": confidence,
                "headmate":   headmate,
            }
        }

        path = _lora_queue_path(headmate)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        log_event("PsychProcessor", "LORA_RECORD_WRITTEN",
            headmate=headmate,
            confidence=confidence,
        )

    except Exception as e:
        log_error("PsychProcessor", "failed to write LoRA record", exc=e)


# ── Confidence scoring ────────────────────────────────────────────────────────

def _score_confidence(data: dict, stage2_output: str, stage3_output: str) -> float:
    """
    Heuristic confidence score for the daily observation.
    Used to gate LoRA training queue inclusion.
    Higher when: more events, more historical context, longer outputs.
    """
    score = 0.3  # base

    # More events = more signal
    if data["event_count"] >= 10: score += 0.2
    elif data["event_count"] >= 5: score += 0.1

    # Historical context available
    if len(data["recent_conversations"]) >= 3: score += 0.15
    elif len(data["recent_conversations"]) >= 1: score += 0.05

    # Similarities found = pattern richness
    all_sims = data["sequential_similarities"] + data["random_similarities"]
    if len(all_sims) >= 3: score += 0.15
    elif len(all_sims) >= 1: score += 0.05

    # Output quality — longer means more was found
    if len(stage2_output) > 800: score += 0.1
    if len(stage3_output) > 500: score += 0.1

    return min(1.0, score)


# ── Main processor ────────────────────────────────────────────────────────────

class PsychProcessor:
    """
    Singleton. Daily psychological batch processor.
    """

    async def run_for(
        self,
        headmate:     str,
        session_file: Optional[str] = None,
    ) -> None:
        """
        Run the full pipeline for one headmate.
        If session_file not provided, uses the most recent session from today.
        """
        headmate = headmate.lower()
        date_str = tz_now().strftime("%Y-%m-%d")

        log_event("PsychProcessor", "STARTING",
            headmate=headmate,
            date=date_str,
        )

        # Resolve session file if not provided
        if not session_file:
            session_file = self._get_todays_session_file(headmate)
            if not session_file:
                log_event("PsychProcessor", "NO_SESSION_FILE",
                    headmate=headmate,
                )
                return

        # ── Stage 1 ───────────────────────────────────────────────────────────
        log_event("PsychProcessor", "STAGE1_START", headmate=headmate)
        data = _gather_stage1(headmate, session_file)

        if not data["events"]:
            log_event("PsychProcessor", "NO_EVENTS", headmate=headmate)
            return

        log_event("PsychProcessor", "STAGE1_COMPLETE",
            headmate=headmate,
            events=data["event_count"],
            recent_convs=len(data["recent_conversations"]),
            similarities=len(
                data["sequential_similarities"] +
                data["random_similarities"]
            ),
        )

        # ── Stage 2 ───────────────────────────────────────────────────────────
        log_event("PsychProcessor", "STAGE2_START", headmate=headmate)
        psych_profile = _read_psych_profile(headmate)

        stage2_prompt  = _build_stage2_prompt(data, psych_profile)
        stage2_output  = await _call_reasoning_model(stage2_prompt)

        if not stage2_output:
            log_event("PsychProcessor", "STAGE2_EMPTY", headmate=headmate)
            return

        log_event("PsychProcessor", "STAGE2_COMPLETE",
            headmate=headmate,
            output_length=len(stage2_output),
        )

        # ── Stage 3 ───────────────────────────────────────────────────────────
        log_event("PsychProcessor", "STAGE3_START", headmate=headmate)

        stage3_prompt = _build_stage3_prompt(data, stage2_output, psych_profile)
        stage3_output = await _call_reasoning_model(stage3_prompt)

        if not stage3_output:
            log_event("PsychProcessor", "STAGE3_EMPTY", headmate=headmate)
            # Still write what we have from stage 2
            stage3_output = "Stage 3 analysis unavailable for this session."

        log_event("PsychProcessor", "STAGE3_COMPLETE",
            headmate=headmate,
            output_length=len(stage3_output),
        )

        # ── Write output ──────────────────────────────────────────────────────
        _append_daily_observation(
            headmate=headmate,
            date_str=date_str,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            data=data,
        )

        # ── LoRA queue ────────────────────────────────────────────────────────
        confidence = _score_confidence(data, stage2_output, stage3_output)
        _write_lora_record(
            headmate=headmate,
            data=data,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            confidence=confidence,
        )

        log_event("PsychProcessor", "COMPLETE",
            headmate=headmate,
            date=date_str,
            confidence=confidence,
            lora_queued=confidence >= LORA_MIN_CONFIDENCE,
        )

    async def run_all(self) -> None:
        """
        Run the processor for all headmates who had activity today.
        Called by the scheduler.
        """
        headmates = self._get_active_headmates_today()

        if not headmates:
            log_event("PsychProcessor", "NO_ACTIVITY_TODAY")
            return

        log_event("PsychProcessor", "RUN_ALL_START",
            headmates=headmates,
        )

        for headmate in headmates:
            try:
                await self.run_for(headmate)
            except Exception as e:
                log_error("PsychProcessor",
                    f"run_for failed for {headmate}", exc=e)

        log_event("PsychProcessor", "RUN_ALL_COMPLETE",
            headmates=headmates,
        )

    def schedule(self, scheduler) -> None:
        """
        Register the daily batch job with APScheduler.
        Fires at 3am local time.
        """
        scheduler.add_job(
            self._scheduled_run,
            trigger="cron",
            hour=3,
            minute=0,
            id="psych_batch_daily",
            replace_existing=True,
            misfire_grace_time=3600,  # if server was down, run within 1hr
        )
        log_event("PsychProcessor", "SCHEDULED", hour=3)

    async def _scheduled_run(self) -> None:
        """Wrapper for scheduler — catches all exceptions."""
        try:
            await self.run_all()
        except Exception as e:
            log_error("PsychProcessor", "scheduled run failed", exc=e)

    def _get_todays_session_file(self, headmate: str) -> Optional[str]:
        """Find the most recent session file for a headmate today."""
        date_str   = tz_now().strftime("%Y-%m-%d")
        index_path = CONVERSATIONS_DIR / date_str / "index.json"

        if not index_path.exists():
            return None

        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for session in reversed(index.get("sessions", [])):
                if headmate in [h.lower() for h in session.get("hosts", [])]:
                    return session.get("filename")
        except Exception as e:
            log_error("PsychProcessor", "failed to get today's session", exc=e)

        return None

    def _get_active_headmates_today(self) -> list[str]:
        """Return list of headmates who had activity in today's sessions."""
        date_str   = tz_now().strftime("%Y-%m-%d")
        index_path = CONVERSATIONS_DIR / date_str / "index.json"

        if not index_path.exists():
            return []

        try:
            index    = json.loads(index_path.read_text(encoding="utf-8"))
            headmates = set()
            for session in index.get("sessions", []):
                for host in session.get("hosts", []):
                    headmates.add(host.lower())
            return list(headmates)
        except Exception as e:
            log_error("PsychProcessor", "failed to get active headmates", exc=e)
            return []


# ── Singleton ─────────────────────────────────────────────────────────────────
psych_processor = PsychProcessor()
