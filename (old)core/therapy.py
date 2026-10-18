"""
core/therapy.py
Longitudinal therapy agent.

Not a chatbot. Not a crisis line. A presence that watches.

It reads every session passively, builds a longitudinal picture of each
headmate over time, and intervenes only when it has something worth saying.
It never talks directly to the user through Gizmo uninvited.
It influences, nudges, informs. Gizmo stays the face.

What it does:

PASSIVE OBSERVATION
  Reads extraction output after every session.
  Builds per-headmate behavioral timelines.
  Never interrupts. Just watches and learns.

PATTERN WATCHING
  Watches for things the pattern engine doesn't catch:
    - Multi-session trends (3 low-energy Sundays in a row)
    - Correlation clusters (work stress → specific behavior → withdrawal)
    - Absence signals (she used to mention X daily, hasn't in 2 weeks)
    - Recovery patterns (how long after intense sessions to come back)
    - Cross-headmate signals (system-wide stress vs individual)

NUDGE GENERATION
  When a pattern is solid enough, hands something to Gizmo.
  Not "Kaylee is depressed" — more like:
    "she's been tired for 3 sessions, ask how she's sleeping"
  Gizmo delivers it naturally. Never clinical.

FOLLOW-THROUGH TRACKING
  Logs when Gizmo acts on a nudge.
  Checks if the thing was addressed.
  "Last week you said you'd try X — did that happen?"

WEEKLY REPORT
  Per-headmate, sanitized for external therapist.
  What happened, what patterns are active, what's worth discussing.
  Meaning survives. Raw content doesn't.

Therapy data lives in:
  wellbeing table     — observations, needs, limits, patterns
  pattern_engine      — behavioral patterns with feed/break state
  personality table   — per-headmate change requests and drift notes
  preference_qa table — questions asked, answers received

Nudges are delivered via session_manager queued messages.
Reports are written to wellbeing table with tag therapy_report.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


# ── Nudge thresholds ──────────────────────────────────────────────────────────

MIN_SESSIONS_FOR_TREND    = 3    # need N sessions to call something a trend
MIN_CONFIDENCE_TO_NUDGE   = 0.60 # confidence before sending nudge to Gizmo
NUDGE_COOLDOWN_HOURS      = 48   # don't nudge about same thing twice in N hours
ABSENCE_THRESHOLD_DAYS    = 14   # topic not mentioned in N days = absence signal
LOW_VALENCE_THRESHOLD     = -0.25
HIGH_CHAOS_THRESHOLD      = 0.55
REPORT_INTERVAL_DAYS      = 7


# ── Nudge dataclass ───────────────────────────────────────────────────────────

@dataclass
class Nudge:
    """
    A suggestion for Gizmo to weave into conversation.
    Never delivered directly. Always through Gizmo's voice.
    """
    headmate:    str
    nudge_type:  str            # check_in / follow_up / observation / question
    message:     str            # what Gizmo should say/do — casual, not clinical
    reasoning:   str            # why (internal, not shown to user)
    confidence:  float
    urgency:     str            # low / medium / high
    context:     str            # when to surface it
    expires_at:  float          # don't surface after this
    source_data: dict = field(default_factory=dict)


# ── Therapy agent ─────────────────────────────────────────────────────────────

class TherapyAgent:

    def __init__(self):
        self._running        = False
        self._nudge_log:     dict[str, list] = {}  # headmate → [{nudge, sent_at}]
        self._follow_ups:    dict[str, list] = {}  # headmate → [{topic, promised_at}]
        log_event("TherapyAgent", "INIT")

    # ── Post-session observation ──────────────────────────────────────────────

    async def observe_session(
        self,
        session_id: str,
        headmate:   str,
        llm,
    ) -> None:
        """
        Called after every session closes.
        Reads session data and updates the longitudinal picture.
        Generates nudges if warranted.
        Never raises — fire and forget.
        """
        try:
            await self._observe(session_id, headmate, llm)
        except Exception as e:
            log_error("TherapyAgent", f"observe_session failed: {e}", exc=e)

    async def _observe(
        self,
        session_id: str,
        headmate:   str,
        llm,
    ) -> None:
        from core.store import store

        # Load session data
        session = store.get("sessions", session_id)
        if not session:
            return

        # Recent sessions for trend analysis
        recent_sessions = store.query("sessions",
            headmate=headmate.lower(),
            active=1,
            order_by="opened_at DESC",
            limit=10,
        )

        if len(recent_sessions) < 2:
            return  # not enough history yet

        # Emotion trend across recent sessions
        emotion_trend = await self._analyze_emotion_trend(headmate, recent_sessions)

        # Absence detection
        absences = await self._detect_absences(headmate, recent_sessions)

        # Recovery pattern
        recovery = await self._analyze_recovery(headmate, recent_sessions)

        # Follow-up check — did something we flagged get addressed?
        await self._check_follow_ups(headmate, session, llm)

        # Generate nudges from observations
        nudges = await self._generate_nudges(
            headmate=headmate,
            session=session,
            emotion_trend=emotion_trend,
            absences=absences,
            recovery=recovery,
            llm=llm,
        )

        # Queue nudges through session manager
        for nudge in nudges:
            await self._queue_nudge(nudge)

        log_event("TherapyAgent", "SESSION_OBSERVED",
            session=session_id[:8],
            headmate=headmate,
            nudges=len(nudges),
            emotion_trend=emotion_trend.get("direction", "unknown"),
        )

    # ── Emotion trend analysis ────────────────────────────────────────────────

    async def _analyze_emotion_trend(
        self,
        headmate:        str,
        recent_sessions: list,
    ) -> dict:
        """
        Analyze emotion trend across recent sessions.
        Returns trend direction, magnitude, and notable shifts.
        """
        from core.store import store

        # Get emotion log across these sessions
        session_ids = [s["id"] for s in recent_sessions[:6]]
        all_emotion = []

        for sid in session_ids:
            points = store.query("emotion_log",
                session_id=sid,
                headmate=headmate.lower(),
                active=1,
                limit=20,
            )
            all_emotion.extend(points)

        if len(all_emotion) < 4:
            return {"direction": "insufficient data", "magnitude": 0.0}

        # Sort by time
        all_emotion.sort(key=lambda e: e.get("created_at", 0))

        # Split into recent half vs older half
        mid          = len(all_emotion) // 2
        older        = all_emotion[:mid]
        newer        = all_emotion[mid:]

        older_val    = sum(e.get("valence", 0) for e in older) / len(older)
        newer_val    = sum(e.get("valence", 0) for e in newer) / len(newer)
        older_chaos  = sum(e.get("chaos", 0) for e in older) / len(older)
        newer_chaos  = sum(e.get("chaos", 0) for e in newer) / len(newer)

        val_delta    = newer_val - older_val
        chaos_delta  = newer_chaos - older_chaos

        direction = "stable"
        if val_delta > 0.15:    direction = "improving"
        elif val_delta < -0.15: direction = "declining"

        if newer_chaos > HIGH_CHAOS_THRESHOLD:
            direction = "fragmented" if direction == "stable" else direction + "_fragmented"

        return {
            "direction":    direction,
            "magnitude":    abs(val_delta),
            "valence_avg":  round(newer_val, 2),
            "chaos_avg":    round(newer_chaos, 2),
            "val_delta":    round(val_delta, 2),
            "chaos_delta":  round(chaos_delta, 2),
            "low_valence":  newer_val < LOW_VALENCE_THRESHOLD,
            "high_chaos":   newer_chaos > HIGH_CHAOS_THRESHOLD,
        }

    # ── Absence detection ─────────────────────────────────────────────────────

    async def _detect_absences(
        self,
        headmate:        str,
        recent_sessions: list,
    ) -> list[dict]:
        """
        Detect topics/entities that used to appear regularly but have gone quiet.
        Returns list of absence signals worth noting.
        """
        from core.store import store

        absences = []
        now      = time.time()
        cutoff   = now - (ABSENCE_THRESHOLD_DAYS * 24 * 3600)

        # Topics seen in older sessions but not recent ones
        if len(recent_sessions) < MIN_SESSIONS_FOR_TREND:
            return absences

        older_sessions  = recent_sessions[MIN_SESSIONS_FOR_TREND:]
        newer_sessions  = recent_sessions[:MIN_SESSIONS_FOR_TREND]

        older_topics    = set()
        newer_topics    = set()

        for s in older_sessions:
            for t in (s.get("topics") or []):
                older_topics.add(t)

        for s in newer_sessions:
            for t in (s.get("topics") or []):
                newer_topics.add(t)

        # Topics that disappeared
        vanished = older_topics - newer_topics
        for topic in vanished:
            # Check if it was frequent before
            freq = sum(
                1 for s in older_sessions
                if topic in (s.get("topics") or [])
            )
            if freq >= 2:
                absences.append({
                    "type":    "topic_absence",
                    "topic":   topic,
                    "freq":    freq,
                    "last_seen": older_sessions[-1].get("opened_at", 0),
                    "note":    f"{topic} came up {freq}x before, hasn't in {MIN_SESSIONS_FOR_TREND}+ sessions",
                })

        return absences[:3]  # don't overwhelm

    # ── Recovery pattern analysis ─────────────────────────────────────────────

    async def _analyze_recovery(
        self,
        headmate:        str,
        recent_sessions: list,
    ) -> dict:
        """
        Analyze recovery patterns after intense sessions.
        How long does it take to come back? What helps?
        """
        from core.store import store

        if len(recent_sessions) < 3:
            return {}

        # Find intense sessions followed by lower-intensity sessions
        recovery_data = []

        for i in range(len(recent_sessions) - 1):
            current  = recent_sessions[i]
            previous = recent_sessions[i + 1]

            curr_mood = current.get("mood", "")
            prev_mood = previous.get("mood", "")

            # Look for: previous was intense, current is calmer
            intense_words = {"intense", "deep", "heavy", "raw", "charged"}
            calm_words    = {"calm", "light", "easy", "soft", "grounded"}

            prev_intense  = any(w in (prev_mood or "").lower() for w in intense_words)
            curr_calm     = any(w in (curr_mood or "").lower() for w in calm_words)

            if prev_intense and curr_calm:
                time_between = (
                    current.get("opened_at", 0) - previous.get("closed_at", 0)
                ) / 3600  # hours

                recovery_data.append({
                    "recovery_hours": round(time_between, 1),
                    "prev_mood":      prev_mood,
                    "curr_mood":      curr_mood,
                })

        if not recovery_data:
            return {}

        avg_recovery = sum(r["recovery_hours"] for r in recovery_data) / len(recovery_data)

        return {
            "avg_recovery_hours": round(avg_recovery, 1),
            "data_points":        len(recovery_data),
            "note": (
                f"typically returns to baseline in ~{avg_recovery:.0f}h "
                f"after intense sessions"
            ),
        }

    # ── Follow-up tracking ────────────────────────────────────────────────────

    async def _check_follow_ups(
        self,
        headmate: str,
        session:  dict,
        llm,
    ) -> None:
        """
        Check if previously flagged follow-ups were addressed this session.
        Log what was addressed, carry forward what wasn't.
        """
        from core.store import store

        pending = self._follow_ups.get(headmate.lower(), [])
        if not pending:
            return

        session_summary = session.get("summary", "") or ""
        session_topics  = session.get("topics", []) or []

        still_pending = []
        for item in pending:
            topic      = item.get("topic", "")
            promised   = item.get("promised_at", 0)
            days_ago   = (time.time() - promised) / 86400

            # Check if addressed — topic appeared in session
            addressed = (
                topic.lower() in session_summary.lower() or
                topic.lower() in [t.lower() for t in session_topics]
            )

            if addressed:
                log_event("TherapyAgent", "FOLLOW_UP_ADDRESSED",
                    headmate=headmate,
                    topic=topic,
                    days_ago=round(days_ago, 1),
                )
            elif days_ago > 14:
                # Expired — don't keep chasing
                log_event("TherapyAgent", "FOLLOW_UP_EXPIRED",
                    headmate=headmate,
                    topic=topic,
                )
            else:
                still_pending.append(item)

        self._follow_ups[headmate.lower()] = still_pending

    # ── Nudge generation ──────────────────────────────────────────────────────

    async def _generate_nudges(
        self,
        headmate:     str,
        session:      dict,
        emotion_trend: dict,
        absences:     list,
        recovery:     dict,
        llm,
    ) -> list[Nudge]:
        """
        Given observations, decide if anything is worth nudging Gizmo about.
        One nudge maximum per session. Quality over quantity.
        """
        from core.store import store

        nudges   = []
        now      = time.time()

        # ── Declining trend nudge ─────────────────────────────────────────────
        if (emotion_trend.get("declining") or emotion_trend.get("low_valence")):
            if not self._recently_nudged(headmate, "declining_trend"):
                sessions_declining = [
                    s for s in store.query("sessions",
                        headmate=headmate.lower(),
                        active=1,
                        order_by="opened_at DESC",
                        limit=MIN_SESSIONS_FOR_TREND,
                    )
                    if s.get("mood", "") in ("heavy", "low", "sad", "tired", "quiet")
                ]

                if len(sessions_declining) >= MIN_SESSIONS_FOR_TREND:
                    nudge = await self._craft_nudge(
                        headmate=headmate,
                        nudge_type="check_in",
                        observation=(
                            f"energy has been low for {len(sessions_declining)} sessions, "
                            f"valence trend: {emotion_trend.get('val_delta', 0):+.2f}"
                        ),
                        suggested_angle="how she's been doing, sleep, energy",
                        urgency="medium",
                        llm=llm,
                    )
                    if nudge:
                        nudges.append(nudge)

        # ── High chaos nudge ──────────────────────────────────────────────────
        if emotion_trend.get("high_chaos") and not self._recently_nudged(headmate, "high_chaos"):
            nudge = await self._craft_nudge(
                headmate=headmate,
                nudge_type="check_in",
                observation=(
                    f"chaos elevated across sessions "
                    f"(avg={emotion_trend.get('chaos_avg', 0):.2f})"
                ),
                suggested_angle="what's been on her mind, feels scattered",
                urgency="medium",
                llm=llm,
            )
            if nudge:
                nudges.append(nudge)

        # ── Absence nudge ─────────────────────────────────────────────────────
        for absence in absences[:1]:  # max one absence nudge
            if not self._recently_nudged(headmate, f"absence_{absence['topic']}"):
                nudge = await self._craft_nudge(
                    headmate=headmate,
                    nudge_type="observation",
                    observation=absence["note"],
                    suggested_angle=f"how {absence['topic']} has been going",
                    urgency="low",
                    llm=llm,
                )
                if nudge:
                    nudges.append(nudge)

        # ── Follow-up nudge ───────────────────────────────────────────────────
        unresolved = session.get("unresolved")
        if unresolved and not self._recently_nudged(headmate, "unresolved"):
            nudge = await self._craft_nudge(
                headmate=headmate,
                nudge_type="follow_up",
                observation=f"last session left unresolved: {unresolved[:100]}",
                suggested_angle="check if that got resolved",
                urgency="low",
                llm=llm,
            )
            if nudge:
                nudges.append(nudge)
                # Track for follow-up
                self._follow_ups.setdefault(headmate.lower(), []).append({
                    "topic":       unresolved[:60],
                    "promised_at": now,
                })

        # Return highest urgency nudge only — never overwhelm
        if not nudges:
            return []

        urgency_order = {"high": 0, "medium": 1, "low": 2}
        nudges.sort(key=lambda n: urgency_order.get(n.urgency, 3))
        return nudges[:1]

    async def _craft_nudge(
        self,
        headmate:        str,
        nudge_type:      str,
        observation:     str,
        suggested_angle: str,
        urgency:         str,
        llm,
    ) -> Optional[Nudge]:
        """
        Ask the LLM to craft a natural nudge for Gizmo to deliver.
        Never clinical. Never direct. Woven into conversation.
        """
        from core.store import store

        # Load headmate personality context
        personality = store.get_personality(
            headmate=headmate.lower(),
            aspect="with_headmate",
        )
        persona_text = "\n".join(
            r.get("text", "") for r in personality[:2]
        ) or ""

        raw = await _call_llm(llm,
            system=(
                "You craft subtle nudges for Gizmo to weave into conversation. "
                "Never clinical. Never direct. One sentence. Natural. "
                "Gizmo delivers these in his own voice. JSON only."
            ),
            user=f"""Headmate: {headmate}
Observation: {observation}
Suggested angle: {suggested_angle}
Nudge type: {nudge_type}
Gizmo's relationship with this person:
{persona_text or '(not established yet)'}

Write a nudge Gizmo can weave naturally into the next conversation.
Not a question he has to ask. A thing he might notice or bring up.
One sentence. Casual. In character.

Return JSON:
{{
  "nudge": "the thing Gizmo says/does — one sentence, natural",
  "context": "when to surface this — casual moment / after she mentions work / etc",
  "confidence": 0.0-1.0
}}""",
            tokens=200,
            temp=0.4,
        )

        data = _parse_json_safe(raw, {"nudge": "", "confidence": 0.0})
        if not data.get("nudge"):
            return None

        confidence = data.get("confidence", 0.5)
        if confidence < MIN_CONFIDENCE_TO_NUDGE:
            return None

        nudge = Nudge(
            headmate=headmate,
            nudge_type=nudge_type,
            message=data["nudge"],
            reasoning=observation,
            confidence=confidence,
            urgency=urgency,
            context=data.get("context", "natural moment"),
            expires_at=time.time() + (72 * 3600),  # expires in 3 days
        )

        # Log nudge
        self._log_nudge(headmate, nudge_type, observation)

        log_event("TherapyAgent", "NUDGE_CRAFTED",
            headmate=headmate,
            type=nudge_type,
            urgency=urgency,
            confidence=confidence,
            preview=data["nudge"][:60],
        )

        return nudge

    # ── Nudge delivery ────────────────────────────────────────────────────────

    async def _queue_nudge(self, nudge: Nudge) -> None:
        """
        Queue nudge through session manager for natural delivery.
        Gizmo surfaces it when the moment is right.
        """
        try:
            from core.session_manager import session_manager
            from core.store import store

            # Find active session for this headmate
            sessions = list(session_manager._sessions.values())
            target_session = None
            for sess in sessions:
                if sess.headmate and sess.headmate.lower() == nudge.headmate.lower():
                    target_session = sess
                    break

            if target_session:
                # Session is active — add to pending queue
                target_session.pending_questions.append({
                    "type":     "therapy_nudge",
                    "message":  nudge.message,
                    "context":  nudge.context,
                    "urgency":  nudge.urgency,
                    "expires":  nudge.expires_at,
                })
            else:
                # No active session — write to store for next session preload
                store.write("wellbeing", {
                    "headmate":    nudge.headmate.lower(),
                    "category":    "therapy_nudge",
                    "observation": nudge.message,
                    "context":     nudge.context,
                    "register":    "therapy",
                    "confidence":  nudge.confidence,
                    "source":      "therapy_agent",
                    "tags":        f"nudge,{nudge.headmate.lower()},{nudge.nudge_type}",
                })

            log_event("TherapyAgent", "NUDGE_QUEUED",
                headmate=nudge.headmate,
                type=nudge.nudge_type,
            )

        except Exception as e:
            log_error("TherapyAgent", "nudge queue failed", exc=e)

    # ── Nudge cooldown ────────────────────────────────────────────────────────

    def _recently_nudged(self, headmate: str, nudge_key: str) -> bool:
        """Check if we recently sent a nudge of this type for this headmate."""
        log = self._nudge_log.get(headmate.lower(), [])
        cutoff = time.time() - (NUDGE_COOLDOWN_HOURS * 3600)
        for entry in log:
            if entry.get("key") == nudge_key and entry.get("sent_at", 0) > cutoff:
                return True
        return False

    def _log_nudge(self, headmate: str, nudge_type: str, observation: str) -> None:
        key = headmate.lower()
        self._nudge_log.setdefault(key, []).append({
            "key":         nudge_type,
            "observation": observation[:80],
            "sent_at":     time.time(),
        })
        # Trim log to last 20 entries
        self._nudge_log[key] = self._nudge_log[key][-20:]

    # ── Weekly report ─────────────────────────────────────────────────────────

    async def generate_weekly_report(
        self,
        headmate: str,
        llm,
    ) -> Optional[str]:
        """
        Generate a weekly therapy report for a headmate.
        Sanitized for external sharing — meaning survives, raw content doesn't.
        Returns the report text, also written to store.
        """
        from core.store import store

        week_ago = time.time() - (REPORT_INTERVAL_DAYS * 24 * 3600)

        # Sessions this week
        all_sessions = store.query("sessions",
            headmate=headmate.lower(),
            active=1,
            order_by="opened_at DESC",
            limit=20,
        )
        week_sessions = [
            s for s in all_sessions
            if s.get("opened_at", 0) >= week_ago
        ]

        if not week_sessions:
            return None

        # Emotion trend
        emotion_trend = await self._analyze_emotion_trend(headmate, week_sessions)

        # Active patterns
        patterns = store.get_patterns(headmate=headmate)
        flagged_patterns = [p for p in patterns if p.get("therapy_flag")]
        active_feed      = [p for p in patterns if p.get("action") == "feed"]
        active_break     = [p for p in patterns if p.get("action") == "break"]

        # Wellbeing observations this week
        wb_all = store.query("wellbeing",
            headmate=headmate.lower(),
            active=1,
            order_by="created_at DESC",
            limit=30,
        )
        wb_week = [w for w in wb_all if w.get("created_at", 0) >= week_ago]

        # Group wellbeing by category
        wb_by_category: dict = {}
        for w in wb_week:
            cat = w.get("category", "unknown")
            wb_by_category.setdefault(cat, []).append(w.get("observation", ""))

        # Format for LLM
        sessions_text = "\n".join(
            f"- [{s.get('mood','?')}] {s.get('summary','')[:100]}"
            for s in week_sessions[:5]
        )

        pattern_text = ""
        if flagged_patterns:
            pattern_text = "Flagged patterns:\n" + "\n".join(
                f"- {p.get('pattern_type','?')}: {p.get('approach','')[:80]}"
                for p in flagged_patterns[:3]
            )

        wb_text = "\n".join(
            f"[{cat}] " + "; ".join(obs[:60] for obs in obs_list[:2])
            for cat, obs_list in wb_by_category.items()
            if cat not in ("therapy_report", "therapy_nudge")
        )

        raw = await _call_llm(llm,
            system=(
                "You write weekly mental health summaries for therapists. "
                "Clinical, specific, non-judgmental. "
                "Sanitize intimate content — preserve clinical meaning. "
                "Write as if briefing a therapist who knows their patient. "
                "No identifying info beyond first name."
            ),
            user=f"""Patient: {headmate.title()}
Week of: {tz_now().strftime('%Y-%m-%d')}
Sessions this week: {len(week_sessions)}

Emotional trend: {emotion_trend.get('direction','unknown')} 
  (valence avg={emotion_trend.get('valence_avg',0):+.2f}, 
   chaos avg={emotion_trend.get('chaos_avg',0):.2f})

Session summaries:
{sessions_text or '(none)'}

{pattern_text}

Wellbeing observations:
{wb_text or '(none on file)'}

Write a 4-6 sentence weekly summary suitable for therapist review.
Cover: emotional state, behavioral patterns, needs, anything worth discussing.
Sanitize intimate content — the therapist needs clinical context, not explicit detail.
Note anything that has improved or worsened from prior weeks if visible.""",
            tokens=400,
            temp=0.3,
        )

        if not raw or not raw.strip():
            return None

        report = raw.strip()

        # Write to store
        store.write("wellbeing", {
            "headmate":    headmate.lower(),
            "category":    "therapy_report",
            "observation": report,
            "context":     f"Week of {tz_now().strftime('%Y-%m-%d')}",
            "register":    "clinical",
            "source":      "therapy_agent",
            "confidence":  0.9,
            "tags":        f"therapy_report,{headmate.lower()},weekly",
        })

        log_event("TherapyAgent", "WEEKLY_REPORT_WRITTEN",
            headmate=headmate,
            sessions=len(week_sessions),
            preview=report[:80],
        )

        return report

    # ── Real-time session monitoring ──────────────────────────────────────────

    async def monitor_session(
        self,
        session_id: str,
        headmate:   str,
        brief_data: dict,
        llm,
    ) -> Optional[str]:
        """
        Called during active sessions for real-time monitoring.
        Returns a therapy note for the director if warranted.
        Never blocks. Returns None if nothing to flag.

        Watches for:
          - Acute distress signals
          - Intensity exceeding known edge
          - Pattern combinations that correlate with bad outcomes
        """
        from core.store import store

        register     = brief_data.get("register", "neutral")
        valence      = brief_data.get("valence", 0.0)
        stress       = brief_data.get("stress_level", "none")
        intensity    = brief_data.get("intensity", 0.3)

        notes = []

        # Acute distress
        if register in ("crisis", "distress") or (valence < -0.6 and stress == "crisis"):
            notes.append("acute distress signal — prioritize grounding over engagement")

        # Edge approaching
        patterns = store.get_patterns(headmate=headmate, action="feed")
        for p in patterns:
            edge = p.get("edge_intensity")
            if edge and intensity >= edge - 0.1:
                notes.append(
                    f"approaching edge for {p.get('pattern_type','?')} pattern "
                    f"(edge={edge:.1f}, current={intensity:.1f}) — "
                    f"watch for: {', '.join((p.get('watch_for') or [])[:2])}"
                )

        # Check limits
        limits = store.get_wellbeing(headmate, category="limit", limit=10)
        for limit in limits:
            obs = limit.get("observation", "").lower()
            msg = brief_data.get("message", "").lower()
            # Simple keyword overlap — not perfect but fast
            limit_words = set(obs.split()[:5])
            if len(limit_words & set(msg.split())) >= 2:
                notes.append(f"limit proximity: {obs[:60]}")

        if not notes:
            return None

        return " | ".join(notes)

    # ── Background loop ───────────────────────────────────────────────────────

    async def start(self, llm) -> None:
        """Start the weekly report background loop."""
        self._running = True
        log_event("TherapyAgent", "STARTED")

        while self._running:
            # Run weekly reports every 7 days
            await asyncio.sleep(REPORT_INTERVAL_DAYS * 24 * 3600)
            try:
                from core.store import store
                # Get all known headmates
                entities = store.query("entities",
                    entity_type="headmate",
                    active=1,
                    limit=50,
                )
                for entity in entities:
                    name = entity.get("name", "")
                    if name:
                        await self.generate_weekly_report(name, llm)
            except Exception as e:
                log_error("TherapyAgent", "weekly report loop failed", exc=e)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _call_llm(
    llm,
    system: str,
    user:   str,
    tokens: int   = 300,
    temp:   float = 0.2,
) -> str:
    try:
        return await llm.generate(
            [{"role": "user", "content": user}],
            system_prompt=system,
            max_new_tokens=tokens,
            temperature=temp,
        )
    except Exception as e:
        log_error("TherapyAgent", f"LLM call failed: {e}", exc=None)
        return ""


def _parse_json_safe(raw: str, fallback: dict) -> dict:
    if not raw:
        return fallback
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception:
        return fallback


# ── Singleton ─────────────────────────────────────────────────────────────────

therapy_agent = TherapyAgent()
