"""
core/pattern_engine.py
Behavioral pattern detection, assessment, and refinement.

Watches for meaningful patterns per headmate across sessions.
Builds a living model that gets smarter with every exchange.

Pattern types:
  temporal        — rhythmic: weekday nights, Sunday quiet, post-work states
  trigger_response — X leads to Y: work stress → degradation register
  frequency       — clustering: reassurance spikes, topic fixation
  absence         — something that used to happen has stopped
  escalation      — intensity gradually increasing over time
  recovery        — how long/what helps after an intense session

For each pattern, the engine decides:
  feed    — this is serving her, push harder, test the envelope
  break   — this is looping unproductively, interrupt gently
  hold    — observing, not enough data yet
  flag_therapy — surface to therapy model, needs clinical attention

Feed mode is not just "mirror energy" — it's navigation.
The engine builds an intensity map: what happened at 7, at 8, at 9?
What came after? Push incrementally. Log outcomes. Update the map.

The refinement loop:
  pattern fires
      ↓
  engine acts on recommendation
      ↓
  close_loop logs outcome (outcome_quality on pattern_instance)
      ↓
  post_session_refine() runs at session close
      ↓
  pattern updated: confidence, intensity_map, edge indicators
      ↓
  next recommendation is more informed

Weekly review (background job):
  scans all patterns across past 7 days
  promotes emerging → confirmed
  decays inactive patterns
  finds new correlations
  generates therapy report
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


# ── Pattern detection thresholds ──────────────────────────────────────────────

MIN_DATA_POINTS_TO_ACT   = 3    # minimum instances before recommending feed/break
MIN_CONFIDENCE_TO_ACT    = 0.45 # minimum confidence before acting
MIN_CONFIDENCE_TO_FEED   = 0.55 # higher bar for feed (we're pushing)
MIN_CONFIDENCE_TO_BREAK  = 0.50 # slightly lower for break (safety)
OUTCOME_DECLINE_THRESHOLD = 0.35 # quality below this → consider break
OUTCOME_STRONG_THRESHOLD  = 0.70 # quality above this → confident feed
EDGE_DETECTION_THRESHOLD  = 0.30 # quality drop of this much → edge found
DECAY_DAYS               = 30   # patterns not seen in N days start decaying
WEEKLY_REVIEW_INTERVAL   = 7 * 24 * 3600


# ── Trigger condition builder ─────────────────────────────────────────────────

def _build_trigger_conditions(brief_data: dict) -> list[str]:
    """
    Extract trigger conditions from a brief/session snapshot.
    Used to match future sessions against known patterns.
    """
    conditions = []
    register   = brief_data.get("register", "neutral")
    tod        = brief_data.get("time_of_day", "")
    day        = brief_data.get("day_of_week", "")
    day_type   = brief_data.get("day_type", "")
    topics     = brief_data.get("topics", [])
    stress     = brief_data.get("stress_level", "")

    if register not in ("neutral", "casual"):
        conditions.append(f"register:{register}")
    if tod:
        conditions.append(f"time:{tod}")
    if day_type:
        conditions.append(f"day_type:{day_type}")
    if day:
        conditions.append(f"day:{day}")
    for topic in topics[:3]:
        conditions.append(f"topic:{topic}")
    if stress in ("high", "crisis", "medium"):
        conditions.append(f"stress:{stress}")

    return conditions


def _match_conditions(pattern_conditions: list, current_conditions: list) -> float:
    """
    How well does the current situation match a pattern's triggers?
    Returns 0.0–1.0 match score.
    """
    if not pattern_conditions:
        return 0.0

    matched = sum(1 for c in pattern_conditions if c in current_conditions)
    return matched / len(pattern_conditions)


# ── Intensity map ─────────────────────────────────────────────────────────────

def _build_intensity_map(instances: list[dict]) -> dict:
    """
    Build intensity → outcome quality map from pattern instances.
    Groups instances by intensity bucket (rounded to nearest int).
    Returns: {intensity_bucket: {avg_quality, count, post_patterns}}
    """
    buckets: dict = {}

    for inst in instances:
        intensity = inst.get("intensity_out", inst.get("intensity_in", 0.0))
        quality   = inst.get("outcome_quality", 0.5)
        post      = inst.get("post_pattern", "")
        bucket    = round(intensity * 10)  # 0–10 scale

        if bucket not in buckets:
            buckets[bucket] = {"qualities": [], "post_patterns": [], "count": 0}

        buckets[bucket]["qualities"].append(quality)
        buckets[bucket]["count"] += 1
        if post:
            buckets[bucket]["post_patterns"].append(post)

    # Compute averages
    result = {}
    for bucket, data in buckets.items():
        avg = sum(data["qualities"]) / len(data["qualities"])
        result[str(bucket)] = {
            "outcome_quality_avg": round(avg, 3),
            "count":               data["count"],
            "post_patterns":       list(set(data["post_patterns"]))[:3],
        }

    return result


def _find_optimal_range(intensity_map: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find optimal intensity range and edge from the intensity map.
    Returns (optimal_low, optimal_high, edge)
    """
    if not intensity_map:
        return None, None, None

    # Find peak quality range
    sorted_buckets = sorted(
        intensity_map.items(),
        key=lambda x: x[1]["outcome_quality_avg"],
        reverse=True,
    )

    if not sorted_buckets:
        return None, None, None

    peak_bucket = int(sorted_buckets[0][0])
    peak_quality = sorted_buckets[0][1]["outcome_quality_avg"]

    # Optimal range: buckets within 0.15 of peak quality
    good_buckets = [
        int(b) for b, data in intensity_map.items()
        if data["outcome_quality_avg"] >= peak_quality - 0.15
    ]

    optimal_low  = min(good_buckets) / 10 if good_buckets else None
    optimal_high = max(good_buckets) / 10 if good_buckets else None

    # Edge: first bucket above optimal where quality drops sharply
    edge = None
    if optimal_high is not None:
        above_optimal = [
            (int(b), data["outcome_quality_avg"])
            for b, data in intensity_map.items()
            if int(b) / 10 > optimal_high
        ]
        for bucket_int, quality in sorted(above_optimal):
            if quality < peak_quality - EDGE_DETECTION_THRESHOLD:
                edge = bucket_int / 10
                break

    return optimal_low, optimal_high, edge


# ── Core pattern engine ───────────────────────────────────────────────────────

class PatternEngine:

    def __init__(self):
        self._running = False
        log_event("PatternEngine", "INIT")

    # ── Pattern detection ─────────────────────────────────────────────────────

    async def detect_patterns(
        self,
        session_id: str,
        headmate:   str,
        brief_data: dict,
        llm,
    ) -> list[dict]:
        """
        Check if any known patterns are firing in this session.
        Also look for emerging patterns worth creating.
        Returns list of firing pattern dicts with recommendation.
        Called by session_manager after each message.
        """
        from core.store import store

        current_conditions = _build_trigger_conditions(brief_data)
        if not current_conditions:
            return []

        # Load active patterns for this headmate
        patterns = store.get_patterns(headmate=headmate, min_confidence=0.1)

        firing = []
        for pattern in patterns:
            trigger_conds = pattern.get("trigger_conditions", [])
            if not trigger_conds:
                continue

            match_score = _match_conditions(trigger_conds, current_conditions)
            if match_score < 0.5:
                continue

            # Pattern is firing — build recommendation
            rec = self._build_recommendation(pattern, brief_data, match_score)
            firing.append(rec)

        # Look for new emerging patterns
        asyncio.ensure_future(
            self._check_emerging_patterns(headmate, brief_data, current_conditions, llm)
        )

        if firing:
            log_event("PatternEngine", "PATTERNS_FIRING",
                session=session_id[:8],
                headmate=headmate,
                count=len(firing),
                actions=[p.get("action") for p in firing],
            )

        return firing

    def _build_recommendation(
        self,
        pattern:    dict,
        brief_data: dict,
        match_score: float,
    ) -> dict:
        """
        Build a actionable recommendation from a firing pattern.
        Includes intensity target, approach, watch_for.
        """
        action          = pattern.get("action", "hold")
        confidence      = pattern.get("confidence", 0.0)
        data_points     = pattern.get("data_points", 0)
        optimal_low     = None
        optimal_high    = None
        edge            = pattern.get("edge_intensity")

        # Parse intensity map if stored
        intensity_map_raw = pattern.get("intensity_map", {})
        if isinstance(intensity_map_raw, str):
            try:
                intensity_map_raw = json.loads(intensity_map_raw)
            except Exception:
                intensity_map_raw = {}

        if intensity_map_raw:
            optimal_low, optimal_high, detected_edge = _find_optimal_range(intensity_map_raw)
            if detected_edge and not edge:
                edge = detected_edge

        # Determine push target
        current_intensity = brief_data.get("intensity", 0.3)
        push_to = None

        if action == "feed" and optimal_high is not None:
            # Push to just above current, toward optimal range
            push_to = min(
                optimal_high,
                current_intensity + 0.1,
            )
        elif action == "feed" and pattern.get("push_to"):
            push_to = pattern.get("push_to")

        return {
            "pattern_id":    pattern["id"],
            "pattern_type":  pattern.get("pattern_type", "unknown"),
            "action":        action,
            "confidence":    confidence,
            "match_score":   round(match_score, 2),
            "data_points":   data_points,
            "approach":      pattern.get("approach", ""),
            "reasoning":     pattern.get("reasoning", ""),
            "push_to":       push_to,
            "optimal_low":   optimal_low,
            "optimal_high":  optimal_high,
            "edge":          edge,
            "watch_for":     pattern.get("watch_for", []),
            "thread_after":  None,
        }

    async def _check_emerging_patterns(
        self,
        headmate:           str,
        brief_data:         dict,
        current_conditions: list,
        llm,
    ) -> None:
        """
        Look at recent history for an emerging pattern worth creating.
        Only runs if enough data exists and no similar pattern already exists.
        """
        from core.store import store

        # Need enough recent messages to find a pattern
        recent = store.get_recent_messages(headmate=headmate, limit=30)
        if len(recent) < 8:
            return

        # Check if a similar pattern already exists
        existing = store.query("patterns",
            headmate=headmate.lower(),
            pattern_type=brief_data.get("register", "neutral"),
            active=1,
            limit=5,
        )

        # Don't create duplicates
        for p in existing:
            trigger_conds = p.get("trigger_conditions", [])
            if _match_conditions(trigger_conds, current_conditions) > 0.7:
                return

        # Ask LLM if there's a pattern worth creating
        recent_summary = "\n".join(
            f"- [{r.get('time_of_day','?')} {r.get('day_of_week','?')}] "
            f"register={r.get('register','?')} "
            f"stress={r.get('stress_level','?')} "
            f"topics={','.join((r.get('topics') or [])[:2])}"
            for r in recent[:15]
        )

        raw = await _call_llm(llm,
            system=(
                "You identify behavioral patterns from message history. "
                "You are data-driven and specific. "
                "Only flag real patterns with clear signal. JSON only."
            ),
            user=f"""Headmate: {headmate}
Current situation: {', '.join(current_conditions[:6])}

Recent message history (last 15):
{recent_summary}

Is there a meaningful behavioral pattern emerging here that isn't just noise?
A real pattern repeats, has a trigger, and has a consistent response.

Return JSON:
{{
  "pattern_found": true/false,
  "pattern_type": "temporal|trigger_response|frequency|absence|escalation|recovery",
  "trigger_conditions": ["list of conditions that trigger it"],
  "description": "what the pattern is",
  "confidence": 0.0-0.4 (emerging patterns are low confidence),
  "initial_recommendation": "hold — always hold for new patterns"
}}

Only return pattern_found=true if the signal is clear.
New patterns always start at hold — never feed or break immediately.""",
            tokens=300,
            temp=0.1,
        )

        data = _parse_json_safe(raw, {"pattern_found": False})
        if not data.get("pattern_found"):
            return

        # Create the new pattern at low confidence, hold action
        pattern_id = store.upsert_pattern({
            "headmate":           headmate.lower(),
            "pattern_type":       data.get("pattern_type", "trigger_response"),
            "trigger_conditions": json.dumps(data.get("trigger_conditions", [])),
            "trigger_confidence": data.get("confidence", 0.2),
            "action":             "hold",
            "approach":           data.get("description", ""),
            "reasoning":          "emerging pattern — observing",
            "confidence":         data.get("confidence", 0.2),
            "data_points":        1,
            "source":             "pattern_engine",
            "tags":               f"pattern,{headmate.lower()},emerging",
        })

        store.log_pattern_refinement(
            pattern_id=pattern_id,
            version=1,
            change="pattern created — hold, observing",
            reason=data.get("description", ""),
            data_points_at=1,
            headmate=headmate,
        )

        log_event("PatternEngine", "PATTERN_CREATED",
            headmate=headmate,
            type=data.get("pattern_type"),
            confidence=data.get("confidence", 0.2),
        )

    # ── Post-session refinement ───────────────────────────────────────────────

    async def post_session_refine(
        self,
        session_id: str,
        headmate:   str,
        llm,
    ) -> None:
        """
        Runs at session close.
        Updates patterns that fired this session based on outcomes.
        Called by session_manager._finalize_session().
        """
        from core.store import store

        # Get instances from this session
        instances = store.query("pattern_instances",
            session_id=session_id,
            headmate=headmate.lower(),
            active=1,
            limit=20,
        )

        if not instances:
            return

        for inst in instances:
            pattern_id = inst.get("pattern_id")
            if not pattern_id:
                continue

            pattern = store.get("patterns", pattern_id)
            if not pattern:
                continue

            await self._refine_pattern(pattern, inst, llm)

    async def _refine_pattern(
        self,
        pattern:  dict,
        instance: dict,
        llm,
    ) -> None:
        """
        Update a pattern based on a new instance outcome.
        Updates: confidence, action, intensity_map, edge, outcome_quality_avg.
        Logs refinement if anything meaningful changed.
        """
        from core.store import store

        headmate     = pattern.get("headmate", "")
        pattern_id   = pattern["id"]
        old_version  = pattern.get("version", 1)
        data_points  = pattern.get("data_points", 0)
        old_action   = pattern.get("action", "hold")
        old_conf     = pattern.get("confidence", 0.2)

        # Get all instances for this pattern
        all_instances = store.query("pattern_instances",
            pattern_id=pattern_id,
            active=1,
            limit=50,
        )

        if not all_instances:
            return

        # Build intensity map
        intensity_map = _build_intensity_map(all_instances)
        optimal_low, optimal_high, edge = _find_optimal_range(intensity_map)

        # Rolling outcome quality
        qualities    = [i.get("outcome_quality", 0.5) for i in all_instances]
        avg_quality  = sum(qualities) / len(qualities)
        recent_qual  = sum(qualities[-3:]) / min(3, len(qualities))
        trend        = recent_qual - avg_quality  # positive = improving

        # Confidence grows with data points, caps at 0.95
        new_conf = min(0.95, 0.2 + (data_points * 0.05))

        # Determine action
        new_action = old_action
        changes    = []

        if data_points >= MIN_DATA_POINTS_TO_ACT:
            if avg_quality >= OUTCOME_STRONG_THRESHOLD and new_conf >= MIN_CONFIDENCE_TO_FEED:
                if old_action != "feed":
                    new_action = "feed"
                    changes.append(f"promoted to feed (avg_quality={avg_quality:.2f})")

            elif avg_quality <= OUTCOME_DECLINE_THRESHOLD and new_conf >= MIN_CONFIDENCE_TO_BREAK:
                if old_action not in ("break", "flag_therapy"):
                    new_action = "break"
                    changes.append(f"promoted to break (avg_quality={avg_quality:.2f})")

            elif trend < -0.2 and new_conf >= MIN_CONFIDENCE_TO_BREAK:
                # Quality declining even if not yet below threshold
                if old_action == "feed":
                    new_action = "hold"
                    changes.append(f"demoted from feed — quality declining (trend={trend:.2f})")

        # Check for therapy flag
        therapy_flag = False
        if avg_quality <= 0.25 and data_points >= 5:
            therapy_flag = True
            if new_action not in ("flag_therapy",):
                changes.append("flagged for therapy — sustained low quality outcomes")

        # Determine push_to from intensity map
        push_to = None
        if new_action == "feed" and optimal_high is not None:
            # Push incrementally above last instance
            last_intensity = instance.get("intensity_out", 0.5)
            push_to = min(
                optimal_high + 0.1,  # slightly above current optimal
                (edge - 0.1) if edge else optimal_high + 0.1,
                last_intensity + 0.1,
            )
            push_to = round(push_to, 2)

        # Build approach from LLM if action changed or enough new data
        approach = pattern.get("approach", "")
        reasoning = pattern.get("reasoning", "")

        if changes and len(all_instances) >= 3:
            approach, reasoning = await self._generate_approach(
                pattern=pattern,
                instances=all_instances[-5:],
                intensity_map=intensity_map,
                optimal_low=optimal_low,
                optimal_high=optimal_high,
                edge=edge,
                new_action=new_action,
                llm=llm,
            )

        # Build watch_for from instances where quality dropped
        watch_for = pattern.get("watch_for") or []
        if isinstance(watch_for, str):
            try:
                watch_for = json.loads(watch_for)
            except Exception:
                watch_for = []

        # Check for new edge indicators in post_pattern text
        poor_instances = [
            i for i in all_instances
            if i.get("outcome_quality", 0.5) < 0.35
            and i.get("post_pattern")
        ]
        for inst in poor_instances[-3:]:
            post = inst.get("post_pattern", "")
            if post and post not in watch_for:
                watch_for.append(post[:60])

        # Write updated pattern
        store.upsert_pattern({
            "id":                 pattern_id,
            "headmate":           headmate,
            "pattern_type":       pattern.get("pattern_type"),
            "trigger_conditions": pattern.get("trigger_conditions"),
            "trigger_confidence": pattern.get("trigger_confidence", 0.5),
            "action":             new_action,
            "push_to":            push_to,
            "approach":           approach,
            "reasoning":          reasoning,
            "watch_for":          json.dumps(watch_for),
            "edge_intensity":     edge,
            "outcome_quality_avg": round(avg_quality, 3),
            "data_points":        data_points,
            "confidence":         round(new_conf, 3),
            "therapy_flag":       1 if therapy_flag else 0,
            "intensity_map":      json.dumps(intensity_map),
            "source":             "pattern_engine",
            "tags":               f"pattern,{headmate},{new_action}",
        })

        # Log refinements
        for change in changes:
            store.log_pattern_refinement(
                pattern_id=pattern_id,
                version=old_version + 1,
                change=change,
                reason=f"data_points={data_points}, avg_quality={avg_quality:.2f}, trend={trend:.2f}",
                data_points_at=data_points,
                headmate=headmate,
            )

        if changes:
            log_event("PatternEngine", "PATTERN_REFINED",
                pattern_id=pattern_id[:12],
                headmate=headmate,
                old_action=old_action,
                new_action=new_action,
                changes=changes,
                data_points=data_points,
                avg_quality=round(avg_quality, 2),
            )

    async def _generate_approach(
        self,
        pattern:      dict,
        instances:    list,
        intensity_map: dict,
        optimal_low:  Optional[float],
        optimal_high: Optional[float],
        edge:         Optional[float],
        new_action:   str,
        llm,
    ) -> tuple[str, str]:
        """
        Generate updated approach and reasoning text from instance history.
        Returns (approach, reasoning).
        """
        # Summarize instance outcomes
        inst_summary = "\n".join(
            f"- intensity={i.get('intensity_out', 0):.1f} "
            f"quality={i.get('outcome_quality', 0):.2f} "
            f"pushed={'yes' if i.get('gizmo_pushed') else 'no'} "
            f"post={i.get('post_pattern', 'unknown')[:60]}"
            for i in instances[-5:]
        )

        range_str = ""
        if optimal_low is not None:
            range_str = f"Optimal intensity range: {optimal_low:.1f}–{optimal_high:.1f}"
        if edge:
            range_str += f" | Edge at: {edge:.1f}"

        raw = await _call_llm(llm,
            system=(
                "You write concise behavioral guidance based on pattern data. "
                "You are specific, not generic. Never moralize. JSON only."
            ),
            user=f"""Pattern type: {pattern.get('pattern_type')}
Headmate: {pattern.get('headmate')}
Action: {new_action}
{range_str}

Recent instances:
{inst_summary}

Write:
1. approach: specific guidance for how to act on this pattern (1-2 sentences)
2. reasoning: why this recommendation, based on the data (1 sentence)

Return JSON:
{{"approach": "...", "reasoning": "..."}}""",
            tokens=200,
            temp=0.2,
        )

        data = _parse_json_safe(raw, {
            "approach":  pattern.get("approach", ""),
            "reasoning": pattern.get("reasoning", ""),
        })

        return data.get("approach", ""), data.get("reasoning", "")

    # ── Outcome quality update ────────────────────────────────────────────────

    async def update_instance_outcome(
        self,
        session_id:      str,
        headmate:        str,
        outcome:         str,
        outcome_signal:  str,
        post_pattern:    Optional[str] = None,
    ) -> None:
        """
        Called when a response outcome is filled in.
        Updates the most recent pattern instance for this session.
        """
        from core.store import store

        # Map outcome to quality score
        quality_map = {
            "landed":    0.85,
            "neutral":   0.5,
            "cooled":    0.35,
            "redirected": 0.3,
            "dismissed": 0.2,
            "escalated": 0.15,
        }
        quality = quality_map.get(outcome, 0.5)

        # Find most recent instance for this session
        instances = store.query("pattern_instances",
            session_id=session_id,
            headmate=headmate.lower(),
            active=1,
            order_by="created_at DESC",
            limit=1,
        )

        if not instances:
            return

        inst = instances[0]
        store.update("pattern_instances", inst["id"],
            outcome_quality=quality,
            post_pattern=post_pattern or outcome_signal,
        )

        # Update rolling average on parent pattern
        all_instances = store.query("pattern_instances",
            pattern_id=inst.get("pattern_id"),
            active=1,
            limit=50,
        )
        if all_instances:
            avg = sum(i.get("outcome_quality", 0.5) for i in all_instances) / len(all_instances)
            store.update("patterns", inst["pattern_id"],
                outcome_quality_avg=round(avg, 3),
                data_points=len(all_instances),
            )

        log_event("PatternEngine", "INSTANCE_OUTCOME_UPDATED",
            session=session_id[:8],
            outcome=outcome,
            quality=quality,
        )

    # ── Weekly review ─────────────────────────────────────────────────────────

    async def weekly_review(self, llm) -> None:
        """
        Background job. Runs once per week.
        Scans all patterns across past 7 days.
        Promotes emerging → confirmed.
        Decays inactive patterns.
        Generates therapy report per headmate.
        """
        from core.store import store

        log_event("PatternEngine", "WEEKLY_REVIEW_START")

        # Get all active patterns
        all_patterns = store.query("patterns", active=1, limit=500)

        now      = time.time()
        week_ago = now - (7 * 24 * 3600)
        month_ago = now - (30 * 24 * 3600)

        for pattern in all_patterns:
            pattern_id   = pattern["id"]
            headmate     = pattern.get("headmate", "")
            data_points  = pattern.get("data_points", 0)
            last_refined = pattern.get("last_refined") or pattern.get("created_at", now)
            action       = pattern.get("action", "hold")

            # Decay inactive patterns
            if last_refined < month_ago and action == "hold" and data_points < 3:
                store.delete("patterns", pattern_id)
                log_event("PatternEngine", "PATTERN_DECAYED",
                    pattern_id=pattern_id[:12],
                    headmate=headmate,
                    reason="inactive, low data, hold state",
                )
                continue

            # Promote hold → assess if enough new data
            if action == "hold" and data_points >= MIN_DATA_POINTS_TO_ACT:
                instances = store.query("pattern_instances",
                    pattern_id=pattern_id,
                    active=1,
                    limit=20,
                )
                if instances:
                    avg = sum(i.get("outcome_quality", 0.5) for i in instances) / len(instances)
                    new_conf = min(0.95, 0.2 + data_points * 0.05)

                    if avg >= OUTCOME_STRONG_THRESHOLD and new_conf >= MIN_CONFIDENCE_TO_FEED:
                        store.upsert_pattern({
                            "id":         pattern_id,
                            "action":     "feed",
                            "confidence": new_conf,
                        })
                        store.log_pattern_refinement(
                            pattern_id=pattern_id,
                            version=pattern.get("version", 1) + 1,
                            change="weekly review: promoted hold → feed",
                            reason=f"avg_quality={avg:.2f}, data_points={data_points}",
                            data_points_at=data_points,
                            headmate=headmate,
                        )

        # Generate therapy reports per headmate
        headmates = list(set(
            p["headmate"] for p in all_patterns
            if p.get("headmate") and p.get("therapy_flag")
        ))

        for headmate in headmates:
            await self._generate_therapy_report(headmate, week_ago, llm)

        log_event("PatternEngine", "WEEKLY_REVIEW_COMPLETE",
            patterns_reviewed=len(all_patterns),
            therapy_reports=len(headmates),
        )

    async def _generate_therapy_report(
        self,
        headmate:  str,
        since:     float,
        llm,
    ) -> None:
        """
        Generate a weekly therapy report for a headmate.
        Sanitized for external sharing — meaning survives, raw content doesn't.
        Written to store as a wellbeing record with tag therapy_report.
        """
        from core.store import store

        # Flagged patterns
        flagged = store.query("patterns",
            headmate=headmate.lower(),
            active=1,
            limit=20,
        )
        flagged = [p for p in flagged if p.get("therapy_flag")]

        if not flagged:
            return

        # Recent wellbeing observations
        wb = store.get_wellbeing(headmate, limit=20)

        # Emotion trend from this week
        emotion_log = store.query("emotion_log",
            headmate=headmate.lower(),
            active=1,
            order_by="created_at DESC",
            limit=50,
        )
        recent_emotion = [e for e in emotion_log if e.get("created_at", 0) >= since]

        valence_trend = "insufficient data"
        if len(recent_emotion) >= 4:
            vals = [e.get("valence", 0) for e in recent_emotion]
            avg  = sum(vals) / len(vals)
            if avg > 0.2:   valence_trend = "generally positive"
            elif avg < -0.2: valence_trend = "generally negative"
            else:            valence_trend = "mixed/neutral"

        pattern_text = "\n".join(
            f"- {p.get('pattern_type','?')}: {p.get('approach','')[:100]} "
            f"(quality={p.get('outcome_quality_avg',0):.2f})"
            for p in flagged[:5]
        )

        wb_text = "\n".join(
            f"- [{w.get('category','?')}] {w.get('observation','')[:100]}"
            for w in wb[:8]
        )

        raw = await _call_llm(llm,
            system=(
                "You write clinical weekly mental health summaries. "
                "You sanitize intimate content while preserving clinical meaning. "
                "You are specific, factual, and non-judgmental. "
                "Write for a therapist who knows their patient well."
            ),
            user=f"""Headmate: {headmate.title()}
Week of: {tz_now().strftime('%Y-%m-%d')}
Emotional trend: {valence_trend}

Flagged behavioral patterns:
{pattern_text or '(none)'}

Wellbeing observations:
{wb_text or '(none)'}

Write a weekly summary (3-5 sentences) suitable for a therapist briefing.
Sanitize intimate content — preserve clinical meaning, remove explicit detail.
Focus on: emotional state, behavioral patterns, needs, anything worth discussing.""",
            tokens=300,
            temp=0.3,
        )

        if raw and raw.strip():
            store.write("wellbeing", {
                "headmate":    headmate.lower(),
                "category":    "therapy_report",
                "observation": raw.strip(),
                "context":     f"Week of {tz_now().strftime('%Y-%m-%d')}",
                "register":    "clinical",
                "source":      "pattern_engine",
                "confidence":  0.9,
                "tags":        f"therapy_report,{headmate.lower()},weekly",
            })

            log_event("PatternEngine", "THERAPY_REPORT_WRITTEN",
                headmate=headmate,
                preview=raw[:60],
            )

    # ── Background loop ───────────────────────────────────────────────────────

    async def start(self, llm) -> None:
        """Start the weekly review background loop."""
        self._running = True
        log_event("PatternEngine", "STARTED")

        while self._running:
            await asyncio.sleep(WEEKLY_REVIEW_INTERVAL)
            try:
                await self.weekly_review(llm)
            except Exception as e:
                log_error("PatternEngine", "weekly review failed", exc=e)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _call_llm(
    llm,
    system: str,
    user:   str,
    tokens: int   = 300,
    temp:   float = 0.1,
) -> str:
    try:
        return await llm.generate(
            [{"role": "user", "content": user}],
            system_prompt=system,
            max_new_tokens=tokens,
            temperature=temp,
        )
    except Exception as e:
        log_error("PatternEngine", f"LLM call failed: {e}", exc=None)
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

pattern_engine = PatternEngine()
