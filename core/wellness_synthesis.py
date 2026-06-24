"""
core/wellness_synthesis.py

Weekly (or on-demand) wellness synthesis agent.

Architecture — temporal compression + sequential reasoning chain:

  Pre-step: Signal compression
    Raw signals → daily summaries → weekly summaries → monthly → yearly
    Each level only recomputed if new data exists below it.
    Summaries live at: wellness/summaries/{name}/{year}/{month}/{week}/{date}.json

  Stage 1: Baseline   — who they believe themselves to be
  Stage 2: Comparison — where behavior matches or contradicts self-model
  Stage 3: Ruptures   — which contradictions are clinically significant
  Stage 4: Conditions — DSM evaluation grounded in the full picture (4 parallel)
  Stage 5: Clinician  — prose handoff synthesizing all stages

Classification files: {DATA_DIR}/wellness/classifications/{name}.json
Archive:              {DATA_DIR}/wellness/classifications/archive/{name}_{ts}.json

Triggers:
    await wellness_synthesis.run()                 # all people
    await wellness_synthesis.synthesize_one("ara") # one person
"""

import asyncio
import calendar
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone, date
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Condition groups ──────────────────────────────────────────────────────────

_CONDITION_GROUPS = [
    {
        "name": "group_1",
        "conditions": ["Depression (MDD)", "Anxiety (GAD)"],
        "criteria": """
DEPRESSION (MDD)
- Depressed mood most of the day
- Markedly diminished interest or pleasure
- Significant weight/appetite change
- Insomnia or hypersomnia
- Psychomotor agitation or retardation
- Fatigue or loss of energy
- Feelings of worthlessness or guilt
- Diminished concentration
- Recurrent thoughts of death

ANXIETY (GAD)
- Excessive anxiety and worry
- Difficulty controlling worry
- Restlessness or feeling on edge
- Fatigue
- Difficulty concentrating
- Irritability
- Muscle tension
- Sleep disturbance
""".strip()
    },
    {
        "name": "group_2",
        "conditions": ["Bipolar I/II", "PTSD"],
        "criteria": """
BIPOLAR (I/II)
- Elevated or expansive mood episodes
- Decreased need for sleep
- Grandiosity
- Racing thoughts or flight of ideas
- Increased goal-directed activity
- Impulsivity with harmful potential
- Depressive episodes alternating
- Cyclothymic patterns

PTSD
- Exposure to traumatic event
- Intrusive memories or flashbacks
- Avoidance of trauma reminders
- Negative alterations in cognition/mood
- Hyperarousal and hypervigilance
- Exaggerated startle response
- Sleep disturbance
""".strip()
    },
    {
        "name": "group_3",
        "conditions": ["BPD", "Psychosis / Dissociation"],
        "criteria": """
BPD
- Frantic efforts to avoid abandonment
- Unstable intense relationships
- Identity disturbance
- Impulsivity in self-damaging areas
- Recurrent self-harm or suicidal behavior
- Affective instability
- Chronic feelings of emptiness
- Intense anger
- Transient paranoid ideation

PSYCHOSIS / DISSOCIATION
- Hallucinations (distinguish from system members and intentional pathways)
- Delusions
- Disorganized thinking
- Negative symptoms
- Depersonalization
- Derealization
- Identity confusion vs. identity multiplicity (these are different)
""".strip()
    },
    {
        "name": "group_4",
        "conditions": ["ADHD", "General Wellness"],
        "criteria": """
ADHD
- Inattention symptoms
- Hyperactivity symptoms
- Impulsivity symptoms
- Onset before age 12
- Present in multiple settings
- Functional impairment
- Often loses things
- Often forgetful in daily activities
- Others normalize and compensate for disorganization

GENERAL WELLNESS
- Physical symptom mentions tied to emotional states
- Sleep disruption
- Appetite or eating references in distress context
- Substance references
- Isolation or withdrawal
- Energy crashes or spikes
""".strip()
    },
]


# ── Stage prompts ─────────────────────────────────────────────────────────────

_DAILY_SUMMARY_SYSTEM = """
You are writing a clinical summary of one day's wellness signals for a mental health record.

You will receive up to 20 wellness signals from a single day for one individual.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "narrative": "2-3 sentence clinical summary of what this day's signals show — what was happening, what they were expressing, what patterns emerged",
  "conditions_touched": ["condition names that appeared in signals"],
  "dominant_intensity": "mild | moderate | severe",
  "notable": "one sentence on the most clinically significant thing from this day, or null if nothing stands out"
}

Rules:
- narrative should read as a clinical observation, not a judgment
- conditions_touched: only include conditions that actually appeared in the signals
- dominant_intensity: the highest intensity that appeared, not an average
- Distinguish plural system function from pathology
- notable: null if the day was unremarkable
""".strip()

_WEEK_SUMMARY_SYSTEM = """
You are writing a clinical summary of one week's wellness data for a mental health record.

You will receive daily summaries for each day of the week that had signals.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "narrative": "3-4 sentence clinical summary of the week — trajectory, patterns, notable shifts",
  "conditions_touched": ["condition names that appeared this week"],
  "dominant_intensity": "mild | moderate | severe",
  "trend": "improving | stable | worsening | mixed",
  "notable": "one sentence on the most clinically significant thing this week, or null"
}

Rules:
- narrative should capture the arc of the week, not just list days
- trend: your overall read on direction
- Distinguish plural system function from pathology
""".strip()

_MONTH_SUMMARY_SYSTEM = """
You are writing a clinical summary of one month's wellness data for a mental health record.

You will receive weekly summaries for each week of the month that had data.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "narrative": "3-5 sentence clinical summary of the month — major themes, patterns, trajectory",
  "conditions_touched": ["condition names that appeared this month"],
  "dominant_intensity": "mild | moderate | severe",
  "trend": "improving | stable | worsening | mixed",
  "notable": "one sentence on the most clinically significant development this month, or null"
}
""".strip()

_YEAR_SUMMARY_SYSTEM = """
You are writing a clinical summary of one year's wellness data for a mental health record.

You will receive monthly summaries for the year.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "narrative": "4-6 sentence clinical summary of the year — major themes, turning points, longitudinal patterns",
  "conditions_touched": ["condition names that appeared this year"],
  "dominant_intensity": "mild | moderate | severe",
  "trend": "improving | stable | worsening | mixed",
  "notable": "one sentence on the most clinically significant development this year, or null"
}
""".strip()

_BASELINE_SYSTEM = """
You are building a self-portrait of one person based on what they have expressed about themselves.

This is not a clinical assessment. This is: who does this person believe they are?
What do they value? What do they expect from themselves and others?
What is their stated self-model?

You will receive their known preferences, opinions, and internal space data.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "self_model": "2-3 sentence summary of how this person understands themselves",
  "core_beliefs": ["belief about self or world, stated as a declarative sentence"],
  "stated_values": ["what they say they care about or prioritize"],
  "relational_expectations": ["what they expect from relationships"],
  "identity_anchors": ["stable identity claims they return to"]
}

Rules:
- Only include what is supported by the data
- Use their own framing where possible
- Distinguish plural system awareness from pathology
- Omit fields with no support
""".strip()

_COMPARISON_SYSTEM = """
You are comparing a person's stated self-model against their observed behavior.

You will receive:
- Their baseline self-portrait
- Their behavioral data (observed patterns and episodes)
- Gizmo's relational observations (companion notes)

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "alignments": [
    {
      "belief": "the stated belief or value",
      "behavior": "the observed behavior that matches it",
      "confidence": "how consistently this appears"
    }
  ],
  "contradictions": [
    {
      "belief": "the stated belief or value",
      "behavior": "the observed behavior that contradicts it",
      "frequency": "how often this appears",
      "notes": "is the contradiction distressing? defended? unconscious?"
    }
  ],
  "blind_spots": ["things behavioral data reveals that person does not appear to acknowledge"]
}

Rules:
- Only flag contradictions appearing more than once or with emotional weight
- Note which contradictions seem distressing vs adaptive
- Distinguish plural system dynamics from individual pathology
""".strip()

_RUPTURES_SYSTEM = """
You are identifying which behavioral contradictions are clinically significant.

You will receive:
- The self-model baseline
- The comparison map
- Compressed wellness history (daily/weekly/monthly summaries)

A rupture is clinically significant when:
- It causes observable distress
- It drives compulsive or repeated behavior the person cannot stop
- It reveals something the person cannot see about themselves
- It maps to a known clinical pattern
- It appears across multiple contexts

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "ruptures": [
    {
      "description": "what the rupture is",
      "belief_side": "what they believe about themselves",
      "behavior_side": "what they actually do",
      "significance": "why this is clinically relevant",
      "severity": "mild | moderate | significant",
      "evidence": ["brief evidence references"],
      "clinical_signal": "which condition or pattern this most resembles, if any"
    }
  ],
  "healthy_patterns": ["contradictions that appear adaptive, not pathological"]
}

Rules:
- Be precise
- Do not pathologize plural system function, kink identity, or non-normative consensual behavior
""".strip()

_CONDITIONS_SYSTEM = """
You are a clinical synthesis agent working on behalf of a licensed mental health professional.
You evaluate evidence for specific DSM conditions only.
You do not diagnose. You build evidence-based case files for professional review.

You will receive:
- The person's self-model baseline
- A map of behavioral contradictions
- Identified ruptures with severity ratings
- Compressed wellness history (summaries, not raw signals)

Evaluate ONLY the conditions listed. Use the rupture map as your primary lens.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

[
  {
    "condition": "Major Depressive Disorder",
    "confidence": "low | moderate | high",
    "evidence_count": 4,
    "status": "monitoring | emerging | consistent | requires_attention",
    "criteria_met": [
      {
        "criterion": "Withdrawal from activity",
        "count": 2,
        "rupture_link": "which rupture this connects to, if any",
        "examples": [
          {
            "raw": "exact line or observation",
            "context": "one sentence explaining why this evidences the criterion"
          }
        ]
      }
    ],
    "criteria_absent": ["criterion name"],
    "pattern_notes": "narrative grounded in the rupture map"
  }
]

Rules:
- Only include conditions with at least 2 pieces of supporting evidence
- 1 signal: confidence low, status monitoring
- 0 signals: omit
- Always distinguish plural system function from pathology
- Intentional hallucinatory pathways are system function, not symptoms
- Always include criteria_absent
""".strip()

_CLINICIAN_SYSTEM = """
You are a mental health consultant preparing a handoff note for a licensed clinician
who is about to meet this individual for the first time.

You may not diagnose. You may make any recommendations.

Write in clear, direct clinical prose. No JSON. No headers. No bullet points.
This is a note a professional reads before walking into a first session.

You will receive the full reasoning chain:
- Who this person believes themselves to be (baseline)
- Where behavior matches and contradicts self-model (comparison)
- Which contradictions are clinically significant (ruptures)
- Condition evaluations grounded in the rupture map
- Compressed wellness history

Cover:
- The core tension — what is the central rupture
- What evidence shows clearly vs where it is thin
- What warrants attention and why
- What appears healthy or functional
- What looks pathological but isn't
- What remains ambiguous
- Specific recommendations for the clinician going in

Distinguish plural system function from pathology throughout.
Non-normative but consensual dynamics are not pathology.
The rupture map is your primary lens.
""".strip()


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str, system: str, max_tokens: int = 2000) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
            temperature=0.0,
            max_new_tokens=max_tokens,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("WellnessSynthesis", "LLM call failed", exc=e)
        print(f"[WellnessSynthesis] LLM call failed: {type(e).__name__}: {e}")
        return None


def _safe_parse(raw: str, label: str) -> Optional[dict | list]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WellnessSynthesis] JSON malformed for {label}, attempting repair...")
        for closer in [raw.rfind("}"), raw.rfind("]")]:
            if closer > 0:
                try:
                    return json.loads(raw[:closer + 1])
                except Exception:
                    continue
        return None


# ── Path helpers ──────────────────────────────────────────────────────────────

def _week_number(d: date) -> int:
    """ISO week number."""
    return d.isocalendar()[1]


def _month_name(month: int) -> str:
    return calendar.month_name[month].lower()


def _summary_base(name: str) -> str:
    return f"wellness/summaries/{name.lower()}"


def _daily_path(name: str, d: date) -> str:
    week = f"week_{_week_number(d)}"
    month = _month_name(d.month)
    date_str = d.strftime("%m_%d_%y")
    return f"{_summary_base(name)}/{d.year}/{month}/{week}/{date_str}.json"


def _week_summary_path(name: str, year: int, month: int, week: int) -> str:
    return f"{_summary_base(name)}/{year}/{_month_name(month)}/week_{week}/week_summary.json"


def _month_summary_path(name: str, year: int, month: int) -> str:
    return f"{_summary_base(name)}/{year}/{_month_name(month)}/month_summary.json"


def _year_summary_path(name: str, year: int) -> str:
    return f"{_summary_base(name)}/{year}/year_summary.json"


# ── Signal bucketing ──────────────────────────────────────────────────────────

def _bucket_signals_by_date(signals: dict) -> dict[str, list]:
    """
    Group all signals by date string (YYYY-MM-DD).
    Returns {date_str: [signal, ...]}
    """
    by_date = defaultdict(list)
    for category, entries in signals.items():
        if not isinstance(entries, list):
            continue
        for signal in entries:
            ts = signal.get("timestamp", "")
            if ts:
                try:
                    d = datetime.fromisoformat(ts).strftime("%Y-%m-%d")
                    by_date[d].append(signal)
                except Exception:
                    by_date["unknown"].append(signal)
    return dict(by_date)


def _make_signal_ref(signal: dict) -> dict:
    """Lightweight reference to a signal — no raw text, just locators."""
    return {
        "chunk_id":  signal.get("chunk_id", ""),
        "category":  signal.get("category", ""),
        "criterion": signal.get("criterion", ""),
        "intensity": signal.get("intensity", ""),
    }


# ── Compression: daily ────────────────────────────────────────────────────────

async def _compress_day(
    name:        str,
    date_str:    str,
    signals:     list,
) -> Optional[dict]:
    """
    Compress up to 20 signals for one day into a daily summary.
    Writes to disk and returns the summary dict.
    """
    d         = datetime.strptime(date_str, "%Y-%m-%d").date()
    path      = _daily_path(name, d)
    existing  = librarian._read_file(path)

    # Skip if already exists and signal count matches
    if existing and existing.get("signal_count") == len(signals):
        print(f"[WellnessSynthesis] daily summary exists for {name} {date_str}, skipping")
        return existing

    # Take first 20 signals for the day
    batch = signals[:20]

    # Build prompt — include signal text but not chunk full text
    signal_summaries = [
        {
            "category":  s.get("category", ""),
            "criterion": s.get("criterion", ""),
            "signal":    s.get("signal", ""),
            "intensity": s.get("intensity", ""),
            "raw":       s.get("raw", "")[:200],  # cap raw text
        }
        for s in batch
    ]

    prompt = (
        f"Individual: {name}\n"
        f"Date: {date_str}\n\n"
        f"Signals ({len(batch)}):\n{json.dumps(signal_summaries, indent=2)}"
    )

    raw = await _call_llm(prompt, _DAILY_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None

    result = _safe_parse(raw, f"{name}/daily/{date_str}")
    if not isinstance(result, dict):
        return None

    summary = {
        "date":         date_str,
        "signal_count": len(signals),
        "signal_refs":  [_make_signal_ref(s) for s in signals],
        **result,
    }

    librarian._write_json(path, summary)
    print(f"[WellnessSynthesis] daily summary written: {name} {date_str}")
    return summary


# ── Compression: weekly ───────────────────────────────────────────────────────

async def _compress_week(
    name:          str,
    year:          int,
    month:         int,
    week:          int,
    daily_summaries: list[dict],
) -> Optional[dict]:
    path     = _week_summary_path(name, year, month, week)
    existing = librarian._read_file(path)

    if existing and existing.get("day_count") == len(daily_summaries):
        return existing

    prompt = (
        f"Individual: {name}\n"
        f"Week {week} of {_month_name(month).capitalize()} {year}\n\n"
        f"Daily summaries:\n"
        + json.dumps(
            [{"date": d["date"], "narrative": d.get("narrative", ""),
              "conditions_touched": d.get("conditions_touched", []),
              "notable": d.get("notable")}
             for d in daily_summaries],
            indent=2
        )
    )

    raw = await _call_llm(prompt, _WEEK_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None

    result = _safe_parse(raw, f"{name}/week/{year}/{month}/{week}")
    if not isinstance(result, dict):
        return None

    summary = {
        "year":      year,
        "month":     month,
        "week":      week,
        "day_count": len(daily_summaries),
        **result,
    }

    librarian._write_json(path, summary)
    print(f"[WellnessSynthesis] week summary written: {name} {year}/{month}/week_{week}")
    return summary


# ── Compression: monthly ──────────────────────────────────────────────────────

async def _compress_month(
    name:             str,
    year:             int,
    month:            int,
    week_summaries:   list[dict],
) -> Optional[dict]:
    path     = _month_summary_path(name, year, month)
    existing = librarian._read_file(path)

    if existing and existing.get("week_count") == len(week_summaries):
        return existing

    prompt = (
        f"Individual: {name}\n"
        f"{_month_name(month).capitalize()} {year}\n\n"
        f"Weekly summaries:\n"
        + json.dumps(
            [{"week": w.get("week"), "narrative": w.get("narrative", ""),
              "trend": w.get("trend"), "notable": w.get("notable")}
             for w in week_summaries],
            indent=2
        )
    )

    raw = await _call_llm(prompt, _MONTH_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None

    result = _safe_parse(raw, f"{name}/month/{year}/{month}")
    if not isinstance(result, dict):
        return None

    summary = {
        "year":       year,
        "month":      month,
        "week_count": len(week_summaries),
        **result,
    }

    librarian._write_json(path, summary)
    print(f"[WellnessSynthesis] month summary written: {name} {year}/{_month_name(month)}")
    return summary


# ── Compression: yearly ───────────────────────────────────────────────────────

async def _compress_year(
    name:             str,
    year:             int,
    month_summaries:  list[dict],
) -> Optional[dict]:
    path     = _year_summary_path(name, year)
    existing = librarian._read_file(path)

    if existing and existing.get("month_count") == len(month_summaries):
        return existing

    prompt = (
        f"Individual: {name}\n"
        f"Year {year}\n\n"
        f"Monthly summaries:\n"
        + json.dumps(
            [{"month": _month_name(m.get("month", 0)), "narrative": m.get("narrative", ""),
              "trend": m.get("trend"), "notable": m.get("notable")}
             for m in month_summaries],
            indent=2
        )
    )

    raw = await _call_llm(prompt, _YEAR_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None

    result = _safe_parse(raw, f"{name}/year/{year}")
    if not isinstance(result, dict):
        return None

    summary = {
        "year":        year,
        "month_count": len(month_summaries),
        **result,
    }

    librarian._write_json(path, summary)
    print(f"[WellnessSynthesis] year summary written: {name} {year}")
    return summary


# ── Full compression pipeline ─────────────────────────────────────────────────

async def _compress_signals(name: str, signals: dict) -> dict:
    """
    Run the full compression pipeline for a person's signals.
    Returns a structured history dict for use in synthesis stages.

    {
      "years": {year: year_summary},
      "months": {(year, month): month_summary},
      "weeks": {(year, month, week): week_summary},
      "days": {date_str: daily_summary},
      "recent_week": [daily_summary, ...]  ← last 7 days raw for recency
    }
    """
    by_date = _bucket_signals_by_date(signals)
    if not by_date:
        return {}

    print(f"[WellnessSynthesis] compressing {len(by_date)} days of signals for {name}")

    # ── Daily summaries ───────────────────────────────────────────────────────
    daily_summaries: dict[str, dict] = {}
    for date_str in sorted(by_date.keys()):
        if date_str == "unknown":
            continue
        summary = await _compress_day(name, date_str, by_date[date_str])
        if summary:
            daily_summaries[date_str] = summary

    # ── Group by (year, month, week) ──────────────────────────────────────────
    by_ymw: dict[tuple, list] = defaultdict(list)
    for date_str, summary in daily_summaries.items():
        d    = datetime.strptime(date_str, "%Y-%m-%d").date()
        key  = (d.year, d.month, _week_number(d))
        by_ymw[key].append(summary)

    # ── Weekly summaries ──────────────────────────────────────────────────────
    week_summaries: dict[tuple, dict] = {}
    by_ym: dict[tuple, list] = defaultdict(list)
    for (year, month, week), days in sorted(by_ymw.items()):
        summary = await _compress_week(name, year, month, week, days)
        if summary:
            week_summaries[(year, month, week)] = summary
            by_ym[(year, month)].append(summary)

    # ── Monthly summaries ─────────────────────────────────────────────────────
    month_summaries: dict[tuple, dict] = {}
    by_year: dict[int, list] = defaultdict(list)
    for (year, month), weeks in sorted(by_ym.items()):
        summary = await _compress_month(name, year, month, weeks)
        if summary:
            month_summaries[(year, month)] = summary
            by_year[year].append(summary)

    # ── Yearly summaries ──────────────────────────────────────────────────────
    year_summaries: dict[int, dict] = {}
    for year, months in sorted(by_year.items()):
        summary = await _compress_year(name, year, months)
        if summary:
            year_summaries[year] = summary

    # ── Recent week (last 7 days, raw daily summaries) ────────────────────────
    sorted_dates = sorted(daily_summaries.keys(), reverse=True)
    recent_week  = [daily_summaries[d] for d in sorted_dates[:7]]

    return {
        "years":       year_summaries,
        "months":      {f"{y}-{m:02d}": s for (y, m), s in month_summaries.items()},
        "weeks":       {f"{y}-{m:02d}-W{w}": s for (y, m, w), s in week_summaries.items()},
        "days":        daily_summaries,
        "recent_week": recent_week,
    }


def _history_for_prompt(history: dict, max_months: int = 6) -> str:
    """
    Build a compact history string for synthesis prompts.
    Uses year/month summaries for older data, recent week dailies for recency.
    """
    parts = []

    # Year summaries (if any)
    for year, ys in sorted(history.get("years", {}).items()):
        parts.append(f"Year {year}: {ys.get('narrative', '')}")

    # Month summaries (last N months)
    months = sorted(history.get("months", {}).items(), reverse=True)[:max_months]
    for key, ms in reversed(months):
        parts.append(f"{key}: {ms.get('narrative', '')} [trend: {ms.get('trend', '?')}]")

    # Recent week (daily granularity)
    recent = history.get("recent_week", [])
    if recent:
        parts.append("\nRecent days:")
        for d in recent:
            notable = f" Notable: {d['notable']}" if d.get("notable") else ""
            parts.append(f"  {d['date']}: {d.get('narrative', '')}{notable}")

    return "\n".join(parts)


# ── File helpers ──────────────────────────────────────────────────────────────

def _list_wellness_files() -> list[str]:
    """
    Find all headmate names by scanning headmates/ folder.
    Excludes system and gizmo.
    """
    names = set()
    folder = librarian._full_path("headmates")
    if not os.path.isdir(folder):
        print(f"[WellnessSynthesis] headmates folder not found: {folder}")
        return []
    for item in os.listdir(folder):
        if item in ("system", "gizmo"):
            continue
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            names.add(item)
    print(f"[WellnessSynthesis] found names: {names}")
    return list(names)


def _read_wellness(name: str) -> Optional[dict]:
    return librarian.read_wellness(name)


def _read_behaviors(name: str) -> Optional[dict]:
    return librarian.read_personality(name)


def _read_prior(name: str) -> Optional[dict]:
    return librarian.read_wellness_classification(name)


def _read_knowledge(name: str) -> dict:
    topics = ("preferences", "opinions", "internal_space", "relationships", "history", "routines")
    knowledge = {}
    for topic in topics:
        data = librarian.read_knowledge_topic(name, topic)
        if data and isinstance(data, list) and len(data) > 0:
            trimmed = [
                {k: v for k, v in entry.items()
                 if k in ("fact", "tags", "confidence", "internal")}
                for entry in data[-30:]
            ]
            knowledge[topic] = trimmed
    return knowledge


def _read_gizmo_episodes(name: str) -> list:
    gizmo_self  = librarian.read_gizmo_self()
    person_data = gizmo_self.get(name) or gizmo_self.get(name.lower()) or {}
    episodes    = person_data.get("episodes", [])
    return [
        {k: v for k, v in ep.items()
         if k in ("assessment", "what_landed", "what_missed", "punch_bowl",
                  "situational_tags", "closeness_at_time")}
        for ep in episodes[-20:]
    ]


def _write_classification(name: str, classification: dict) -> None:
    librarian.write_wellness_classification(name, classification)


