"""
core/wellness_synthesis.py

Weekly (or on-demand) wellness synthesis agent.

Architecture — temporal compression + sequential reasoning chain:

  Pre-step: Signal compression
    Raw signals → daily summaries → weekly summaries → monthly → yearly
    Each level only recomputed if new data exists below it.

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


_CONDITION_GROUPS = [
    {
        "name": "group_1",
        "conditions": ["Depression (MDD)", "Anxiety (GAD)"],
        "criteria": "DEPRESSION (MDD)\n- Depressed mood most of the day\n- Markedly diminished interest or pleasure\n- Significant weight/appetite change\n- Insomnia or hypersomnia\n- Fatigue or loss of energy\n- Feelings of worthlessness or guilt\n- Diminished concentration\n- Recurrent thoughts of death\n\nANXIETY (GAD)\n- Excessive anxiety and worry\n- Difficulty controlling worry\n- Restlessness or feeling on edge\n- Fatigue\n- Difficulty concentrating\n- Irritability\n- Muscle tension\n- Sleep disturbance"
    },
    {
        "name": "group_2",
        "conditions": ["Bipolar I/II", "PTSD"],
        "criteria": "BIPOLAR (I/II)\n- Elevated or expansive mood episodes\n- Decreased need for sleep\n- Grandiosity\n- Racing thoughts or flight of ideas\n- Increased goal-directed activity\n- Impulsivity with harmful potential\n- Depressive episodes alternating\n\nPTSD\n- Exposure to traumatic event\n- Intrusive memories or flashbacks\n- Avoidance of trauma reminders\n- Negative alterations in cognition/mood\n- Hyperarousal and hypervigilance\n- Exaggerated startle response\n- Sleep disturbance"
    },
    {
        "name": "group_3",
        "conditions": ["BPD", "Psychosis / Dissociation"],
        "criteria": "BPD\n- Frantic efforts to avoid abandonment\n- Unstable intense relationships\n- Identity disturbance\n- Impulsivity in self-damaging areas\n- Recurrent self-harm or suicidal behavior\n- Affective instability\n- Chronic feelings of emptiness\n- Intense anger\n\nPSYCHOSIS / DISSOCIATION\n- Hallucinations (distinguish from system members and intentional pathways)\n- Delusions\n- Disorganized thinking\n- Depersonalization\n- Derealization\n- Identity confusion vs. identity multiplicity (these are different)"
    },
    {
        "name": "group_4",
        "conditions": ["ADHD", "General Wellness"],
        "criteria": "ADHD\n- Inattention symptoms\n- Hyperactivity symptoms\n- Impulsivity symptoms\n- Present in multiple settings\n- Functional impairment\n- Often loses things\n- Often forgetful in daily activities\n- Others normalize and compensate for disorganization\n\nGENERAL WELLNESS\n- Physical symptom mentions tied to emotional states\n- Sleep disruption\n- Appetite or eating references in distress context\n- Substance references\n- Isolation or withdrawal\n- Energy crashes or spikes"
    },
]

_DAILY_SUMMARY_SYSTEM = "You are writing a clinical summary of one day's wellness signals for a mental health record.\nReturn ONLY valid JSON. No markdown. No explanation.\n\n{\"narrative\": \"2-3 sentence clinical summary\", \"conditions_touched\": [\"condition names\"], \"dominant_intensity\": \"mild | moderate | severe\", \"notable\": \"one sentence or null\"}\n\nDistinguish plural system function from pathology."

_WEEK_SUMMARY_SYSTEM = "You are writing a clinical summary of one week's wellness data.\nReturn ONLY valid JSON. No markdown. No explanation.\n\n{\"narrative\": \"3-4 sentence summary\", \"conditions_touched\": [\"condition names\"], \"dominant_intensity\": \"mild | moderate | severe\", \"trend\": \"improving | stable | worsening | mixed\", \"notable\": \"one sentence or null\"}"

_MONTH_SUMMARY_SYSTEM = "You are writing a clinical summary of one month's wellness data.\nReturn ONLY valid JSON. No markdown. No explanation.\n\n{\"narrative\": \"3-5 sentence summary\", \"conditions_touched\": [\"condition names\"], \"dominant_intensity\": \"mild | moderate | severe\", \"trend\": \"improving | stable | worsening | mixed\", \"notable\": \"one sentence or null\"}"

_YEAR_SUMMARY_SYSTEM = "You are writing a clinical summary of one year's wellness data.\nReturn ONLY valid JSON. No markdown. No explanation.\n\n{\"narrative\": \"4-6 sentence summary\", \"conditions_touched\": [\"condition names\"], \"dominant_intensity\": \"mild | moderate | severe\", \"trend\": \"improving | stable | worsening | mixed\", \"notable\": \"one sentence or null\"}"

_BASELINE_SYSTEM = "You are building a self-portrait of one person based on what they have expressed about themselves.\nThis is: who does this person believe they are?\n\nReturn ONLY valid JSON. No markdown. No explanation.\n\n{\"self_model\": \"2-3 sentence summary\", \"core_beliefs\": [\"declarative sentences\"], \"stated_values\": [\"what they prioritize\"], \"relational_expectations\": [\"what they expect from relationships\"], \"identity_anchors\": [\"stable identity claims\"]}\n\nOnly include what is supported by data. Distinguish plural system awareness from pathology."

_COMPARISON_SYSTEM = "You are comparing a person's stated self-model against their observed behavior.\n\nReturn ONLY valid JSON. No markdown. No explanation.\n\n{\"alignments\": [{\"belief\": \"\", \"behavior\": \"\", \"confidence\": \"\"}], \"contradictions\": [{\"belief\": \"\", \"behavior\": \"\", \"frequency\": \"\", \"notes\": \"\"}], \"blind_spots\": [\"things data reveals person does not acknowledge\"]}\n\nOnly flag contradictions appearing more than once or with emotional weight. Distinguish plural system dynamics from individual pathology."

_RUPTURES_SYSTEM = "You are identifying which behavioral contradictions are clinically significant.\n\nA rupture is significant when it causes observable distress, drives compulsive behavior, reveals something the person cannot see, maps to a clinical pattern, or appears across multiple contexts.\n\nReturn ONLY valid JSON. No markdown. No explanation.\n\n{\"ruptures\": [{\"description\": \"\", \"belief_side\": \"\", \"behavior_side\": \"\", \"significance\": \"\", \"severity\": \"mild | moderate | significant\", \"evidence\": [\"\"], \"clinical_signal\": \"\"}], \"healthy_patterns\": [\"adaptive contradictions\"]}\n\nDo not pathologize plural system function, kink identity, or non-normative consensual behavior."

_CONDITIONS_SYSTEM = "You are a clinical synthesis agent evaluating DSM conditions for a licensed professional. You do not diagnose.\n\nEvaluate ONLY the listed conditions. Use the rupture map as your primary lens.\n\nReturn ONLY valid JSON array. No markdown.\n\n[{\"condition\": \"\", \"confidence\": \"low | moderate | high\", \"evidence_count\": 0, \"status\": \"monitoring | emerging | consistent | requires_attention\", \"criteria_met\": [{\"criterion\": \"\", \"count\": 0, \"rupture_link\": \"\", \"examples\": [{\"raw\": \"\", \"context\": \"\"}]}], \"criteria_absent\": [\"\"], \"pattern_notes\": \"\"}]\n\nOnly include conditions with 2+ supporting pieces of evidence. Always distinguish plural system function from pathology."

_CLINICIAN_SYSTEM = "You are preparing a handoff note for a licensed clinician meeting this individual for the first time.\n\nWrite in clear clinical prose. No JSON. No headers. No bullet points.\n\nCover: the core tension/central rupture, what evidence shows clearly vs where it is thin, what warrants attention, what appears healthy, what looks pathological but is not, what remains ambiguous, specific recommendations.\n\nDistinguish plural system function from pathology throughout. Non-normative consensual dynamics are not pathology. The rupture map is your primary lens."


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


def _week_number(d: date) -> int:
    return d.isocalendar()[1]

def _month_name(month: int) -> str:
    return calendar.month_name[month].lower()

def _summary_base(name: str) -> str:
    return f"wellness/summaries/{name.lower()}"

def _daily_path(name: str, d: date) -> str:
    return f"{_summary_base(name)}/{d.year}/{_month_name(d.month)}/week_{_week_number(d)}/{d.strftime('%m_%d_%y')}.json"

def _week_summary_path(name: str, year: int, month: int, week: int) -> str:
    return f"{_summary_base(name)}/{year}/{_month_name(month)}/week_{week}/week_summary.json"

def _month_summary_path(name: str, year: int, month: int) -> str:
    return f"{_summary_base(name)}/{year}/{_month_name(month)}/month_summary.json"

def _year_summary_path(name: str, year: int) -> str:
    return f"{_summary_base(name)}/{year}/year_summary.json"


def _bucket_signals_by_date(signals: dict) -> dict[str, list]:
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
    return {k: signal.get(k, "") for k in ("chunk_id", "category", "criterion", "intensity")}


async def _compress_day(name: str, date_str: str, signals: list) -> Optional[dict]:
    d        = datetime.strptime(date_str, "%Y-%m-%d").date()
    path     = _daily_path(name, d)
    existing = librarian._read_file(path)
    if existing and existing.get("signal_count") == len(signals):
        return existing
    batch = signals[:20]
    prompt = (
        f"Individual: {name}\nDate: {date_str}\n\n"
        f"Signals ({len(batch)}):\n"
        + json.dumps([{"category": s.get("category",""), "criterion": s.get("criterion",""),
                       "signal": s.get("signal",""), "intensity": s.get("intensity",""),
                       "raw": s.get("raw","")[:200]} for s in batch], indent=2)
    )
    raw = await _call_llm(prompt, _DAILY_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None
    result = _safe_parse(raw, f"{name}/daily/{date_str}")
    if not isinstance(result, dict):
        return None
    summary = {"date": date_str, "signal_count": len(signals),
               "signal_refs": [_make_signal_ref(s) for s in signals], **result}
    librarian._write_json(path, summary)
    return summary


async def _compress_week(name: str, year: int, month: int, week: int, days: list) -> Optional[dict]:
    path     = _week_summary_path(name, year, month, week)
    existing = librarian._read_file(path)
    if existing and existing.get("day_count") == len(days):
        return existing
    prompt = (f"Individual: {name}\nWeek {week} of {_month_name(month).capitalize()} {year}\n\n"
              f"Daily summaries:\n" + json.dumps(
                  [{"date": d["date"], "narrative": d.get("narrative",""),
                    "conditions_touched": d.get("conditions_touched",[]),
                    "notable": d.get("notable")} for d in days], indent=2))
    raw = await _call_llm(prompt, _WEEK_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None
    result = _safe_parse(raw, f"{name}/week/{year}/{month}/{week}")
    if not isinstance(result, dict):
        return None
    summary = {"year": year, "month": month, "week": week, "day_count": len(days), **result}
    librarian._write_json(path, summary)
    return summary


async def _compress_month(name: str, year: int, month: int, weeks: list) -> Optional[dict]:
    path     = _month_summary_path(name, year, month)
    existing = librarian._read_file(path)
    if existing and existing.get("week_count") == len(weeks):
        return existing
    prompt = (f"Individual: {name}\n{_month_name(month).capitalize()} {year}\n\n"
              f"Weekly summaries:\n" + json.dumps(
                  [{"week": w.get("week"), "narrative": w.get("narrative",""),
                    "trend": w.get("trend"), "notable": w.get("notable")} for w in weeks], indent=2))
    raw = await _call_llm(prompt, _MONTH_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None
    result = _safe_parse(raw, f"{name}/month/{year}/{month}")
    if not isinstance(result, dict):
        return None
    summary = {"year": year, "month": month, "week_count": len(weeks), **result}
    librarian._write_json(path, summary)
    return summary


async def _compress_year(name: str, year: int, months: list) -> Optional[dict]:
    path     = _year_summary_path(name, year)
    existing = librarian._read_file(path)
    if existing and existing.get("month_count") == len(months):
        return existing
    prompt = (f"Individual: {name}\nYear {year}\n\n"
              f"Monthly summaries:\n" + json.dumps(
                  [{"month": _month_name(m.get("month",0)), "narrative": m.get("narrative",""),
                    "trend": m.get("trend"), "notable": m.get("notable")} for m in months], indent=2))
    raw = await _call_llm(prompt, _YEAR_SUMMARY_SYSTEM, max_tokens=400)
    if not raw:
        return None
    result = _safe_parse(raw, f"{name}/year/{year}")
    if not isinstance(result, dict):
        return None
    summary = {"year": year, "month_count": len(months), **result}
    librarian._write_json(path, summary)
    return summary


async def _compress_signals(name: str, signals: dict) -> dict:
    by_date = _bucket_signals_by_date(signals)
    if not by_date:
        return {}
    print(f"[WellnessSynthesis] compressing {len(by_date)} days for {name}")

    daily: dict[str, dict] = {}
    for date_str in sorted(by_date.keys()):
        if date_str == "unknown":
            continue
        s = await _compress_day(name, date_str, by_date[date_str])
        if s:
            daily[date_str] = s

    by_ymw: dict[tuple, list] = defaultdict(list)
    for date_str, s in daily.items():
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        by_ymw[(d.year, d.month, _week_number(d))].append(s)

    weeks: dict[tuple, dict] = {}
    by_ym: dict[tuple, list] = defaultdict(list)
    for (y, m, w), days in sorted(by_ymw.items()):
        s = await _compress_week(name, y, m, w, days)
        if s:
            weeks[(y, m, w)] = s
            by_ym[(y, m)].append(s)

    months: dict[tuple, dict] = {}
    by_year: dict[int, list]  = defaultdict(list)
    for (y, m), ws in sorted(by_ym.items()):
        s = await _compress_month(name, y, m, ws)
        if s:
            months[(y, m)] = s
            by_year[y].append(s)

    years: dict[int, dict] = {}
    for y, ms in sorted(by_year.items()):
        s = await _compress_year(name, y, ms)
        if s:
            years[y] = s

    sorted_dates = sorted(daily.keys(), reverse=True)
    return {
        "years":       years,
        "months":      {f"{y}-{m:02d}": s for (y, m), s in months.items()},
        "weeks":       {f"{y}-{m:02d}-W{w}": s for (y, m, w), s in weeks.items()},
        "days":        daily,
        "recent_week": [daily[d] for d in sorted_dates[:7]],
    }


def _history_for_prompt(history: dict, max_months: int = 6) -> str:
    parts = []
    for year, ys in sorted(history.get("years", {}).items()):
        parts.append(f"Year {year}: {ys.get('narrative','')}")
    for key, ms in list(reversed(sorted(history.get("months", {}).items())))[:max_months]:
        parts.append(f"{key}: {ms.get('narrative','')} [trend: {ms.get('trend','?')}]")
    recent = history.get("recent_week", [])
    if recent:
        parts.append("\nRecent days:")
        for d in recent:
            notable = f" Notable: {d['notable']}" if d.get("notable") else ""
            parts.append(f"  {d['date']}: {d.get('narrative','')}{notable}")
    return "\n".join(parts)


def _list_wellness_files() -> list[str]:
    names  = set()
    folder = librarian._full_path("headmates")
    if not os.path.isdir(folder):
        print(f"[WellnessSynthesis] headmates folder not found: {folder}")
        return []
    for item in os.listdir(folder):
        if item in ("system", "gizmo"):
            continue
        if os.path.isdir(os.path.join(folder, item)):
            names.add(item)
    print(f"[WellnessSynthesis] found names: {names}")
    return list(names)


def _read_wellness(name: str) -> Optional[dict]:
    if hasattr(librarian, "read_wellness"):
        return librarian.read_wellness(name)
    return librarian._read_file(f"wellness/{name.lower()}.json")

def _read_behaviors(name: str) -> Optional[dict]:
    if hasattr(librarian, "read_personality"):
        return librarian.read_personality(name)
    return librarian._read_file(f"behaviors/{name.lower()}.json")

def _read_knowledge(name: str) -> dict:
    topics = ("preferences", "opinions", "internal_space", "relationships", "history", "routines")
    knowledge = {}
    for topic in topics:
        if hasattr(librarian, "read_knowledge_topic"):
            data = librarian.read_knowledge_topic(name, topic)
        else:
            data = librarian._read_file(f"headmates/{name.lower()}/knowledge/{topic}.json")
        if data and isinstance(data, list) and data:
            knowledge[topic] = [
                {k: v for k, v in e.items() if k in ("fact","tags","confidence","internal")}
                for e in data[-30:]
            ]
    return knowledge

def _read_gizmo_episodes(name: str) -> list:
    if hasattr(librarian, "read_gizmo_self"):
        gizmo_self = librarian.read_gizmo_self()
    else:
        gizmo_self = librarian._read_file("behaviors/gizmo_self.json") or {}
    person_data = gizmo_self.get(name) or gizmo_self.get(name.lower()) or {}
    keep = ("assessment","what_landed","what_missed","punch_bowl","situational_tags","closeness_at_time")
    return [{k: v for k, v in ep.items() if k in keep} for ep in person_data.get("episodes", [])[-20:]]

def _write_classification(name: str, classification: dict) -> None:
    if hasattr(librarian, "write_wellness_classification"):
        librarian.write_wellness_classification(name, classification)
        return
    path     = f"wellness/classifications/{name.lower()}.json"
    existing = librarian._read_file(path)
    if existing:
        ts       = existing.get("last_synthesized", datetime.now(timezone.utc).isoformat())
        ts_clean = ts.replace(":", "-").replace(".", "-")[:19]
        librarian._write_json(f"wellness/classifications/archive/{name.lower()}_{ts_clean}.json", existing)
    librarian._write_json(path, classification)
    print(f"[WellnessSynthesis] classification written for {name}")


async def _stage_baseline(name: str, knowledge: dict, behaviors: dict) -> Optional[dict]:
    prompt = (f"Individual: {name}\n\nKnowledge:\n{json.dumps(knowledge, indent=2)}\n\n"
              f"Behavioral personality:\n{json.dumps(behaviors.get('Personality',{}), indent=2)}")
    raw = await _call_llm(prompt, _BASELINE_SYSTEM, max_tokens=600)
    return _safe_parse(raw, f"{name}/baseline") if raw else None

async def _stage_comparison(name: str, baseline: dict, behaviors: dict, gizmo_episodes: list) -> Optional[dict]:
    prompt = (f"Individual: {name}\n\nSelf-model baseline:\n{json.dumps(baseline, indent=2)}\n\n"
              f"Behavioral data:\n{json.dumps({'Personality': behaviors.get('Personality',{}), 'Episodes': behaviors.get('Episodes',[])[-20:]}, indent=2)}\n\n"
              f"Gizmo relational observations:\n{json.dumps(gizmo_episodes, indent=2)}")
    raw = await _call_llm(prompt, _COMPARISON_SYSTEM, max_tokens=800)
    return _safe_parse(raw, f"{name}/comparison") if raw else None

async def _stage_ruptures(name: str, baseline: dict, comparison: dict, history: dict) -> Optional[dict]:
    prompt = (f"Individual: {name}\n\nSelf-model baseline:\n{json.dumps(baseline, indent=2)}\n\n"
              f"Behavioral comparison:\n{json.dumps(comparison, indent=2)}\n\n"
              f"Compressed wellness history:\n{_history_for_prompt(history)}")
    raw = await _call_llm(prompt, _RUPTURES_SYSTEM, max_tokens=800)
    return _safe_parse(raw, f"{name}/ruptures") if raw else None

async def _stage_conditions(name: str, baseline: dict, comparison: dict, ruptures: dict, history: dict, group: dict) -> list:
    prompt = (f"Individual: {name}\n\nEvaluate ONLY: {', '.join(group['conditions'])}\n\n"
              f"DSM criteria:\n{group['criteria']}\n\n"
              f"Baseline:\n{json.dumps(baseline, indent=2)}\n\n"
              f"Contradictions:\n{json.dumps(comparison, indent=2)}\n\n"
              f"Ruptures:\n{json.dumps(ruptures, indent=2)}\n\n"
              f"History:\n{_history_for_prompt(history)}")
    raw    = await _call_llm(prompt, _CONDITIONS_SYSTEM, max_tokens=1200)
    result = _safe_parse(raw, f"{name}/{group['name']}") if raw else None
    return result if isinstance(result, list) else []

async def _stage_clinician(name: str, baseline: dict, comparison: dict, ruptures: dict, conditions: list, history: dict) -> str:
    prompt = (f"Individual: {name}\n\nBaseline:\n{json.dumps(baseline, indent=2)}\n\n"
              f"Comparison:\n{json.dumps(comparison, indent=2)}\n\n"
              f"Ruptures:\n{json.dumps(ruptures, indent=2)}\n\n"
              f"Conditions:\n{json.dumps(conditions, indent=2)}\n\n"
              f"History:\n{_history_for_prompt(history)}")
    raw = await _call_llm(prompt, _CLINICIAN_SYSTEM, max_tokens=1200)
    return raw.strip() if raw else "Clinician stage failed."


class WellnessSynthesis:

    async def synthesize_one(self, name: str) -> Optional[dict]:
        print(f"[WellnessSynthesis] synthesizing {name}...")

        signals        = _read_wellness(name) or {}
        behaviors      = _read_behaviors(name) or {}
        knowledge      = _read_knowledge(name)
        gizmo_episodes = _read_gizmo_episodes(name)

        total = sum(len(v) for v in signals.values() if isinstance(v, list))
        print(f"[WellnessSynthesis] {name}: {total} signals, behaviors: {bool(behaviors)}, knowledge: {bool(knowledge)}")

        if total < 1 and not behaviors and not knowledge:
            print(f"[WellnessSynthesis] no data for {name}, skipping")
            return None

        history    = await _compress_signals(name, signals) if signals else {}
        baseline   = await _stage_baseline(name, knowledge, behaviors) or {}
        comparison = await _stage_comparison(name, baseline, behaviors, gizmo_episodes) or {}
        ruptures   = await _stage_ruptures(name, baseline, comparison, history) or {}

        group_results = await asyncio.gather(*[
            _stage_conditions(name, baseline, comparison, ruptures, history, group)
            for group in _CONDITION_GROUPS
        ])
        all_conditions  = [c for group in group_results for c in group]
        clinician_notes = await _stage_clinician(name, baseline, comparison, ruptures, all_conditions, history)

        classification = {
            "last_synthesized": datetime.now(timezone.utc).isoformat(),
            "observations":     total,
            "baseline":         baseline,
            "ruptures":         ruptures.get("ruptures", []),
            "healthy_patterns": ruptures.get("healthy_patterns", []),
            "conditions":       all_conditions,
            "clinician_notes":  clinician_notes,
        }

        _write_classification(name, classification)
        return classification

    async def run(self) -> dict:
        log_event("WellnessSynthesis", "START")
        names   = _list_wellness_files()
        results = {}
        print(f"[WellnessSynthesis] found {len(names)} names: {names}")
        for name in names:
            result = await self.synthesize_one(name)
            results[name] = "synthesized" if result else "skipped"
        log_event("WellnessSynthesis", "COMPLETE", processed=len(results))
        print(f"[WellnessSynthesis] complete: {results}")
        return results


wellness_synthesis = WellnessSynthesis()
