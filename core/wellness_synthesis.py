"""
core/wellness_synthesis.py

Weekly (or on-demand) wellness synthesis agent.

Architecture:
  - 4 parallel condition group passes (2 conditions each)
  - 1 final assembly pass (system patterns + clinician notes)
  - Results merged into one classification file

Classification files: {DATA_DIR}/wellness/classifications/{name}.json
Archive:              {DATA_DIR}/wellness/classifications/archive/{name}_{ts}.json

Triggers:
    await wellness_synthesis.run()              # all people
    await wellness_synthesis.synthesize_one("ara")  # one person
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
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


# ── Prompts ───────────────────────────────────────────────────────────────────

_GROUP_SYSTEM = """
You are a clinical synthesis agent working on behalf of a licensed mental health professional.
You evaluate evidence for specific DSM conditions only.
You do not diagnose. You build evidence-based case files for professional review.

You will receive wellness signals and behavioral data for one individual.
Evaluate ONLY the conditions listed. Ignore all others.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

Return exactly this structure — an array of condition objects:
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
        "examples": [
          {
            "chunk_id": "sess_abc-001",
            "raw": "exact line that evidenced this",
            "context": "one sentence explaining why this evidences the criterion"
          }
        ]
      }
    ],
    "criteria_absent": ["criterion name"],
    "pattern_notes": "Narrative note about the overall pattern"
  }
]

Rules:
- Only include conditions where you have at least 2 pieces of supporting evidence
- Conditions with 1 signal: return them with confidence "low" and status "monitoring"
- Conditions with 0 signals: omit entirely
- Always distinguish plural system function from pathology
- Ego-syntonic perceptual experiences differ from psychotic hallucination
- Intentional hallucinatory pathways are system function, not symptoms
- Always include criteria_absent for flagged conditions
""".strip()

_ASSEMBLY_SYSTEM = """
You are a clinical synthesis agent working on behalf of a licensed mental health professional.
You receive pre-evaluated condition results and behavioral data for one individual.
Your job is to identify system-level patterns and write the clinician handoff notes.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

Return exactly this structure:
{
  "system_patterns": [
    {
      "pattern": "pattern name",
      "confidence": "low | moderate | high",
      "evidence_count": 3,
      "notes": "narrative description"
    }
  ],
  "clinician_notes": "Overall synthesis narrative. What stands out. What warrants attention. What appears healthy. What remains ambiguous. Written as a handoff to a professional walking in for the first time."
}

Rules:
- system_patterns: recurring dynamics that are clinically relevant but not pathological
- Distinguish plural system function from pathology throughout
- clinician_notes should be the most useful thing a clinician reads before a first session
- Note what evidence is strong vs thin
- Note what warrants further exploration
""".strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str, system: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
            temperature=0.0,
            max_new_tokens=8000,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("WellnessSynthesis", "LLM call failed", exc=e)
        print(f"[WellnessSynthesis] LLM call failed: {type(e).__name__}: {e}")
        return None


def _safe_parse(raw: str, name: str) -> Optional[dict | list]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WellnessSynthesis] JSON malformed for {name}, attempting repair...")
        try:
            last_brace = max(raw.rfind("}"), raw.rfind("]"))
            if last_brace > 0:
                return json.loads(raw[:last_brace + 1])
        except Exception:
            pass
        return None


# ── File helpers ──────────────────────────────────────────────────────────────

def _list_wellness_files() -> list[str]:
    """
    Find all names with wellness data.
    New structure: wellness/{name}/ subdirectories
    Also checks behaviors/ for names with behavioral data only.
    Skips classifications/ and system/.
    """
    names = set()

    # New structure — wellness/{name}/ directories
    wellness_root = librarian._full_path("wellness")
    if os.path.isdir(wellness_root):
        for entry in os.listdir(wellness_root):
            full = os.path.join(wellness_root, entry)
            if os.path.isdir(full) and entry not in ("classifications", "system"):
                names.add(entry)

    # Legacy flat files — wellness/{name}.json (old pipeline)
    if os.path.isdir(wellness_root):
        for fname in os.listdir(wellness_root):
            if fname.endswith(".json") and fname not in ("classifications",):
                names.add(fname[:-5])

    # Behaviors — names with behavioral data but maybe no wellness yet
    behaviors_root = librarian._full_path("behaviors")
    if os.path.isdir(behaviors_root):
        for fname in os.listdir(behaviors_root):
            if fname.endswith(".json"):
                names.add(fname[:-5])

    # Strip known non-person names
    names.discard("gizmo")
    names.discard("system")

    print(f"[WellnessSynthesis] found names: {names}")
    return list(names)


def _read_wellness(name: str) -> dict:
    """
    Read all wellness signals for a person.
    New structure: wellness/{name}/{domain}.json — merged into one dict keyed by domain.
    Falls back to legacy flat file wellness/{name}.json if no directory exists.
    Returns {} if nothing found.
    """
    # New structure — per-domain files in wellness/{name}/
    person_dir = librarian._full_path(f"wellness/{name.lower()}")
    if os.path.isdir(person_dir):
        merged = {}
        for fname in os.listdir(person_dir):
            if not fname.endswith(".json"):
                continue
            domain = fname[:-5]
            data   = librarian._read_file(f"wellness/{name.lower()}/{fname}")
            if data and isinstance(data.get("signals"), list):
                merged[domain] = data["signals"]
        if merged:
            return merged

    # Legacy fallback — flat wellness/{name}.json
    legacy = librarian._read_file(f"wellness/{name}.json")
    if legacy:
        return legacy

    return {}


def _read_behaviors(name: str) -> Optional[dict]:
    return librarian._read_file(f"behaviors/{name.lower()}.json")


def _read_prior(name: str) -> Optional[dict]:
    return librarian._read_file(f"wellness/classifications/{name.lower()}.json")


def _write_classification(name: str, classification: dict) -> None:
    n = name.lower()
    existing = librarian._read_file(f"wellness/classifications/{n}.json")
    if existing:
        ts = existing.get("last_synthesized", datetime.now(timezone.utc).isoformat())
        ts_clean = ts.replace(":", "-").replace(".", "-")[:19]
        librarian._write_json(f"wellness/classifications/archive/{n}_{ts_clean}.json", existing)
        print(f"[WellnessSynthesis] archived previous classification for {n}")
    librarian._write_json(f"wellness/classifications/{n}.json", classification)
    print(f"[WellnessSynthesis] classification written for {n}")


# ── Group pass ────────────────────────────────────────────────────────────────

async def _run_group(
    group:    dict,
    name:     str,
    signals:  dict,
    behaviors: dict,
) -> list:
    prompt = (
        f"Individual: {name}\n\n"
        f"Evaluate ONLY these conditions: {', '.join(group['conditions'])}\n\n"
        f"DSM criteria for these conditions:\n{group['criteria']}\n\n"
        f"Wellness signals:\n{json.dumps(signals, indent=2)}\n\n"
        f"Behavioral data:\n{json.dumps(behaviors, indent=2)}"
    )
    raw = await _call_llm(prompt, _GROUP_SYSTEM)
    if not raw:
        return []
    result = _safe_parse(raw, f"{name}/{group['name']}")
    if isinstance(result, list):
        return result
    return []


# ── Assembly pass ─────────────────────────────────────────────────────────────

async def _run_assembly(
    name:       str,
    conditions: list,
    signals:    dict,
    behaviors:  dict,
    prior:      Optional[dict],
) -> dict:
    prompt = (
        f"Individual: {name}\n\n"
        f"Pre-evaluated conditions:\n{json.dumps(conditions, indent=2)}\n\n"
        f"Wellness signals:\n{json.dumps(signals, indent=2)}\n\n"
        f"Behavioral data:\n{json.dumps({'Personality': behaviors.get('Personality', {})}, indent=2)}"
    )
    if prior:
        prompt += f"\n\nPrior classification (build on this):\n{json.dumps(prior, indent=2)}"

    raw = await _call_llm(prompt, _ASSEMBLY_SYSTEM)
    if not raw:
        return {"system_patterns": [], "clinician_notes": "Assembly pass failed."}
    result = _safe_parse(raw, f"{name}/assembly")
    if isinstance(result, dict):
        return result
    return {"system_patterns": [], "clinician_notes": "Assembly parse failed."}


# ── Synthesis ─────────────────────────────────────────────────────────────────

class WellnessSynthesis:

    async def synthesize_one(self, name: str) -> Optional[dict]:
        print(f"[WellnessSynthesis] synthesizing {name}...")

        signals   = _read_wellness(name) or {}
        behaviors = _read_behaviors(name) or {}
        # Count all signals across all domains
        total = sum(
            len(v) for v in signals.values()
            if isinstance(v, list)
        )

        print(f"[WellnessSynthesis] {name}: {total} wellness signals, behaviors: {bool(behaviors)}")

        if total < 1 and not behaviors:
            print(f"[WellnessSynthesis] no data for {name}, skipping")
            return None

        prior = _read_prior(name)

        # ── Parallel condition group passes ───────────────────────────────────
        group_results = await asyncio.gather(*[
            _run_group(group, name, signals, behaviors)
            for group in _CONDITION_GROUPS
        ])

        # Flatten all condition results
        all_conditions = [c for group in group_results for c in group]

        # ── Assembly pass ─────────────────────────────────────────────────────
        assembly = await _run_assembly(name, all_conditions, signals, behaviors, prior)

        # ── Build final classification ────────────────────────────────────────
        classification = {
            "last_synthesized": datetime.now(timezone.utc).isoformat(),
            "observations":     total,
            "conditions":       all_conditions,
            "system_patterns":  assembly.get("system_patterns", []),
            "clinician_notes":  assembly.get("clinician_notes", ""),
        }

        _write_classification(name, classification)
        return classification

    async def run(self) -> dict:
        log_event("WellnessSynthesis", "START")
        names   = _list_wellness_files()
        results = {}

        print(f"[WellnessSynthesis] found {len(names)} files: {names}")

        for name in names:
            result = await self.synthesize_one(name)
            results[name] = "synthesized" if result else "skipped"

        log_event("WellnessSynthesis", "COMPLETE", processed=len(results))
        print(f"[WellnessSynthesis] complete: {results}")
        return results


wellness_synthesis = WellnessSynthesis()