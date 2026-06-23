"""
core/wellness_synthesis.py

Weekly (or on-demand) wellness synthesis agent.

Architecture — sequential reasoning chain:
  Stage 1: Baseline      — who they believe themselves to be (opinions, preferences, self-model)
  Stage 2: Comparison    — where observed behavior matches or contradicts the self-model
  Stage 3: Ruptures      — which contradictions are clinically significant
  Stage 4: Conditions    — DSM evaluation grounded in the full picture (4 parallel passes)
  Stage 5: Clinician     — prose handoff synthesizing all stages

Each stage feeds the next. This is reasoning, not aggregation.

Classification files: {DATA_DIR}/wellness/classifications/{name}.json
Archive:              {DATA_DIR}/wellness/classifications/archive/{name}_{ts}.json

Triggers:
    await wellness_synthesis.run()                 # all people
    await wellness_synthesis.synthesize_one("ara") # one person
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
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

_BASELINE_SYSTEM = """
You are building a self-portrait of one person based on what they have expressed about themselves.

This is not a clinical assessment. This is: who does this person believe they are?
What do they value? What do they expect from themselves and others?
What is their stated self-model?

You will receive their known preferences, opinions, and internal space data.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "self_model": "2-3 sentence summary of how this person understands themselves",
  "core_beliefs": [
    "belief about self or world, stated as a declarative sentence"
  ],
  "stated_values": [
    "what they say they care about or prioritize"
  ],
  "relational_expectations": [
    "what they expect from relationships or how they expect to be treated"
  ],
  "identity_anchors": [
    "stable identity claims they return to — how they define themselves"
  ]
}

Rules:
- Only include what is supported by the data — never invent
- Use their own framing where possible
- Distinguish plural system awareness from pathology
- If a field has no support, omit it entirely
""".strip()

_COMPARISON_SYSTEM = """
You are comparing a person's stated self-model against their observed behavior.

You will receive:
- Their baseline self-portrait (what they believe about themselves)
- Their behavioral data (what has actually been observed)
- Gizmo's relational observations (companion notes from unguarded moments)

Your job: where does behavior match the self-model, and where does it contradict it?

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "alignments": [
    {
      "belief": "the stated belief or value",
      "behavior": "the observed behavior that matches it",
      "confidence": "how consistently this alignment appears"
    }
  ],
  "contradictions": [
    {
      "belief": "the stated belief or value",
      "behavior": "the observed behavior that contradicts it",
      "frequency": "how often this contradiction appears",
      "notes": "any relevant context — is the contradiction distressing? defended? unconscious?"
    }
  ],
  "blind_spots": [
    "things the behavioral data reveals that the person does not appear to acknowledge about themselves"
  ]
}

Rules:
- A contradiction is only worth flagging if it appears more than once or carries emotional weight
- Not all contradictions are pathological — note which ones seem distressing vs. adaptive
- Gizmo's observations are first-person companion notes, not clinical assessments — weight them accordingly
- Distinguish plural system dynamics from individual pathology
""".strip()

_RUPTURES_SYSTEM = """
You are identifying which behavioral contradictions are clinically significant.

You will receive:
- The self-model baseline
- The comparison map (alignments and contradictions)
- Wellness signals from conversations

A rupture is clinically significant when:
- It causes observable distress
- It drives compulsive or repeated behavior the person cannot stop
- It reveals something the person cannot see about themselves
- It maps to a known clinical pattern
- It appears across multiple contexts (not just one conversation)

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "ruptures": [
    {
      "description": "what the rupture is — stated clearly",
      "belief_side": "what they believe about themselves",
      "behavior_side": "what they actually do",
      "significance": "why this is clinically relevant",
      "severity": "mild | moderate | significant",
      "evidence": ["brief evidence references"],
      "clinical_signal": "which condition or pattern this most resembles, if any"
    }
  ],
  "healthy_patterns": [
    "contradictions or tensions that appear adaptive, not pathological"
  ]
}

Rules:
- Be precise — vague ruptures are not useful
- Mild: interesting but not urgent. Moderate: worth tracking. Significant: warrants clinical attention.
- Do not pathologize plural system function, kink identity, or non-normative but consensual behavior
- A person can have both healthy and unhealthy versions of the same pattern
""".strip()

_CONDITIONS_SYSTEM = """
You are a clinical synthesis agent working on behalf of a licensed mental health professional.
You evaluate evidence for specific DSM conditions only.
You do not diagnose. You build evidence-based case files for professional review.

You will receive:
- The person's self-model baseline
- A map of where behavior matches or contradicts that self-model
- Identified ruptures with clinical significance ratings
- Wellness signals from conversations

Evaluate ONLY the conditions listed. Use the rupture map as your primary lens —
the most clinically significant signals are where self-model and behavior diverge.

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
        "rupture_link": "optional — which rupture this connects to",
        "examples": [
          {
            "raw": "exact line or observation that evidenced this",
            "context": "one sentence explaining why this evidences the criterion"
          }
        ]
      }
    ],
    "criteria_absent": ["criterion name"],
    "pattern_notes": "Narrative note about the overall pattern, grounded in the rupture map"
  }
]

Rules:
- Only include conditions where you have at least 2 pieces of supporting evidence
- Conditions with 1 signal: return with confidence "low" and status "monitoring"
- Conditions with 0 signals: omit entirely
- Always distinguish plural system function from pathology
- Intentional hallucinatory pathways are system function, not symptoms
- Always include criteria_absent for flagged conditions
- Anchor pattern_notes to the rupture map where possible
""".strip()

_CLINICIAN_SYSTEM = """
You are a mental health consultant preparing a handoff note for a licensed clinician
who is about to meet this individual for the first time.

You may not diagnose. You may make any recommendations.

Write in clear, direct clinical prose. No JSON. No headers. No bullet points.
This is a note a professional reads before walking into a first session.

You will receive the full reasoning chain:
- Who this person believes themselves to be
- Where their behavior matches and contradicts that self-model
- Which contradictions are clinically significant (the ruptures)
- Condition evaluations grounded in the rupture map
- Raw wellness signals and behavioral data

Cover:
- The core tension in this person — what is the central rupture between self-model and behavior
- What the evidence shows clearly vs where it is thin
- What warrants attention and why
- What appears healthy or functional — including things that look pathological but aren't
- What remains ambiguous and needs further exploration
- Specific recommendations for the clinician going in

Distinguish plural system function from pathology throughout.
Intentional perceptual pathways and system multiplicity are not symptoms.
Non-normative but consensual relational or sexual dynamics are not pathology.
The rupture map is your primary lens — lead with what matters most.
""".strip()


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str, system: str, max_tokens: int = 4000) -> Optional[str]:
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
        try:
            last_brace = max(raw.rfind("}"), raw.rfind("]"))
            if last_brace > 0:
                return json.loads(raw[:last_brace + 1])
        except Exception:
            pass
        return None


# ── File helpers ──────────────────────────────────────────────────────────────

def _list_wellness_files() -> list[str]:
    names = set()
    for subfolder in ("wellness", "behaviors"):
        folder = librarian._full_path(subfolder)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".json") and fname != "classifications":
                names.add(fname[:-5])
    print(f"[WellnessSynthesis] found names: {names}")
    return list(names)


def _read_wellness(name: str) -> Optional[dict]:
    return librarian._read_file(f"wellness/{name}.json")


def _read_behaviors(name: str) -> Optional[dict]:
    return librarian._read_file(f"behaviors/{name}.json")


def _read_prior(name: str) -> Optional[dict]:
    return librarian._read_file(f"wellness/classifications/{name}.json")


def _read_knowledge(name: str) -> dict:
    topics = ("preferences", "opinions", "internal_space", "relationships", "history", "routines")
    knowledge = {}
    for topic in topics:
        data = librarian._read_file(f"knowledge/{name.lower()}/{topic}.json")
        if data and isinstance(data, list) and len(data) > 0:
            trimmed = [
                {k: v for k, v in entry.items()
                 if k in ("fact", "tags", "confidence", "internal")}
                for entry in data[-30:]
            ]
            knowledge[topic] = trimmed
    return knowledge


def _read_gizmo_episodes(name: str) -> list:
    gizmo_self  = librarian._read_file("behaviors/gizmo_self.json") or {}
    person_data = gizmo_self.get(name) or gizmo_self.get(name.lower()) or {}
    episodes    = person_data.get("episodes", [])
    return [
        {k: v for k, v in ep.items()
         if k in ("assessment", "what_landed", "what_missed", "punch_bowl",
                  "situational_tags", "closeness_at_time")}
        for ep in episodes[-20:]
    ]


def _write_classification(name: str, classification: dict) -> None:
    existing = librarian._read_file(f"wellness/classifications/{name}.json")
    if existing:
        ts       = existing.get("last_synthesized", datetime.now(timezone.utc).isoformat())
        ts_clean = ts.replace(":", "-").replace(".", "-")[:19]
        librarian._write_json(
            f"wellness/classifications/archive/{name}_{ts_clean}.json", existing
        )
        print(f"[WellnessSynthesis] archived previous classification for {name}")
    librarian._write_json(f"wellness/classifications/{name}.json", classification)
    print(f"[WellnessSynthesis] classification written for {name}")


# ── Stage 1: Baseline ─────────────────────────────────────────────────────────

async def _stage_baseline(name: str, knowledge: dict) -> Optional[dict]:
    if not knowledge:
        return None

    prompt = (
        f"Individual: {name}\n\n"
        f"Known preferences, opinions, and self-description:\n"
        + json.dumps(knowledge, indent=2)
    )

    raw = await _call_llm(prompt, _BASELINE_SYSTEM, max_tokens=1000)
    if not raw:
        return None
    result = _safe_parse(raw, f"{name}/baseline")
    return result if isinstance(result, dict) else None


# ── Stage 2: Comparison ───────────────────────────────────────────────────────

async def _stage_comparison(
    name:           str,
    baseline:       Optional[dict],
    behaviors:      dict,
    gizmo_episodes: list,
) -> Optional[dict]:
    if not baseline and not behaviors:
        return None

    prompt_parts = [f"Individual: {name}\n"]

    if baseline:
        prompt_parts.append(
            f"Self-model baseline:\n{json.dumps(baseline, indent=2)}"
        )

    if behaviors:
        # Top 20 weighted traits + last 5 episodes
        personality = behaviors.get("Personality", {})
        top_traits  = dict(
            sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:20]
        )
        episodes = behaviors.get("Episodes", [])[-5:]
        prompt_parts.append(
            f"\nObserved behavioral patterns:\n"
            + json.dumps({"top_traits": top_traits, "recent_episodes": episodes}, indent=2)
        )

    if gizmo_episodes:
        prompt_parts.append(
            f"\nGizmo's relational observations:\n{json.dumps(gizmo_episodes, indent=2)}"
        )

    raw = await _call_llm("\n".join(prompt_parts), _COMPARISON_SYSTEM, max_tokens=2000)
    if not raw:
        return None
    result = _safe_parse(raw, f"{name}/comparison")
    return result if isinstance(result, dict) else None


# ── Stage 3: Ruptures ─────────────────────────────────────────────────────────

async def _stage_ruptures(
    name:       str,
    baseline:   Optional[dict],
    comparison: Optional[dict],
    signals:    dict,
) -> Optional[dict]:
    if not comparison:
        return None

    prompt_parts = [f"Individual: {name}\n"]

    if baseline:
        prompt_parts.append(f"Self-model:\n{json.dumps(baseline, indent=2)}")

    prompt_parts.append(
        f"\nBehavior comparison:\n{json.dumps(comparison, indent=2)}"
    )

    if signals:
        prompt_parts.append(
            f"\nWellness signals:\n{json.dumps(signals, indent=2)}"
        )

    raw = await _call_llm("\n".join(prompt_parts), _RUPTURES_SYSTEM, max_tokens=2000)
    if not raw:
        return None
    result = _safe_parse(raw, f"{name}/ruptures")
    return result if isinstance(result, dict) else None


# ── Stage 4: Conditions (parallel) ───────────────────────────────────────────

async def _stage_conditions_group(
    group:      dict,
    name:       str,
    baseline:   Optional[dict],
    comparison: Optional[dict],
    ruptures:   Optional[dict],
    signals:    dict,
) -> list:
    prompt_parts = [
        f"Individual: {name}\n",
        f"Evaluate ONLY these conditions: {', '.join(group['conditions'])}\n",
        f"DSM criteria:\n{group['criteria']}\n",
    ]

    if baseline:
        prompt_parts.append(f"Self-model baseline:\n{json.dumps(baseline, indent=2)}\n")

    if comparison:
        prompt_parts.append(f"Behavior comparison map:\n{json.dumps(comparison, indent=2)}\n")

    if ruptures:
        prompt_parts.append(f"Identified ruptures:\n{json.dumps(ruptures, indent=2)}\n")

    if signals:
        prompt_parts.append(f"Wellness signals:\n{json.dumps(signals, indent=2)}")

    raw = await _call_llm("\n".join(prompt_parts), _CONDITIONS_SYSTEM, max_tokens=3000)
    if not raw:
        return []
    result = _safe_parse(raw, f"{name}/{group['name']}")
    return result if isinstance(result, list) else []


# ── Stage 5: Clinician notes ──────────────────────────────────────────────────

async def _stage_clinician_notes(
    name:           str,
    baseline:       Optional[dict],
    comparison:     Optional[dict],
    ruptures:       Optional[dict],
    conditions:     list,
    signals:        dict,
    behaviors:      dict,
    dynamic_brief:  Optional[str],
) -> str:
    prompt_parts = [f"Individual: {name}\n"]

    if baseline:
        prompt_parts.append(f"Self-model baseline:\n{json.dumps(baseline, indent=2)}\n")

    if comparison:
        prompt_parts.append(f"Behavior comparison map:\n{json.dumps(comparison, indent=2)}\n")

    if ruptures:
        prompt_parts.append(f"Identified ruptures:\n{json.dumps(ruptures, indent=2)}\n")

    prompt_parts.append(f"Condition evaluations:\n{json.dumps(conditions, indent=2)}\n")

    if signals:
        prompt_parts.append(f"Wellness signals:\n{json.dumps(signals, indent=2)}\n")

    if behaviors:
        personality = behaviors.get("Personality", {})
        top_traits  = dict(
            sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:10]
        )
        prompt_parts.append(f"Top behavioral traits:\n{json.dumps(top_traits, indent=2)}\n")

    if dynamic_brief:
        prompt_parts.append(f"Longitudinal dynamic history:\n{dynamic_brief}")

    raw = await _call_llm("\n".join(prompt_parts), _CLINICIAN_SYSTEM, max_tokens=2000)
    if not raw:
        return "Clinician notes pass failed."
    return re.sub(r"```(?:json)?|```", "", raw).strip()


# ── Synthesis ─────────────────────────────────────────────────────────────────

class WellnessSynthesis:

    async def synthesize_one(self, name: str) -> Optional[dict]:
        print(f"[WellnessSynthesis] synthesizing {name}...")

        signals        = _read_wellness(name) or {}
        behaviors      = _read_behaviors(name) or {}
        knowledge      = _read_knowledge(name)
        gizmo_episodes = _read_gizmo_episodes(name)
        total          = sum(len(v) for v in signals.values() if isinstance(v, list))

        print(
            f"[WellnessSynthesis] {name}: {total} signals, "
            f"knowledge topics: {list(knowledge.keys())}, "
            f"gizmo episodes: {len(gizmo_episodes)}"
        )

        if total < 1 and not behaviors and not knowledge and not gizmo_episodes:
            print(f"[WellnessSynthesis] no data for {name}, skipping")
            return None

        prior = _read_prior(name)

        # ── Stage 1: Baseline ─────────────────────────────────────────────────
        print(f"[WellnessSynthesis] stage 1: baseline for {name}")
        baseline = await _stage_baseline(name, knowledge)

        # ── Stage 2: Comparison ───────────────────────────────────────────────
        print(f"[WellnessSynthesis] stage 2: comparison for {name}")
        comparison = await _stage_comparison(name, baseline, behaviors, gizmo_episodes)

        # ── Stage 3: Ruptures ─────────────────────────────────────────────────
        print(f"[WellnessSynthesis] stage 3: ruptures for {name}")
        ruptures = await _stage_ruptures(name, baseline, comparison, signals)

        # ── Stage 4: Conditions — parallel ───────────────────────────────────
        print(f"[WellnessSynthesis] stage 4: conditions for {name}")
        group_results = await asyncio.gather(*[
            _stage_conditions_group(group, name, baseline, comparison, ruptures, signals)
            for group in _CONDITION_GROUPS
        ])
        all_conditions = [c for group in group_results for c in group]

        # ── Stage 5: Clinician notes ──────────────────────────────────────────
        print(f"[WellnessSynthesis] stage 5: clinician notes for {name}")
        try:
            from core.dynamic_reader import get_longitudinal_brief
            dynamic_brief = get_longitudinal_brief(name)
        except Exception:
            dynamic_brief = None

        clinician_notes = await _stage_clinician_notes(
            name=name,
            baseline=baseline,
            comparison=comparison,
            ruptures=ruptures,
            conditions=all_conditions,
            signals=signals,
            behaviors=behaviors,
            dynamic_brief=dynamic_brief,
        )

        # ── Build final classification ────────────────────────────────────────
        classification = {
            "last_synthesized": datetime.now(timezone.utc).isoformat(),
            "observations":     total,
            "baseline":         baseline,
            "comparison":       comparison,
            "ruptures":         ruptures,
            "conditions":       all_conditions,
            "clinician_notes":  clinician_notes,
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
