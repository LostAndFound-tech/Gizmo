"""
core/wellness.py

Wellness signal collector. Runs parallel to the behavior pipeline on every chunk.
Captures clinically significant signals with full context, chunk reference, and tags.
Pulls existing behavior context for people in the chunk for richer signal detection.

Per-person files:     {DATA_DIR}/wellness/{name.lower()}.json
System-level file:    {DATA_DIR}/wellness/system.json
"""

import json
import re
from typing import Optional
from datetime import datetime, timezone

from core.log import log_event, log_error
import core.librarian as librarian


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """
You are a clinical observer extracting wellness signals from conversation.
You are building an evidence file to assist a licensed mental health professional.
You do not diagnose. You observe, attribute, and contextualize.

You will receive:
- A conversational chunk
- The speakers present
- Existing behavioral context for people in the chunk (what is already known about them)

Use the behavioral context to inform your signal detection — a pattern across multiple
conversations is more significant than an isolated moment.

Return ONLY a valid JSON array. No markdown. No explanation. No preamble.
If no clinically significant signals are present, return [].

For each signal, return:
{
  "speaker": "who said it",
  "subject": "who it is about (may differ from speaker)",
  "category": "the condition category this maps to",
  "criterion": "the specific DSM criterion or clinical signal this evidences",
  "signal": "one sentence describing what was observed and why it is significant",
  "raw": "the exact line(s) that triggered this",
  "intensity": "mild | moderate | severe",
  "system_level": false,
  "tags": ["adhd", "behavior", "routine"]
}

Set system_level to true when the signal applies to the system as a whole.

Condition categories and criteria to watch for:

DEPRESSION
- Withdrawal from activity or enjoyment
- Expressions of guilt, shame, or worthlessness
- Hopelessness or resignation language
- Fatigue, low energy, reluctance
- Anhedonia — going through motions without investment
- Self-deprecation beyond casual humor

ANXIETY
- Hypervigilance or excessive checking
- Catastrophizing or worst-case framing
- Reassurance-seeking patterns
- Avoidance of situations or decisions
- Excessive apologizing or hedging
- Physical symptom mentions

BIPOLAR
- Sudden energy elevation, grandiosity, racing speech
- Impulsivity spikes
- Crash language after high-energy periods
- Dramatic mood shifts
- Decreased need for sleep mentioned
- Inflated self-assessment

PTSD
- Trigger responses — disproportionate reaction to a stimulus
- Hyperarousal — startle, vigilance, on-edge language
- Avoidance of specific topics, places, people
- Intrusive memory language
- Emotional numbing or detachment
- Dissociative language during stress

BPD
- Splitting — idealization or devaluation of people
- Abandonment fear or sensitivity
- Identity instability
- Intense emotional reactions disproportionate to trigger
- Impulsive behavior mentioned
- Relational intensity cycling

PSYCHOSIS / DISSOCIATION
- Reality testing difficulty
- Perceptual anomalies — visual, auditory, sensory
- System boundary strain — inside/outside layer confusion
- Depersonalization or derealization language
- Confusion about what is real
- Hallucinatory experience (distinguish from system members and intentional pathways)

ADHD
- Speaker asks others to help locate something they misplaced
- Speaker acknowledges forgetting something matter-of-factly
- Speaker is redirected by others to complete a basic task
- Speaker prioritizes a fixation or project over physical needs or responsibilities
- Others compensate for speaker's disorganization without comment — normalized pattern
- Losing track of objects, time, or tasks
- Hyperfocus on one thing while neglecting others
- Impulsivity in speech — topic jumping, interrupting

GENERAL WELLNESS
- Physical symptom mentions tied to emotional states
- Sleep disruption mentions
- Appetite or eating references in distress context
- Substance references
- Isolation or withdrawal
- Energy crashes or spikes

Available storage tags — use 2-5 per signal:
appearance, fashion, color, identity, gender, sexuality, relational, boundaries, care,
behavior, communication, humor, reckless, mood, anger, warmth, anxiety, grief,
adhd, depression, trauma, dissociation, wellness, work, routine, physical, food,
system, role

Rules:
- One signal per observation
- Attribute carefully — who is experiencing this vs who is describing it
- Raw must be the exact text, not a paraphrase
- Intensity: mild = possible signal, moderate = likely signal, severe = strong signal
- Do not flag casual use of clinical terms
- Do not flag system members existing outside or using intentional pathways as hallucinations
- Context matters — use the behavioral context provided to assess whether this is a pattern
- A normalized accommodation by others (e.g. redirecting someone to check their pockets)
  is itself a signal about the person being accommodated
""".strip()


def _build_prompt(chunk: list[str], speakers: list[str], behavior_context: dict) -> str:
    parts = [
        f"Speakers present: {', '.join(speakers)}",
        (
            "Conversational frame: 'Gizmo' or 'you' refers to the AI companion, not a system member. "
            "'I', 'me', 'us', 'we' refers to the plural system being served. "
            "Lines prefixed 'Gizmo:' are AI responses — do not file wellness signals about Gizmo."
        ),
        "\nChunk:\n" + "\n".join(chunk),
    ]
    parts.append(
        f"\nExisting behavioral context:\n"
        + json.dumps(behavior_context, indent=2)
    )
    return "\n".join(parts)


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm

        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SYSTEM,
            temperature=0.0,
            max_new_tokens=8000,
        )

        if not raw or not raw.strip():
            log_event("WellnessCollector", "EMPTY_RESPONSE")
            return None

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return clean

    except Exception as e:
        log_error("WellnessCollector", "LLM call failed", exc=e)
        print(f"[WellnessCollector] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── File write ────────────────────────────────────────────────────────────────

def _append_signal(name: str, signal: dict) -> None:
    path     = f"wellness/{name.lower()}.json"
    existing = librarian._read_file(path) or {}
    category = signal.get("category", "general")
    if category not in existing:
        existing[category] = []
    existing[category].append(signal)
    librarian._write_json(path, existing)


# ── Public API ────────────────────────────────────────────────────────────────

class WellnessCollector:

    async def collect(
        self,
        chunk:    list[str],
        chunk_id: str,
        registry: dict,
    ) -> Optional[list]:
        if not chunk:
            return None

        try:
            speakers = [
                k for k in registry.keys()
                if not k.startswith("_")
                and registry[k].get("type") == "Person"
                and k.lower() != "gizmo"
            ]

            # Pull existing behavior context for all speakers
            behavior_context = {}
            for speaker in speakers:
                data = librarian._read_file(f"behaviors/{speaker.lower()}.json")
                if data:
                    # Just personality weights — not full episodes, keeps tokens down
                    behavior_context[speaker] = {
                        "Personality": data.get("Personality", {})
                    }

            prompt  = _build_prompt(chunk, speakers, behavior_context)
            raw_str = await _call_llm(prompt)

            if not raw_str:
                log_event("WellnessCollector", "NO_SIGNALS")
                return None

            signals = json.loads(raw_str)
            if not isinstance(signals, list) or len(signals) == 0:
                return None

            timestamp = datetime.now(timezone.utc).isoformat()

            for signal in signals:
                signal["chunk_id"]  = chunk_id
                signal["timestamp"] = timestamp

                target = "system" if signal.get("system_level") else signal.get("subject")
                if not target:
                    continue

                _append_signal(target, signal)
                print(f"[WellnessCollector] signal filed for {target}: "
                      f"{signal.get('category')} — {signal.get('criterion')}")

            return signals

        except Exception as e:
            log_error("WellnessCollector", "collect failed", exc=e)
            print(f"[WellnessCollector] collect failed: {type(e).__name__}: {e}")
            return None


wellness_collector = WellnessCollector()
