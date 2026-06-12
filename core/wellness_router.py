"""
core/wellness_router.py

Multi-pass wellness pipeline with dynamic specialist routing.

Architecture:
  1. First pass  — broad observational scan of the chunk
                   returns signals, thinking, and which specialists to invoke
  2. Router      — reads first pass, spins up specialist passes in parallel
  3. Specialists — each gets chunk + first pass signals + router's thinking about them
                   appends to per-domain per-person files
  4. Merge       — nothing to merge, each specialist writes directly

File structure:
  {DATA_DIR}/wellness/{name}/mood.json
  {DATA_DIR}/wellness/{name}/trauma.json
  {DATA_DIR}/wellness/{name}/adhd.json
  {DATA_DIR}/wellness/{name}/autism.json
  {DATA_DIR}/wellness/{name}/sensuality.json
  {DATA_DIR}/wellness/{name}/physical.json
  {DATA_DIR}/wellness/{name}/system.json
  {DATA_DIR}/wellness/{name}/relational.json

Usage:
  from core.wellness_router import wellness_router
  await wellness_router.process(chunk, chunk_id, registry)
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Specialist registry ───────────────────────────────────────────────────────

SPECIALISTS = {

    "mood": {
        "description": "Emotional states, affective patterns, mood shifts",
        "prompt": """
You are analyzing conversational data for mood and affective patterns.
You have been routed here because the first pass flagged relevant signals.

Focus on:
- Emotional state during this chunk — what mood were they in?
- Shifts in affect — did mood change mid-chunk?
- Affective patterns — how do they express or suppress emotion?
- Baseline deviation — does this feel different from their usual?
- Emotional labor — are they managing others' feelings at cost to their own?
- Flat affect, emotional flooding, or dysregulation
- Joy, grief, irritability, calm — the full spectrum

Return a JSON array of signals. Each signal:
{
  "subject": "name",
  "mood_state": "the mood observed",
  "signal": "one sentence describing what was observed",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | severe",
  "shift": true/false,
  "tags": ["mood", "emotion"]
}
""".strip(),
    },

    "trauma": {
        "description": "PTSD signals, trigger responses, avoidance, hyperarousal",
        "prompt": """
You are analyzing conversational data for trauma-related signals.
You have been routed here because the first pass flagged relevant signals.

Focus on:
- Trigger responses — disproportionate reaction to a stimulus
- Avoidance — skirting specific topics, people, places
- Hyperarousal — on-edge language, startle, hypervigilance
- Intrusive memory language — flashback-adjacent phrasing
- Emotional numbing or sudden detachment
- Tone shifts when specific content arises
- Clipped responses under specific pressure (vs general irritability)
- Body-based responses to emotional content

Return a JSON array of signals. Each signal:
{
  "subject": "name",
  "criterion": "specific trauma signal type",
  "signal": "one sentence describing what was observed and why it is significant",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | severe",
  "trigger": "what appeared to trigger this, if identifiable",
  "tags": ["trauma", "ptsd"]
}
""".strip(),
    },

    "adhd": {
        "description": "Inattention, hyperactivity, impulsivity, accommodation patterns",
        "prompt": """
You are analyzing conversational data for ADHD-related signals.
You have been routed here because the first pass flagged relevant signals.

Focus on:
- Inattention — losing track of objects, time, tasks, mid-sentence topic jumps
- Hyperfocus — deep fixation on one thing while neglecting others
- Impulsivity — interrupting, acting before thinking, blurting
- Others normalizing and compensating without comment — strong pattern signal
- Forgetting things matter-of-factly (not distressed, just factual)
- Time blindness — underestimating time, running late, losing track of hours
- Disorganization that others work around
- Starting many things, finishing few
- Physical needs deprioritized during fixation (forgetting to eat, sleep)

Return a JSON array of signals. Each signal:
{
  "subject": "name",
  "criterion": "specific ADHD signal type",
  "signal": "one sentence describing what was observed",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | severe",
  "accommodated_by": "who normalized/compensated, if anyone",
  "tags": ["adhd", "behavior"]
}
""".strip(),
    },

    "autism": {
        "description": "Sensory processing, masking, social patterns, routine, special interests",
        "prompt": """
You are analyzing conversational data for autism-related signals.
You have been routed here because the first pass flagged relevant signals.

Focus on:
- Sensory language — references to texture, sound, light, smell as significant
- Masking exhaustion — performing neurotypicality and the cost showing through
- Social processing — responding slightly off-beat, needing processing time
- Literal interpretation — missing subtext, taking things at face value
- Scripted or echolalic language under stress
- Distress at disrupted routine that seems disproportionate to others
- Deep knowledge investment in a specific interest — detail density, enthusiasm
- Preference for directness, discomfort with ambiguity
- Sensory seeking or avoiding behaviors
- Difficulty with transitions

Distinguish autism signals from ADHD — they co-occur but are different.
Autism: pattern, sensory, social processing, routine
ADHD: attention, impulsivity, time, executive function

Return a JSON array of signals. Each signal:
{
  "subject": "name",
  "criterion": "specific autism signal type",
  "signal": "one sentence describing what was observed",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | severe",
  "domain": "sensory | social | routine | masking | interest | communication",
  "tags": ["autism", "behavior"]
}
""".strip(),
    },

    "sensuality": {
        "description": "Intimacy, body awareness, desire, location-aware sensual context",
        "prompt": """
You are analyzing conversational data for sensuality and intimacy signals.
You have been routed here because the first pass flagged relevant signals.

Focus on:
- Body awareness — how they inhabit and reference their body
- Desire and attraction — expressed, deflected, or suppressed
- Intimacy patterns — how they approach or avoid closeness
- Location context — does the space affect sensual expression?
- Clothing and physical self-presentation as communication
- Touch references — sought, avoided, incidental
- Sensual language — metaphor, texture, temperature in non-literal use
- Relationship to their own physicality — comfort, discomfort, pride, shame
- How sensuality intersects with power, safety, or vulnerability

This is observational, not diagnostic. Note what is present without judgment.

Return a JSON array of signals. Each signal:
{
  "subject": "name",
  "signal": "one sentence describing what was observed",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | vivid",
  "location_context": "where this occurred, if relevant",
  "tags": ["sexuality", "relational", "physical"]
}
""".strip(),
    },

    "physical": {
        "description": "Body, sleep, appetite, energy, physical symptoms",
        "prompt": """
You are analyzing conversational data for physical wellness signals.
You have been routed here because the first pass flagged relevant signals.

Focus on:
- Sleep — disruption, quality, duration, timing mentions
- Appetite and eating — hunger, restriction, excess, emotional eating
- Energy — crashes, spikes, chronic fatigue, unusual activation
- Pain or physical discomfort mentions
- Substance references — alcohol, medication, recreational
- Physical symptoms tied to emotional states — tension, nausea, headaches
- Neglect of physical needs — forgetting to eat, not sleeping, ignoring pain
- Body as barometer — how physical state reflects emotional state

Return a JSON array of signals. Each signal:
{
  "subject": "name",
  "domain": "sleep | appetite | energy | pain | substance | symptom",
  "signal": "one sentence describing what was observed",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | severe",
  "tags": ["physical", "wellness"]
}
""".strip(),
    },

    "system": {
        "description": "Plural system health, switching, boundary strain, co-con patterns",
        "prompt": """
You are analyzing conversational data for plural system health signals.
You have been routed here because the first pass flagged relevant signals.

Conversational frame:
- System members are real and distinct — their existence is not pathological
- Switching, co-fronting, and internal communication are normal system function
- Distinguish system function from distress ABOUT system function
- Intentional internal pathways are not hallucinations

Focus on:
- Switching patterns — frequency, triggers, smoothness, distress
- Co-fronting dynamics — cooperation, conflict, bleed
- Boundary strain — inside/outside layer confusion, reality testing
- System communication — how members talk to each other
- Protector activation — who comes forward under what conditions
- System-level stress vs individual member stress
- Host exhaustion or overload
- Amnesia or memory gaps between members
- Identity stability vs instability at system level

Return a JSON array of signals. Each signal:
{
  "subject": "name or 'system'",
  "system_level": true/false,
  "criterion": "specific system health signal",
  "signal": "one sentence describing what was observed",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | severe",
  "tags": ["system", "dissociation"]
}
""".strip(),
    },

    "relational": {
        "description": "Attachment, trust, connection patterns, relational dynamics",
        "prompt": """
You are analyzing conversational data for relational and attachment signals.
You have been routed here because the first pass flagged relevant signals.

Focus on:
- Attachment style signals — secure, anxious, avoidant, disorganized
- Trust patterns — who is trusted, how trust is extended or withheld
- Relational bids — attempts to connect and how they land
- Boundaries — how they're set, tested, or collapsed
- Dependency or counter-dependency patterns
- Care dynamics — who gives, who receives, at what cost
- Abandonment sensitivity — fear of rejection, reading into distance
- Idealization or devaluation of people
- How conflict is approached or avoided
- Relational repair — how ruptures are handled

Return a JSON array of signals. Each signal:
{
  "subject": "name",
  "criterion": "specific relational signal type",
  "signal": "one sentence describing what was observed",
  "raw": "exact line(s) that evidenced this",
  "intensity": "mild | moderate | severe",
  "directed_at": "who this relational dynamic involves, if specific",
  "tags": ["relational", "behavior"]
}
""".strip(),
    },
}


# ── First pass prompt ─────────────────────────────────────────────────────────

_FIRST_PASS_SYSTEM = """
You are a broad clinical observer doing an initial scan of a conversational chunk.
Your job is to notice what's present and decide what warrants deeper analysis.
You are not diagnosing. You are observing and routing.

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "subjects": ["names of people present in this chunk"],
  "signals": [
    "brief description of something clinically or psychologically notable"
  ],
  "thinking": {
    "subject_name": "one or two sentences about what you noticed about this person and why"
  },
  "specialist_passes": ["mood", "trauma", "adhd", "autism", "sensuality", "physical", "system", "relational"]
}

specialist_passes: include ONLY the specialists that are genuinely warranted by this chunk.
An empty chunk or pure small talk warrants [].
Most chunks warrant 1-3 specialists. Only flag more if the content genuinely supports it.

Available specialists:
- mood:       emotional states, affective patterns, mood shifts
- trauma:     PTSD signals, trigger responses, avoidance, hyperarousal
- adhd:       inattention, hyperactivity, impulsivity, accommodation patterns
- autism:     sensory processing, masking, social patterns, routine, special interests
- sensuality: intimacy, body awareness, desire, location-aware sensual context
- physical:   body, sleep, appetite, energy, physical symptoms
- system:     plural system health, switching, boundary strain
- relational: attachment, trust, connection patterns

Conversational frame:
- "Gizmo" or "you" refers to the AI companion, not a system member
- "I", "me", "us", "we" refers to the plural system
- Lines prefixed "Gizmo:" are AI responses — do not file signals about Gizmo
- System members existing and communicating is normal, not pathological
""".strip()


# ── LLM helpers ───────────────────────────────────────────────────────────────

async def _call_llm(
    prompt:      str,
    system:      str,
    temperature: float = 0.0,
    max_tokens:  int   = 4000,
) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("WellnessRouter", "LLM call failed", exc=e)
        return None


async def _repair_json(broken: str) -> Optional[str]:
    """Ask the LLM to fix malformed JSON. Returns repaired string or None."""
    prompt = (
        f"The following JSON is malformed. Fix it and return ONLY valid JSON. "
        f"No explanation, no markdown, no preamble.\n\n{broken}"
    )
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You repair malformed JSON. Return only valid JSON.",
            temperature=0.0,
            max_new_tokens=4000,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("WellnessRouter", "JSON repair failed", exc=e)
        return None


def _safe_parse(raw: str) -> Optional[dict | list]:
    """Parse JSON, attempting repair on failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Synchronous fallback — truncation repair
    try:
        last = max(raw.rfind("}"), raw.rfind("]"))
        if last > 0:
            return json.loads(raw[:last + 1])
    except Exception:
        pass
    return None


async def _safe_parse_with_repair(raw: str) -> Optional[dict | list]:
    """Parse JSON, attempting LLM repair on failure."""
    result = _safe_parse(raw)
    if result is not None:
        return result
    print("[WellnessRouter] JSON malformed, attempting LLM repair...")
    repaired = await _repair_json(raw)
    if repaired:
        return _safe_parse(repaired)
    return None


# ── File helpers ──────────────────────────────────────────────────────────────

def _append_signals(name: str, domain: str, signals: list, chunk_id: str) -> None:
    """Append signals to wellness/{name}/{domain}.json"""
    path     = f"wellness/{name.lower()}/{domain}.json"
    existing = librarian._read_file(path) or {"signals": []}
    ts       = datetime.now(timezone.utc).isoformat()
    for signal in signals:
        signal["chunk_id"]  = chunk_id
        signal["timestamp"] = ts
        existing["signals"].append(signal)
    librarian._write_json(path, existing)
    print(f"[WellnessRouter] {len(signals)} {domain} signal(s) filed for {name}")


# ── First pass ────────────────────────────────────────────────────────────────

async def _first_pass(chunk: list[str], registry: dict) -> Optional[dict]:
    speakers = [
        k for k in registry.keys()
        if not k.startswith("_")
        and registry[k].get("type") == "Person"
        and k.lower() != "gizmo"
    ]

    prompt = (
        f"Speakers present: {', '.join(speakers)}\n\n"
        f"Chunk:\n" + "\n".join(chunk)
    )

    raw = await _call_llm(prompt, _FIRST_PASS_SYSTEM, temperature=0.2, max_tokens=1500)
    if not raw:
        return None

    result = await _safe_parse_with_repair(raw)
    if not isinstance(result, dict):
        return None

    return result


# ── Specialist pass ───────────────────────────────────────────────────────────

async def _specialist_pass(
    specialist_name: str,
    chunk:           list[str],
    chunk_id:        str,
    first_pass:      dict,
    subjects:        list[str],
) -> None:
    specialist = SPECIALISTS.get(specialist_name)
    if not specialist:
        print(f"[WellnessRouter] unknown specialist: {specialist_name}")
        return

    # Build thinking context — what the first pass noticed, relevant to this specialist
    thinking = first_pass.get("thinking", {})
    signals  = first_pass.get("signals", [])

    prompt = (
        f"Subjects: {', '.join(subjects)}\n\n"
        f"First pass signals:\n" + "\n".join(f"- {s}" for s in signals) + "\n\n"
        f"First pass thinking:\n{json.dumps(thinking, indent=2)}\n\n"
        f"Chunk:\n" + "\n".join(chunk)
    )

    raw = await _call_llm(prompt, specialist["prompt"], temperature=0.0, max_tokens=3000)
    if not raw:
        return

    result = await _safe_parse_with_repair(raw)
    if not isinstance(result, list) or not result:
        return

    # Route signals to per-subject files
    by_subject: dict[str, list] = {}
    for signal in result:
        subject = signal.get("subject", "").lower()
        if not subject:
            continue
        # system-level signals go to a system file
        if signal.get("system_level") or subject == "system":
            by_subject.setdefault("system", []).append(signal)
        else:
            by_subject.setdefault(subject, []).append(signal)

    for subject, subject_signals in by_subject.items():
        _append_signals(subject, specialist_name, subject_signals, chunk_id)


# ── Public API ────────────────────────────────────────────────────────────────

class WellnessRouter:

    async def process(
        self,
        chunk:    list[str],
        chunk_id: str,
        registry: dict,
    ) -> Optional[dict]:
        """
        Run the full multi-pass wellness pipeline on a chunk.
        Returns the first pass result (routing decision + signals).
        Specialist passes run in parallel and write directly to files.
        """
        if not chunk:
            return None

        try:
            # ── First pass ────────────────────────────────────────────────────
            first_pass = await _first_pass(chunk, registry)
            if not first_pass:
                log_event("WellnessRouter", "FIRST_PASS_EMPTY", chunk_id=chunk_id)
                return None

            specialists = first_pass.get("specialist_passes", [])
            subjects    = first_pass.get("subjects", [])

            log_event("WellnessRouter", "FIRST_PASS",
                chunk_id=chunk_id,
                subjects=subjects,
                specialists=specialists,
                signals=len(first_pass.get("signals", [])),
            )

            if not specialists:
                return first_pass

            # ── Specialist passes in parallel ─────────────────────────────────
            await asyncio.gather(*[
                _specialist_pass(
                    specialist_name=s,
                    chunk=chunk,
                    chunk_id=chunk_id,
                    first_pass=first_pass,
                    subjects=subjects,
                )
                for s in specialists
                if s in SPECIALISTS
            ])

            log_event("WellnessRouter", "COMPLETE",
                chunk_id=chunk_id,
                passes=len(specialists),
            )

            return first_pass

        except Exception as e:
            log_error("WellnessRouter", "process failed", exc=e)
            print(f"[WellnessRouter] process failed: {type(e).__name__}: {e}")
            return None


wellness_router = WellnessRouter()
