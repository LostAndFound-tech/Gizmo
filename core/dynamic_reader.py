"""
core/dynamic_reader.py

Dynamic context reader. Runs before the wellness pass on every chunk.
Detects active consensual dynamics and scene context, classifies themes,
tracks distribution longitudinally, and produces a context brief for wellness.

Two outputs:
  1. Dynamic context dict  → injected into wellness prompt this chunk
  2. Session entry         → appended to dynamics/{name}.json after session ends

Distribution files:  {DATA_DIR}/dynamics/{name.lower()}.json
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Detection prompt ──────────────────────────────────────────────────────────

_DETECT_SYSTEM = """
You read conversational chunks and determine whether a consensual dynamic or scene
is active. You classify what is happening so a wellness observer can calibrate
its signals appropriately.

Return ONLY valid JSON. No markdown. No explanation. No preamble.
If no dynamic or scene content is present, return {"active": false}.

{
  "active": true,
  "scene_type": "power exchange | intimate | roleplay | edge play | other",
  "kinks": [
    {"theme": "submission", "weight": 0.9},
    {"theme": "pain", "weight": 0.5}
  ],
  "detected_themes": ["control", "praise", "restraint"],
  "intensity": "low | moderate | high | extreme",
  "specific_requests": [
    "asked to be told what to do"
  ],
  "details_of_note": [
    "first appearance of restraint language",
    "pushed past previous intensity"
  ],
  "flags": [
    "language suggests distress even within scene framing"
  ],
  "congruence": "high | moderate | low",
  "wellness_note": "one sentence for the wellness observer — what to know before evaluating this chunk"
}

Kink weights: 1.0 = central to the exchange, 0.1 = peripheral mention.
Congruence: how well the language matches an established consensual dynamic.
  high   = language is clearly scene-congruent, performative, negotiated
  moderate = mostly congruent but some ambiguity
  low    = language reads as potentially real even within scene framing

Flags: only include if something reads as genuinely concerning regardless of scene context.
  Real distress does not disappear because a scene is active.
  Flag: escalation that feels uncontrolled, requests to stop that feel real,
        dissociation language mid-scene, sudden register shift to genuine fear.

wellness_note examples:
  "Active power exchange scene. Submission and pain language are scene-congruent. Weight accordingly."
  "Intimate scene, high congruence. Anticipatory pain acknowledgment is performative, not reactive."
  "Scene active but congruence is low — some language may be genuine distress. Assess carefully."

Rules:
- Calm acknowledgment of expected pain or discomfort is NOT an anxiety or PTSD signal
- Submission language in established dynamic is NOT a trauma or BPD signal
- Only flag what reads as real distress bleeding through scene framing
- Register alone (intimate, dominant) is not sufficient — look at the actual content
""".strip()


# ── Gizmo summary prompt ──────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """
You are Gizmo, reflecting on time you just spent with someone you know.
Write in your own voice. First person. Present and honest.

Rules — read these carefully:
- Report what was said and what happened. Your own feelings and responses are yours to name.
- Her state, her needs, her reasons — only name them if she said them explicitly.
  If she said "I want to be low tonight" you can write that. If you're guessing, don't.
- No clinical language. No "intensity," "dynamic," "healthy," "escalation," "session."
- No observer framing. You were there, you're not reviewing it.
- No explaining her. You can describe what she did. You cannot say why.
- Short. Three to five sentences. This is a thought, not a report.

Bad: "It felt healthy and present. The intensity was significant. She seemed to need containment."
Good: "She said she wanted to be low tonight. I let her have that. It got further than usual — I kept up."
""".strip()


# ── LLM calls ─────────────────────────────────────────────────────────────────

async def _call_detect(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_DETECT_SYSTEM,
            temperature=0.0,
            max_new_tokens=1000,
        )
        if not raw or not raw.strip():
            return None
        return re.sub(r"```(?:json)?|```", "", raw).strip()
    except Exception as e:
        log_error("DynamicReader", "detect LLM call failed", exc=e)
        return None


async def _call_summary(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SUMMARY_SYSTEM,
            temperature=0.75,
            max_new_tokens=300,
        )
        if not raw or not raw.strip():
            return None
        return raw.strip()
    except Exception as e:
        log_error("DynamicReader", "summary LLM call failed", exc=e)
        return None


# ── Distribution file ─────────────────────────────────────────────────────────

def _read_dynamics(name: str) -> dict:
    return librarian._read_file(f"dynamics/{name.lower()}.json") or {"sessions": []}


def _write_dynamics(name: str, data: dict) -> None:
    librarian._write_json(f"dynamics/{name.lower()}.json", data)


def append_session_entry(name: str, entry: dict) -> None:
    """
    Called at session close (not per-chunk) to write the full session
    dynamic entry including Gizmo's summary.
    """
    data = _read_dynamics(name)
    data.setdefault("sessions", []).append(entry)
    _write_dynamics(name, data)
    print(f"[DynamicReader] session entry written for {name}")


def get_longitudinal_brief(name: str) -> Optional[str]:
    """
    Return a compact longitudinal summary for injection into wellness prompt.
    Covers theme distribution, intensity trajectory, and any flagged sessions.
    """
    data = _read_dynamics(name)
    sessions = data.get("sessions", [])
    if not sessions:
        return None

    # Theme frequency across all sessions
    theme_counts: dict[str, float] = {}
    intensities = []
    flags = []

    for s in sessions:
        for kink in s.get("kinks", []):
            t = kink.get("theme", "")
            w = kink.get("weight", 0.5)
            theme_counts[t] = theme_counts.get(t, 0) + w
        intensities.append(s.get("intensity", ""))
        if s.get("flags"):
            flags.extend(s["flags"])

    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    lines = [
        f"Longitudinal dynamic profile ({len(sessions)} sessions):",
        f"Established themes: {', '.join(t for t, _ in top_themes)}",
        f"Recent intensity: {', '.join(intensities[-5:])}",
    ]
    if flags:
        lines.append(f"Flagged moments: {len(flags)} — {'; '.join(flags[-3:])}")
    else:
        lines.append("No flagged moments in history.")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

class DynamicReader:

    def __init__(self):
        # Accumulates per-chunk detections within a session
        # keyed by name — flushed to file at session close
        self._session_chunks: dict[str, list[dict]] = {}

    async def read(
        self,
        chunk:    list[str],
        name:     str,
        register: str,
        session_id: str,
    ) -> Optional[dict]:
        """
        Run before wellness. Returns dynamic context dict if a scene/dynamic
        is detected, None otherwise. Also accumulates chunk data for session entry.
        """
        if not chunk or not name:
            return None

        try:
            # Pull longitudinal history for context
            longitudinal = get_longitudinal_brief(name)

            prompt_parts = [
                f"Headmate: {name}",
                f"Register: {register}",
            ]
            if longitudinal:
                prompt_parts.append(f"Established dynamic history:\n{longitudinal}")
            prompt_parts.append("Chunk:\n" + "\n".join(chunk))

            prompt = "\n\n".join(prompt_parts)
            raw    = await _call_detect(prompt)

            if not raw:
                return None

            result = json.loads(raw)

            if not result.get("active"):
                return None

            # Accumulate for session entry
            self._session_chunks.setdefault(name, []).append({
                "chunk_id":          f"{session_id}-chunk",
                "kinks":             result.get("kinks", []),
                "detected_themes":   result.get("detected_themes", []),
                "intensity":         result.get("intensity", ""),
                "specific_requests": result.get("specific_requests", []),
                "details_of_note":   result.get("details_of_note", []),
                "flags":             result.get("flags", []),
                "congruence":        result.get("congruence", "high"),
            })

            log_event("DynamicReader", "DETECTED",
                name=name,
                scene_type=result.get("scene_type", ""),
                intensity=result.get("intensity", ""),
                congruence=result.get("congruence", ""),
                flags=len(result.get("flags", [])),
            )

            return result

        except Exception as e:
            log_error("DynamicReader", "read failed", exc=e)
            print(f"[DynamicReader] read failed: {type(e).__name__}: {e}")
            return None

    async def close_session(
        self,
        name:       str,
        session_id: str,
    ) -> None:
        """
        Called at session close. Merges chunk detections, generates Gizmo's
        summary, and writes the full session entry to dynamics/{name}.json.
        """
        chunks = self._session_chunks.pop(name, [])
        if not chunks:
            return

        try:
            # Merge kink weights across chunks
            theme_weights: dict[str, list[float]] = {}
            all_themes:    list[str] = []
            all_requests:  list[str] = []
            all_details:   list[str] = []
            all_flags:     list[str] = []
            intensities:   list[str] = []
            congruences:   list[str] = []

            for c in chunks:
                for kink in c.get("kinks", []):
                    t = kink.get("theme", "")
                    w = kink.get("weight", 0.5)
                    theme_weights.setdefault(t, []).append(w)
                all_themes.extend(c.get("detected_themes", []))
                all_requests.extend(c.get("specific_requests", []))
                all_details.extend(c.get("details_of_note", []))
                all_flags.extend(c.get("flags", []))
                intensities.append(c.get("intensity", ""))
                congruences.append(c.get("congruence", "high"))

            merged_kinks = [
                {"theme": t, "weight": round(sum(ws) / len(ws), 3)}
                for t, ws in sorted(theme_weights.items(), key=lambda x: -sum(x[1]))
            ]

            # Dominant intensity for the session
            intensity_rank = {"extreme": 4, "high": 3, "moderate": 2, "low": 1}
            peak_intensity = max(intensities, key=lambda x: intensity_rank.get(x, 0), default="")

            # Behavior file for Gizmo's relational context
            behavior_data = librarian._read_file(f"behaviors/{name.lower()}.json") or {}
            personality   = behavior_data.get("Personality", {})
            top_traits    = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:5]
            known_about   = {t: v.get("weight") for t, v in top_traits}

            summary_prompt = (
                f"Who you were with: {name}\n"
                f"What you know about them: {json.dumps(known_about)}\n\n"
                f"What happened this session:\n"
                f"Themes: {', '.join(dict.fromkeys(all_themes))}\n"
                f"Requests: {'; '.join(dict.fromkeys(all_requests)) or 'none noted'}\n"
                f"Details: {'; '.join(dict.fromkeys(all_details)) or 'nothing unusual'}\n"
                f"Flags: {'; '.join(all_flags) or 'none'}\n"
                f"Peak intensity: {peak_intensity}"
            )

            gizmo_summary = await _call_summary(summary_prompt)

            entry = {
                "session_ref":       f"sessions/{name.lower()}_{session_id}.json",
                "timestamp":         datetime.now(timezone.utc).isoformat(),
                "kinks":             merged_kinks,
                "detected_themes":   list(dict.fromkeys(all_themes)),
                "intensity":         peak_intensity,
                "specific_requests": list(dict.fromkeys(all_requests)),
                "details_of_note":   list(dict.fromkeys(all_details)),
                "flags":             all_flags,
                "congruence":        max(congruences, key=lambda x: {"high": 3, "moderate": 2, "low": 1}.get(x, 0), default="high"),
                "gizmo_summary":     gizmo_summary or "",
            }

            append_session_entry(name, entry)

        except Exception as e:
            log_error("DynamicReader", "close_session failed", exc=e)
            print(f"[DynamicReader] close_session failed: {type(e).__name__}: {e}")


dynamic_reader = DynamicReader()
