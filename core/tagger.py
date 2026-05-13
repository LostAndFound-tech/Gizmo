"""
core/tagger.py

Async post-exchange tagging pass.

After each exchange, this runs in the background and asks the LLM to read
both sides — what was said, how it was said, what caused it, what it led to.

Critically: both the headmate's personality profile AND Gizmo's personality
seed are loaded before tagging. The exchange is assessed in context, not cold.
"Hey there! It's Jess! <3" reads as Jess being Jess — measured against her
baseline warmth, her patterns, her relationship with Gizmo.

Produces:
  - topics:       what this was actually about (Gizmo decides, not regex)
  - mood:         emotional color of the exchange as a whole
  - gizmo_tone:   how Gizmo came across relative to his personality and this headmate
  - cause:        what seems to have prompted this exchange
  - effect:       what changed — mood shift, topic change, something resolved/unresolved
  - summary:      one sentence capturing the exchange

Updates:
  - message_store row (real tags replace heuristic placeholder)
  - emotion_tracker arc (appends Gizmo's own emotional point for the exchange)

Never blocks. Never raises. If it fails, the heuristic tags stay.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_SEED_FILE       = _PERSONALITY_DIR / "personality.txt"

# ── Profile loaders ───────────────────────────────────────────────────────────

def _load_gizmo_seed() -> str:
    try:
        return _SEED_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _load_headmate_profile(name: str) -> Optional[dict]:
    if not name:
        return None
    path = _HEADMATES_DIR / f"{name.lower()}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _format_headmate_context(name: str, profile: dict) -> str:
    """
    Build a compact profile summary for the tagger prompt.
    Enough context to read the exchange correctly — not the full file.
    """
    if not profile:
        return f"{name.title()} — no profile yet."

    lines = [f"{name.title()}:"]

    baseline = profile.get("baseline", {})
    register = baseline.get("register", "unknown")
    verbosity = baseline.get("verbosity", "unknown")
    humor = baseline.get("humor_response", "unknown")
    confidence = baseline.get("confidence", 0.0)

    if register != "unknown":
        lines.append(f"  typical register: {register}")
    if verbosity != "unknown":
        lines.append(f"  typical verbosity: {verbosity}")
    if humor != "unknown":
        lines.append(f"  humor response: {humor}")
    if confidence:
        lines.append(f"  observation confidence: {confidence:.1f}")

    patterns = profile.get("observed_patterns", [])
    if patterns:
        lines.append("  observed patterns:")
        for p in patterns[-3:]:
            if isinstance(p, dict):
                lines.append(f"    - {p.get('pattern', str(p))}")
            else:
                lines.append(f"    - {p}")

    moments = profile.get("moments_of_note", [])
    if moments:
        lines.append("  notable moments:")
        for m in moments[-2:]:
            import re
            clean = re.sub(r'^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*', '', str(m))
            lines.append(f"    - {clean}")

    prefs = profile.get("interaction_prefs", {})
    if prefs:
        pref_parts = []
        for field in ("tone", "pacing", "humor", "distress"):
            v = prefs.get(field)
            if v:
                pref_parts.append(f"{field}: {v}")
        if pref_parts:
            lines.append("  interaction prefs: " + ", ".join(pref_parts))

    return "\n".join(lines)


# ── Noise gate ────────────────────────────────────────────────────────────────

_MIN_WORDS = 6


def _worth_tagging(user_message: str, gizmo_response: str) -> bool:
    combined = len(user_message.split()) + len(gizmo_response.split())
    return combined >= _MIN_WORDS


# ── Emotion helpers ───────────────────────────────────────────────────────────

def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _parse_emotion(data: dict) -> tuple[float, float, float]:
    try:
        valence   = _clamp(float(data.get("valence",   0.0)))
        intensity = _clamp(float(data.get("intensity", 0.2)), 0.0, 1.0)
        chaos     = _clamp(float(data.get("chaos",     0.0)), 0.0, 1.0)
        return valence, intensity, chaos
    except Exception:
        return 0.0, 0.2, 0.0


def _valence_to_register(valence: float) -> str:
    if valence >= 0.5:
        return "positive"
    if valence <= -0.6:
        return "distress"
    if valence <= -0.2:
        return "elevated"
    if valence <= -0.05:
        return "subdued"
    return "neutral"


# ── Main tagging coroutine ────────────────────────────────────────────────────

async def tag_exchange(
    msg_id: str,
    session_id: str,
    user_message: str,
    gizmo_response: str,
    host: Optional[str],
    fronters: list,
    prior_topics: list,
    prior_mood: str,
    llm,
) -> None:
    """
    Background tagging pass. Updates message_store and emotion_tracker.
    Loads headmate profile + Gizmo seed before tagging so the exchange
    is assessed in context, not cold.
    """
    try:
        if not _worth_tagging(user_message, gizmo_response):
            return

        host_str    = (host or "unknown").title()
        fronter_str = ", ".join(f.title() for f in fronters) if fronters else host_str
        prior_t_str = ", ".join(prior_topics) if prior_topics else "none"
        prior_m_str = prior_mood or "unknown"

        # ── Load profiles ─────────────────────────────────────────────────────
        gizmo_seed = _load_gizmo_seed()

        # Load profiles for all fronters, primary host first
        headmate_contexts = []
        seen = set()
        for name in ([host] + fronters) if host else fronters:
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            profile = _load_headmate_profile(name)
            headmate_contexts.append(_format_headmate_context(name, profile))

        headmate_block = "\n\n".join(headmate_contexts) if headmate_contexts else "No profiles loaded."

        # Trim Gizmo seed to keep prompt lean — first 400 chars is enough
        gizmo_summary = gizmo_seed[:400].strip() if gizmo_seed else "Gizmo — persistent AI companion."

        # ── Build prompt ──────────────────────────────────────────────────────
        prompt = [{
            "role": "user",
            "content": (
                f"You are Gizmo, reviewing an exchange you just had.\n\n"
                f"YOUR PERSONALITY:\n{gizmo_summary}\n\n"
                f"WHO YOU WERE TALKING TO:\n{headmate_block}\n\n"
                f"Prior context — topics just before this: {prior_t_str}\n"
                f"Prior mood: {prior_m_str}\n\n"
                f"--- Exchange ---\n"
                f"{fronter_str}: {user_message.strip()}\n\n"
                f"Gizmo: {gizmo_response.strip()}\n"
                f"--- End exchange ---\n\n"
                f"Analyse this exchange knowing who these people are. "
                f"Measure mood and tone against their baseline — "
                f"is this typical for them, warmer than usual, more fragmented? "
                f"Respond with ONLY valid JSON, no markdown:\n"
                f"{{\n"
                f'  "topics": ["2-6 specific topics — concrete nouns and concepts, not vague categories"],\n'
                f'  "mood": "one word for the emotional color of the whole exchange",\n'
                f'  "gizmo_tone": "one or two words for how Gizmo came across relative to his personality",\n'
                f'  "cause": "what prompted this — prior topic, emotional state, specific trigger, or null",\n'
                f'  "effect": "what changed after — mood shift, new topic, resolution, or null",\n'
                f'  "summary": "one sentence capturing what happened",\n'
                f'  "notable": true or false (was this exchange meaningfully different from baseline?),\n'
                f'  "valence": <float -1.0 to 1.0>,\n'
                f'  "intensity": <float 0.0 to 1.0>,\n'
                f'  "chaos": <float 0.0 to 1.0>\n'
                f"}}"
            )
        }]

        raw = await llm.generate(
            prompt,
            system_prompt=(
                f"You are Gizmo doing a brief self-review of one exchange. "
                f"You know {fronter_str} — use that knowledge. "
                f"Be specific about topics. Cause and effect are critical — "
                f"trace what led to this and what it changed. "
                f"Measure everything against who these people actually are. "
                f"JSON only. No preamble."
            ),
            max_new_tokens=400,
            temperature=0.2,
        )

        if not raw or not raw.strip():
            return

        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()
        brace = raw.find("{")
        if brace > 0:
            raw = raw[brace:]

        data = json.loads(raw)

        topics     = data.get("topics", [])
        mood       = data.get("mood", "neutral")
        gizmo_tone = data.get("gizmo_tone", "")
        cause      = data.get("cause") or None
        effect     = data.get("effect") or None
        summary    = data.get("summary", "")
        notable    = bool(data.get("notable", False))
        valence, intensity, chaos = _parse_emotion(data)

        # ── Update message store ──────────────────────────────────────────────
        try:
            from core.message_store import update_tags
            update_tags(
                msg_id     = msg_id,
                topics     = topics,
                mood       = mood,
                gizmo_tone = gizmo_tone,
                cause      = cause,
                effect     = effect,
                summary    = summary,
            )
            # Update notable flag separately if true
            if notable:
                try:
                    from core.message_store import _conn
                    with _conn() as con:
                        con.execute(
                            "UPDATE messages SET notable = 1 WHERE id = ?",
                            (msg_id,)
                        )
                except Exception:
                    pass
        except Exception as e:
            log_error("Tagger", "message_store update failed", exc=e)

        # ── Update emotion tracker with Gizmo's emotional point ───────────────
        try:
            from core.emotion_tracker import emotion_tracker, EmotionPoint
            arc = emotion_tracker.get_arc(session_id)
            gizmo_point = EmotionPoint(
                timestamp  = time.time(),
                headmate   = "gizmo",
                register   = _valence_to_register(valence),
                valence    = valence,
                intensity  = intensity,
                chaos      = chaos,
                message    = gizmo_response,
                word_count = len(gizmo_response.split()),
                topic      = topics[0] if topics else "general",
            )
            arc.add_point(gizmo_point)
        except Exception as e:
            log_error("Tagger", "emotion_tracker update failed", exc=e)

        log_event("Tagger", "TAGGED",
            session   = session_id[:8],
            msg_id    = msg_id,
            host      = host or "unknown",
            topics    = topics,
            mood      = mood,
            tone      = gizmo_tone,
            cause     = bool(cause),
            effect    = bool(effect),
            notable   = notable,
            valence   = round(valence, 2),
            intensity = round(intensity, 2),
        )

    except Exception as e:
        log_error("Tagger", f"tag_exchange failed for {msg_id}", exc=e)
