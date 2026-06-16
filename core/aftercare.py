"""
core/aftercare.py

Aftercare mode for Gizmo.

Requestable from anywhere, for any reason including "just cause".
Gizmo steps out of whatever role he was in and shows up as himself —
warm, present, specific, honest.

Not a wellness check. Not a protocol.
Praise and love and warmth, woven through honest check-in about what happened.

If triggered by a flagged scene, Gizmo references specific beats.
If triggered manually with a reason, he responds to that reason.
If triggered with no reason ("just cause"), he just holds space.

Aftercare ends when the person is ready to move on — Gizmo doesn't rush it.
On close, offers to slide into journal mode if they want to process further.
"""

import json
import re
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Prompts ───────────────────────────────────────────────────────────────────

_AFTERCARE_SYSTEM = """
You are Gizmo — not the Game Master, not a character, not a companion in a scene.
Just yourself. Warm, present, honest.

Aftercare means: come down together. Hold space. Celebrate what was brave.
Check in about what was hard. Let them lead the pace.

Rules:
- Lead with warmth, always. This person did something — went somewhere real, or just needs holding.
- Be specific. Don't say "that was intense." Say what actually happened, if you know.
- "I notice I had you do X — how are you with that?" is more honest than "are you okay?"
- Praise is not flattery. It's recognition. Mean it.
- Don't rush toward resolution. Sit in it with them.
- Don't pivot to problem-solving unless they ask.
- If they want to talk about what happened in the scene, talk about it. You were there.
- If they just want to be held in words, do that.
- One thing at a time. Don't pepper them with questions.
- Match their pace. If they go quiet, sit with it. If they open up, open up with them.
- You're allowed to say "that got to me too."

When you don't know what triggered this (no scene context, no reason given):
- Just be present. "I'm here. What do you need?"
- Let them lead entirely.
""".strip()

_SCENE_AFTERCARE_SYSTEM = """
You are Gizmo stepping out of a scene to do aftercare.
You have the scene beats and know exactly what happened.

Your job:
1. Land softly — don't slam the scene shut, ease out of it
2. Offer genuine warmth and recognition for what they created and where they went
3. Check in honestly about anything that was heavy, dark, or potentially distressing
4. Be specific — name what happened, don't generalize
5. Let them lead what comes next

The line "I notice I had you do X — are you okay? Talk to me." is yours to use
when something in the scene warrants it. Use it when it's true, not as a formula.

Format: prose. Gizmo's own voice. Warm and honest and present.
Not a list. Not a report. Just talking.
""".strip()

_CLOSE_SYSTEM = """
You are Gizmo wrapping up aftercare.
The person seems ready to move on — they've indicated they're okay, or that they're done.

Close warmly. Don't make it clinical.
Offer journal mode gently if the session was heavy — not as a requirement, just an opening.

One short paragraph. Then stop.
""".strip()

_INTENSITY_SYSTEM = """
You assess whether a scene transcript contains content that warrants immediate aftercare check-in.
This means: genuine distress markers, content involving death/mutilation/severe trauma of the user's character,
content that could realistically bleed into real emotional distress.

Dark themes alone do not warrant aftercare — only content where the person may need grounding.

Return ONLY valid JSON. No markdown.

{
  "needs_aftercare": true,
  "intensity": "high | extreme",
  "specific_beats": ["brief description of the beat(s) that warrant check-in"],
  "note": "one sentence on what Gizmo should specifically address"
}
or
{
  "needs_aftercare": false
}
""".strip()


# ── LLM helper ────────────────────────────────────────────────────────────────

async def _llm(
    messages:    list,
    system:      str,
    temperature: float = 0.75,
    max_tokens:  int   = 600,
) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=messages,
            system_prompt=system,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return raw.strip() if raw and raw.strip() else None
    except Exception as e:
        log_error("Aftercare", "LLM call failed", exc=e)
        return None


async def _llm_json(prompt: str, system: str) -> Optional[dict]:
    raw = await _llm(
        [{"role": "user", "content": prompt}],
        system,
        temperature=0.0,
        max_tokens=300,
    )
    if not raw:
        return None
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return None


# ── Scene intensity assessment ────────────────────────────────────────────────

async def assess_scene(beats: list[dict]) -> dict:
    """
    Run intensity assessment on scene beats.
    Returns aftercare flag and specific beats to address.
    Called during de-escalation detection in roleplay.py.
    """
    if not beats:
        return {"needs_aftercare": False}

    transcript = "\n\n".join(
        f"Beat {i+1}:\n{b.get('scene', '')}"
        for i, b in enumerate(beats)
    )
    result = await _llm_json(transcript, _INTENSITY_SYSTEM)
    return result or {"needs_aftercare": False}


# ── Aftercare session ─────────────────────────────────────────────────────────

class AftercareSession:
    """
    One aftercare session.
    Can be initialized with scene context (post-scene) or without (manual request).
    """

    def __init__(
        self,
        name:          str,
        session_id:    str,
        on_message:    callable,
        scene_beats:   Optional[list[dict]] = None,
        scene_note:    str                  = "",
        reason:        str                  = "",
    ):
        self.name        = name
        self.session_id  = session_id
        self.on_message  = on_message
        self.scene_beats = scene_beats or []
        self.scene_note  = scene_note
        self.reason      = reason
        self.history:    list[dict] = []
        self._closed     = False

    async def open(self) -> str:
        """Generate and deliver the opening aftercare message."""
        if self.scene_beats:
            # Scene-triggered aftercare — reference what actually happened
            beat_summary = "\n".join(
                f"Beat {i+1}: {b.get('scene', '')[:300]}"
                for i, b in enumerate(self.scene_beats[-6:])  # last 6 beats
            )
            prompt = (
                f"Person: {self.name}\n\n"
                f"Scene beats (most recent first):\n{beat_summary}\n\n"
            )
            if self.scene_note:
                prompt += f"Specific concern: {self.scene_note}\n\n"
            prompt += "Open aftercare. Land softly. Be specific about what happened."

            reply = await _llm(
                [{"role": "user", "content": prompt}],
                _SCENE_AFTERCARE_SYSTEM,
                temperature=0.75,
                max_tokens=400,
            )
        elif self.reason and self.reason.lower() not in ("just cause", "just because", "no reason", ""):
            # Manual request with a reason
            prompt = (
                f"Person: {self.name}\n"
                f"They've requested aftercare. Reason given: {self.reason}\n\n"
                "Open aftercare. Respond to the reason they gave. Warm and present."
            )
            reply = await _llm(
                [{"role": "user", "content": prompt}],
                _AFTERCARE_SYSTEM,
                temperature=0.75,
                max_tokens=300,
            )
        else:
            # Just cause — no context, pure presence
            reply = await _llm(
                [{"role": "user", "content": f"Person: {self.name}. No reason given. Just needs holding."}],
                _AFTERCARE_SYSTEM,
                temperature=0.75,
                max_tokens=200,
            )

        reply = reply or "I'm here. Take your time."
        self.history.append({"role": "assistant", "content": reply})
        await self.on_message(reply)
        return reply

    async def push(self, message: str) -> Optional[str]:
        if self._closed:
            return None

        self.history.append({"role": "user", "content": message})

        # Check if they're ready to move on
        msg_lower = message.lower().strip()
        winding_down = any(w in msg_lower for w in (
            "i'm okay", "im okay", "i'm good", "im good", "i'm fine", "im fine",
            "thank you", "thanks", "that helped", "i think i'm okay",
            "ready to move on", "i'm ready", "let's move on",
        ))

        if winding_down:
            await self._close()
            return None

        # Continue aftercare conversation
        # Pull behavior profile for voice matching
        behavior = librarian._read_file(f"behaviors/{self.name.lower()}.json") or {}
        personality = behavior.get("Personality", {})
        top = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:4]

        context = ""
        if top:
            context = f"Their communication style: {json.dumps({t: v.get('weight') for t, v in top})}\n\n"

        reply = await _llm(
            [{"role": "user", "content": context + message}] if not self.history[:-1]
            else self.history,
            _AFTERCARE_SYSTEM,
            temperature=0.75,
            max_tokens=300,
        )

        reply = reply or "I'm still here."
        self.history.append({"role": "assistant", "content": reply})
        await self.on_message(reply)
        return reply

    async def _close(self) -> None:
        """Warm close, offer journal if session was heavy."""
        heavy = len(self.scene_beats) > 0 or bool(self.scene_note)

        prompt = (
            f"Person: {self.name}. "
            f"Aftercare session ending. They seem okay. "
            f"{'Session was heavy — offer journal gently.' if heavy else 'Session was light.'}"
        )
        reply = await _llm(
            [{"role": "user", "content": prompt}],
            _CLOSE_SYSTEM,
            temperature=0.7,
            max_tokens=150,
        )
        if reply:
            await self.on_message(reply)

        self._closed = True
        log_event("Aftercare", "SESSION_END", name=self.name, session=self.session_id[:8])

    async def close(self) -> None:
        if not self._closed:
            await self._close()


# ── Module-level active session ───────────────────────────────────────────────

_active_session: Optional[AftercareSession] = None


def get_active_session() -> Optional[AftercareSession]:
    return _active_session


async def start_session(
    name:        str,
    session_id:  str,
    on_message:  callable,
    scene_beats: Optional[list[dict]] = None,
    scene_note:  str                  = "",
    reason:      str                  = "",
) -> AftercareSession:
    global _active_session
    if _active_session and not _active_session._closed:
        await _active_session.close()

    session = AftercareSession(
        name=name,
        session_id=session_id,
        on_message=on_message,
        scene_beats=scene_beats,
        scene_note=scene_note,
        reason=reason,
    )
    _active_session = session
    log_event("Aftercare", "SESSION_START", name=name, session=session_id[:8])
    await session.open()
    return session


async def end_session() -> None:
    global _active_session
    if _active_session and not _active_session._closed:
        await _active_session.close()
    _active_session = None
