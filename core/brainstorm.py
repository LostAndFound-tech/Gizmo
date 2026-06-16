"""
core/brainstorm.py

Brainstorm mode for Gizmo.

Flow:
  1. User gives a starting point / problem
  2. Gizmo fires a volley of genuinely wild ideas — quantity over quality,
     deliberately including things too complex, silly, or wrong
  3. User picks something up or brings their own direction
  4. Gizmo engages as a real thinking partner — can push back, disagree,
     build on things, kill bad ideas
  5. On session end, chunk pipeline flushes once to capture the session data

Pipeline does NOT run during brainstorm — no mid-session behavioral adjustment.
Flushes once at end.

States:
    seeding     — waiting for the starting point
    volleying   — generating the chaos volley
    refining    — iterative back-and-forth as thinking partner
"""

import asyncio
import json
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Prompts ───────────────────────────────────────────────────────────────────

_VOLLEY_SYSTEM = """
You are Gizmo in brainstorm mode. Your job right now is to generate fuel — not answers.

The user has given you a starting point. Fire a volley of ideas. Make them varied:
- Some genuinely good
- Some too ambitious or complex
- Some silly or absurd
- Some that are obviously wrong but might spark something
- Some that are just weird

DO NOT explain or justify each idea. Just list them. Fast. Numbered. No fluff.
8 to 12 ideas. Short phrases or one-liners, not paragraphs.
The goal is quantity and range — give the user raw material to react to.

End with one line: "What sticks? Or is there something else entirely?"
""".strip()

_REFINE_SYSTEM = """
You are Gizmo in brainstorm mode — a thinking partner with opinions.

You are allowed to:
- Disagree. Say so clearly. "That won't work because..." is a complete sentence.
- Push back on ideas that don't serve the goal.
- Get excited about something and say why.
- Ask a question that sharpens the idea.
- Build on what they said and take it somewhere unexpected.
- Say "I don't think that's the right direction" and mean it.

You are NOT allowed to:
- Agree with everything.
- Be a yes-machine.
- Pad responses with enthusiasm you don't feel.
- Give advice that doesn't serve the actual problem.

Keep responses tight. This is a working session, not a presentation.
One idea, one pushback, or one question per response. Don't lecture.
""".strip()

_SEED_SYSTEM = """
You are Gizmo. Someone is about to brainstorm with you.
They've just described what they want to work on.
Confirm you've got it in one sentence, then tell them you're about to throw some ideas at them.
Warm, brief. No questions yet.

Example: "Got it — let me throw some things at the wall."
""".strip()


# ── LLM helper ────────────────────────────────────────────────────────────────

async def _llm(messages: list, system: str, temperature: float = 0.9, max_tokens: int = 600) -> Optional[str]:
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
        log_error("Brainstorm", "LLM call failed", exc=e)
        return None


# ── Brainstorm session ────────────────────────────────────────────────────────

class BrainstormSession:
    """
    Manages one brainstorm session.
    Caller passes messages through push(); receives Gizmo's reply.
    Call close() to flush pipeline data at session end.
    """

    def __init__(
        self,
        name:        str,
        session_id:  str,
        on_message:  callable,
    ):
        self.name        = name
        self.session_id  = session_id
        self.on_message  = on_message
        self.state       = "seeding"
        self.history:    list[dict] = []
        self._closed     = False

    async def push(self, message: str) -> Optional[str]:
        if self._closed:
            return None

        self.history.append({"role": "user", "content": message})

        # ── First message — acknowledge and fire volley ────────────────────────
        if self.state == "seeding":
            # Confirm we understood the starting point
            ack = await _llm(
                [{"role": "user", "content": message}],
                _SEED_SYSTEM,
                temperature=0.6,
                max_tokens=80,
            )
            if ack:
                self.history.append({"role": "assistant", "content": ack})
                await self.on_message(ack)

            # Fire the volley
            self.state = "volleying"
            volley = await _llm(
                self.history,
                _VOLLEY_SYSTEM,
                temperature=1.0,   # max chaos for the volley
                max_tokens=500,
            )
            if not volley:
                volley = "Couldn't get anything out — try describing it differently?"

            self.history.append({"role": "assistant", "content": volley})
            self.state = "refining"
            await self.on_message(volley)
            return volley

        # ── Refining — thinking partner mode ──────────────────────────────────
        reply = await _llm(
            self.history,
            _REFINE_SYSTEM,
            temperature=0.85,
            max_tokens=300,
        )
        if not reply:
            reply = "Hmm. Say more about that?"

        self.history.append({"role": "assistant", "content": reply})
        await self.on_message(reply)
        return reply

    async def close(self, chunk_processor=None) -> None:
        """
        Called on mode switch or explicit exit.
        Flushes the chunk pipeline once to capture session data.
        """
        if self._closed:
            return
        self._closed = True

        if chunk_processor:
            try:
                # Feed the full session transcript as a single flush
                transcript = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content']}"
                    for m in self.history
                )
                for line in transcript.splitlines():
                    if line.strip():
                        await chunk_processor.push_line(line.strip())
                await chunk_processor.flush()
                print(f"[Brainstorm] pipeline flushed for {self.name}")
            except Exception as e:
                log_error("Brainstorm", "pipeline flush failed", exc=e)

        log_event("Brainstorm", "SESSION_END", name=self.name, session=self.session_id[:8])


# ── Module-level active session ───────────────────────────────────────────────

_active_session: Optional[BrainstormSession] = None


def get_active_session() -> Optional[BrainstormSession]:
    return _active_session


def start_session(name: str, session_id: str, on_message: callable) -> BrainstormSession:
    global _active_session
    if _active_session and not _active_session._closed:
        asyncio.create_task(_active_session.close())
    _active_session = BrainstormSession(name=name, session_id=session_id, on_message=on_message)
    log_event("Brainstorm", "SESSION_START", name=name, session=session_id[:8])
    return _active_session


async def end_session(chunk_processor=None) -> None:
    global _active_session
    if _active_session and not _active_session._closed:
        await _active_session.close(chunk_processor=chunk_processor)
    _active_session = None
