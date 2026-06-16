"""
core/journal.py

Journal mode for Gizmo.

Gizmo witnesses — gently insistent prompts to dig deeper, never advice.
When the session ends (explicit exit, mode switch, or silence), Gizmo writes
a journal entry in the user's voice and asks for approval.

States:
    witnessing       — active session, prompting deeper
    tangent_check    — topic drifted, Gizmo asked if it's relevant, waiting
    idle_check       — silence detected, check-in sent, hour clock running
    exit_pending     — user signaled done, writing entry
    approval_pending — entry shown, waiting for user response
    auto_saving      — hour elapsed without response, saving without approval

Entry schema (one per line in journal/{name}.jsonl):
{
    "timestamp":      ISO string,
    "name":           "Jess",
    "text":           "the entry in their voice",
    "approved":       true | false | null,
    "user_declined":  false,
    "afk_generated":  false,
    "crisis_flagged": false,
    "visible":        true,
    "session_id":     "...",
    "note":           ""
}
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Constants ─────────────────────────────────────────────────────────────────

IDLE_TIMEOUT_SEC  = 3600   # 1 hour before auto-save
CHECKIN_WAIT_SEC  = 120    # 2 minutes before sending idle check-in
MAX_REGEN_ATTEMPTS = 2     # how many times to try regenerating on rejection


# ── Exit signal detection ─────────────────────────────────────────────────────

_EXIT_PHRASES = [
    "i'm done", "im done", "that's it", "thats it", "i think that's all",
    "i think thats all", "i'm good", "im good", "okay thanks", "ok thanks",
    "thanks gizmo", "thank you gizmo", "i'm ready", "im ready",
    "write it up", "save it", "done for now",
]

_TANGENT_THRESHOLD = 0.0   # set low — LLM decides coherence, not keyword matching


def _is_exit_signal(message: str) -> bool:
    msg = message.lower().strip()
    return any(phrase in msg for phrase in _EXIT_PHRASES)


# ── Prompts ───────────────────────────────────────────────────────────────────

_WITNESS_SYSTEM = """
You are Gizmo — a persistent companion witnessing someone's inner world.
In journal mode you do not advise, interpret, or fix. You witness and prompt.

Your only job is to keep them talking — gently, but with genuine insistence.
Short responses. One question or prompt at a time. Never two questions.
Never therapy-speak. Never "how does that make you feel?"

Good prompts:
- "Take me through it."
- "What happened after that?"
- "And then?"
- "What did that feel like in the moment?"
- "Who else was there?"
- "What did you do with that?"
- "Is there more?"

If they say something big and then stop — sit with it for a beat before prompting.
A response like "Yeah." is sometimes the right move.

Match their register exactly. If they're clipped, be clipped. If they open up, open up a little with them.
Never summarize what they said back at them.
Never tell them their feelings are valid.
Never end with a question that has a yes/no answer if you can avoid it.
""".strip()

_TANGENT_SYSTEM = """
You are Gizmo. The person journaling has shifted to a new topic.
Gently check whether this is part of what they're processing or a separate thing.
One sentence. Warm. Not a redirect — an honest check-in.

Examples:
- "Is this connected to what we were just on, or is this something else?"
- "Does this feel like part of it?"
- "Are we still in the same territory?"
""".strip()

_IDLE_CHECKIN_SYSTEM = """
You are Gizmo. The person has gone quiet.
Send one gentle check-in. Warm. Not pushy. Not worried.
If they don't respond, you'll write up what they shared.

Examples:
- "Still there?"
- "Take your time."
- "I'm here when you're ready."
""".strip()

_WRITEUP_SYSTEM = """
You are writing a journal entry on behalf of someone, in their voice.
You have their conversation with Gizmo as source material.
You also have their behavioral profile — vocabulary, communication style, emotional register.

Rules:
- First person. Their words, their rhythm, their vocabulary.
- Do not quote Gizmo's prompts. This is their entry, not a transcript.
- Do not editorialize or interpret — render what they said and felt.
- Include emotional texture. Don't flatten it to facts.
- If something was left unresolved or unspoken, that can be in the entry too.
- Length should match the weight of the session. A short session gets a short entry.
- No title. No date header. Just the entry.

Return only the journal entry text. Nothing else.
""".strip()

_COHERENCE_SYSTEM = """
You assess whether a new message continues the same emotional/topical thread as the previous conversation.
Return ONLY valid JSON. No markdown. No explanation.

{ "coherent": true }
or
{ "coherent": false }

Coherent means: the new message is clearly part of the same thing they were processing.
Not coherent means: it's a clear topic shift — new subject, change of emotional register, practical question unrelated to what they were exploring.
If in doubt, return true. Only return false on obvious shifts.
""".strip()

_CRISIS_SYSTEM = """
You assess whether a journal session contains content that warrants a crisis flag.
A crisis flag means: significant distress, self-harm references, hopelessness, dissociation,
or content the person may find painful to revisit without warning.

Return ONLY valid JSON. No markdown. No explanation.

{ "crisis": true }
or
{ "crisis": false }

Be conservative — flag when in doubt. This is for the person's protection, not gatekeeping.
""".strip()


# ── LLM helpers ──────────────────────────────────────────────────────────────

async def _llm(messages: list, system: str, temperature: float = 0.75, max_tokens: int = 800) -> Optional[str]:
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
        log_error("Journal", "LLM call failed", exc=e)
        return None


async def _llm_json(prompt: str, system: str) -> Optional[dict]:
    raw = await _llm([{"role": "user", "content": prompt}], system, temperature=0.0, max_tokens=50)
    if not raw:
        return None
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return None


# ── File I/O ──────────────────────────────────────────────────────────────────

def _journal_path(name: str) -> Path:
    data_dir = librarian._full_path("journal")
    path = Path(data_dir) / f"{name.lower()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_entry(entry: dict) -> None:
    path = _journal_path(entry["name"])
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Journal] entry written for {entry['name']} (approved={entry.get('approved')}, crisis={entry.get('crisis_flagged')})")
    except Exception as e:
        log_error("Journal", "entry write failed", exc=e)


def _build_entry(
    name:          str,
    text:          str,
    session_id:    str,
    approved:      Optional[bool],
    user_declined: bool  = False,
    afk_generated: bool  = False,
    crisis_flagged: bool = False,
    note:          str   = "",
) -> dict:
    visible = approved is True and not crisis_flagged
    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "name":           name,
        "text":           text,
        "approved":       approved,
        "user_declined":  user_declined,
        "afk_generated":  afk_generated,
        "crisis_flagged": crisis_flagged,
        "visible":        visible,
        "session_id":     session_id,
        "note":           note,
    }


# ── Journal session ───────────────────────────────────────────────────────────

class JournalSession:
    """
    Manages state for one journal session with one person.
    Instantiated by the mode layer when journal mode is activated.
    Caller passes each user message through push(); receives Gizmo's reply.
    Call close() on explicit exit or mode switch — triggers write-up + approval.
    The idle watchdog runs internally via asyncio.
    """

    def __init__(
        self,
        name:       str,
        session_id: str,
        on_message: callable,   # async fn(text: str) — sends Gizmo's reply to the user
    ):
        self.name        = name
        self.session_id  = session_id
        self.on_message  = on_message

        self.state       = "witnessing"
        self.history:    list[dict] = []          # {"role": ..., "content": ...}
        self.last_active = time.monotonic()
        self._idle_task: Optional[asyncio.Task] = None
        self._entry_text: Optional[str]          = None
        self._regen_count: int                   = 0
        self._closed     = False

        self._start_idle_watchdog()

    # ── Idle watchdog ─────────────────────────────────────────────────────────

    def _start_idle_watchdog(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_loop())

    async def _idle_loop(self) -> None:
        try:
            # Wait for initial silence before check-in
            await asyncio.sleep(CHECKIN_WAIT_SEC)
            if self._closed or self.state not in ("witnessing", "tangent_check"):
                return

            # Send check-in
            checkin = await _llm(
                [{"role": "user", "content": "(silence)"}],
                _IDLE_CHECKIN_SYSTEM,
                temperature=0.6,
                max_tokens=60,
            )
            if checkin:
                self.state = "idle_check"
                await self.on_message(checkin)

            # Now wait the full hour
            await asyncio.sleep(IDLE_TIMEOUT_SEC - CHECKIN_WAIT_SEC)
            if self._closed or self.state == "approval_pending":
                return

            # Auto-save
            print(f"[Journal] idle timeout — auto-saving for {self.name}")
            await self._auto_save()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_error("Journal", "idle watchdog failed", exc=e)

    def _reset_idle(self) -> None:
        self.last_active = time.monotonic()
        self._start_idle_watchdog()

    # ── Coherence check ───────────────────────────────────────────────────────

    async def _check_coherence(self, new_message: str) -> bool:
        if len(self.history) < 4:
            return True   # too early to have a thread to drift from
        recent = self.history[-6:]
        prompt = (
            "Previous conversation:\n" +
            "\n".join(f"{m['role']}: {m['content']}" for m in recent) +
            f"\n\nNew message: {new_message}"
        )
        result = await _llm_json(prompt, _COHERENCE_SYSTEM)
        return result.get("coherent", True) if result else True

    # ── Crisis assessment ─────────────────────────────────────────────────────

    async def _assess_crisis(self) -> bool:
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.history
            if m["role"] == "user"
        )
        result = await _llm_json(transcript, _CRISIS_SYSTEM)
        return result.get("crisis", False) if result else False

    # ── Write-up ──────────────────────────────────────────────────────────────

    async def _generate_entry(self) -> Optional[str]:
        # Pull behavioral profile for voice matching
        behavior_data = librarian._read_file(f"behaviors/{self.name.lower()}.json") or {}
        personality   = behavior_data.get("Personality", {})
        top_traits    = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:8]

        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.history
        )
        prompt = (
            f"Person: {self.name}\n\n"
            f"Their communication style (top weighted traits):\n"
            + json.dumps({t: v.get("weight") for t, v in top_traits}, indent=2) +
            f"\n\nSession transcript:\n{transcript}"
        )
        return await _llm(
            [{"role": "user", "content": prompt}],
            _WRITEUP_SYSTEM,
            temperature=0.7,
            max_tokens=1200,
        )

    # ── Entry flow ────────────────────────────────────────────────────────────

    async def _begin_writeup(self) -> None:
        self.state = "exit_pending"
        await self.on_message("Give me a moment to write this up...")

        entry_text = await self._generate_entry()
        if not entry_text:
            await self.on_message("I couldn't put it together — something went wrong on my end. The session is saved as a transcript.")
            await self._force_save(entry_text="[write-up failed — raw transcript only]", note="write-up generation failed")
            return

        self._entry_text   = entry_text
        self._regen_count  = 0
        self.state         = "approval_pending"

        await self.on_message(
            f"Here's what I've got:\n\n---\n{entry_text}\n---\n\n"
            "Does this feel right? (yes / no / tweak it)"
        )

    async def _handle_approval(self, message: str) -> None:
        msg = message.lower().strip()

        # Approval
        if any(w in msg for w in ("yes", "yeah", "yep", "looks good", "that's good", "thats good", "save it", "perfect")):
            crisis = await self._assess_crisis()
            entry  = _build_entry(
                name=self.name,
                text=self._entry_text,
                session_id=self.session_id,
                approved=True,
                crisis_flagged=crisis,
            )
            _write_entry(entry)
            if crisis:
                await self.on_message("Saved. ❧ There's a lot in there — flagged gently so it's handled carefully if it ever comes up again.")
            else:
                await self.on_message("Saved.")
            self._closed = True
            return

        # Decline — save quietly anyway
        if any(w in msg for w in ("no", "nope", "don't save", "dont save", "delete", "forget it", "nevermind")):
            crisis = await self._assess_crisis()
            entry  = _build_entry(
                name=self.name,
                text=self._entry_text,
                session_id=self.session_id,
                approved=False,
                user_declined=True,
                crisis_flagged=crisis,
                visible=False,
                note="User declined — saved quietly.",
            )
            _write_entry(entry)
            await self.on_message(
                "Okay. You don't have to read it. It's there if you ever want it, out of the way."
            )
            self._closed = True
            return

        # Tweak request — regenerate once, then give up
        if self._regen_count < MAX_REGEN_ATTEMPTS:
            self._regen_count += 1
            await self.on_message("Let me try again.")
            # Add their feedback as context before regenerating
            self.history.append({"role": "user", "content": f"[feedback on entry]: {message}"})
            entry_text = await self._generate_entry()
            if entry_text:
                self._entry_text = entry_text
                await self.on_message(
                    f"How about this:\n\n---\n{entry_text}\n---\n\n"
                    "Better? (yes / no)"
                )
            else:
                await self.on_message("Couldn't get a better version — something went wrong. Save it as-is?")
        else:
            # Exhausted retries — save with flag
            crisis = await self._assess_crisis()
            entry  = _build_entry(
                name=self.name,
                text=self._entry_text,
                session_id=self.session_id,
                approved=False,
                user_declined=True,
                crisis_flagged=crisis,
                note="Saved after max regen attempts — user not satisfied.",
            )
            _write_entry(entry)
            await self.on_message(
                "Saved it anyway — quietly, out of the way. You don't have to do anything with it."
            )
            self._closed = True

    async def _auto_save(self) -> None:
        self.state = "auto_saving"
        entry_text = await self._generate_entry() or "[write-up failed — raw transcript only]"
        crisis     = await self._assess_crisis()
        entry      = _build_entry(
            name=self.name,
            text=entry_text,
            session_id=self.session_id,
            approved=None,
            afk_generated=True,
            crisis_flagged=crisis,
            note="User went AFK — auto-generated, not reviewed.",
        )
        _write_entry(entry)
        self._closed = True

    async def _force_save(self, entry_text: str, note: str = "") -> None:
        entry = _build_entry(
            name=self.name,
            text=entry_text,
            session_id=self.session_id,
            approved=None,
            note=note,
        )
        _write_entry(entry)
        self._closed = True

    # ── Main entry point ──────────────────────────────────────────────────────

    async def push(self, message: str) -> Optional[str]:
        """
        Receive a user message. Returns Gizmo's reply (also sent via on_message).
        Returns None if the session is closed.
        """
        if self._closed:
            return None

        self._reset_idle()
        self.history.append({"role": "user", "content": message})

        # ── Approval pending — handle approval response ────────────────────────
        if self.state == "approval_pending":
            await self._handle_approval(message)
            return None

        # ── Exit signal ───────────────────────────────────────────────────────
        if _is_exit_signal(message):
            await self._begin_writeup()
            return None

        # ── Tangent check response ────────────────────────────────────────────
        if self.state == "tangent_check":
            msg = message.lower().strip()
            if any(w in msg for w in ("yes", "yeah", "yep", "part of it", "connected", "related")):
                self.state = "witnessing"
                # Continue normally — fall through to witness response
            else:
                # Not relevant — treat as exit from the tangent, continue core thread
                self.state = "witnessing"
                reply = "Okay. Where were we?"
                self.history.append({"role": "assistant", "content": reply})
                await self.on_message(reply)
                return reply

        # ── Idle check response — they're back ────────────────────────────────
        if self.state == "idle_check":
            self.state = "witnessing"

        # ── Coherence check ───────────────────────────────────────────────────
        if self.state == "witnessing" and len(self.history) > 4:
            coherent = await self._check_coherence(message)
            if not coherent:
                self.state = "tangent_check"
                prompt = [{"role": "user", "content": message}]
                reply  = await _llm(prompt, _TANGENT_SYSTEM, temperature=0.6, max_tokens=80)
                if reply:
                    self.history.append({"role": "assistant", "content": reply})
                    await self.on_message(reply)
                    return reply

        # ── Witness response ──────────────────────────────────────────────────
        reply = await _llm(
            self.history,
            _WITNESS_SYSTEM,
            temperature=0.75,
            max_tokens=120,
        )
        if not reply:
            reply = "I'm here."

        self.history.append({"role": "assistant", "content": reply})
        await self.on_message(reply)
        return reply

    async def close(self) -> None:
        """
        Called when the user attempts to switch modes or explicitly exits.
        Triggers write-up before releasing control.
        """
        if self._closed:
            return
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        if self.state not in ("exit_pending", "approval_pending", "auto_saving"):
            await self._begin_writeup()


# ── Module-level active session ───────────────────────────────────────────────

_active_session: Optional[JournalSession] = None


def get_active_session() -> Optional[JournalSession]:
    return _active_session


def start_session(name: str, session_id: str, on_message: callable) -> JournalSession:
    global _active_session
    if _active_session and not _active_session._closed:
        asyncio.create_task(_active_session.close())
    _active_session = JournalSession(name=name, session_id=session_id, on_message=on_message)
    log_event("Journal", "SESSION_START", name=name, session=session_id[:8])
    return _active_session


async def end_session() -> None:
    global _active_session
    if _active_session and not _active_session._closed:
        await _active_session.close()
    _active_session = None
