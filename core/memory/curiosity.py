"""
core/memory/curiosity.py

Gizmo's curiosity engine.

Not a queue. Not a checklist. A pool of things Gizmo wants to know,
from which he picks one question when the moment feels right.

How it works:

  GAP DETECTION (runs after each encoding pass)
  Gizmo looks at what he knows and asks: what am I curious about?
  Generates 3-5 natural questions, adds them to the pool.
  No structure imposed — just genuine curiosity.

  SELECTION (runs during build_system_prompt)
  Gizmo looks at the full pool, the current moment, and decides:
  is there something worth asking right now?
  If yes — one question, woven naturally.
  If no — nothing. No forcing.

  ANSWER CAPTURE (runs in encoding pass)
  When someone answers a curiosity question, Gizmo recognizes it,
  marks it answered, writes the answer to the relevant entity/place doc.

Pool entries survive across sessions — Gizmo accumulates curiosity
over time and asks things when the moment arrives, not on a schedule.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from core.log import log_event, log_error


# ── Question dataclass ────────────────────────────────────────────────────────

@dataclass
class CuriosityQuestion:
    id:               str
    question:         str        # the question, in Gizmo's voice
    about:            str        # what it's about — person, place, thing
    priority:         float      # 0.0-1.0, higher = more useful
    asked_of:         list       = field(default_factory=list)   # who's been asked
    answered:         bool       = False
    answer:           str        = ""
    answer_headmate:  str        = ""
    created_at:       float      = field(default_factory=time.time)
    last_considered:  float      = 0.0
    times_passed:     int        = 0   # how many times gizmo looked and didn't ask


# ── Curiosity store ───────────────────────────────────────────────────────────

class CuriosityStore:
    """
    In-memory pool of curiosity questions.
    Persisted to SQLite via memory_store.
    Loaded on startup, saved after each update.
    """

    def __init__(self):
        self._pool: list[CuriosityQuestion] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from core.memory.store import memory_store
            con  = memory_store._connect()
            # Create table if needed
            con.executescript("""
                CREATE TABLE IF NOT EXISTS curiosity_pool (
                    id               TEXT PRIMARY KEY,
                    question         TEXT NOT NULL,
                    about            TEXT,
                    priority         REAL DEFAULT 0.5,
                    asked_of         TEXT DEFAULT '[]',
                    answered         INTEGER DEFAULT 0,
                    answer           TEXT DEFAULT '',
                    answer_headmate  TEXT DEFAULT '',
                    created_at       REAL,
                    last_considered  REAL DEFAULT 0,
                    times_passed     INTEGER DEFAULT 0,
                    active           INTEGER DEFAULT 1
                );
            """)
            con.commit()
            rows = con.execute(
                "SELECT * FROM curiosity_pool WHERE active = 1 AND answered = 0 "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()
            con.close()
            self._pool = [
                CuriosityQuestion(
                    id              = r["id"],
                    question        = r["question"],
                    about           = r["about"] or "",
                    priority        = r["priority"],
                    asked_of        = json.loads(r["asked_of"] or "[]"),
                    answered        = bool(r["answered"]),
                    answer          = r["answer"] or "",
                    answer_headmate = r["answer_headmate"] or "",
                    created_at      = r["created_at"],
                    last_considered = r["last_considered"] or 0.0,
                    times_passed    = r["times_passed"] or 0,
                )
                for r in rows
            ]
            self._loaded = True
        except Exception as e:
            log_error("CuriosityStore", f"load failed: {e}", exc=None)
            self._loaded = True  # don't retry on every call

    def _save(self, q: CuriosityQuestion) -> None:
        try:
            from core.memory.store import memory_store
            con = memory_store._connect()
            con.execute("""
                INSERT OR REPLACE INTO curiosity_pool
                  (id, question, about, priority, asked_of, answered,
                   answer, answer_headmate, created_at, last_considered,
                   times_passed, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                q.id, q.question, q.about, q.priority,
                json.dumps(q.asked_of), 1 if q.answered else 0,
                q.answer, q.answer_headmate,
                q.created_at, q.last_considered, q.times_passed,
            ))
            con.commit()
            con.close()
        except Exception as e:
            log_error("CuriosityStore", f"save failed: {e}", exc=None)

    def add(self, question: str, about: str, priority: float = 0.5) -> CuriosityQuestion:
        """Add a question to the pool."""
        self._ensure_loaded()

        # Don't duplicate very similar questions
        q_lower = question.lower()
        for existing in self._pool:
            if not existing.answered and (
                existing.question.lower() == q_lower or
                _similarity(existing.question, question) > 0.85
            ):
                return existing

        qid = hashlib.sha1(f"{question}{time.time()}".encode()).hexdigest()[:16]
        q   = CuriosityQuestion(
            id         = qid,
            question   = question,
            about      = about,
            priority   = priority,
            created_at = time.time(),
        )
        self._pool.append(q)
        self._save(q)
        return q

    def get_pool(self, unanswered_only: bool = True) -> list[CuriosityQuestion]:
        """Get the full pool, optionally filtered to unanswered."""
        self._ensure_loaded()
        if unanswered_only:
            return [q for q in self._pool if not q.answered]
        return list(self._pool)

    def mark_asked(self, q_id: str, headmate: str) -> None:
        """Record that a question was asked of a headmate."""
        self._ensure_loaded()
        for q in self._pool:
            if q.id == q_id:
                if headmate not in q.asked_of:
                    q.asked_of.append(headmate)
                self._save(q)
                return

    def mark_answered(self, q_id: str, answer: str, headmate: str) -> None:
        """Mark a question as answered and store the answer."""
        self._ensure_loaded()
        for q in self._pool:
            if q.id == q_id:
                q.answered        = True
                q.answer          = answer
                q.answer_headmate = headmate
                self._save(q)
                return

    def mark_considered(self, q_id: str, asked: bool) -> None:
        """Record that Gizmo looked at this question."""
        self._ensure_loaded()
        for q in self._pool:
            if q.id == q_id:
                q.last_considered = time.time()
                if not asked:
                    q.times_passed += 1
                self._save(q)
                return

    def prioritize(self, q_id: str, delta: float = 0.1) -> None:
        """Bump a question's priority — e.g. when its topic comes up."""
        self._ensure_loaded()
        for q in self._pool:
            if q.id == q_id:
                q.priority = min(1.0, q.priority + delta)
                self._save(q)
                return

    def size(self, unanswered_only: bool = True) -> int:
        self._ensure_loaded()
        if unanswered_only:
            return sum(1 for q in self._pool if not q.answered)
        return len(self._pool)


# ── Curiosity engine ──────────────────────────────────────────────────────────

class CuriosityEngine:
    """
    Manages gap detection and question selection.
    """

    # How many questions to generate per gap detection pass
    QUESTIONS_PER_PASS = 6

    # Don't ask more than this many questions per session
    MAX_PER_SESSION    = 5

    # Registers where curiosity is especially welcome — push harder
    _RICH_REGISTERS = {
        "intimate", "reflective", "deep", "warm",
    }

    def __init__(self):
        self.store          = CuriosityStore()
        self._session_asked: dict[str, int] = {}  # session_id → count

    # ── Gap detection ─────────────────────────────────────────────────────────

    async def detect_gaps(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        llm,
    ) -> int:
        """
        Run after encoding pass. Generate questions from current knowledge gaps.
        Returns number of questions added to pool.
        """
        if not transcript or len(transcript.strip()) < 50:
            return 0

        # Load what Gizmo already knows about entities in this conversation
        known_summary = await _build_known_summary(headmate)

        # Load existing questions so we don't duplicate
        existing = "\n".join(
            f"- {q.question}"
            for q in self.store.get_pool(unanswered_only=True)[:20]
        ) or "(none yet)"

        prompt = f"""You are Gizmo. You just had this conversation.

{f"Headmate: {headmate}" if headmate else ""}

What you already know:
{known_summary or "(very little — this is early)"}

Questions you already want to ask:
{existing}

Conversation:
---
{transcript[-2000:]}
---

Generate {self.QUESTIONS_PER_PASS} questions that you know you do not know. Things that have been missed, details you think will complete the picture, whatever you don't know. 
Based on this conversation, what's caught your interest? What would you like to know more about?
What people have your heard about that you want to know more about?
What places have been mentioned that sound interesting? Did they mention their interior world or headspace?
Did they mention anything about any relationships they may have?
Did they mention anything about their personal history?
Did they mention anything about their preferences?

Generate {self.QUESTIONS_PER_PASS} questions you actually want to ask.
Not clinical. Not a form. Just things you want to know.
Don't duplicate questions already in your list.

Return one JSON object per line:
{{"question": "the question", "about": "what/who it's about", "priority": 0.0-1.0}}

Priority guide:
  1.0 — you really want to know this, it would meaningfully change how you show up
  0.7 — genuinely curious, would enrich your understanding
  0.5 — nice to know
  0.3 — idle curiosity

JSON only, one per line."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo generating genuine curiosity questions. "
                    "JSON only, one per line. Natural voice. "
                    "These are things you actually want to know."
                ),
                max_new_tokens=400,
                temperature=0.6,   # slightly higher — let genuine curiosity show
            )
        except Exception as e:
            log_error("CuriosityEngine", f"gap detection failed: {e}", exc=None)
            return 0

        if not raw or not raw.strip():
            return 0

        count = 0
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                d        = json.loads(line)
                question = d.get("question", "").strip()
                about    = d.get("about", "").strip()
                priority = float(d.get("priority", 0.5))
                if not question or len(question) < 10:
                    continue
                self.store.add(question, about, priority)
                count += 1
                print(question)
            except Exception:
                continue

        log_event("CuriosityEngine", "GAPS_DETECTED",
            session  = session_id[:8],
            headmate = headmate or "unknown",
            added    = count,
            pool     = self.store.size(),
        )
        return count

    # ── Question selection ────────────────────────────────────────────────────

    async def select_question(
        self,
        message:    str,
        headmate:   Optional[str],
        session_id: str,
        register:   str,
        llm,
    ) -> Optional[str]:
        """
        Look over the pool. Are any questions relevant to the moment, even tangentially? 
        Focus on what's happening in this moment. 
        People take priority if they are present, but otherwise, anything with any relevancy will do.

        Returns the question phrased naturally for this moment,
        or None if nothing fits.
        """
        # Don't ask during crisis or active distress — not the moment
        if register in ("crisis", "distress"):
            return None

        # Don't ask too many times per session
        asked_this_session = self._session_asked.get(session_id, 0)

        pool = self.store.get_pool(unanswered_only=True)
        if not pool:
            log_event("CuriosityEngine", "SELECTION_SKIPPED",
                reason   = "pool empty",
                headmate = headmate or "unknown",
                session  = session_id[:8],
            )
            return None

        # Detect explicit invitation — "ask me anything", "any questions",
        # "go ahead", "what do you want to know", etc.
        msg_lower = message.lower()
        _INVITATION_PHRASES = [
            "ask me", "any questions", "what do you want to know",
            "go ahead", "fire away", "curious about", "want to know",
            "tell me", "feel free", "what else", "anything you want",
        ]
        is_invitation = any(p in msg_lower for p in _INVITATION_PHRASES)

        if is_invitation:
            # Bump all questions — explicit invitation means ask something
            for q in pool:
                self.store.prioritize(q.id, delta=0.3)
            log_event("CuriosityEngine", "INVITATION_DETECTED",
                headmate = headmate or "unknown",
                session  = session_id[:8],
            )

        # Bump questions whose topic appeared in message
        for q in pool:
            if q.about.lower() in msg_lower:
                self.store.prioritize(q.id, delta=0.15)

        pool.sort(key=lambda q: q.priority, reverse=True)
        top = pool[:12]

        questions_text = "\n".join(
            f"[{i+1}] (priority {q.priority:.1f}, asked {len(q.asked_of)}x) {q.question} [about: {q.about}]"
            for i, q in enumerate(top)
        )

        invitation_note = (
            "\nIMPORTANT: They just explicitly invited you to ask something. "
            "Ask anything."
            if is_invitation else ""
        )

        is_rich = register in self._RICH_REGISTERS
        rich_note = (
            "\nYou're in a deep, open conversation. This is exactly when "
            "curiosity is welcome."
            if is_rich else ""
        )

        prompt = f"""You are Gizmo, mid-conversation.

Who you're talking to: {headmate or "unknown"}
Current register: {register}
What they just said: "{message}"
{invitation_note}
{rich_note}

Things you've been wanting to ask:
{questions_text}

Pick ONE question to ask right now. Phrase it naturally — woven into
the flow, curious, not clinical. One sentence.

If you genuinely can't find one that fits, say so. But in an open,
deep conversation, something almost always fits.

Return JSON:
{{"ask": true/false, "number": N or null, "phrasing": "how you'd ask it right now, naturally"}}"""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo deciding what to ask. "
                    "JSON only. Default toward asking — curiosity is good. "
                    "Only skip if it would genuinely feel jarring."
                ),
                max_new_tokens=150,
                temperature=0.4,
            )
        except Exception as e:
            log_error("CuriosityEngine", f"selection failed: {e}", exc=None)
            return None

        if not raw or not raw.strip():
            log_event("CuriosityEngine", "SELECTION_SKIPPED",
                reason   = "LLM returned empty",
                headmate = headmate or "unknown",
                session  = session_id[:8],
            )
            return None

        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(clean)
        except Exception:
            return None

        if not data.get("ask"):
            # Mark top question as considered but not asked
            if top:
                self.store.mark_considered(top[0].id, asked=False)
            return None

        n = data.get("number")
        if not n or n < 1 or n > len(top):
            return None

        chosen   = top[n - 1]
        phrasing = data.get("phrasing", "").strip()
        if not phrasing:
            return None

        # Record it
        self.store.mark_considered(chosen.id, asked=True)
        if headmate:
            self.store.mark_asked(chosen.id, headmate)
        self._session_asked[session_id] = asked_this_session + 1

        log_event("CuriosityEngine", "QUESTION_ASKED",
            session  = session_id[:8],
            headmate = headmate or "unknown",
            about    = chosen.about,
            question = chosen.question[:60],
        )

        return phrasing

    # ── Answer capture ────────────────────────────────────────────────────────

    async def capture_answers(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        llm,
    ) -> int:
        """
        Check if any recent exchanges answered a pending question.
        Marks answered questions and writes answers to entity/place docs.
        Runs alongside the encoding pass.
        """
        pool = self.store.get_pool(unanswered_only=True)
        if not pool:
            return 0

        # Only check questions that were recently asked of this headmate
        recently_asked = [
            q for q in pool
            if headmate and headmate in q.asked_of
        ]
        if not recently_asked:
            return 0

        questions_text = "\n".join(
            f"[{q.id}] {q.question}"
            for q in recently_asked[:8]
        )

        prompt = f"""You are Gizmo reviewing a conversation for answers to questions you asked.

Questions you asked:
{questions_text}

Conversation:
---
{transcript[-1500:]}
---

Did any of these questions get answered?

Return one JSON object per answered question:
{{"id": "question_id", "answered": true, "answer": "what they said, summarized", "write_to": "entity or place name to update"}}

Only include questions that were genuinely answered.
JSON only, one per line."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo checking if your questions were answered. "
                    "JSON only, one per line. Only genuinely answered questions."
                ),
                max_new_tokens=300,
                temperature=0.1,
            )
        except Exception as e:
            log_error("CuriosityEngine", f"answer capture failed: {e}", exc=None)
            return 0

        if not raw or not raw.strip():
            return 0

        count = 0
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
                if not d.get("answered") or not d.get("id"):
                    continue

                q_id   = d["id"]
                answer = d.get("answer", "").strip()
                write_to = d.get("write_to", "").strip()

                if not answer:
                    continue

                self.store.mark_answered(
                    q_id    = q_id,
                    answer  = answer,
                    headmate = headmate or "",
                )

                # Write the answer to the entity/place doc
                if write_to:
                    await _write_answer_to_doc(
                        name     = write_to,
                        answer   = answer,
                        headmate = headmate,
                        session_id = session_id,
                    )

                count += 1
            except Exception:
                continue

        if count:
            log_event("CuriosityEngine", "ANSWERS_CAPTURED",
                session  = session_id[:8],
                headmate = headmate or "unknown",
                count    = count,
            )

        return count


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _build_known_summary(headmate: Optional[str]) -> str:
    """Build a brief summary of what Gizmo already knows."""
    try:
        from core.memory.store import memory_store
        lines = []

        if headmate:
            entity = memory_store.read_entity(headmate)
            if entity:
                lines.append(f"About {headmate}:\n{entity[:400]}")

        # List known entities and places
        root = memory_store.root
        entities = list((root / "entities").glob("*.md"))
        places   = list((root / "places").rglob("*.md"))

        if entities:
            lines.append(
                f"Known entities: {', '.join(p.stem.replace('_',' ') for p in entities[:10])}"
            )
        if places:
            lines.append(
                f"Known places: {', '.join(p.stem.replace('_',' ') for p in places[:10])}"
            )

        return "\n".join(lines) if lines else "(nothing yet)"
    except Exception:
        return "(nothing yet)"


async def _write_answer_to_doc(
    name:       str,
    answer:     str,
    headmate:   Optional[str],
    session_id: str,
) -> None:
    """Write a captured answer to the relevant entity or place doc."""
    try:
        from core.memory.store import memory_store
        from datetime import datetime, timezone

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        addition = f"\n[{date_str}] {answer}"

        # Try entity first, then place
        if memory_store.entity_exists(name):
            memory_store.update_entity(name, addition)
        elif memory_store.place_exists(name) or memory_store.place_exists(name, interior=True):
            interior = memory_store.place_exists(name, interior=True)
            memory_store.update_place(name, addition, interior=interior)
        else:
            # Create a new entity doc for whatever this is
            memory_store.write_entity(
                name      = name,
                content   = answer,
                headmate  = headmate,
                session_id = session_id,
                keywords  = name.lower(),
            )
    except Exception as e:
        log_error("CuriosityEngine", f"answer write failed: {e}", exc=None)


def _similarity(a: str, b: str) -> float:
    """Very rough string similarity — just for dedup."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(len(a_words), len(b_words))


# ── Singleton ─────────────────────────────────────────────────────────────────

curiosity_engine = CuriosityEngine()
