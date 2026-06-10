"""
core/chunk_processor.py

Per-chunk pipeline:
  1. Subject discovery   — who/what is named in this chunk
  2. Pronoun resolution  — match pronouns against the live registry
  3. Descriptors + behaviors + wellness in parallel via asyncio.gather
  4. Action buffer       — unpaired actions held up to 3 chunks, then dropped
  5. Merge               — write resolved data into per-person files via librarian

Chunk size is configurable. Chunks fired before reaching chunk_size are
flagged as partial. Partials are checked for relevance against the next
input before being promoted or dropped.

Each chunk gets a unique ID: {session_id}-{sequence:03d}
"""

import asyncio
import json
import re
import time
from typing import Optional

from core.log import log_event, log_error
from core.Descriptor_catcher import descriptor_catcher as describer
from core.BehaviorCatcher import behaviorcatcher as behavior
from core.wellness import wellness_collector as wellness
import core.librarian as librarian


# ── Subject discovery ─────────────────────────────────────────────────────────

_DISCOVERY_SYSTEM = """
You identify subjects and resolve pronouns from a short conversational chunk.
Return ONLY valid JSON. No markdown. No explanation.

{
  "subjects": [
    {"name": "Ember", "type": "Person"},
    {"name": "the warehouse", "type": "Place"}
  ],
  "pronouns": [
    {"pronoun": "she", "resolves_to": "Ember"},
    {"pronoun": "it", "resolves_to": "the warehouse"}
  ]
}

Conversational frame — always apply this:
- "Gizmo" or "you" (when addressed) refers to the AI companion, not a system member
- "I", "me", "us", "we" refers to the plural system being served
- Lines prefixed "Gizmo:" are responses from the AI companion
- Gizmo is never a subject for wellness or behavioral assessment
- Do not include Gizmo as a subject unless explicitly needed for pronoun resolution

Rules:
- subjects: every named person, place, or object explicitly mentioned (excluding Gizmo)
- pronouns: every pronoun present and your best resolution given the chunk
- If a pronoun cannot be resolved from the chunk alone, set resolves_to to null
- type is one of: Person, Place, Object, Animal
""".strip()

_RELEVANCE_SYSTEM = """
You determine whether a new conversational input continues or responds to a previous partial chunk.
Return ONLY valid JSON. No markdown. No explanation.

{ "relevant": true }
or
{ "relevant": false }

A new input is relevant if it:
- References the same people or subjects
- Continues the same topic or thread
- Contains a response to something said in the partial

If in doubt, return false.
""".strip()


async def _discover_subjects(chunk: list[str], registry: dict, host: str = "unknown") -> tuple[list[dict], list[dict]]:
    try:
        from core.llm import llm
        known  = list(registry.keys())
        prompt = (
            f"Known subjects so far: {json.dumps(known)}\n\n"
            f"Current speaker: {host}\n"
            f"'I', 'me', 'my' in this chunk refers to {host} unless another name is used.\n\n"
            f"Chunk:\n" + "\n".join(chunk)
        )
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_DISCOVERY_SYSTEM,
            temperature=0.0,
            max_new_tokens=2000,
        )
        if not raw or not raw.strip():
            return [], []
        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        return parsed.get("subjects", []), parsed.get("pronouns", [])
    except Exception as e:
        log_error("ChunkProcessor", "subject discovery failed", exc=e)
        print(f"[ChunkProcessor] discovery failed: {type(e).__name__}: {e}")
        return [], []


async def _check_relevance(partial_lines: list[str], new_lines: list[str]) -> bool:
    try:
        from core.llm import llm
        prompt = (
            f"Partial chunk:\n" + "\n".join(partial_lines) +
            f"\n\nNew input:\n" + "\n".join(new_lines)
        )
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_RELEVANCE_SYSTEM,
            temperature=0.0,
            max_new_tokens=50,
        )
        if not raw or not raw.strip():
            return False
        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        return bool(parsed.get("relevant", False))
    except Exception as e:
        log_error("ChunkProcessor", "relevance check failed", exc=e)
        print(f"[ChunkProcessor] relevance check failed: {type(e).__name__}: {e}")
        return False


# ── Action buffer helpers ─────────────────────────────────────────────────────

def _make_pending(subject: str, actions: list[str]) -> list[dict]:
    return [{"subject": subject, "action": a, "age": 0} for a in actions]


def _age_buffer(buffer: list[dict], max_age: int = 3) -> list[dict]:
    aged = []
    for entry in buffer:
        entry["age"] += 1
        if entry["age"] < max_age:
            aged.append(entry)
        else:
            print(f"[ChunkProcessor] dropping stale action: {entry['subject']} → {entry['action']}")
    return aged


def _remove_matched(buffer: list[dict], behavior_results: list[dict]) -> list[dict]:
    matched_actions = set()
    for person in behavior_results:
        for episode in person.get("Episodes", []):
            matched_actions.add(episode.get("action", "").lower()[:60])
    return [
        entry for entry in buffer
        if entry["action"].lower()[:60] not in matched_actions
    ]


# ── Chunk processor ───────────────────────────────────────────────────────────

class ChunkProcessor:

    def __init__(
        self,
        session_id:    str,
        host:          str   = "unknown",
        chunk_size:    int   = 4,   # exchanges, not lines
        timeout_sec:   float = 600.0,
    ):
        self.session_id       = session_id
        self.host             = host
        self.chunk_size       = chunk_size
        self.timeout_sec      = timeout_sec
        self.registry:        dict[str, dict]  = {}
        self.results:         list[dict]       = []
        self.action_buffer:   list[dict]       = []
        self._line_buffer:    list[str]        = []
        self._exchange_count: int              = 0
        self._buffer_start:   float            = 0.0
        self._partial:        Optional[list[str]] = None
        self._chunk_seq:      int              = 0

    def _next_chunk_id(self) -> str:
        self._chunk_seq += 1
        return f"{self.session_id}-{self._chunk_seq:03d}"

    def _update_registry(self, subjects: list[dict]) -> None:
        for s in subjects:
            name = s.get("name")
            if name and name not in self.registry:
                self.registry[name] = {"type": s.get("type", "Person")}
                print(f"[ChunkProcessor] new subject: {name}")

    def _apply_pronoun_resolutions(self, resolutions: list[dict]) -> None:
        for r in resolutions:
            pronoun     = r.get("pronoun")
            resolves_to = r.get("resolves_to")
            if pronoun and resolves_to and resolves_to in self.registry:
                self.registry.setdefault("_pronouns", {})[pronoun] = resolves_to

    async def _run_chunk(self, chunk: list[str], partial: bool = False) -> dict:
        # ── Keyphrase triggers ────────────────────────────────────────────────
        text_lower = " ".join(chunk).lower()
        if "run wellness report" in text_lower:
            print("[ChunkProcessor] wellness report triggered")
            from core.wellness_synthesis import wellness_synthesis
            await wellness_synthesis.run()
            return {"chunk_id": "trigger", "chunk": chunk, "trigger": "wellness_report"}

        chunk_id = self._next_chunk_id()
        text     = "\n".join(chunk)
        flag     = "PARTIAL" if partial else "normal"
        print(f"[ChunkProcessor] firing {flag} chunk {chunk_id} ({len(chunk)} lines)")

        self.action_buffer = _age_buffer(self.action_buffer)

        # Seed registry with known host so "I" always resolves
        if self.host and self.host != "unknown" and self.host not in self.registry:
            self.registry[self.host] = {"type": "Person"}

        # ── 1. Subject discovery ──────────────────────────────────────────────
        subjects, pronoun_resolutions = await _discover_subjects(chunk, self.registry, self.host)
        print(f"[DEBUG] host at chunk fire: {repr(self.host)}")
        self._update_registry(subjects)
        self._apply_pronoun_resolutions(pronoun_resolutions)
        print(f"[DEBUG] registry after discovery: {self.registry}")
        print(f"[DEBUG] chunk: {chunk}")

        # ── 2. Descriptors + behaviors + wellness in parallel ─────────────────
        descriptor_dict, behavior_results, wellness_signals = await asyncio.gather(
            describer.extract(
                user_message=text,
                thread=text,
                subject=self.host,
                session_file=self.session_id,
            ),
            behavior.extract(
                user_message=text,
                thread=text,
                subject=self.host,
                session_file=self.session_id,
                pending_actions=self.action_buffer,
            ),
            wellness.collect(
                chunk=chunk,
                chunk_id=chunk_id,
                registry=self.registry,
            ),
        )

        descriptor_dict  = descriptor_dict  or {}
        behavior_results = behavior_results or []
        wellness_signals = wellness_signals or []

        # ── 3. Merge descriptors ──────────────────────────────────────────────
        if descriptor_dict:
            for name, data in descriptor_dict.items():
                librarian.merge_descriptors(name, data)

        # ── 4. Merge behaviors + update action buffer ─────────────────────────
        if behavior_results:
            for person in behavior_results:
                name = person.get("Subject")
                if not name:
                    continue
                librarian.merge_behaviors(name, person)
                new_actions = person.get("Actions", [])
                if new_actions:
                    self.action_buffer.extend(_make_pending(name, new_actions))
            self.action_buffer = _remove_matched(self.action_buffer, behavior_results)

        result = {
            "chunk_id":       chunk_id,
            "chunk":          chunk,
            "partial":        partial,
            "subjects":       [k for k in self.registry.keys() if not k.startswith("_")],
            "descriptors":    descriptor_dict,
            "behaviors":      behavior_results,
            "wellness":       wellness_signals,
            "pending_buffer": len(self.action_buffer),
        }

        self.results.append(result)
        return result

    async def push_line(self, line: str) -> Optional[dict]:
        line = line.strip()
        if not line:
            return None

        if self._partial and len(self._line_buffer) == 0:
            relevant = await _check_relevance(self._partial, [line])
            if relevant:
                print("[ChunkProcessor] partial is relevant — prepending to new input")
                self._line_buffer = self._partial + self._line_buffer
            else:
                print("[ChunkProcessor] partial not relevant — dropping")
            self._partial = None

        if len(self._line_buffer) == 0:
            self._buffer_start = time.monotonic()
            self._exchange_count = 0

        self._line_buffer.append(line)

        # Every 2 lines = 1 exchange
        if len(self._line_buffer) % 2 == 0:
            self._exchange_count += 1

        # Fire on chunk_size exchanges
        if self._exchange_count >= self.chunk_size:
            chunk = self._line_buffer.copy()
            self._line_buffer.clear()
            self._exchange_count = 0
            return await self._run_chunk(chunk, partial=False)

        elapsed = time.monotonic() - self._buffer_start
        if elapsed >= self.timeout_sec and self._line_buffer:
            chunk = self._line_buffer.copy()
            self._line_buffer.clear()
            self._partial = chunk
            return await self._run_chunk(chunk, partial=True)

        return None

    async def process(self, chunk: list[str]) -> dict:
        """Direct chunk injection — bypasses line buffer."""
        return await self._run_chunk(chunk, partial=False)

    async def flush(self) -> Optional[dict]:
        if self._line_buffer:
            chunk = self._line_buffer.copy()
            self._line_buffer.clear()
            self._exchange_count = 0
            timed_out = self._partial is not None
            self._partial = None
            return await self._run_chunk(chunk, partial=timed_out)
        print(
            f"[ChunkProcessor] session complete. "
            f"registry: {[k for k in self.registry.keys() if not k.startswith('_')]} | "
            f"dropped buffer: {len(self.action_buffer)}"
        )
        return None