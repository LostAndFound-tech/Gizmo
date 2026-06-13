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
import os
import re
import time
from typing import Optional

from core.log import log_event, log_error
from core.Descriptor_catcher import descriptor_catcher as describer
from core.BehaviorCatcher import behaviorcatcher as behavior
from core.wellness_router import wellness_router as wellness
from core.scene_tracker import scene_tracker as scene
from core.preference_catcher import preference_catcher
import core.librarian as librarian


# ── Known headmates ───────────────────────────────────────────────────────────

def _get_known_headmates() -> list[str]:
    """
    Return names of all headmates with existing behavior or descriptor files.
    These are the only people who can be created as Person subjects.
    New named people (coworkers, friends) are allowed if they look like proper names.
    Generic descriptors (naked girl, the speaker) are never allowed as subjects.
    """
    names = set()
    data_dir = os.environ.get("DATA_DIR", "./data")
    for subfolder in ("behaviors", "descriptors"):
        folder = os.path.join(data_dir, subfolder)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".json"):
                names.add(fname[:-5].lower())
    return sorted(names)


# ── Subject discovery ─────────────────────────────────────────────────────────

_DISCOVERY_SYSTEM = """
You identify subjects and resolve pronouns from a short conversational chunk.
Return ONLY valid JSON. No markdown. No explanation.

{
  "subjects": [
    {"name": "Jess", "type": "Person", "resolution": "known_headmate"},
    {"name": "Ara", "type": "Person", "resolution": "known_headmate"},
    {"name": "the warehouse", "type": "Place", "resolution": "new"},
    {"name": "beer bottle", "type": "Object", "resolution": "new"}
  ],
  "pronouns": [
    {"pronoun": "she", "resolves_to": "Jess"},
    {"pronoun": "it", "resolves_to": "the warehouse"}
  ]
}

Conversational frame — always apply this:
- "Gizmo" or "you" (when addressed) refers to the AI companion, not a system member
- "I", "me", "us", "we" refers to the plural system being served
- Lines prefixed "Gizmo:" are responses from the AI companion
- Gizmo is never a subject for wellness or behavioral assessment

SUBJECT RULES — read carefully:

For PEOPLE:
- Only use real names (proper nouns) as Person subjects — e.g. "Jess", "Ara", "Kaylee"
- If someone is described with a generic descriptor ("naked girl", "little girl", "the speaker",
  "unknown", "user", "inferior", "sir", "the girl") — do NOT create a subject for that descriptor
  Instead, resolve it to the nearest known headmate via pronouns
- If "I", "me", "my" appears, it resolves to the current speaker (provided below) — not a new subject
- Pronouns (she, her, he, him, they, them, we, us, you, it) are NEVER subject names
- A new named person (someone not in the known list, but referred to by a real name like "Marcus" or
  "Dr. Chen") is allowed as a Person subject with resolution "new_person"
- If unsure whether something is a name or a descriptor, omit it

For PLACES:
- Include rooms, buildings, locations, spaces explicitly mentioned
- "middle space", "headspace", "the office", "Gizmo's space" are all valid Place subjects

For OBJECTS:
- Include specific objects that are described or interacted with
- "beer bottle", "collar", "the tank" are valid Object subjects

resolution field:
- "known_headmate" — name matches a known headmate
- "new_person" — a real proper name not yet in the known list
- "new" — a new place or object

CRITICAL: Do not invent subjects. Only return subjects explicitly present in the chunk text.
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
        known_headmates = _get_known_headmates()
        known_in_registry = list(registry.keys())

        prompt = (
            f"Known headmates (the only people who exist in this system): {json.dumps(known_headmates)}\n\n"
            f"Subjects already in registry this session: {json.dumps(known_in_registry)}\n\n"
            f"Current speaker: {host}\n"
            f"'I', 'me', 'my' in this chunk refers to {host} — do NOT create a new subject for these pronouns.\n\n"
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

        subjects  = parsed.get("subjects", [])
        pronouns  = parsed.get("pronouns", [])

        # ── Post-filter: enforce rules the LLM may have missed ────────────────
        _PRONOUN_BLOCKLIST = {
            "i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves",
        }
        _DESCRIPTOR_BLOCKLIST = {
            "the speaker", "unknown speaker", "unknown", "user", "speaker",
            "system member", "unknown system member", "unknown headmate",
            "the girl", "a girl", "the boy", "a boy", "the person",
            "little girl", "naked girl", "inferior", "sir", "ma'am",
            "the user", "a person", "someone", "anyone",
        }

        filtered = []
        for s in subjects:
            name = (s.get("name") or "").strip()
            name_lower = name.lower()

            # Never allow pronouns
            if name_lower in _PRONOUN_BLOCKLIST:
                print(f"[ChunkProcessor] filtered pronoun subject: {name}")
                continue

            # Never allow generic descriptors
            if name_lower in _DESCRIPTOR_BLOCKLIST:
                print(f"[ChunkProcessor] filtered descriptor subject: {name}")
                continue

            # For Person type — must be a known headmate OR a proper name (capitalized, not a common noun)
            if s.get("type") == "Person":
                is_known = name_lower in known_headmates
                is_proper = (
                    name and
                    name[0].isupper() and
                    len(name.split()) <= 3 and
                    name_lower not in _DESCRIPTOR_BLOCKLIST
                )
                if not is_known and not is_proper:
                    print(f"[ChunkProcessor] filtered non-proper person subject: {name}")
                    continue

            filtered.append(s)

        return filtered, pronouns

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
        chunk_size:    int   = 4,
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
        known_traits   = {}
        known_episodes = []
        for subject_name in [k for k in self.registry.keys() if not k.startswith("_")]:
            traits = librarian.get_known_traits(subject_name)
            if traits:
                known_traits[subject_name] = traits
        if self.host and self.host != "unknown":
            host_data = librarian._read_file(f"behaviors/{self.host.lower()}.json") or {}
            known_episodes = host_data.get("Episodes", [])

        descriptor_dict, behavior_results, wellness_signals, _, pref_result = await asyncio.gather(
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
                known_traits=known_traits,
                known_episodes=known_episodes,
            ),
            wellness.process(
                chunk=chunk,
                chunk_id=chunk_id,
                registry=self.registry,
            ),
            scene.update(
                chunk=chunk,
                chunk_id=chunk_id,
                name=self.host,
                session_id=self.session_id,
            ),
            preference_catcher.extract(
                chunk=chunk,
                session_id=self.session_id,
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
            "preferences":    pref_result or {},
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

        if len(self._line_buffer) % 2 == 0:
            self._exchange_count += 1

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
