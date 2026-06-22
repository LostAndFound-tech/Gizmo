"""
core/knowledge_writer.py

Extracts discrete facts from conversational exchanges and writes them to
Gizmo's knowledge store.

Store layout:
  {DATA_DIR}/knowledge/
    vocabulary.json                  ← shared tag vocabulary (source of truth)
    index.json                       ← flat index of every fact ever written
    system/
      external/
        home.json                    ← shared physical space
        people.json                  ← external people
        work.json                    ← jobs, workplace, coworkers
        pets.json                    ← animals
        [topic].json                 ← expands as needed
    {headmate}/
      internal_space.json            ← headmate's internal experience
      preferences.json               ← likes, dislikes, habits
      [topic].json                   ← expands as needed
    gizmo/
      threads.json                   ← things Gizmo wants to follow up on

Entry schema:
  {
    "fact":       "Jess rearranged her living room today",
    "source":     "jess",
    "internal":   false,
    "place":      "living room",           ← optional
    "objects":    ["cat palace", "coffee table"],  ← optional
    "tags":       ["living-room", "furniture", "cats"],
    "confidence": "stated | implied | uncertain",
    "route":      "system/external/home",
    "ts":         "2026-06-22T18:00:00+00:00"
  }

Runs in the chunk_processor parallel gather alongside descriptors,
behaviors, and wellness. Fire and forget — never blocks the response.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Vocabulary ────────────────────────────────────────────────────────────────

_VOCAB_FILE = "knowledge/vocabulary.json"
_INDEX_FILE = "knowledge/index.json"


def _read_vocabulary() -> list[str]:
    data = librarian._read_file(_VOCAB_FILE)
    if isinstance(data, dict):
        return data.get("tags", [])
    return []


def _write_vocabulary(tags: list[str]) -> None:
    librarian._write_json(_VOCAB_FILE, {"tags": sorted(tags)})


def _read_index() -> list[dict]:
    data = librarian._read_file(_INDEX_FILE)
    if isinstance(data, list):
        return data
    return []


def _write_index(index: list[dict]) -> None:
    librarian._write_json(_INDEX_FILE, index)


# ── Routing ───────────────────────────────────────────────────────────────────

# Known external topic buckets. LLM can propose new ones — we normalise them.
_EXTERNAL_BUCKETS = {
    "home", "people", "work", "pets", "places",
    "routines", "events", "objects", "media", "food",
}

# Known headmate-level topic buckets.
_HEADMATE_BUCKETS = {
    "internal_space", "preferences", "history",
    "relationships", "opinions", "routines",
}


def _normalise_route(raw_route: str, known_headmates: list[str]) -> str:
    """
    Normalise a raw route string from the LLM into a valid file path
    relative to the knowledge/ directory.

    Expected LLM outputs (examples):
      "system/external/home"
      "jess/preferences"
      "gizmo/threads"

    Falls back to "system/external/general" on anything unparseable.
    """
    raw = raw_route.strip().lower().strip("/")
    parts = raw.split("/")

    # gizmo/threads
    if parts[0] == "gizmo":
        sub = parts[1] if len(parts) > 1 else "threads"
        return f"gizmo/{sub}"

    # system/external/[topic]
    if parts[0] == "system":
        topic = parts[-1] if len(parts) >= 3 else "general"
        # Sanitise topic name
        topic = re.sub(r"[^a-z0-9_-]", "-", topic)
        return f"system/external/{topic}"

    # [headmate]/[topic]
    if parts[0] in [h.lower() for h in known_headmates]:
        topic = parts[1] if len(parts) > 1 else "general"
        topic = re.sub(r"[^a-z0-9_-]", "-", topic)
        return f"{parts[0]}/{topic}"

    return "system/external/general"


_RETRIEVAL_SYSTEM = """
You are matching a conversational message against a tag vocabulary.
Return ONLY a valid JSON array of matching tags. No markdown. No explanation.

Return only tags from the provided vocabulary that genuinely apply to this message.
If nothing matches, return [].
""".strip()

async def get_relevant_tags(message: str, vocabulary: list[str]) -> list[str]:
    try:
        from core.llm import llm
        prompt = f"Vocabulary:\n{json.dumps(vocabulary)}\n\nMessage:\n{message}"
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_RETRIEVAL_SYSTEM,
            temperature=0.0,
            max_new_tokens=200,
        )
        if not raw or not raw.strip():
            return []
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("KnowledgeWriter", "tag retrieval failed", exc=e)
        return []


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """
You are Gizmo's knowledge intake pass. Your job is to extract discrete, reusable facts
from a conversational exchange and route them to the right part of Gizmo's knowledge store.

You will receive:
- The exchange (user message + Gizmo's response)
- Who is speaking (the headmate)
- Known headmates in the system
- The current tag vocabulary

Return ONLY a valid JSON array. No markdown. No explanation. No preamble.
If there is nothing worth storing as a long-term fact, return [].

For each fact, return:
{
  "fact":       "one clear sentence stating what is true",
  "source":     "who said this — headmate name or 'gizmo'",
  "internal":   false,
  "place":      "the physical location this fact is about, if any",
  "objects":    ["specific named objects involved, if any"],
  "tags":       ["tag1", "tag2"],
  "confidence": "stated | implied | uncertain",
  "route":      "system/external/home"
}

ROUTING RULES:
- Facts about the shared physical space (rooms, furniture, layout) → system/external/home
- Facts about external people (Willow, friends, coworkers) → system/external/people
- Facts about work, jobs, workplace → system/external/work
- Facts about pets or animals → system/external/pets
- Facts about places outside the home → system/external/places
- Facts about recurring events or schedules → system/external/routines
- Facts about a headmate's internal experience, feelings about their own space → {headmate}/internal_space
- Facts about a headmate's tastes, preferences, habits → {headmate}/preferences
- Facts about a headmate's opinions → {headmate}/opinions
- Things Gizmo wants to follow up on or is curious about → gizmo/threads
- When in doubt about system vs headmate: if it affects or involves the whole household, system. If it's specific to one headmate's experience of something, headmate.

TAGGING RULES:
- Match existing vocabulary tags first — only coin new tags if nothing fits
- New tags: lowercase, single word or hyphenated phrase, specific not generic
- Use 2–6 tags per fact
- Tags should name the thing, not describe the act: "coffee-table" not "furniture-acquisition"
- Place and objects fields are for named things — don't repeat them as tags unless genuinely useful

WHAT TO EXTRACT:
- Named objects that will come up again (cat palace, coffee table, the lamp)
- Physical layout facts (where things are, what a room looks like)
- External people and their attributes
- Events that happened (rearranged the room, started a new job)
- Preferences and habits revealed through casual mention
- Things Gizmo should remember to ask about later

WHAT TO SKIP:
- Pure emotional states with no factual content (already handled by wellness/behavior)
- Anything too vague to be useful ("she seemed happy")
- Gizmo's own responses unless they reveal something worth tracking
- Greetings, filler, pleasantries

CONFIDENCE:
- stated:   they said it explicitly
- implied:  reasonable inference from what was said
- uncertain: ambiguous — worth noting but not reliable

internal field:
- false: fact about the external world, shared space, or other people
- true:  fact about a headmate's inner experience, internal system space, or personal felt sense
""".strip()


def _build_prompt(
    exchange:        str,
    speaker:         str,
    known_headmates: list[str],
    vocabulary:      list[str],
) -> str:
    return (
        f"Speaker: {speaker}\n"
        f"Known headmates: {', '.join(known_headmates)}\n\n"
        f"Current tag vocabulary:\n{json.dumps(vocabulary)}\n\n"
        f"Exchange:\n{exchange}"
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[list]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SYSTEM,
            temperature=0.0,
            max_new_tokens=4000,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        if not isinstance(parsed, list):
            return None
        return parsed
    except Exception as e:
        log_error("KnowledgeWriter", "LLM call failed", exc=e)
        print(f"[KnowledgeWriter] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── File write ────────────────────────────────────────────────────────────────

def _append_to_knowledge_file(route: str, entry: dict) -> None:
    path     = f"knowledge/{route}.json"
    existing = librarian._read_file(path)
    if not isinstance(existing, list):
        existing = []
    existing.append(entry)
    librarian._write_json(path, existing)


def _update_index(entries: list[dict]) -> None:
    index = _read_index()
    index.extend(entries)
    _write_index(index)


# ── Public API ────────────────────────────────────────────────────────────────

class KnowledgeWriter:

    async def extract(
        self,
        user_message:    str,
        gizmo_response:  str,
        speaker:         str,
        known_headmates: list[str],
        session_id:      str = "",
    ) -> Optional[list[dict]]:
        """
        Extract facts from an exchange and write them to the knowledge store.
        Returns the list of entries written, or None if nothing extracted.
        """
        if not user_message.strip():
            return None

        exchange = f"{speaker}: {user_message.strip()}"
        if gizmo_response and gizmo_response.strip():
            exchange += f"\nGizmo: {gizmo_response.strip()}"

        try:
            vocabulary = _read_vocabulary()
            prompt     = _build_prompt(exchange, speaker, known_headmates, vocabulary)
            facts      = await _call_llm(prompt)

            if not facts:
                log_event("KnowledgeWriter", "NO_FACTS", speaker=speaker, session=session_id[:8])
                return None

            ts           = datetime.now(timezone.utc).isoformat()
            new_tags     = []
            written      = []

            for fact in facts:
                if not isinstance(fact, dict) or not fact.get("fact"):
                    continue

                # Normalise route
                raw_route = fact.get("route", "system/external/general")
                route     = _normalise_route(raw_route, known_headmates)

                # Collect new tags
                for tag in fact.get("tags", []):
                    if tag and tag not in vocabulary and tag not in new_tags:
                        new_tags.append(tag)

                entry = {
                    "fact":       fact.get("fact", ""),
                    "source":     fact.get("source", speaker),
                    "internal":   fact.get("internal", False),
                    "tags":       fact.get("tags", []),
                    "confidence": fact.get("confidence", "stated"),
                    "route":      route,
                    "session_id": session_id,
                    "ts":         ts,
                }

                # Optional fields — only include if present
                if fact.get("place"):
                    entry["place"] = fact["place"]
                if fact.get("objects"):
                    entry["objects"] = fact["objects"]

                _append_to_knowledge_file(route, entry)
                written.append(entry)
                print(f"[KnowledgeWriter] → {route}: {entry['fact'][:60]}")

            # Grow vocabulary with genuinely new tags
            if new_tags:
                vocabulary.extend(new_tags)
                _write_vocabulary(vocabulary)
                print(f"[KnowledgeWriter] new tags coined: {new_tags}")

            # Update flat index
            if written:
                _update_index(written)
                log_event("KnowledgeWriter", "FACTS_WRITTEN",
                    count=len(written),
                    speaker=speaker,
                    session=session_id[:8],
                    new_tags=len(new_tags),
                )

            return written or None

        except Exception as e:
            log_error("KnowledgeWriter", "extract failed", exc=e)
            print(f"[KnowledgeWriter] extract failed: {type(e).__name__}: {e}")
            return None


knowledge_writer = KnowledgeWriter()
