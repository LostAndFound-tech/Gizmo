"""
core/story_engine.py

Story generation engine. Two input modes, one output: narrative prose.

HISTORY FEED
  Give it a transcript or scene log beats.
  Chunks through it, narrates each beat, assembles into a story.
  Used for: scene write-ups, session retrospectives, journal-to-story.

PROMPT FEED
  Give it a premise and genre.
  Generates a beat-by-beat transcript first, then narrates it.
  Used for: "write me a romance novel about X", original stories.

Three passes:
  1. Beat extraction  — what happened, in order
  2. Beat narration   — each beat rendered in the requested genre/voice
  3. Assembly         — stitch beats into coherent prose, smooth transitions

Stories saved to: {DATA_DIR}/stories/{name}/{timestamp}_{genre}.txt
Also returned as a string for immediate delivery.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Constants ─────────────────────────────────────────────────────────────────

BEAT_CHUNK_SIZE = 8   # lines per beat extraction chunk
MAX_BEATS       = 40  # cap to prevent runaway on huge transcripts


# ── Genre voices ──────────────────────────────────────────────────────────────

# Maps genre label → voice instruction for the narration pass
GENRE_VOICES = {
    "romance novel":     "Lush, emotional, heavy on internal sensation and longing. Euphemistic where needed but never clinical. The reader should feel everything.",
    "literary fiction":  "Precise, restrained, psychologically rich. Every detail earns its place. Subtext over statement.",
    "pulp fiction":      "Fast, punchy, visceral. Short sentences. High contrast. The kind of prose that moves like a freight train.",
    "gothic horror":     "Dark, atmospheric, dread-soaked. Beauty and wrongness intertwined. The body as something uncanny.",
    "erotica":           "Explicit, present-tense immediacy. Sensation-forward. The reader is inside it, not watching.",
    "fairy tale":        "Once-upon-a-time distance, archetypal language, moral weight underneath the surface. Strange and inevitable.",
    "noir":              "First person, weary, sardonic. The world is corrupt and beautiful and everyone knows it.",
    "children's book":   "Simple language, wonder-forward, gentle. Nothing darker than the world allows. Warmth above all.",
    "nature documentary":"Observational, precise, quietly awestruck. As if narrated by someone who has seen everything and is still moved.",
    "default":           "Clear, present, in Gizmo's voice — warm but not soft, specific, honest. Reads like he's telling you what happened.",
}


# ── File I/O ──────────────────────────────────────────────────────────────────

def _story_path(name: str, genre: str, session_id: str = "") -> Path:
    base = Path(librarian._full_path("stories")) / name.lower()
    base.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug  = re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
    fname = f"{ts}_{slug}"
    if session_id:
        fname += f"_{session_id[:8]}"
    return base / f"{fname}.txt"


def _save_story(name: str, genre: str, text: str, session_id: str = "") -> Path:
    path = _story_path(name, genre, session_id)
    try:
        path.write_text(text, encoding="utf-8")
        print(f"[StoryEngine] saved: {path}")
    except Exception as e:
        log_error("StoryEngine", "save failed", exc=e)
    return path


# ── LLM helper ────────────────────────────────────────────────────────────────

async def _llm(
    prompt:      str,
    system:      str,
    temperature: float = 0.85,
    max_tokens:  int   = 2000,
) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return raw.strip() if raw and raw.strip() else None
    except Exception as e:
        log_error("StoryEngine", "LLM call failed", exc=e)
        return None


# ── Pass 0: Transcript generation (prompt feed only) ─────────────────────────

_TRANSCRIPT_SYSTEM = """
You generate a beat-by-beat transcript from a story premise.
The transcript is raw material for a story engine — not the story itself.

Return a simple numbered list of beats. Each beat is one thing that happens.
Concrete, specific, in chronological order. No prose, no dialogue yet.
20-30 beats for a short story, 40+ for a longer one.

Example:
1. She enters the bookshop out of the rain, shaking water from her coat.
2. He looks up from behind the counter, startled.
3. Their eyes meet across a shelf of poetry collections.
...

Return only the numbered list. Nothing else.
""".strip()

async def _generate_transcript(premise: str, genre: str) -> Optional[list[str]]:
    prompt = f"Genre: {genre}\n\nPremise: {premise}\n\nGenerate the beat list."
    raw = await _llm(prompt, _TRANSCRIPT_SYSTEM, temperature=0.8, max_tokens=1500)
    if not raw:
        return None
    beats = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if line:
            beats.append(line)
    return beats if beats else None


# ── Pass 1: Beat extraction (history feed) ────────────────────────────────────

_BEAT_EXTRACT_SYSTEM = """
You extract story beats from a raw transcript or scene log.
A beat is one meaningful thing that happens — action, emotional shift, dialogue moment, sensation.

Return a simple numbered list of beats. One per line. Concrete and specific.
Strip speaker labels, formatting, stage directions — just what happened.
Do not editorialize. Do not interpret. Just extract.

Return only the numbered list.
""".strip()

async def _extract_beats(transcript_chunk: str) -> list[str]:
    raw = await _llm(transcript_chunk, _BEAT_EXTRACT_SYSTEM, temperature=0.0, max_tokens=1000)
    if not raw:
        return []
    beats = []
    for line in raw.splitlines():
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if line:
            beats.append(line)
    return beats


# ── Pass 2: Beat narration ─────────────────────────────────────────────────────

_NARRATION_SYSTEM_TEMPLATE = """
You are narrating a story in the following genre and voice:

{voice}

You will receive a list of story beats — what happened, in order.
Render them as flowing narrative prose. Not a list. Not summaries.
Actual story prose.

Rules:
- Stay in the genre voice throughout
- You can expand beats into full scenes or compress them into a sentence — use judgment
- Maintain consistent POV (third person close unless genre demands otherwise)
- Never break the fictional frame
- End your section on a strong beat, not mid-action

Return only the prose. No headers, no beat numbers, no meta-commentary.
""".strip()

async def _narrate_beats(beats: list[str], genre: str, story_so_far: str = "") -> Optional[str]:
    voice  = GENRE_VOICES.get(genre.lower(), GENRE_VOICES["default"])
    system = _NARRATION_SYSTEM_TEMPLATE.format(voice=voice)
    prompt = ""
    if story_so_far:
        prompt += f"Story so far (for continuity — do not repeat):\n{story_so_far[-800:]}\n\n"
    prompt += "Beats to narrate:\n" + "\n".join(f"{i+1}. {b}" for i, b in enumerate(beats))
    return await _llm(prompt, system, temperature=0.9, max_tokens=2000)


# ── Pass 3: Assembly ──────────────────────────────────────────────────────────

_ASSEMBLY_SYSTEM = """
You are assembling story sections into a single coherent narrative.
The sections were written in sequence — your job is to smooth the joins,
ensure consistent voice and POV throughout, and make it read as one piece.

Do not add new plot. Do not summarize. Do not cut content.
Just smooth, unify, and make it whole.

Return the complete assembled story. Nothing else.
""".strip()

async def _assemble(sections: list[str], genre: str) -> Optional[str]:
    if len(sections) == 1:
        return sections[0]
    voice  = GENRE_VOICES.get(genre.lower(), GENRE_VOICES["default"])
    prompt = (
        f"Genre/voice: {voice}\n\n"
        "Story sections to assemble:\n\n"
        + "\n\n---\n\n".join(sections)
    )
    return await _llm(prompt, _ASSEMBLY_SYSTEM, temperature=0.7, max_tokens=4000)


# ── Character context ─────────────────────────────────────────────────────────

def _get_character_context(name: str) -> str:
    """Pull appearance and top personality traits for a person."""
    descriptor = librarian._read_file(f"descriptors/{name.lower()}.json") or {}
    behavior   = librarian._read_file(f"behaviors/{name.lower()}.json") or {}
    personality = behavior.get("Personality", {})
    top = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:5]

    parts = []
    if descriptor:
        # Pull appearance fields
        appearance = {k: v for k, v in descriptor.items()
                      if k in ("Hair", "Eyes", "Skin", "Face", "Body", "Height", "Build")}
        if appearance:
            parts.append(f"{name}: {json.dumps(appearance)}")
    if top:
        parts.append(f"{name} personality: {', '.join(t for t, _ in top)}")

    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

class StoryEngine:

    async def from_history(
        self,
        transcript:  str | list,
        genre:       str,
        name:        str        = "",
        session_id:  str        = "",
        characters:  list[str]  = None,
    ) -> Optional[str]:
        """
        Generate a story from a transcript or list of scene beats.

        transcript: raw text or list of beat strings
        genre:      genre label (see GENRE_VOICES)
        name:       primary character name (for file saving and context)
        session_id: for file naming
        characters: list of character names to pull context for
        """
        log_event("StoryEngine", "START_HISTORY", genre=genre, name=name)

        # Normalize input
        if isinstance(transcript, list):
            # Already a beat list
            all_beats = transcript[:MAX_BEATS]
        else:
            # Raw text — chunk and extract beats
            lines      = [l for l in transcript.splitlines() if l.strip()]
            all_beats  = []
            for i in range(0, len(lines), BEAT_CHUNK_SIZE):
                chunk = "\n".join(lines[i:i + BEAT_CHUNK_SIZE])
                beats = await _extract_beats(chunk)
                all_beats.extend(beats)
                if len(all_beats) >= MAX_BEATS:
                    break
            all_beats = all_beats[:MAX_BEATS]

        if not all_beats:
            print("[StoryEngine] no beats extracted")
            return None

        print(f"[StoryEngine] {len(all_beats)} beats — narrating in {BEAT_CHUNK_SIZE}-beat chunks")

        # Build character context
        char_context = ""
        for char in (characters or ([name] if name else [])):
            char_context += _get_character_context(char) + "\n"

        # Narrate in chunks
        sections     = []
        story_so_far = ""
        chunk_size   = 10  # beats per narration pass

        for i in range(0, len(all_beats), chunk_size):
            chunk   = all_beats[i:i + chunk_size]
            section = await _narrate_beats(chunk, genre, story_so_far)
            if section:
                sections.append(section)
                story_so_far += "\n\n" + section

        if not sections:
            return None

        # Assembly pass
        story = await _assemble(sections, genre)
        if not story:
            story = "\n\n".join(sections)

        # Save
        if name:
            _save_story(name, genre, story, session_id)

        log_event("StoryEngine", "COMPLETE_HISTORY",
            genre=genre, name=name, beats=len(all_beats), sections=len(sections))

        return story

    async def from_prompt(
        self,
        premise:     str,
        genre:       str,
        name:        str       = "",
        session_id:  str       = "",
        characters:  list[str] = None,
    ) -> Optional[str]:
        """
        Generate a story from a premise.
        First generates a beat transcript, then narrates it.
        """
        log_event("StoryEngine", "START_PROMPT", genre=genre, name=name)

        # Pass 0: generate transcript
        print(f"[StoryEngine] generating transcript from premise...")
        beats = await _generate_transcript(premise, genre)
        if not beats:
            print("[StoryEngine] transcript generation failed")
            return None

        print(f"[StoryEngine] {len(beats)} beats generated")

        # Feed into history path
        return await self.from_history(
            transcript=beats,
            genre=genre,
            name=name,
            session_id=session_id,
            characters=characters,
        )

    async def retell(
        self,
        story_path:  str | Path,
        new_genre:   str,
        name:        str       = "",
        session_id:  str       = "",
        changes:     str       = "",
    ) -> Optional[str]:
        """
        Retell an existing story in a new genre with optional detail changes.

        story_path: path to the existing story file
        new_genre:  new genre label
        changes:    freeform description of what to change ("make them strangers",
                    "she doesn't submit at the end", etc.)
        """
        log_event("StoryEngine", "RETELL", genre=new_genre, name=name)

        try:
            original = Path(story_path).read_text(encoding="utf-8")
        except Exception as e:
            log_error("StoryEngine", "retell read failed", exc=e)
            return None

        voice  = GENRE_VOICES.get(new_genre.lower(), GENRE_VOICES["default"])
        system = f"""
You are retelling an existing story in a new genre and voice.

New genre/voice: {voice}

{"Changes to make: " + changes if changes else "Keep all plot events the same."}

Rules:
- Same events, same emotional arc, same ending — unless changes specify otherwise
- Fully rewrite in the new voice — do not paste or lightly edit the original
- Return only the retold story
""".strip()

        prompt = f"Original story:\n\n{original}\n\nRetell this in the new genre."
        story  = await _llm(prompt, system, temperature=0.9, max_tokens=4000)

        if story and name:
            _save_story(name, f"retell_{new_genre}", story, session_id)

        return story

    def list_stories(self, name: str) -> list[dict]:
        """List saved stories for a person."""
        base = Path(librarian._full_path("stories")) / name.lower()
        if not base.exists():
            return []
        stories = []
        for f in sorted(base.glob("*.txt"), reverse=True):
            stories.append({
                "path":     str(f),
                "filename": f.name,
                "size":     f.stat().st_size,
                "created":  datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
        return stories


story_engine = StoryEngine()
