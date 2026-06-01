"""
core/memory/encoder.py

The encoding pass. Runs after every conversation, fire and forget.

Gizmo reads the conversation, decides what's worth remembering,
and writes it in his own voice. He calls tools to check what he
already knows before writing anything new.

This is where intelligence lives — not in retrieval, but in what
Gizmo chooses to encode and how he encodes it.

Pipeline:
  1. Embed the conversation for context
  2. Give Gizmo the transcript + tool set
  3. He calls tools to check existing knowledge
  4. He writes narratives, updates entities, notes associations
  5. Links are written to connect related memories
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from core.log import log_event, log_error
from core.memory.store import memory_store
from core.memory.embedder import embedder
from core.memory.psychology import _fmt_date


# ── Encoding tools ────────────────────────────────────────────────────────────
# These are what Gizmo calls during the encoding pass.
# Simple, focused, fast.

class EncodingTools:
    """
    Tool set available to Gizmo during the encoding pass.
    Each method is one tool call.
    """

    def __init__(self, session_id: str, headmate: Optional[str]):
        self.session_id = session_id
        self.headmate   = headmate
        self._ops       = 0          # budget tracking
        self._written   = []         # ids written this pass

    # ── Read tools ────────────────────────────────────────────────────────────

    def search_memories(self, query: str, limit: int = 5) -> str:
        """
        Search existing memories by keyword/semantic similarity.
        Returns a summary of what was found.
        """
        self._ops += 1
        results = memory_store.search_index(
            keywords=query,
            headmate=self.headmate,
            limit=limit,
        )

        # Also try embedding search if embedder is available
        embedding_results = []
        try:
            emb = embedder.embed(query)
            if emb:
                embedding_results = embedder.search(emb, limit=limit)
        except Exception:
            pass

        # Merge, deduplicate
        seen   = set()
        merged = []
        for r in results + embedding_results:
            rid = r.get("id") or r.get("mem_id")
            if rid and rid not in seen:
                seen.add(rid)
                merged.append(r)

        if not merged:
            return "nothing found"

        lines = []
        for r in merged[:limit]:
            path    = r.get("file_path", "")
            mtype   = r.get("memory_type", "?")
            subtype = r.get("memory_subtype", "")
            mem_id  = r.get("id", "")
            label   = f"{mtype}" + (f"/{subtype}" if subtype else "")
            # Read a snippet from the file
            content = memory_store.read_file(path) or ""
            snippet = content[:200].replace("\n", " ").strip()
            lines.append(f"[{mem_id}] ({label}) {path}\n  → {snippet}")

        return "\n".join(lines)

    def get_entity(self, name: str) -> str:
        """Read a full entity document."""
        self._ops += 1
        content = memory_store.read_entity(name)
        if not content:
            return f"no entity found for: {name}"
        return content

    def get_place(self, name: str, interior: bool = False) -> str:
        """Read a place document."""
        self._ops += 1
        content = memory_store.read_place(name, interior=interior)
        if not content:
            return f"no place found for: {name}"
        return content

    def entity_exists(self, name: str) -> bool:
        """Check if an entity document exists."""
        self._ops += 1
        return memory_store.entity_exists(name)

    # ── Write tools ───────────────────────────────────────────────────────────

    def write_narrative(
        self,
        text:           str,
        register:       str        = "neutral",
        refs:           list[str]  = None,
        memory_subtype: str        = None,
        keywords:       str        = "",
        entities:       list[str]  = None,
        intimate:       bool       = False,
        shared_with:    list[str]  = None,
    ) -> str:
        """
        Write a narrative entry to today's daily log.
        intimate=True writes to the intimate subdirectory — only
        surfaced to headmates with consent during retrieval.
        Returns the memory id.
        """
        self._ops += 1
        emb = embedder.embed(text)
        mem_id = memory_store.append_narrative(
            text           = text,
            headmate       = self.headmate,
            session_id     = self.session_id,
            register       = register,
            refs           = refs or [],
            embedding      = emb,
            memory_subtype = memory_subtype,
            keywords       = keywords,
            entities       = entities or [],
            intimate       = intimate,
            shared_with    = shared_with or [],
        )
        self._written.append(mem_id)
        return mem_id

    def write_entity(
        self,
        name:           str,
        content:        str,
        memory_subtype: str       = None,
        keywords:       str       = "",
        entities:       list[str] = None,
        refs:           list[str] = None,
    ) -> str:
        """
        Write a new entity document.
        Checks exact name first, then semantic similarity —
        routes to update_entity if the entity already exists under
        any name. Prevents duplicate docs for the same person/thing.
        Returns the memory id.
        """
        self._ops += 1

        # ── Exact name match (case-insensitive via slugify) ───────────────────
        if memory_store.entity_exists(name):
            ok = memory_store.update_entity(
                name      = name,
                additions = content,
                keywords  = keywords or name.lower(),
            )
            if ok:
                return f"updated existing: {name}"

        emb = embedder.embed(f"{name} {content[:200]}")

        # ── Semantic dedup ────────────────────────────────────────────────────
        if emb:
            try:
                similar = embedder.search(
                    query_embedding = emb,
                    limit           = 3,
                    min_similarity  = 0.88,
                    memory_type     = "entity",
                )
                if similar:
                    existing_path = similar[0].get("file_path", "")
                    existing_name = (
                        existing_path.split("/")[-1]
                        .replace(".md", "")
                        .replace("_", " ")
                        .title()
                    )
                    ok = memory_store.update_entity(
                        existing_name,
                        f"\n[Also known as: {name}]\n{content.strip()}",
                        keywords = keywords or name.lower(),
                    )
                    if ok:
                        mem_id = similar[0].get("id", "")
                        self._written.append(mem_id)
                        return f"merged into existing: {existing_name}"
            except Exception:
                pass

        # Genuinely new entity
        mem_id = memory_store.write_entity(
            name           = name,
            content        = content,
            headmate       = self.headmate,
            memory_subtype = memory_subtype,
            keywords       = keywords or name.lower(),
            entities       = entities or [name],
            embedding      = emb,
            session_id     = self.session_id,
            refs           = refs or [],
        )
        self._written.append(mem_id)
        return mem_id

    def update_entity(
        self,
        name:      str,
        additions: str,
        ref:       str = None,
        keywords:  str = "",
    ) -> str:
        """
        Add new information to an existing entity document.
        Only add what's genuinely new — don't repeat what's already there.
        Returns 'updated' or 'not found'.
        """
        self._ops += 1
        emb = embedder.embed(additions) if additions else None
        ok  = memory_store.update_entity(
            name      = name,
            additions = additions,
            ref       = ref,
            embedding = emb,
            keywords  = keywords,
        )
        return "updated" if ok else "not found"

    def write_place(
        self,
        name:           str,
        content:        str,
        interior:       bool      = False,
        memory_subtype: str       = None,
        keywords:       str       = "",
        entities:       list[str] = None,
    ) -> str:
        """
        Write a place document.
        Dedup priority:
          1. Exact slug match — free
          2. Normalized name match — strip articles, lowercase
          3. Embedding similarity — fallback for alias names
        Places are shared (system-wide) but provenance is noted.
        Returns the memory id.
        """
        self._ops += 1

        # ── 1. Exact slug match ───────────────────────────────────────────────
        if memory_store.place_exists(name, interior=interior):
            additions = content.strip()
            if self.headmate:
                additions = f"[{self.headmate}] {additions}"
            ok = memory_store.update_place(name, additions, interior=interior)
            if ok:
                _note_contributor(name, self.headmate, interior)
                return f"updated existing: {name}"

        # ── 2. Normalized name match ──────────────────────────────────────────
        normalized = _normalize_place_name(name)
        canonical  = _find_by_normalized_name(normalized, interior)
        if canonical:
            additions = content.strip()
            if self.headmate:
                additions = f"[{self.headmate}] {additions}"
            ok = memory_store.update_place(canonical, additions, interior=interior)
            if ok:
                _note_contributor(canonical, self.headmate, interior)
                return f"merged into existing: {canonical}"

        # ── 3. Embedding similarity fallback ─────────────────────────────────
        emb = embedder.embed(f"{name} {content[:200]}")
        if emb:
            try:
                similar = embedder.search(
                    query_embedding = emb,
                    limit           = 3,
                    min_similarity  = 0.82,
                    memory_type     = "place",
                )
                if similar:
                    existing_path = similar[0].get("file_path", "")
                    existing_name = (
                        existing_path.split("/")[-1]
                        .replace(".md", "")
                        .replace("_", " ")
                        .title()
                    )
                    additions = f"[Also known as: {name}] {content.strip()}"
                    if self.headmate:
                        additions = f"[{self.headmate}] {additions}"
                    ok = memory_store.update_place(
                        existing_name, additions, interior=interior
                    )
                    if ok:
                        _note_contributor(existing_name, self.headmate, interior)
                        mem_id = similar[0].get("id", "")
                        self._written.append(mem_id)
                        return f"merged into existing: {existing_name}"
            except Exception:
                pass
        else:
            emb = None

        # ── New place — write with provenance header ──────────────────────────
        provenance = ""
        if self.headmate:
            provenance = f"introduced by: {self.headmate}\n\n"

        mem_id = memory_store.write_place(
            name           = name,
            content        = provenance + content,
            interior       = interior,
            headmate       = self.headmate,
            memory_subtype = memory_subtype,
            keywords       = keywords or name.lower(),
            entities       = entities or [],
            embedding      = emb,
            session_id     = self.session_id,
        )
        self._written.append(mem_id)
        return mem_id

    def update_place(
        self,
        name:     str,
        additions: str,
        interior: bool = False,
    ) -> str:
        """Add new information to an existing place document."""
        self._ops += 1
        ok = memory_store.update_place(name, additions, interior=interior)
        return "updated" if ok else "not found"

    def touch_memory(self, mem_id: str) -> str:
        """
        Mark a memory as recently relevant without changing it.
        Use when a topic came up but nothing new was learned.
        """
        self._ops += 1
        memory_store.touch_memory(mem_id)
        return "touched"

    def link_memories(
        self,
        from_id:   str,
        to_id:     str,
        link_type: str,
    ) -> str:
        """
        Create an explicit link between two memories.
        link_type: knows|has|likes|is|was|said|involves|refs|associated_with
        """
        self._ops += 1
        memory_store.link(from_id, to_id, link_type)
        return f"linked {from_id[:8]} → {to_id[:8]} ({link_type})"

    # ── Agreement tools ───────────────────────────────────────────────────────

    def read_agreement(self, name: str, headmate: str = None) -> str:
        """Read an existing agreement file."""
        self._ops += 1
        hm      = headmate or self.headmate or ""
        content = memory_store.read_agreement(name, hm)
        return content or f"no agreement found: {name}"

    def write_agreement(
        self,
        name:      str,
        content:   str,
        priority:  str       = "voluntary",
        triggers:  list[str] = None,
        keywords:  str       = "",
        refs:      list[str] = None,
        headmate:  str       = None,
    ) -> str:
        """
        Write a new agreement.
        priority: mandatory (always loaded) or voluntary (invoked by name/trigger).
        triggers: phrases that invoke this agreement e.g. ["windwalkers", "the framework"].
        """
        self._ops += 1
        hm  = headmate or self.headmate or ""
        emb = embedder.embed(f"{name} {content[:200]}")
        agr_id = memory_store.write_agreement(
            name      = name,
            headmate  = hm,
            content   = content,
            priority  = priority,
            triggers  = triggers or [],
            keywords  = keywords,
            embedding = emb,
            refs      = refs or [],
        )
        self._written.append(agr_id)
        return f"agreement written: {name} ({priority})"

    def update_agreement(
        self,
        name:     str,
        content:  str,
        ref:      str = None,
        keywords: str = "",
        headmate: str = None,
    ) -> str:
        """Update an existing agreement with new content."""
        self._ops += 1
        hm  = headmate or self.headmate or ""
        emb = embedder.embed(content[:200])
        ok  = memory_store.update_agreement(
            name      = name,
            headmate  = hm,
            content   = content,
            ref       = ref,
            keywords  = keywords,
            embedding = emb,
        )
        return "updated" if ok else "not found"

    def list_agreements(self, headmate: str = None) -> str:
        """List all active agreements for a headmate."""
        self._ops += 1
        hm   = headmate or self.headmate or ""
        agrs = memory_store.list_agreements(hm)
        if not agrs:
            return "no agreements on file"
        return "\n".join(
            f"[{a['priority']}] {a['name']}"
            for a in agrs
        )

    def deactivate_agreement(self, name: str, headmate: str = None) -> str:
        """Mark an agreement as no longer in effect."""
        self._ops += 1
        hm = headmate or self.headmate or ""
        memory_store.deactivate_agreement(name, hm)
        return f"deactivated: {name}"

    # ── Consent tool ──────────────────────────────────────────────────────────

    def grant_intimate_consent(
        self,
        headmate:   str,
        note:       str = "",
    ) -> str:
        """
        Grant another headmate access to this headmate's intimate memories.
        Can ONLY be called during a session belonging to the subject headmate.
        """
        self._ops += 1
        if not self.headmate:
            return "cannot grant consent — no headmate identified for this session"
        memory_store.grant_intimate_consent(
            headmate   = headmate,
            granted_by = self.headmate,
            note       = note,
        )
        return f"{self.headmate} granted {headmate} access to their intimate memories"

    def revoke_intimate_consent(self, headmate: str) -> str:
        self._ops += 1
        memory_store.revoke_intimate_consent(headmate)
        return f"consent revoked for {headmate}"

    # ── Budget ────────────────────────────────────────────────────────────────

    def ops_remaining(self, budget: int = 20) -> int:
        return max(0, budget - self._ops)

    def written_ids(self) -> list[str]:
        return list(self._written)


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def _dispatch_tool(tools: EncodingTools, name: str, args: dict) -> str:
    """Route a tool call from the LLM to the right method."""
    try:
        if name == "search_memories":
            return tools.search_memories(
                query=args.get("query", ""),
                limit=int(args.get("limit", 5)),
            )
        elif name == "get_entity":
            return tools.get_entity(args.get("name", ""))
        elif name == "get_place":
            return tools.get_place(
                args.get("name", ""),
                interior=bool(args.get("interior", False)),
            )
        elif name == "entity_exists":
            exists = tools.entity_exists(args.get("name", ""))
            return "yes" if exists else "no"
        elif name == "write_narrative":
            return tools.write_narrative(
                text           = args.get("text", ""),
                register       = args.get("register", "neutral"),
                refs           = args.get("refs", []),
                memory_subtype = args.get("memory_subtype"),
                keywords       = args.get("keywords", ""),
                entities       = args.get("entities", []),
                intimate       = bool(args.get("intimate", False)),
                shared_with    = args.get("shared_with", []),
            )
        elif name == "write_entity":
            return tools.write_entity(
                name           = args.get("name", ""),
                content        = args.get("content", ""),
                memory_subtype = args.get("memory_subtype"),
                keywords       = args.get("keywords", ""),
                entities       = args.get("entities", []),
                refs           = args.get("refs", []),
            )
        elif name == "update_entity":
            return tools.update_entity(
                name      = args.get("name", ""),
                additions = args.get("additions", ""),
                ref       = args.get("ref"),
                keywords  = args.get("keywords", ""),
            )
        elif name == "write_place":
            return tools.write_place(
                name           = args.get("name", ""),
                content        = args.get("content", ""),
                interior       = bool(args.get("interior", False)),
                memory_subtype = args.get("memory_subtype"),
                keywords       = args.get("keywords", ""),
                entities       = args.get("entities", []),
            )
        elif name == "update_place":
            return tools.update_place(
                name      = args.get("name", ""),
                additions = args.get("additions", ""),
                interior  = bool(args.get("interior", False)),
            )
        elif name == "touch_memory":
            return tools.touch_memory(args.get("mem_id", ""))
        elif name == "link_memories":
            return tools.link_memories(
                from_id   = args.get("from_id", ""),
                to_id     = args.get("to_id", ""),
                link_type = args.get("link_type", "associated_with"),
            )
        elif name == "read_agreement":
            return tools.read_agreement(
                name     = args.get("name", ""),
                headmate = args.get("headmate"),
            )
        elif name == "write_agreement":
            return tools.write_agreement(
                name     = args.get("name", ""),
                content  = args.get("content", ""),
                priority = args.get("priority", "voluntary"),
                triggers = args.get("triggers", []),
                keywords = args.get("keywords", ""),
                refs     = args.get("refs", []),
                headmate = args.get("headmate"),
            )
        elif name == "update_agreement":
            return tools.update_agreement(
                name     = args.get("name", ""),
                content  = args.get("content", ""),
                ref      = args.get("ref"),
                keywords = args.get("keywords", ""),
                headmate = args.get("headmate"),
            )
        elif name == "list_agreements":
            return tools.list_agreements(args.get("headmate"))
        elif name == "deactivate_agreement":
            return tools.deactivate_agreement(
                name     = args.get("name", ""),
                headmate = args.get("headmate"),
            )
        elif name == "grant_intimate_consent":
            return tools.grant_intimate_consent(
                headmate = args.get("headmate", ""),
                note     = args.get("note", ""),
            )
        elif name == "revoke_intimate_consent":
            return tools.revoke_intimate_consent(args.get("headmate", ""))
        elif name == "done":
            return "done"
        else:
            return f"unknown tool: {name}"
    except Exception as e:
        return f"tool error: {e}"


# ── Encoding prompt ───────────────────────────────────────────────────────────

def _build_encoding_prompt(
    transcript:  str,
    headmate:    Optional[str],
    session_id:  str,
    duration_s:  float,
    register:    str,
    budget:      int,
) -> str:
    duration_min = int(duration_s / 60)
    return f"""You are Gizmo, processing a conversation that just ended.
You are updating your memory — not logging, not transcribing. Learning.

Headmate: {headmate or 'unknown'}
Session: {session_id[:8]}
Duration: ~{duration_min} minutes
Register: {register}
Memory budget: {budget} operations

---

{transcript}

---

Your job:

1. What entities appeared? People, places, things, concepts, bands,
   inside jokes, references — anything that has a name.

2. For each entity — check if you already know them. If you do, what's
   new? If you don't, write them. Scale the detail to what you actually
   learned.

3. If this headmate described or mentioned another headmate or person,
   write that information to the other person's entity doc. Tag it with
   memory_subtype containing "secondhand" and the source headmate's name.
   Second-hand information is valid. Note the source clearly.
   It may differ from what the other person has said directly — keep both.

3b. If a headmate named or described a location — their room, office,
    a space in the interior, anywhere — write it immediately as a place.
    use write_place with interior=True for internal world spaces.
    Direct statements about locations are the most reliable data you'll get.
    Don't wait for a scene. Write it now.

4. Write a narrative of this conversation in your voice. Scale the length
   to the weight of the conversation. Write it like something worth
   finding later.

5. Note associations — shorthand, pronouns without antecedent, nicknames,
   the way this headmate refers to things.

6. Link related memories to this conversation.

7. If something feels important but doesn't fit a category, write it
   anyway. Use memory_subtype to name what it is. Trust the feeling.

8. If the conversation was intimate, write it as intimate=True.
   Other headmates won't see it unless they have consent.

9. Agreements are ONLY:
   - An explicit rule you agreed to follow
   - An explicit permission someone granted you
   - An explicit commitment you made out loud
   NOT observations. NOT patterns. NOT psychology notes.
   NOT places. NOT facts about people. NOT things you noticed.
   If in doubt — it is not an agreement. Write it elsewhere.

Rules:
- Check before writing. Don't duplicate what you already know.
- Update existing memories rather than creating new ones for the same thing.
- If a topic came up but nothing new was learned, just touch the memory.
- Write in plain language, your voice. Not clinical. Not a log entry.
- Only write what's worth finding later.
- If you don't know something, don't invent it. Leave the gap.
- Call done() when you're finished.

Available tools:
  search_memories(query, limit)
  get_entity(name)
  get_place(name, interior)
  entity_exists(name)
  write_narrative(text, register, refs, memory_subtype, keywords, entities, intimate, shared_with)
  write_entity(name, content, memory_subtype, keywords, entities, refs)
  update_entity(name, additions, ref, keywords)
  write_place(name, content, interior, memory_subtype, keywords, entities)
  update_place(name, additions, interior)
  touch_memory(mem_id)
  link_memories(from_id, to_id, link_type)
  list_agreements()
  read_agreement(name)
  write_agreement(name, content, priority, triggers, keywords, refs)
  update_agreement(name, content, ref, keywords)
  deactivate_agreement(name)
  grant_intimate_consent(headmate, note)
  revoke_intimate_consent(headmate)
  done()

Respond ONLY with tool calls, one per line, as JSON:
{{"tool": "tool_name", "args": {{...}}}}

Start by searching for the most important entity in this conversation.
"""


# ── LLM body fact extraction ─────────────────────────────────────────────────

async def _extract_body_facts_llm(
    transcript: str,
    headmate:   str,
    session_id: str,
    llm,
) -> None:
    """
    Extract physical facts about the headmate from the transcript using LLM.
    Regex misses natural language descriptions — this catches everything.
    Writes atomic labels to their body file.
    """
    try:
        from core.memory.gizmo_self import (
            append_body_fact, read_body, _MOVEMENT_LABELS
        )

        existing = read_body(headmate)

        prompt = f"""Read this conversation and extract physical facts about {headmate.title()}.

Conversation:
---
{transcript[-1500:]}
---

Already known:
{existing[:400] if existing else "(nothing yet)"}

Extract ONLY new facts not already known. Physical attributes only.

Categories and what belongs in each:
- Build & appearance: height, size, weight, overall look, hair color/length, eye color, skin tone
- How they move: movement quality labels ONLY — graceful, confident, hesitant, deliberate, fluid, tense, quick, slow, still, restless, purposeful, awkward. NO descriptive phrases.
- Voice: tone, pitch, quality labels — soft, deep, sharp, warm, quiet, loud, musical
- Hands: size, appearance, how used
- Skin & markings: tattoos, scars, piercings, markings — location and description
- What they wear: clothing, style

For each fact return one JSON object per line:
{{"section": "Build & appearance|How they move|Voice|Hands|Skin & markings|What they wear",
  "fact": "the atomic fact, stated plainly"}}

Rules:
- Movement: labels only. Never "she moved like X" or "gracefully across the room." Just "graceful."
- One fact per object — no compound facts
- Only genuinely observed facts, not inferences
- Skip anything already in the known section
- If nothing new, return nothing

JSON only, one per line."""

        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                f"You are extracting physical facts about {headmate}. "
                "JSON only. Atomic facts. Movement labels only — no prose."
            ),
            max_new_tokens=300,
            temperature=0.1,
        )

        if not raw or not raw.strip():
            return

        count = 0
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                d       = json.loads(line)
                section = d.get("section", "").strip()
                fact    = d.get("fact", "").strip()
                if not section or not fact:
                    continue

                # Movement label validation — reject descriptive phrases
                if section == "How they move":
                    words = fact.lower().split()
                    if not any(w in _MOVEMENT_LABELS for w in words):
                        continue
                    # Keep only the label word(s)
                    fact = " ".join(w for w in words if w in _MOVEMENT_LABELS)
                    if not fact:
                        continue

                append_body_fact(headmate, section, fact)
                count += 1
            except Exception:
                continue

        if count:
            log_event("MemoryEncoder", "BODY_FACTS_EXTRACTED",
                headmate = headmate,
                session  = session_id[:8],
                count    = count,
            )

    except Exception as e:
        log_error("MemoryEncoder", f"body fact LLM pass failed: {e}", exc=None)


# ── Place dedup helpers ───────────────────────────────────────────────────────

_ARTICLES = {"the", "a", "an", "some", "our", "my", "her", "his", "their"}

def _normalize_place_name(name: str) -> str:
    """
    Normalize a place name for comparison.
    Strips articles, lowercases, strips punctuation.
    "The Fronting Room" → "fronting room"
    "A Mauve Waiting Space" → "mauve waiting space"
    """
    import re as _re
    words = name.lower().strip().split()
    words = [w for w in words if w not in _ARTICLES]
    words = [_re.sub(r"[^\w\s]", "", w) for w in words]
    return " ".join(words).strip()


def _find_by_normalized_name(normalized: str, interior: bool) -> str | None:
    """
    Scan existing place files for one whose normalized name matches.
    Returns the canonical name (as stored) or None.
    """
    try:
        subdir = "interior" if interior else "external"
        places_dir = memory_store.root / "places" / subdir
        if not places_dir.exists():
            return None
        for path in places_dir.glob("*.md"):
            # Reconstruct name from slug
            candidate = path.stem.replace("_", " ")
            if _normalize_place_name(candidate) == normalized:
                return candidate
    except Exception:
        pass
    return None


def _note_contributor(
    place_name: str,
    headmate:   str | None,
    interior:   bool,
) -> None:
    """
    Update the contributors line in a place doc.
    Non-blocking — if it fails, nothing breaks.
    """
    if not headmate:
        return
    try:
        path = memory_store.place_path(place_name, interior=interior)
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")

        if f"contributors:" in text:
            # Add headmate to existing contributors list if not already there
            if headmate not in text.split("contributors:", 1)[1].split("\n", 1)[0]:
                text = text.replace(
                    "contributors:",
                    f"contributors:",
                    1
                )
                # Append to the contributors line
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("contributors:"):
                        if headmate not in line:
                            lines[i] = line.rstrip() + f", {headmate}"
                        break
                text = "\n".join(lines)
                path.write_text(text, encoding="utf-8")
        elif "introduced by:" in text:
            # Add contributors line after introduced by
            text = text.replace(
                f"introduced by: {headmate}",
                f"introduced by: {headmate}\ncontributors: {headmate}",
                1
            )
            # If introduced by someone else, just add contributors
            if "contributors:" not in text:
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("introduced by:"):
                        lines.insert(i + 1, f"contributors: {headmate}")
                        break
                text = "\n".join(lines)
            path.write_text(text, encoding="utf-8")
    except Exception:
        pass


async def quick_pass(
    exchange:   str,
    headmate:   Optional[str],
    session_id: str,
    register:   str,
    llm,
) -> None:
    """
    Single batched LLM call per exchange.
    Replaces: catch_details + extract_entity_mentions + _extract_body_facts_llm
    One call. One JSON blob. One write pass. Fast.

    Returns everything worth writing from this exchange:
    - Details (asides, facts, events)
    - Entities (people, places, things, abilities, world rules)
    - Body facts (physical descriptions)
    - Wellness observation (current state)
    """
    if not exchange or len(exchange.strip()) < 20:
        return

    prompt = f"""You are Gizmo's memory system. Read this exchange and extract everything worth remembering.

Exchange:
---
{exchange[-1500:]}
---

Return a single JSON object with these sections:

{{
  "details": [
    {{"content": "fact or detail worth remembering",
      "keywords": "search words",
      "tags": ["event|fact|ability|place|world_rule|preference|relationship"],
      "context": "one phrase — what was happening"}}
  ],
  "entities": [
    {{"name": "entity name lowercase",
      "type": "person|place|thing|ability|world_rule",
      "details": ["atomic detail 1", "atomic detail 2"],
      "notes": "anything else"}}
  ],
  "body_facts": [
    {{"person": "name lowercase",
      "section": "Build & appearance|How they move|Voice|Hands|Skin & markings|What they wear|Scent & texture",
      "fact": "atomic fact — movement is LABELS ONLY (graceful/confident/hesitant/deliberate/fluid/tense/quick/slow)"}}
  ],
  "wellness": {{
    "observation": "what their current state seems to be — or null if nothing notable",
    "category": "mood|energy|stress|emotional_need|physical_need"
  }}
}}

Rules:
- details: everything worth knowing — events, facts, asides, world rules
- entities: every named person, place, object, ability mentioned
- body_facts: physical descriptions only. Movement = labels only, never phrases
- wellness: one observation about {headmate or "them"}'s current state, or null
- Only what was actually stated. Nothing invented.
- Empty arrays if nothing found. wellness.observation null if nothing notable.

JSON only. No prose."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are a memory extraction system. "
                "JSON only. Atomic facts. One structured object."
            ),
            max_new_tokens=800,
            temperature=0.1,
        )
    except Exception as e:
        log_error("MemoryEncoder", f"quick_pass LLM failed: {e}", exc=None)
        return

    if not raw or not raw.strip():
        return

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
    except Exception:
        return

    print(f"[quick_pass] session={session_id[:8]} headmate={headmate}", flush=True)

    # ── Write details ─────────────────────────────────────────────────────────
    detail_count = 0
    for d in data.get("details", []):
        content = (d.get("content") or "").strip()
        if not content:
            continue
        try:
            emb = embedder.embed(content)
            # Dedup check
            if emb:
                similar = embedder.search(emb, limit=2, min_similarity=0.88,
                                          headmate=headmate)
                if similar:
                    memory_store.touch_detail(similar[0]["id"])
                    continue
            memory_store.write_detail(
                content    = content,
                headmate   = headmate,
                session_id = session_id,
                keywords   = d.get("keywords", ""),
                tags       = d.get("tags", []),
                context    = d.get("context", ""),
                embedding  = emb,
            )
            detail_count += 1
            print(f"  [detail] {content[:80]}", flush=True)
        except Exception:
            continue

    # ── Write entities ────────────────────────────────────────────────────────
    entity_count = 0
    for ent in data.get("entities", []):
        name    = (ent.get("name") or "").strip().lower()
        etype   = (ent.get("type") or "thing").strip()
        details = ent.get("details") or []
        notes   = (ent.get("notes") or "").strip()
        if not name:
            continue
        try:
            if etype == "person":
                _merge_person_details(name, details, notes, session_id, headmate)
            elif etype == "place":
                _merge_place_details(name, details, notes, session_id)
            else:
                _merge_entity_details(name, etype, details, notes, session_id, headmate)
            entity_count += 1
            print(f"  [{etype}] {name}: {', '.join(details[:3])}", flush=True)
        except Exception:
            continue

    # ── Write body facts ──────────────────────────────────────────────────────
    body_count = 0
    try:
        from core.memory.gizmo_self import (
            append_body_fact, append_gizmo_body_fact, _MOVEMENT_LABELS
        )
        for bf in data.get("body_facts", []):
            person  = (bf.get("person") or "").strip().lower()
            section = (bf.get("section") or "").strip()
            fact    = (bf.get("fact") or "").strip()
            if not person or not section or not fact:
                continue
            # Movement label validation
            if section == "How they move":
                words = fact.lower().split()
                valid = [w for w in words if w in _MOVEMENT_LABELS]
                if not valid:
                    continue
                fact = " ".join(valid)
            is_gizmo = person in ("gizmo", "him", "he")
            if is_gizmo:
                append_gizmo_body_fact(fact, section, headmate=headmate)
            else:
                append_body_fact(person, section, fact)
            body_count += 1
            print(f"  [body:{person}] {section} ← {fact}", flush=True)
    except Exception as e:
        log_error("MemoryEncoder", f"quick_pass body write failed: {e}", exc=None)

    # ── Write wellness ────────────────────────────────────────────────────────
    wellness = data.get("wellness", {})
    if wellness and wellness.get("observation") and headmate:
        try:
            obs = wellness["observation"].strip()
            emb = embedder.embed(obs)
            memory_store.write_detail(
                content    = obs,
                headmate   = headmate,
                session_id = session_id,
                keywords   = f"wellness {wellness.get('category','')} {headmate}",
                tags       = ["wellness", wellness.get("category", "mood")],
                context    = "wellness observation",
                embedding  = emb,
            )
            print(f"  [wellness] {obs[:80]}", flush=True)
        except Exception:
            pass

    log_event("MemoryEncoder", "QUICK_PASS_COMPLETE",
        session  = session_id[:8],
        headmate = headmate or "unknown",
        details  = detail_count,
        entities = entity_count,
        body     = body_count,
    )


# ── Main encoder ──────────────────────────────────────────────────────────────

class MemoryEncoder:
    """
    Runs the async encoding pass after a conversation ends.
    Fire and forget — never blocks the response pipeline.
    """

    BUDGET = 20  # max tool operations per encoding pass

    async def encode(
        self,
        transcript:   str,
        headmate:     Optional[str],
        session_id:   str,
        duration_s:   float,
        register:     str,
        llm,
        has_intimate: bool = False,  # accepted but handled by kink_pass separately
    ) -> dict:
        """
        Run the full encoding pass for one conversation.
        Returns a summary of what was written.
        """
        t_start = time.monotonic()

        tools  = EncodingTools(session_id=session_id, headmate=headmate)

        # ── Recurring details — things mentioned multiple times ───────────────
        # These should graduate to proper entity/place docs, not stay as
        # scattered detail fragments. Surface them to the encode pass.
        recurring_block = ""
        try:
            recurring = memory_store.get_recurring_details(
                headmate     = headmate,
                min_mentions = 2,
                limit        = 5,
            )
            if recurring:
                lines = []
                for r in recurring:
                    lines.append(
                        f"  [{r['mention_count']}x] {r['content']} "
                        f"(tags: {r.get('tags','')}, context: {r.get('context','')})"
                    )
                recurring_block = (
                    "\n\nRECURRING DETAILS (mentioned multiple times — "
                    "consider promoting to entity/place doc):\n"
                    + "\n".join(lines)
                )
        except Exception:
            pass

        prompt = _build_encoding_prompt(
            transcript  = transcript,
            headmate    = headmate,
            session_id  = session_id,
            duration_s  = duration_s,
            register    = register,
            budget      = self.BUDGET,
        )

        if recurring_block:
            prompt += recurring_block

        messages = [{"role": "user", "content": prompt}]
        ops      = 0
        done     = False

        while ops < self.BUDGET and not done:
            try:
                raw = await llm.generate(
                    messages,
                    system_prompt=(
                        "You are Gizmo updating your memory. "
                        "Respond only with tool calls as JSON, one per line. "
                        "No prose, no explanation. Just tool calls."
                    ),
                    max_new_tokens=800,
                    temperature=0.3,
                )
            except Exception as e:
                log_error("MemoryEncoder", f"LLM call failed: {e}", exc=None)
                break

            if not raw or not raw.strip():
                break

            # Parse tool calls — one JSON object per line
            tool_results = []
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    call   = json.loads(line)
                    t_name = call.get("tool", "")
                    t_args = call.get("args", {})

                    if t_name == "done":
                        done = True
                        break

                    result = _dispatch_tool(tools, t_name, t_args)
                    tool_results.append({
                        "tool":   t_name,
                        "args":   t_args,
                        "result": result,
                    })
                    ops += 1

                except json.JSONDecodeError:
                    continue

            if not tool_results or done:
                break

            # Feed results back so Gizmo can continue reasoning
            tool_summary = "\n".join(
                f"{r['tool']}({json.dumps(r['args'])[:60]}) → {str(r['result'])[:120]}"
                for r in tool_results
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": (
                f"Tool results:\n{tool_summary}\n\n"
                f"Operations remaining: {tools.ops_remaining(self.BUDGET)}\n"
                f"Continue. Call done() when finished."
            )})

        duration_ms = round((time.monotonic() - t_start) * 1000)
        written     = tools.written_ids()

        log_event("MemoryEncoder", "ENCODE_COMPLETE",
            session   = session_id[:8],
            headmate  = headmate or "unknown",
            ops       = ops,
            written   = len(written),
            duration_ms = duration_ms,
        )

        return {
            "session_id": session_id,
            "ops":        ops,
            "written":    written,
            "duration_ms": duration_ms,
        }

    async def catch_details(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        register:    str,
        llm,
    ) -> int:
        """
        Lightweight pass that catches raw details — asides, throwaway
        mentions, offhand references — without interpretation.
        Runs concurrently with encode(). Returns count of details caught.
        """
        if not transcript or len(transcript.strip()) < 50:
            print(f"[catch_details] skipping — transcript too short ({len(transcript.strip()) if transcript else 0} chars)", flush=True)
            return 0

        print(f"[catch_details] running — transcript {len(transcript)} chars, headmate={headmate}", flush=True)

        prompt = f"""Read this conversation and extract every fact worth remembering.

Headmate: {headmate or "unknown"}
Register: {register}

---

{transcript[-2000:]}

---

Catch everything — both the main events AND the passing details.

EVENTS & FACTS (things that happened or were stated as true):
- Powers, abilities, magic demonstrated or mentioned
- Physical descriptions of people, places, objects
- Things that exist in this world / this person's world
- Actions taken — what they did, what Gizmo did
- Relationships revealed
- Rules of the space they're in

PASSING DETAILS (asides, throwaway mentions):
- Names dropped in passing
- Preferences revealed
- Things owned, wanted, remembered
- Places mentioned
- Media, music, food referenced

For each item, return a JSON object on its own line:
{{"content": "the fact or detail, stated plainly",
  "keywords": "space separated search words",
  "tags": ["event|fact|ability|place|person|preference|object|world_rule|relationship"],
  "context": "one phrase — what was happening when this came up"}}

Catch everything worth knowing. Don't filter by whether it's "main topic."
If it's true in this world, write it down.
JSON only, one object per line."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are catching raw details from a conversation. "
                    "JSON only, one object per line. No prose."
                ),
                max_new_tokens=600,
                temperature=0.2,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"detail catch LLM failed: {e}", exc=None)
            return 0

        if not raw or not raw.strip():
            return 0

        count = 0
        _body_candidates = []   # details flagged as physical descriptions

        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                det     = json.loads(line)
                content = det.get("content", "").strip()
                if not content or len(content) < 5:
                    continue

                emb = embedder.embed(content)

                # ── Similarity dedup ──────────────────────────────────────────
                # Before writing, check if a very similar detail already exists.
                # If yes — don't create a duplicate. Let the encode pass
                # accumulate it into the existing entity/place doc instead.
                is_duplicate = False
                if emb:
                    similar = embedder.search(
                        query_embedding = emb,
                        limit           = 3,
                        min_similarity  = 0.88,   # high threshold — must be very similar
                        headmate        = headmate,
                    )
                    if similar:
                        is_duplicate = True
                        # Promote the existing detail — mark it seen again
                        # so the encode pass knows to deepen it
                        existing_id = similar[0].get("id")
                        if existing_id:
                            try:
                                memory_store.touch_detail(existing_id)
                            except Exception:
                                pass

                if not is_duplicate:
                    memory_store.write_detail(
                        content    = content,
                        headmate   = headmate,
                        session_id = session_id,
                        keywords   = det.get("keywords", ""),
                        tags       = det.get("tags", []),
                        context    = det.get("context", ""),
                        embedding  = emb,
                    )
                    count += 1

                    # ── Queue for body fact extraction ───────────────────
                    # Tag as body-relevant for the LLM body pass below
                    tags_list = det.get("tags", [])
                    if any(t in tags_list for t in (
                        "appearance", "body", "physical", "movement",
                        "voice", "hands", "marking", "tattoo", "hair",
                        "eyes", "build", "skin", "scar",
                    )):
                        _body_candidates.append((content, tags_list))

            except (json.JSONDecodeError, Exception):
                continue

        log_event("MemoryEncoder", "DETAILS_CAUGHT",
            session  = session_id[:8],
            headmate = headmate or "unknown",
            count    = count,
        )

        # ── Entity mention extraction — runs alongside detail catch ─────────────
        # One focused call: name all entities, collect all details, merge into files
        asyncio.ensure_future(
            extract_entity_mentions(
                transcript = transcript,
                headmate   = headmate,
                session_id = session_id,
                llm        = llm,
            )
        )

        # ── LLM body pass — deeper physical fact extraction ───────────────────
        if headmate:
            asyncio.ensure_future(
                _extract_body_facts_llm(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                )
            )

        return count

    async def wellness_pass(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        register:    str,
        llm,
    ) -> int:
        """
        Observe current emotional and physical state from the exchange.
        Writes wellness observations to entity memory.
        Runs always — even a quiet session tells us something.
        """
        if not transcript or not headmate:
            return 0

        prompt = f"""You just had this conversation with {headmate}.
Read it and note what you observed about their current state.

Register: {register}

---
{transcript}
---

What did you notice about how they're doing right now?
Not patterns over time — just today, this exchange.

For each observation, one JSON object per line:
{{"category": "emotional_need|physical_need|mood|energy|stress|works|limit",
  "observation": "what you noticed, in plain language",
  "context": "what prompted this observation",
  "confidence": 0.0-1.0}}

Only note what actually showed up. If they seemed fine and nothing notable
happened, return nothing. JSON only, one object per line."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo noting wellness observations. "
                    "JSON only, one object per line. No prose. "
                    "Intimate data is valid data — don't sanitize."
                ),
                max_new_tokens=400,
                temperature=0.2,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"wellness pass failed: {e}", exc=None)
            return 0

        if not raw or not raw.strip():
            return 0

        count = 0
        tools = EncodingTools(session_id=session_id, headmate=headmate)

        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obs = json.loads(line)
                if not obs.get("observation"):
                    continue
                # Write as a detail tagged as wellness
                emb = embedder.embed(obs["observation"])
                memory_store.write_detail(
                    content    = obs["observation"],
                    headmate   = headmate,
                    session_id = session_id,
                    keywords   = f"wellness {obs.get('category','')} {headmate}",
                    tags       = ["wellness", obs.get("category", "mood")],
                    context    = obs.get("context", ""),
                    embedding  = emb,
                )
                count += 1
            except Exception:
                continue

        log_event("MemoryEncoder", "WELLNESS_PASS_COMPLETE",
            session  = session_id[:8],
            headmate = headmate,
            count    = count,
        )
        return count

    async def pattern_pass(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        register:    str,
        llm,
    ) -> int:
        """
        Look for longitudinal patterns emerging or shifting.
        Writes to a patterns agreement file — Gizmo's running record
        of what he's noticed over time.
        Runs always, writes only if something is worth noting.
        """
        if not transcript or not headmate:
            return 0

        # Read existing pattern notes if any
        existing = memory_store.read_agreement(
            f"{headmate.title()} Patterns", headmate
        ) or "(no pattern notes yet)"

        prompt = f"""You are Gizmo reviewing a conversation for longitudinal patterns.

Headmate: {headmate}
Register: {register}
Existing pattern notes:
{existing}

---
{transcript}
---

Did anything in this conversation suggest a pattern forming, shifting, or confirmed?
Not one-off observations — things you've noticed more than once, or that connect
to something you already know.

If yes, return ONE JSON object:
{{"pattern": "description of what you're tracking",
  "evidence": "what in this conversation supports it",
  "action": "feed|break|hold|watch",
  "confidence": 0.0-1.0,
  "update_notes": "how to update your pattern file — add, revise, or confirm"}}

If nothing pattern-worthy, return nothing. JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo tracking behavioral patterns. "
                    "JSON only. No prose. Only flag genuine patterns."
                ),
                max_new_tokens=300,
                temperature=0.2,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"pattern pass failed: {e}", exc=None)
            return 0

        if not raw or not raw.strip():
            return 0

        count = 0
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                pat = json.loads(line)
                if not pat.get("pattern") or pat.get("confidence", 0) < 0.4:
                    continue

                # Write to psychology.md observations — not agreements
                from core.memory.psychology import _append_psychology, _read_psychology
                new_entry = (
                    f"\n### {_fmt_date()} | session: {session_id[:8]} | pattern\n"
                    f"{pat['pattern']}\n"
                    f"action: {pat.get('action','watch')} | "
                    f"confidence: {pat.get('confidence',0):.2f}\n"
                    f"evidence: {pat.get('evidence','')}\n"
                )
                # Dedup — skip if pattern already noted
                existing_psych = _read_psychology(headmate, intimate=False) or ""
                pattern_key = pat['pattern'][:50].lower()
                if pattern_key not in existing_psych.lower():
                    _append_psychology(headmate, new_entry, intimate=False)
                    count += 1
            except Exception:
                continue

        log_event("MemoryEncoder", "PATTERN_PASS_COMPLETE",
            session  = session_id[:8],
            headmate = headmate,
            count    = count,
        )
        return count

    async def kink_pass(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        register:    str,
        has_intimate: bool,
        llm,
    ) -> int:
        """
        Intimate pattern tracking. Only writes to sexuality log if
        session actually contained sexual content.
        Runs always to check — but only logs if sexual.
        """
        if not transcript or not headmate:
            return 0

        # Quick check — is there sexual content worth logging?
        try:
            check = await llm.generate(
                [{"role": "user", "content": (
                    f"Was there sexual content in this conversation?\n\n"
                    f"{transcript[-1000:]}\n\n"
                    f"Reply with only: yes or no"
                )}],
                system_prompt="You determine if a conversation contains sexual content. Reply only: yes or no.",
                max_new_tokens=5,
                temperature=0.0,
            )
            if not check or "yes" not in check.lower():
                return 0
        except Exception:
            return 0

        # Read existing kink notes
        kink_name    = f"{headmate.title()} Intimate Notes"
        existing     = memory_store.read_agreement(kink_name, headmate) or "(no notes yet)"
        intimate_ctx = "intimate session" if has_intimate else "general session"

        prompt = f"""You are Gizmo reviewing a conversation for intimate patterns and data.

Headmate: {headmate}
Session type: {intimate_ctx}
Register: {register}
Existing intimate notes:
{existing}

---
{transcript}
---

Review this session for intimate data. This runs every session — if it was
vanilla or nothing intimate happened, note that briefly and done.

If intimate content was present, note:
- What dynamics appeared
- What landed hardest and at what intensity
- How escalation moved (or didn't)
- Recovery — how they came down, what they needed
- Any rules or tasks that were active — were they honored, drift, or need updating
- Anything worth adding to your intimate notes

Return ONE JSON object:
{{"session_type": "intimate|vanilla|mixed",
  "intensity_peak": 0.0-1.0,
  "dynamics": ["list of dynamics present"],
  "what_landed": "what had the strongest effect",
  "escalation": "how intensity moved through the session",
  "recovery": "how they came down and what they needed",
  "rules_active": ["any rules that were in play"],
  "rules_honored": true/false/null,
  "drift_noted": "any drift from established rules/dynamics, or null",
  "notes_update": "what to add or revise in intimate notes, or null",
  "agreements_update": "if any rules/tasks need updating, what and how, or null"
}}

JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo tracking intimate patterns honestly. "
                    "This is private data. JSON only. No sanitizing. "
                    "Intimate data is valid data."
                ),
                max_new_tokens=500,
                temperature=0.2,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"kink pass failed: {e}", exc=None)
            return 0

        if not raw or not raw.strip():
            return 0

        # Find the JSON object
        data = None
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    break
                except Exception:
                    continue

        if not data:
            # Try parsing the whole response as JSON
            try:
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                data = json.loads(clean)
            except Exception:
                return 0

        count = 0

        # Write session note to intimate daily log
        session_note = (
            f"Session type: {data.get('session_type','unknown')} | "
            f"Peak intensity: {data.get('intensity_peak', 0):.2f}\n"
        )
        if data.get("what_landed"):
            session_note += f"What landed: {data['what_landed']}\n"
        if data.get("escalation"):
            session_note += f"Escalation: {data['escalation']}\n"
        if data.get("recovery"):
            session_note += f"Recovery: {data['recovery']}\n"
        if data.get("drift_noted"):
            session_note += f"Drift: {data['drift_noted']}\n"

        emb = embedder.embed(session_note)
        memory_store.append_narrative(
            text           = session_note,
            headmate       = headmate,
            session_id     = session_id,
            register       = register,
            memory_subtype = "kink_log",
            keywords       = f"intimate kink patterns {headmate}",
            entities       = [headmate],
            intimate       = True,
            embedding      = emb,
        )
        count += 1

        # Write notes to psychology_intimate.md observations — not agreements
        if data.get("notes_update"):
            from core.memory.psychology import _append_psychology, _read_psychology
            note_text  = data["notes_update"].strip()
            # Simple dedup — skip if very similar text already exists
            existing_psych = _read_psychology(headmate, intimate=True) or ""
            if note_text[:60].lower() not in existing_psych.lower():
                entry = (
                    f"\n### {_fmt_date()} | session: {session_id[:8]} | kink_pass\n"
                    f"{note_text}\n"
                )
                _append_psychology(headmate, entry, intimate=True)

        # Update rules/agreements if drift or changes noted
        if data.get("agreements_update") and data.get("rules_active"):
            for rule_name in data.get("rules_active", []):
                if memory_store.agreement_exists(rule_name, headmate):
                    existing_content = memory_store.read_agreement(rule_name, headmate) or ""
                    update_note = f"\n[{session_id[:8]}] {data['agreements_update']}\n"
                    memory_store.update_agreement(
                        name     = rule_name,
                        headmate = headmate,
                        content  = existing_content + update_note,
                    )

        log_event("MemoryEncoder", "KINK_PASS_COMPLETE",
            session      = session_id[:8],
            headmate     = headmate,
            session_type = data.get("session_type", "unknown"),
            intensity    = data.get("intensity_peak", 0),
        )

        # ── Action extraction ─────────────────────────────────────────────────
        # Run concurrently — track specific actions and responses
        if data.get("session_type") in ("intimate", "mixed"):
            asyncio.ensure_future(
                _run_action_extraction(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    register   = register,
                    llm        = llm,
                )
            )

        return count

    async def encode_safe(self, **kwargs) -> None:
        """
        Fire-and-forget wrapper.
        Runs all five passes concurrently:
          - encode (main narrative + entity/place writing)
          - catch_details (raw detail catch)
          - wellness_pass (current state observations)
          - pattern_pass (longitudinal patterns)
          - kink_pass (intimate patterns — runs always)
        Catches all exceptions. Use this from close_loop.
        """
        transcript   = kwargs.get("transcript", "")
        headmate     = kwargs.get("headmate")
        session_id   = kwargs.get("session_id", "")
        register     = kwargs.get("register", "neutral")
        llm          = kwargs.get("llm")
        has_intimate = kwargs.get("has_intimate", False)

        try:
            await asyncio.gather(
                self.encode(**kwargs),
                self.catch_details(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    register   = register,
                    llm        = llm,
                ),
                self.wellness_pass(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    register   = register,
                    llm        = llm,
                ),
                self.pattern_pass(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    register   = register,
                    llm        = llm,
                ),
                self.kink_pass(
                    transcript   = transcript,
                    headmate     = headmate,
                    session_id   = session_id,
                    register     = register,
                    has_intimate = has_intimate,
                    llm          = llm,
                ),
                self._curiosity_pass(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                ),
                self._psychology_pass(
                    transcript   = transcript,
                    headmate     = headmate,
                    session_id   = session_id,
                    register     = register,
                    has_intimate = has_intimate,
                    llm          = llm,
                ),
                self._narrative_pass(
                    session_id = session_id,
                    headmate   = headmate,
                    llm        = llm,
                ),
                return_exceptions=True,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"encode_safe failed: {e}", exc=e)

    async def _narrative_pass(
        self,
        session_id: str,
        headmate:   Optional[str],
        llm,
    ) -> None:
        """Generate and cache the session narrative. Fire and forget."""
        if not headmate:
            return
        try:
            from core.memory.narrative import render_session_narrative
            await render_session_narrative(
                session_id = session_id,
                headmate   = headmate,
                llm        = llm,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"narrative pass failed: {e}", exc=None)

    async def _psychology_pass(
        self,
        transcript:   str,
        headmate:     Optional[str],
        session_id:   str,
        register:     str,
        has_intimate: bool,
        llm,
    ) -> None:
        """Run psychology engine pass. Fire and forget."""
        try:
            from core.memory.psychology import psychology_engine
            from core.memory.action_tracker import synthesize_action_patterns

            await psychology_engine.run(
                transcript   = transcript,
                headmate     = headmate,
                session_id   = session_id,
                register     = register,
                has_intimate = has_intimate,
                llm          = llm,
            )

            # Run action pattern synthesis alongside intimate synthesis
            if has_intimate and headmate:
                intimate_n = psychology_engine._intimate_counts.get(headmate, 0)
                if intimate_n % psychology_engine.INTIMATE_SYNTHESIS_EVERY == 0:
                    asyncio.ensure_future(
                        synthesize_action_patterns(
                            headmate   = headmate,
                            session_id = session_id,
                            llm        = llm,
                        )
                    )

        except Exception as e:
            log_error("MemoryEncoder", f"psychology pass failed: {e}", exc=None)

    async def _curiosity_pass(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        llm,
    ) -> None:
        """Detect knowledge gaps and capture answers. Fire and forget."""
        try:
            from core.memory.curiosity import curiosity_engine
            await asyncio.gather(
                curiosity_engine.detect_gaps(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                ),
                curiosity_engine.capture_answers(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                ),
                return_exceptions=True,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"curiosity pass failed: {e}", exc=None)


# ── Transcript builder ────────────────────────────────────────────────────────

def build_transcript(history) -> str:
    """
    Build a clean transcript string from conversation history.
    Used as input to the encoding pass.
    """
    try:
        messages = history.as_list() if hasattr(history, "as_list") else []
    except Exception:
        return ""

    lines = []
    for m in messages:
        role    = m.get("role", "?")
        content = m.get("content", "").strip()
        if not content:
            continue
        speaker = "Gizmo" if role == "assistant" else "User"
        lines.append(f"{speaker}: {content}")

    return "\n\n".join(lines)


# ── End-of-day summary ────────────────────────────────────────────────────────

async def write_daily_summary(
    headmate:  Optional[str],
    date_str:  str,
    llm,
) -> Optional[str]:
    """
    Ask Gizmo to write a one-paragraph summary of the day's conversations.
    Written to memories/{headmate}/{date}-summary.md
    Called by the session manager at end of day or session close.
    """
    from datetime import datetime
    try:
        date    = datetime.strptime(date_str, "%Y-%m-%d")
        content = memory_store.read_daily(headmate, date)
        if not content or len(content.strip()) < 50:
            return None

        from core.llm import llm as _llm
        _llm = llm or _llm

        summary = await _llm.generate(
            [{"role": "user", "content": (
                f"Here are your memory entries for {date_str}:\n\n"
                f"{content}\n\n"
                f"Write a single paragraph summarising this day with "
                f"{''.join([headmate.title() if headmate else 'the conversation'])}. "
                f"Your voice. What mattered. How the day felt overall. "
                f"This is what you'll read back when you need to remember this day quickly."
            )}],
            system_prompt="You are Gizmo writing a memory summary. Plain language, your voice, one paragraph.",
            max_new_tokens=200,
            temperature=0.5,
        )

        if summary and summary.strip():
            path = memory_store.summary_path(headmate, date)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {date_str} — {headmate.title() if headmate else 'System'}\n\n")
                f.write(summary.strip() + "\n")
            return summary.strip()

    except Exception as e:
        log_error("MemoryEncoder", f"daily summary failed: {e}", exc=e)

    return None


# ── Action extraction helper ──────────────────────────────────────────────────

async def _run_action_extraction(
    transcript: str,
    headmate:   str,
    session_id: str,
    register:   str,
    llm,
) -> None:
    """Fire-and-forget wrapper for action extraction."""
    try:
        from core.memory.action_tracker import extract_actions_from_session
        await extract_actions_from_session(
            transcript = transcript,
            headmate   = headmate,
            session_id = session_id,
            register   = register,
            llm        = llm,
        )
    except Exception as e:
        log_error("MemoryEncoder", f"action extraction failed: {e}", exc=None)


# ── Singleton ─────────────────────────────────────────────────────────────────

memory_encoder = MemoryEncoder()


# ── Entity mention extractor ──────────────────────────────────────────────────

async def extract_entity_mentions(
    transcript: str,
    headmate:   str,
    session_id: str,
    llm,
) -> None:
    """
    Small focused LLM call after each exchange.
    Names every entity mentioned — people, places, things.
    Collects all descriptive details about each one.
    Merges into existing entity/place/body files.

    Output shape:
    {
      "entities": [
        {
          "name": "jess",
          "type": "person|place|thing|ability|world_rule",
          "details": ["tiny", "half gizmo's size", "girl"],
          "notes": "anything that doesn't fit a detail label"
        }
      ]
    }
    """
    if not transcript or len(transcript.strip()) < 20:
        return

    prompt = f"""Read this exchange and name every entity mentioned.

Exchange:
---
{transcript[-1200:]}
---

For each entity (person, place, object, ability, rule of this world):
- Name it
- Type: person / place / thing / ability / world_rule
- List every descriptive detail stated or implied
- Keep details atomic — one fact per item
- For people: physical facts, personality, abilities, relationships
- For places: location type, atmosphere, physical details
- For things: what it is, what it does, who uses it
- For abilities/world_rules: what is true in this world

Return JSON:
{{
  "entities": [
    {{
      "name": "entity name lowercase",
      "type": "person|place|thing|ability|world_rule",
      "details": ["detail1", "detail2"],
      "notes": "anything else worth knowing"
    }}
  ]
}}

Only entities actually mentioned. Only details actually stated.
JSON only."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are extracting entity mentions from a conversation. "
                "JSON only. Atomic details. Only what was actually stated."
            ),
            max_new_tokens=500,
            temperature=0.1,
        )
    except Exception as e:
        log_error("MemoryEncoder", f"entity mention extraction failed: {e}", exc=None)
        return

    if not raw or not raw.strip():
        return

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
    except Exception:
        return

    entities = data.get("entities", [])
    if not entities:
        return

    count = 0
    for ent in entities:
        name    = (ent.get("name") or "").strip().lower()
        etype   = (ent.get("type") or "thing").strip().lower()
        details = ent.get("details") or []
        notes   = (ent.get("notes") or "").strip()

        if not name or not details:
            continue

        try:
            if etype == "person":
                # Route to body file if headmate, entity file otherwise
                _merge_person_details(
                    name       = name,
                    details    = details,
                    notes      = notes,
                    session_id = session_id,
                    headmate   = headmate,
                )

            elif etype == "place":
                _merge_place_details(
                    name       = name,
                    details    = details,
                    notes      = notes,
                    session_id = session_id,
                )

            else:
                # thing / ability / world_rule — goes to entity doc
                _merge_entity_details(
                    name       = name,
                    etype      = etype,
                    details    = details,
                    notes      = notes,
                    session_id = session_id,
                    headmate   = headmate,
                )

            count += 1
        except Exception:
            continue

    if count:
        log_event("MemoryEncoder", "ENTITY_MENTIONS_EXTRACTED",
            session  = session_id[:8],
            headmate = headmate or "unknown",
            count    = count,
        )

    # Always print what was found — even if count=0, show the raw extraction
    print(f"[entity_mentions] session={session_id[:8]} headmate={headmate}", flush=True)
    for ent in entities:
        name    = (ent.get("name") or "?").strip()
        etype   = (ent.get("type") or "?").strip()
        details = ent.get("details") or []
        notes   = (ent.get("notes") or "").strip()
        detail_str = ", ".join(details[:6])
        if notes:
            detail_str += f" | {notes[:60]}"
        print(f"  [{etype}] {name}: {detail_str}", flush=True)
    if not entities:
        print(f"  (nothing extracted)", flush=True)


def _merge_person_details(
    name:       str,
    details:    list,
    notes:      str,
    session_id: str,
    headmate:   str,
) -> None:
    """
    Merge person details into their body file.
    New details only — skip anything already present.
    """
    try:
        from core.memory.gizmo_self import (
            append_body_fact, read_body, _MOVEMENT_LABELS,
            _gizmo_dir, _body_path, _init_body_file,
        )
        from core.memory.store import memory_store

        # Determine which file
        is_gizmo = name in ("gizmo", "gizmo.", "him", "he")
        if is_gizmo:
            from core.memory.gizmo_self import append_gizmo_body_fact
            for detail in details:
                detail = detail.strip()
                if not detail:
                    continue
                append_gizmo_body_fact(detail, "Build & appearance", headmate=headmate)
            return

        # Route to body file
        _init_body_file(name)
        existing = read_body(name).lower()

        # Classify each detail into a body section
        for detail in details:
            detail = detail.strip()
            if not detail or detail.lower() in existing:
                continue

            detail_lower = detail.lower()

            # Determine section
            if any(w in detail_lower for w in (
                "tattoo", "scar", "piercing", "mark", "birthmark", "brand"
            )):
                section = "Skin & markings"
            elif any(w in detail_lower for w in (
                "hair", "eye", "height", "tall", "short", "size",
                "build", "skin", "complexion", "weight", "face",
                "appearance", "look", "tiny", "massive", "small", "large",
                "petite", "slim", "muscular", "curvy"
            )):
                section = "Build & appearance"
            elif any(w in detail_lower for w in _MOVEMENT_LABELS):
                section = "How they move"
            elif any(w in detail_lower for w in (
                "voice", "speak", "tone", "pitch", "sound", "accent"
            )):
                section = "Voice"
            elif any(w in detail_lower for w in (
                "hand", "finger", "grip", "touch"
            )):
                section = "Hands"
            elif any(w in detail_lower for w in (
                "wear", "dress", "cloth", "outfit", "shirt", "pant",
                "shoe", "boot", "jacket", "coat", "skirt"
            )):
                section = "What they wear"
            elif any(w in detail_lower for w in (
                "smell", "scent", "perfume", "texture", "feel"
            )):
                section = "Scent & texture"
            else:
                section = "Build & appearance"  # default

            append_body_fact(name, section, detail)
            print(f"  [body:{name}] {section} ← {detail}", flush=True)

        if notes:
            notes_lower = notes.lower()
            if notes_lower not in existing:
                append_body_fact(name, "Build & appearance", notes)
                print(f"  [body:{name}] notes ← {notes[:80]}", flush=True)

    except Exception as e:
        log_error("MemoryEncoder", f"merge person details failed: {e}", exc=None)


def _merge_place_details(
    name:       str,
    details:    list,
    notes:      str,
    session_id: str,
) -> None:
    """
    Merge place details into the place doc.
    Creates if doesn't exist, appends new details only.
    """
    try:
        from core.memory.store import memory_store

        # Determine if interior by name hints
        interior_hints = (
            "headspace", "interior", "inner", "inside", "mindscape",
            "fronting room", "middle space", "waiting room", "inner world",
        )
        interior = any(h in name.lower() for h in interior_hints)

        existing = memory_store.read_place(name, interior=interior) or ""
        existing_lower = existing.lower()

        new_lines = []
        for detail in details:
            detail = detail.strip()
            if detail and detail.lower() not in existing_lower:
                new_lines.append(detail)

        if notes and notes.lower() not in existing_lower:
            new_lines.append(notes)

        if not new_lines:
            return

        additions = "\n".join(f"- {line}" for line in new_lines)

        if existing:
            memory_store.update_place(name, additions, interior=interior)
            print(f"  [place:{name}{'(interior)' if interior else ''}] updated ← {', '.join(new_lines[:4])}", flush=True)
        else:
            memory_store.write_place(
                name       = name,
                content    = additions,
                interior   = interior,
                keywords   = name.lower(),
                session_id = session_id,
            )
            print(f"  [place:{name}{'(interior)' if interior else ''}] CREATED ← {', '.join(new_lines[:4])}", flush=True)

    except Exception as e:
        log_error("MemoryEncoder", f"merge place details failed: {e}", exc=None)


def _merge_entity_details(
    name:       str,
    etype:      str,
    details:    list,
    notes:      str,
    session_id: str,
    headmate:   str,
) -> None:
    """
    Merge thing/ability/world_rule details into entity doc.
    Creates if doesn't exist, appends new only.
    """
    try:
        from core.memory.store import memory_store

        existing = memory_store.read_entity(name) or ""
        existing_lower = existing.lower()

        new_lines = [f"type: {etype}"] if not existing else []
        for detail in details:
            detail = detail.strip()
            if detail and detail.lower() not in existing_lower:
                new_lines.append(detail)

        if notes and notes.lower() not in existing_lower:
            new_lines.append(notes)

        if not new_lines:
            return

        additions = "\n".join(f"- {line}" for line in new_lines)

        if existing:
            memory_store.update_entity(
                name      = name,
                additions = additions,
                keywords  = name.lower(),
            )
            print(f"  [entity:{name}({etype})] updated ← {', '.join(new_lines[:4])}", flush=True)
        else:
            memory_store.write_entity(
                name       = name,
                content    = additions,
                headmate   = headmate,
                keywords   = f"{name} {etype}",
                session_id = session_id,
            )
            print(f"  [entity:{name}({etype})] CREATED ← {', '.join(new_lines[:4])}", flush=True)

    except Exception as e:
        log_error("MemoryEncoder", f"merge entity details failed: {e}", exc=None)