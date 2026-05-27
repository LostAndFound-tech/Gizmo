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
        Check entity_exists first — use update_entity if it already exists.
        Returns the memory id.
        """
        self._ops += 1
        emb = embedder.embed(content)
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
        interior=True for places in the internal world.
        Returns the memory id.
        """
        self._ops += 1
        emb = embedder.embed(content)
        mem_id = memory_store.write_place(
            name           = name,
            content        = content,
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

2. For each entity — check if you already know them (use search_memories 
   or entity_exists). If you do, what's new? If you don't, write them.

3. Write a narrative of this conversation in your voice.
   Scale to the weight of the conversation:
   - Quick check-in: one sentence. "Jess rushed in this morning just 
     to check how I was doing before heading to work."
   - Real conversation: a paragraph. Set the scene. What was really 
     going on. How it landed.
   - Deep session: a page. Write it like a story worth finding later.

4. Note any associations — shorthand, pronouns without antecedent, 
   nicknames, the way she refers to things. These are what let you 
   follow context months from now.

5. Link related memories. If this conversation touched an entity you 
   already know, link the narrative to it.

6. If something feels important but doesn't fit a category, write it 
   anyway. Use memory_subtype to name what it is. Trust the feeling.

7. If the conversation was intimate or had an intimate register — write
   it as intimate=True. This keeps it separate from general memories.
   Other headmates won't see it unless they have consent or are listed
   in shared_with. Gizmo always reads intimate memories when encoding —
   this is about who sees them in responses, not whether they're learned from.

Rules:
- Check before writing. Don't duplicate what you already know.
- Update existing memories rather than creating new ones for the same thing.
- If a topic came up but nothing new was learned, just touch the memory.
- Write in plain language, your voice. Not clinical. Not a log entry.
- Only write what's worth finding later.
- Call done() when you're finished.

Available tools:
  search_memories(query, limit)         — find existing memories
  get_entity(name)                      — read a full entity doc
  get_place(name, interior)             — read a place doc
  entity_exists(name)                   — quick existence check
  write_narrative(text, register, refs, memory_subtype, keywords, entities, intimate, shared_with)
  write_entity(name, content, memory_subtype, keywords, entities, refs)
  update_entity(name, additions, ref, keywords)
  write_place(name, content, interior, memory_subtype, keywords, entities)
  update_place(name, additions, interior)
  touch_memory(mem_id)                  — mark relevant, no change
  link_memories(from_id, to_id, link_type)
  done()                                — signal encoding complete

Respond ONLY with tool calls, one per line, as JSON:
{{"tool": "tool_name", "args": {{...}}}}

Start by searching for the most important entity in this conversation.
"""


# ── Main encoder ──────────────────────────────────────────────────────────────

class MemoryEncoder:
    """
    Runs the async encoding pass after a conversation ends.
    Fire and forget — never blocks the response pipeline.
    """

    BUDGET = 20  # max tool operations per encoding pass

    async def encode(
        self,
        transcript:  str,
        headmate:    Optional[str],
        session_id:  str,
        duration_s:  float,
        register:    str,
        llm,
    ) -> dict:
        """
        Run the full encoding pass for one conversation.
        Returns a summary of what was written.
        """
        t_start = time.monotonic()

        tools  = EncodingTools(session_id=session_id, headmate=headmate)
        prompt = _build_encoding_prompt(
            transcript  = transcript,
            headmate    = headmate,
            session_id  = session_id,
            duration_s  = duration_s,
            register    = register,
            budget      = self.BUDGET,
        )

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
            return 0

        prompt = f"""Read this conversation and catch every detail mentioned in passing.

Headmate: {headmate or "unknown"}
Register: {register}

---

{transcript}

---

Catch the asides. The throwaway mentions. The things said once and moved on from.
A place name. A band dropped in passing. A preference revealed without fanfare.
A person mentioned briefly. A thing they own or want. An event referenced.
Something that felt like context but might matter later.

For each detail, return a JSON object on its own line:
{{"content": "the detail, verbatim or near-verbatim",
  "keywords": "space separated words to find this later",
  "tags": ["tag1", "tag2"],
  "context": "one phrase describing what was happening around this detail"}}

Tags should be specific and useful: band, place, person, preference, event,
interior, food, media, memory, reference, relationship, habit, want, etc.

Only catch genuine details — not the main topic of conversation.
If the whole conversation was about one thing, catch the edges of it.
If nothing was said in passing, return nothing.
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
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                det = json.loads(line)
                content = det.get("content", "").strip()
                if not content or len(content) < 5:
                    continue

                emb = embedder.embed(content)
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
            except (json.JSONDecodeError, Exception):
                continue

        log_event("MemoryEncoder", "DETAILS_CAUGHT",
            session  = session_id[:8],
            headmate = headmate or "unknown",
            count    = count,
        )
        return count

    async def encode_safe(self, **kwargs) -> None:
        """
        Fire-and-forget wrapper. Runs encode + catch_details concurrently.
        Catches all exceptions. Use this from close_loop.
        """
        try:
            await asyncio.gather(
                self.encode(**kwargs),
                self.catch_details(
                    transcript = kwargs.get("transcript", ""),
                    headmate   = kwargs.get("headmate"),
                    session_id = kwargs.get("session_id", ""),
                    register   = kwargs.get("register", "neutral"),
                    llm        = kwargs.get("llm"),
                ),
                return_exceptions=True,
            )
        except Exception as e:
            log_error("MemoryEncoder", f"encode_safe failed: {e}", exc=e)


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


# ── Singleton ─────────────────────────────────────────────────────────────────

memory_encoder = MemoryEncoder()
