"""
core/memory/retriever.py

Pull-based retrieval. Given an incoming message, assembles the memory
context Gizmo needs to respond well.

This is the read side of the memory system. It does not interpret —
it finds, crawls, ranks, and assembles. Gizmo does the rest.

Pipeline:
  1. Embed the incoming message
  2. Vector search — find semantically similar memories
  3. Keyword search — catch exact references the vector might miss
  4. Detail scan — surface any relevant asides from the details table
  5. Crawl refs — follow reference links from top hits outward
  6. Load entity/place docs for anything named in top results
  7. Rank and budget — trim to token budget, highest value first
  8. Assemble — build a clean context block Gizmo reads before responding

The output is a MemoryContext object: structured, ranked, ready to read.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from core.log import log_event, log_error
from core.memory.store import memory_store
from core.memory.embedder import embedder


# ── Context dataclass ─────────────────────────────────────────────────────────

@dataclass
class MemoryContext:
    """
    Assembled memory context for one incoming message.
    Passed to the response LLM as readable context.
    """
    # Agreements — loaded before everything else
    mandatory_agreements:  list[dict] = field(default_factory=list)  # always loaded
    invoked_agreements:    list[dict] = field(default_factory=list)  # loaded on trigger

    # Ranked memory hits — each is a dict with content + metadata
    memories:       list[dict]     = field(default_factory=list)

    # Entity documents loaded — living docs about named things
    entities:       dict[str, str] = field(default_factory=dict)  # name → content

    # Place documents loaded
    places:         dict[str, str] = field(default_factory=dict)  # name → content

    # Raw details that might be relevant
    details:        list[dict]     = field(default_factory=list)

    # Recent daily narrative (last day or two)
    recent_narrative: Optional[str] = None

    # Retrieval metadata
    query:          str   = ""
    headmate:       str   = ""
    retrieved_at:   float = field(default_factory=time.time)
    total_hits:     int   = 0
    token_estimate: int   = 0

    def is_empty(self) -> bool:
        return (
            not self.mandatory_agreements
            and not self.invoked_agreements
            and not self.memories
            and not self.entities
            and not self.places
            and not self.details
            and not self.recent_narrative
        )

    def to_prompt_block(self) -> str:
        """
        Render the context as a readable block for the system prompt.
        Agreements first — always. Memory context after.
        """
        if self.is_empty():
            return ""

        sections = []

        # ── Mandatory agreements — top of everything ──────────────────────────
        if self.mandatory_agreements:
            for agr in self.mandatory_agreements:
                content = agr.get("content", "").strip()
                name    = agr.get("name", "Agreement")
                if content:
                    sections.append(f"[{name} — in effect]\n{content}")

        # ── Invoked voluntary agreements ──────────────────────────────────────
        if self.invoked_agreements:
            for agr in self.invoked_agreements:
                content = agr.get("content", "").strip()
                name    = agr.get("name", "Agreement")
                if content:
                    sections.append(f"[{name}]\n{content}")

        # ── Recent narrative ──────────────────────────────────────────────────
        if self.recent_narrative:
            sections.append("[Recent]\n" + self.recent_narrative.strip())

        # ── Entity docs ───────────────────────────────────────────────────────
        if self.entities:
            for name, content in self.entities.items():
                snippet = content.strip()[:400]
                if len(content.strip()) > 400:
                    snippet += "\n..."
                sections.append(f"[{name}]\n{snippet}")

        # ── Place docs ────────────────────────────────────────────────────────
        if self.places:
            for name, content in self.places.items():
                snippet = content.strip()[:300]
                if len(content.strip()) > 300:
                    snippet += "\n..."
                sections.append(f"[Place: {name}]\n{snippet}")

        # ── Memory hits ───────────────────────────────────────────────────────
        if self.memories:
            mem_lines = []
            for m in self.memories[:6]:
                content  = m.get("content", "").strip()
                mtype    = m.get("memory_type", "")
                subtype  = m.get("memory_subtype", "")
                label    = mtype + (f"/{subtype}" if subtype else "")
                if content:
                    mem_lines.append(f"  [{label}] {content[:200]}")
            if mem_lines:
                sections.append("[Memory]\n" + "\n".join(mem_lines))

        # ── Details ───────────────────────────────────────────────────────────
        if self.details:
            detail_lines = []
            for d in self.details[:4]:
                content = d.get("content", "").strip()
                ctx     = d.get("context", "")
                if content:
                    suffix = f" ({ctx})" if ctx else ""
                    detail_lines.append(f"  {content}{suffix}")
            if detail_lines:
                sections.append("[Details]\n" + "\n".join(detail_lines))

        if not sections:
            return ""

        return "--- memory ---\n" + "\n\n".join(sections) + "\n--- end memory ---"


# ── Retriever ─────────────────────────────────────────────────────────────────

class MemoryRetriever:

    # Token budget for memory context — keeps system prompt lean
    TOKEN_BUDGET = 1200

    # How many hops to follow refs outward from top hits
    CRAWL_DEPTH  = 2

    # Minimum similarity score to include a vector hit
    MIN_SIM      = 0.35

    async def retrieve(
        self,
        message:      str,
        headmate:     Optional[str],
        session_id:   str,
        register:     str   = "neutral",
        fast:         bool  = False,
        intimate_ok:  Optional[bool] = None,  # None = auto-check consent store
    ) -> MemoryContext:
        """
        Main retrieval entry point.
        fast=True skips crawl and details — for rapid-fire messages.
        intimate_ok=True allows intimate memories in context.
        intimate_ok=None auto-checks consent store for the headmate.
        Returns a MemoryContext ready to pass to the response LLM.
        """
        t_start = time.monotonic()
        ctx     = MemoryContext(query=message, headmate=headmate or "")
        fast == False
        if not message or not message.strip():
            return ctx

        # ── 0. Agreements — always first ──────────────────────────────────────
        if headmate:
            # Mandatory — always loaded, no questions asked
            mandatory = memory_store.get_mandatory_agreements(headmate)
            for agr in mandatory:
                content = memory_store.read_agreement(agr["name"], headmate)
                if content:
                    ctx.mandatory_agreements.append({
                        "name":    agr["name"],
                        "content": content,
                        "id":      agr["id"],
                    })

            # Voluntary — only if message triggers them
            triggered = memory_store.match_agreement_trigger(message, headmate)
            for agr in triggered:
                content = memory_store.read_agreement(agr["name"], headmate)
                if content:
                    ctx.invoked_agreements.append({
                        "name":    agr["name"],
                        "content": content,
                        "id":      agr["id"],
                    })

        # ── Resolve intimate access ───────────────────────────────────────────
        if intimate_ok is None:
            intimate_ok = (
                memory_store.has_intimate_consent(headmate)
                if headmate else False
            )

        # ── 1. Embed the message ──────────────────────────────────────────────
        query_emb = embedder.embed(message)

        # ── 2. Vector search ──────────────────────────────────────────────────
        vector_hits = []
        if query_emb:
            vector_hits = embedder.search(
                query_embedding = query_emb,
                limit           = 12,
                min_similarity  = self.MIN_SIM,
                headmate        = headmate,
            )

        # ── 3. Keyword search ─────────────────────────────────────────────────
        keyword_hits = memory_store.search_index(
            keywords    = message,
            headmate    = headmate,
            limit       = 8,
        )
        print("Keywords I caught!")

        # ── 4. Merge and deduplicate ──────────────────────────────────────────
        seen     = set()
        all_hits = []

        for h in vector_hits:
            mid = h.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                h["_score"] = h.get("similarity", 0.5)
                all_hits.append(h)

        for h in keyword_hits:
            mid = h.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                h["_score"] = 0.4  # keyword hits score slightly lower
                all_hits.append(h)

        # Sort by score
        all_hits.sort(key=lambda x: x.get("_score", 0), reverse=True)
        ctx.total_hits = len(all_hits)

        # ── 4b. Filter private memories by consent ────────────────────────────
        # Intimate memories are always included in encoding (learning) but
        # only surfaced in retrieval (responding) if the headmate has consent.
        # shared_with overrides private — if this headmate is on the list, allow.
        def _allowed(hit: dict) -> bool:
            if not hit.get("private"):
                return True
            if intimate_ok:
                return True
            # Check shared_with list
            try:
                shared = json.loads(hit.get("shared_with") or "[]")
                if headmate and headmate.lower() in [s.lower() for s in shared]:
                    return True
            except Exception:
                pass
            return False

        all_hits = [h for h in all_hits if _allowed(h)]
        print("All the stuff I actually caught:")
        for a in all_hits:
            print(a)

        # ── 5. Load content for top hits ──────────────────────────────────────
        loaded_memories = []
        entity_names    = set()
        place_names     = set()

        for hit in all_hits[:10]:
            file_path  = hit.get("file_path", "")
            anchor     = hit.get("anchor")
            memory_type = hit.get("memory_type", "")

            if not file_path:
                continue

            # Track entity and place names for doc loading
            if memory_type == "entity":
                name = _name_from_path(file_path)
                if name:
                    entity_names.add(name)
                continue  # entity content loaded via read_entity below
            if memory_type == "place":
                name = _name_from_path(file_path)
                if name:
                    place_names.add(name)
                continue  # place content loaded via read_place below

            # Load the memory content
            content = _load_anchor(file_path, anchor)
            if content:
                loaded_memories.append({
                    "content":        content,
                    "memory_type":    memory_type,
                    "memory_subtype": hit.get("memory_subtype"),
                    "headmate":       hit.get("headmate"),
                    "file_path":      file_path,
                    "id":             hit.get("id"),
                    "_score":         hit.get("_score", 0),
                })

        ctx.memories = loaded_memories
        print("Memories I loaded")
        print("-----------------")
        for l in loaded_memories:
            print(l)

        # ── 6. Crawl refs from top hits ───────────────────────────────────────
        if not fast and all_hits:
            crawl_ids   = [h["id"] for h in all_hits[:4] if h.get("id")]
            extra_names = await self._crawl_refs(
                mem_ids     = crawl_ids,
                entity_names = entity_names,
                place_names  = place_names,
                depth        = self.CRAWL_DEPTH,
            )
            entity_names.update(extra_names.get("entities", set()))
            place_names.update(extra_names.get("places", set()))

        # ── 7. Load entity docs ───────────────────────────────────────────────
        for name in list(entity_names)[:4]:
            content = memory_store.read_entity(name)
            if content:
                ctx.entities[name] = content
                print("What I'm loading")
                print("----------------")
                print(name, content)

        # ── 8. Load place docs ────────────────────────────────────────────────
        for name in list(place_names)[:3]:
            # Try interior first, then external
            content = (
                memory_store.read_place(name, interior=True)
                or memory_store.read_place(name, interior=False)
            )
            if content:
                ctx.places[name] = content
                print("places I loaded")
                print("---------------")
                print(name, content)

        # ── 8b. Load psychology docs ──────────────────────────────────────────
        if headmate:
            try:
                from core.memory.psychology import load_psychology_for_retrieval
                psych = load_psychology_for_retrieval(
                    headmate    = headmate,
                    intimate_ok = intimate_ok,
                )
                # Only load synthesis section — keep it lean
                if psych.get("conversational"):
                    conv = psych["conversational"]
                    if "## Current Understanding" in conv:
                        section = conv.split("## Current Understanding", 1)[1]
                        section = section.split("## Observations", 1)[0].strip()
                        if section:
                            ctx.entities[f"{headmate.title()} (psychology)"] = section
                if psych.get("intimate") and intimate_ok:
                    intim = psych["intimate"]
                    if "## Current Understanding" in intim:
                        section = intim.split("## Current Understanding", 1)[1]
                        section = section.split("## Observations", 1)[0].strip()
                        if section:
                            ctx.entities[f"{headmate.title()} (intimate)"] = section

                # Load action patterns if intimate
                if intimate_ok:
                    from core.memory.action_tracker import action_tracker
                    pattern_doc = action_tracker.read_pattern_doc(headmate)
                    if pattern_doc:
                        # Just the synthesis section, not the raw summary
                        if "## Raw Summary" in pattern_doc:
                            pattern_doc = pattern_doc.split("## Raw Summary")[0].strip()
                        ctx.entities[f"{headmate.title()} (action patterns)"] = pattern_doc[:400]
            except Exception:
                pass
        if not fast:
            detail_hits = memory_store.search_details(
                keywords = message,
                headmate = headmate,
                limit    = 6,
            )
            print("Details I remember")
            print("------------------")
            for d in detail_hits:
                print(d)
            print("------------------")

            # Also vector search details if embedder available
            if query_emb:
                detail_hits += _vector_search_details(query_emb, headmate, limit=4)

            # Deduplicate by id
            seen_det  = set()
            deduped   = []
            for d in detail_hits:
                did = d.get("id")
                if did and did not in seen_det:
                    seen_det.add(did)
                    deduped.append(d)

            ctx.details = deduped[:6]

        # ── 10. Recent narrative ──────────────────────────────────────────────
        ctx.recent_narrative = self._load_recent_narrative(headmate)
        print("Recent narratives?")
        print("------------------")
        print("yes" if ctx.recent_narrative is not None else "No")
        # ── 10b. Objects — surface in-rotation or meaningfully dormant ────────
        if headmate and intimate_ok:
            try:
                from core.memory.psychology import _load_object_memories
                now     = time.time()
                objects = _load_object_memories(headmate)
                active  = []
                dormant = []

                for obj in objects.values():
                    if obj.frequency < 2:
                        continue  # not established enough
                    days_since = (now - obj.last_used) / 86400
                    if days_since <= 60:
                        active.append(obj)   # in rotation — can reference naturally
                    elif days_since >= 90:
                        dormant.append(obj)  # nostalgia territory

                if active:
                    names = ", ".join(obj.name for obj in active[:4])
                    ctx.details.append({
                        "content": f"Objects currently in rotation: {names}",
                        "context": "can reference naturally",
                        "id":      "active_objects",
                    })
                if dormant:
                    names = ", ".join(obj.name for obj in dormant[:3])
                    ctx.details.append({
                        "content": f"Objects not used in 3+ months: {names}",
                        "context": "nostalgia territory — only if it comes up naturally",
                        "id":      "dormant_objects",
                    })
            except Exception:
                pass

        # ── 11. Budget trim ───────────────────────────────────────────────────
        ctx = _trim_to_budget(ctx, self.TOKEN_BUDGET)

        duration_ms = round((time.monotonic() - t_start) * 1000)

        

        log_event("MemoryRetriever", "RETRIEVED",
            session     = session_id[:8],
            headmate    = headmate or "unknown",
            hits        = ctx.total_hits,
            memories    = len(ctx.memories),
            entities    = len(ctx.entities),
            places      = len(ctx.places),
            details     = len(ctx.details),
            duration_ms = duration_ms,
        )
        print("FULL PROMPT")
        print("------------------")
        print("------------------")
        print("------------------")
        print(log_event)
        print("------------------")
        print("------------------")
        print("------------------")
        return ctx

    async def _crawl_refs(
        self,
        mem_ids:      list[str],
        entity_names: set,
        place_names:  set,
        depth:        int,
    ) -> dict:
        """
        Follow reference links outward from top memory hits.
        Collects entity and place names found along the way.
        Returns {"entities": set, "places": set}
        """
        found_entities = set()
        found_places   = set()
        visited        = set(mem_ids)
        frontier       = list(mem_ids)

        for _ in range(depth):
            if not frontier:
                break
            next_frontier = []

            for mid in frontier:
                links = memory_store.get_links(mid, direction="out")
                for link in links[:3]:
                    to_id = link.get("to_id")
                    if not to_id or to_id in visited:
                        continue
                    visited.add(to_id)

                    row = memory_store.get_by_id(to_id)
                    if not row:
                        continue

                    mtype     = row.get("memory_type", "")
                    file_path = row.get("file_path", "")

                    if mtype == "entity":
                        name = _name_from_path(file_path)
                        if name:
                            found_entities.add(name)
                    elif mtype == "place":
                        name = _name_from_path(file_path)
                        if name:
                            found_places.add(name)
                    else:
                        next_frontier.append(to_id)

            frontier = next_frontier

        return {"entities": found_entities, "places": found_places}

    def _load_recent_narrative(self, headmate: Optional[str]) -> Optional[str]:
        """
        Load the most recent daily narrative or summary.
        Tries today's summary first, then today's raw log, then yesterday's summary.
        """
        from datetime import datetime, timedelta, timezone

        today     = datetime.now(timezone.utc)
        yesterday = today - timedelta(days=1)

        # Try today's summary
        summary = memory_store.read_file(
            str(memory_store.summary_path(headmate, today).relative_to(memory_store.root))
        )
        if summary:
            return summary.strip()[:600]

        # Try today's raw daily log (trimmed)
        daily = memory_store.read_daily(headmate, today)
        if daily and len(daily.strip()) > 20:
            return daily.strip()[-600:]  # tail — most recent entries

        # Try yesterday's summary
        summary_yd = memory_store.read_file(
            str(memory_store.summary_path(headmate, yesterday).relative_to(memory_store.root))
        )
        if summary_yd:
            return summary_yd.strip()[:400]

        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _name_from_path(file_path: str) -> Optional[str]:
    """Extract entity/place name from file path slug."""
    from pathlib import Path
    stem = Path(file_path).stem
    # Convert slug back to readable name
    return stem.replace("_", " ").strip() if stem else None


def _load_anchor(file_path: str, anchor: Optional[str]) -> Optional[str]:
    """
    Load content from a file, optionally at a specific anchor (heading).
    If no anchor, returns first 300 chars.
    """
    content = memory_store.read_file(file_path)
    if not content:
        return None

    if not anchor:
        return content.strip()[:300]

    # Find the heading matching the anchor (e.g. "1603" → "## 16:03")
    lines   = content.splitlines()
    target  = f"## {anchor[:2]}:{anchor[2:]}" if len(anchor) == 4 else f"## {anchor}"
    start   = None

    for i, line in enumerate(lines):
        if line.startswith(target) or (anchor in line and line.startswith("##")):
            start = i
            break

    if start is None:
        return content.strip()[:300]

    # Grab content until the next heading or 400 chars
    block = []
    for line in lines[start + 1:]:
        if line.startswith("##") and block:
            break
        block.append(line)

    text = "\n".join(block).strip()
    return text[:400] if text else content.strip()[:300]


def _vector_search_details(
    query_emb: bytes,
    headmate:  Optional[str],
    limit:     int = 4,
) -> list[dict]:
    """
    Cosine similarity search over the details table.
    Loads all detail embeddings and scores them.
    """
    try:
        con    = memory_store._connect()
        wheres = ["active = 1", "embedding IS NOT NULL"]
        params = []
        if headmate:
            wheres.append("(headmate = ? OR headmate IS NULL)")
            params.append(headmate.lower())

        rows = con.execute(
            f"SELECT * FROM details WHERE {' AND '.join(wheres)}",
            params
        ).fetchall()
        con.close()

        from core.memory.embedder import _bytes_to_floats, _cosine
        query_vec = _bytes_to_floats(query_emb)
        scored    = []

        for row in rows:
            try:
                sim = _cosine(query_vec, _bytes_to_floats(row["embedding"]))
                if sim >= 0.35:
                    d = dict(row)
                    d["_similarity"] = sim
                    scored.append(d)
            except Exception:
                continue

        scored.sort(key=lambda x: x["_similarity"], reverse=True)
        return scored[:limit]

    except Exception as e:
        log_error("MemoryRetriever", f"detail vector search failed: {e}", exc=None)
        return []


def _trim_to_budget(ctx: MemoryContext, budget_tokens: int) -> MemoryContext:
    """
    Trim the context to stay within the token budget.
    Drops lowest-value items first.
    Rough estimate: 1 token ≈ 4 chars.
    """
    char_budget = budget_tokens * 4
    used        = 0

    # Recent narrative is highest value — keep it all
    if ctx.recent_narrative:
        used += len(ctx.recent_narrative)

    # Entities — trim content if needed
    trimmed_entities = {}
    for name, content in ctx.entities.items():
        if used + len(content) > char_budget:
            # Trim to what fits
            remaining = max(0, char_budget - used)
            if remaining > 100:
                trimmed_entities[name] = content[:remaining]
                used += remaining
        else:
            trimmed_entities[name] = content
            used += len(content)
    ctx.entities = trimmed_entities

    # Places
    trimmed_places = {}
    for name, content in ctx.places.items():
        if used + len(content) > char_budget:
            remaining = max(0, char_budget - used)
            if remaining > 100:
                trimmed_places[name] = content[:remaining]
                used += remaining
        else:
            trimmed_places[name] = content
            used += len(content)
    ctx.places = trimmed_places

    # Memories — drop from the bottom
    trimmed_memories = []
    for m in ctx.memories:
        content = m.get("content", "")
        if used + len(content) > char_budget:
            break
        trimmed_memories.append(m)
        used += len(content)
    ctx.memories = trimmed_memories

    # Details — lowest priority
    trimmed_details = []
    for d in ctx.details:
        content = d.get("content", "")
        if used + len(content) > char_budget:
            break
        trimmed_details.append(d)
        used += len(content)
    ctx.details = trimmed_details

    ctx.token_estimate = used // 4
    return ctx


# ── Singleton ─────────────────────────────────────────────────────────────────

memory_retriever = MemoryRetriever()
