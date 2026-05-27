"""
core/memory/store.py

Gizmo's memory store.

Two layers:
  1. Markdown files  — narratives, entity docs, place docs. Human-readable,
                       Gizmo-writable, crawlable by reference.
  2. SQLite index    — embeddings + metadata for fast retrieval. Always derived
                       from the markdown files, never the source of truth.

Directory layout:
  {root}/
    memories/
      {headmate}/
        {YYYY-MM-DD}.md        daily narrative log
        {YYYY-MM-DD}-summary.md  end-of-day summary
    entities/
      {slug}.md                living entity documents
    places/
      interior/
        ...
      external/
        ...
    system/
      {YYYY-MM-DD}.md          multi-headmate or unattributed entries
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

def _default_root() -> Path:
    return Path(os.getenv("GIZMO_MEMORY_ROOT", "/data/gizmo/memory"))


class MemoryStore:

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else _default_root()
        self._db_path = self.root / "index.db"
        self._init_dirs()
        self._init_db()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_dirs(self) -> None:
        for d in [
            self.root / "memories",
            self.root / "entities",
            self.root / "places" / "interior",
            self.root / "places" / "external",
            self.root / "agreements",
            self.root / "system",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def agreement_path(self, headmate: str, name: str) -> Path:
        slug = _slugify(name)
        p    = self.root / "agreements" / headmate.lower()
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{slug}.md"

    def _init_db(self) -> None:
        con = self._connect()
        con.executescript("""
            CREATE TABLE IF NOT EXISTS details (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,       -- the raw detail, verbatim or near-verbatim
                headmate    TEXT,
                session_id  TEXT,
                keywords    TEXT,                -- space-separated, auto-extracted
                tags        TEXT,                -- JSON list
                context     TEXT,                -- what was happening around it
                embedding   BLOB,
                created_at  REAL,
                surfaced    INTEGER DEFAULT 0,   -- promoted to a real memory?
                active      INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_details_headmate  ON details(headmate);
            CREATE INDEX IF NOT EXISTS idx_details_surfaced  ON details(surfaced);
            CREATE INDEX IF NOT EXISTS idx_details_created   ON details(created_at);

            CREATE TABLE IF NOT EXISTS memory_index (
                id            TEXT PRIMARY KEY,
                file_path     TEXT NOT NULL,
                anchor        TEXT,           -- heading anchor within file, e.g. "#1603"
                memory_type   TEXT NOT NULL,  -- narrative|entity|place|fact|association|pattern|preference
                memory_subtype TEXT,          -- gizmo defines freely
                headmate      TEXT,
                entities      TEXT,           -- JSON list
                keywords      TEXT,           -- space-separated
                embedding     BLOB,           -- float32 array, serialised
                created_at    REAL,
                last_accessed REAL,
                access_count  INTEGER DEFAULT 0,
                confidence    REAL DEFAULT 1.0,
                session_id    TEXT,
                private       INTEGER DEFAULT 0,  -- 1 = intimate, filter by consent
                shared_with   TEXT,               -- JSON list of headmates who can see this
                active        INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS intimate_consent (
                id            TEXT PRIMARY KEY,
                headmate      TEXT NOT NULL,      -- who has consent
                granted_by    TEXT,               -- who granted it (headmate name or "system")
                granted_at    REAL,
                note          TEXT,               -- context e.g. "Oren asked directly"
                active        INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_memory_private    ON memory_index(private);
            CREATE INDEX IF NOT EXISTS idx_consent_headmate  ON intimate_consent(headmate);

            CREATE TABLE IF NOT EXISTS agreements (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,        -- human name e.g. "Windwalkers", "Jess's Slave Rules"
                headmate      TEXT NOT NULL,
                priority      TEXT DEFAULT 'voluntary', -- mandatory|voluntary
                triggers      TEXT,                 -- JSON list of invocation phrases
                file_path     TEXT NOT NULL,        -- path to the markdown file
                keywords      TEXT,                 -- space-separated for search
                embedding     BLOB,                 -- embedded from name+content summary
                created_at    REAL,
                last_accessed REAL,
                last_updated  REAL,
                active        INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_agreements_headmate  ON agreements(headmate);
            CREATE INDEX IF NOT EXISTS idx_agreements_priority  ON agreements(priority);
            CREATE INDEX IF NOT EXISTS idx_agreements_name      ON agreements(name);

            CREATE TABLE IF NOT EXISTS memory_links (
                id         TEXT PRIMARY KEY,
                from_id    TEXT NOT NULL,
                to_id      TEXT NOT NULL,
                link_type  TEXT NOT NULL,
                strength   REAL DEFAULT 1.0,
                created_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_headmate   ON memory_index(headmate);
            CREATE INDEX IF NOT EXISTS idx_memory_type       ON memory_index(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memory_accessed   ON memory_index(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_links_from        ON memory_links(from_id);
            CREATE INDEX IF NOT EXISTS idx_links_to          ON memory_links(to_id);
        """)
        con.commit()
        con.close()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path)
        con.row_factory = sqlite3.Row
        return con

    # ── File paths ────────────────────────────────────────────────────────────

    def daily_path(
        self,
        headmate:  Optional[str],
        date:      Optional[datetime] = None,
        intimate:  bool = False,
    ) -> Path:
        date = date or datetime.now(timezone.utc)
        ds   = date.strftime("%Y-%m-%d")
        if headmate:
            if intimate:
                p = self.root / "memories" / headmate.lower() / "intimate"
            else:
                p = self.root / "memories" / headmate.lower()
            p.mkdir(parents=True, exist_ok=True)
            return p / f"{ds}.md"
        else:
            return self.root / "system" / f"{ds}.md"

    def summary_path(self, headmate: Optional[str], date: Optional[datetime] = None) -> Path:
        # Summaries are always general — never intimate
        base = self.daily_path(headmate, date, intimate=False)
        return base.parent / (base.stem + "-summary.md")

    # ── Consent ───────────────────────────────────────────────────────────────

    def grant_intimate_consent(
        self,
        headmate:   str,
        granted_by: str  = "system",
        note:       str  = "",
    ) -> None:
        """
        Grant a headmate access to intimate memories during retrieval.
        This is stored permanently — Gizmo remembers who can see what.
        """
        con = self._connect()
        con.execute("""
            INSERT OR REPLACE INTO intimate_consent
              (id, headmate, granted_by, granted_at, note, active)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            _make_id("consent", headmate.lower()),
            headmate.lower(),
            granted_by,
            time.time(),
            note,
        ))
        con.commit()
        con.close()

    def revoke_intimate_consent(self, headmate: str) -> None:
        con = self._connect()
        con.execute(
            "UPDATE intimate_consent SET active = 0 WHERE headmate = ?",
            (headmate.lower(),)
        )
        con.commit()
        con.close()

    def has_intimate_consent(self, headmate: str) -> bool:
        """Check if a headmate has consent to see intimate memories."""
        con = self._connect()
        row = con.execute(
            "SELECT id FROM intimate_consent WHERE headmate = ? AND active = 1",
            (headmate.lower(),)
        ).fetchone()
        con.close()
        return row is not None

    def list_consent(self) -> list[dict]:
        """List all active intimate consent grants."""
        con  = self._connect()
        rows = con.execute(
            "SELECT * FROM intimate_consent WHERE active = 1 ORDER BY granted_at DESC"
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]

    # ── Agreements ────────────────────────────────────────────────────────────

    def write_agreement(
        self,
        name:      str,
        headmate:  str,
        content:   str,
        priority:  str        = "voluntary",
        triggers:  list[str]  = None,
        keywords:  str        = "",
        embedding: Optional[bytes] = None,
        refs:      list[str]  = None,
    ) -> str:
        """
        Write a new agreement file.
        Returns the agreement id.
        """
        path   = self.agreement_path(headmate, name)
        agr_id = _make_id("agreement", headmate.lower(), _slugify(name))
        now    = time.time()

        trigger_list = triggers or []
        refs_section = ""
        if refs:
            refs_section = "\n## References\n" + "\n".join(f"- {r}" for r in refs) + "\n"

        file_content = (
            f"# {name}\n"
            f"priority: {priority}\n"
            f"headmate: {headmate.lower()}\n"
            f"triggers: {', '.join(trigger_list)}\n"
            f"last_updated: {_fmt_date(now)}\n\n"
            f"{content.strip()}\n"
            + refs_section
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(file_content)

        con = self._connect()
        con.execute("""
            INSERT OR REPLACE INTO agreements
              (id, name, headmate, priority, triggers, file_path,
               keywords, embedding, created_at, last_accessed, last_updated, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            agr_id,
            name,
            headmate.lower(),
            priority,
            json.dumps(trigger_list),
            str(path.relative_to(self.root)),
            (keywords + " " + name.lower() + " " + " ".join(trigger_list)).strip(),
            embedding,
            now, now, now,
        ))
        con.commit()
        con.close()
        return agr_id

    def update_agreement(
        self,
        name:      str,
        headmate:  str,
        content:   str,
        ref:       Optional[str] = None,
        keywords:  str           = "",
        embedding: Optional[bytes] = None,
    ) -> bool:
        """
        Update an existing agreement file.
        Rewrites the content section, preserving header metadata.
        Returns True if found, False if not.
        """
        path = self.agreement_path(headmate, name)
        if not path.exists():
            return False

        existing = path.read_text(encoding="utf-8")
        now      = time.time()

        # Update last_updated in header
        lines = existing.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("last_updated:"):
                new_lines.append(f"last_updated: {_fmt_date(now)}")
            else:
                new_lines.append(line)

        # Find where header ends (first blank line after metadata)
        header_end = 0
        in_header  = True
        for i, line in enumerate(new_lines):
            if in_header and line.startswith("#"):
                continue
            if in_header and (line.startswith("priority:") or
                              line.startswith("headmate:") or
                              line.startswith("triggers:") or
                              line.startswith("last_updated:")):
                header_end = i
                continue
            if in_header and line.strip() == "":
                in_header  = False
                header_end = i
                break

        # Rebuild: header + new content + refs
        header = "\n".join(new_lines[:header_end + 1])
        refs_block = ""
        if "## References" in existing:
            refs_block = "\n## References" + existing.split("## References", 1)[1]
        if ref:
            if refs_block:
                refs_block = refs_block.rstrip() + f"\n- {ref}\n"
            else:
                refs_block = f"\n## References\n- {ref}\n"

        new_content = f"{header}\n\n{content.strip()}\n{refs_block}"

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        agr_id = _make_id("agreement", headmate.lower(), _slugify(name))
        con    = self._connect()
        update = {"last_updated": now, "last_accessed": now}
        if embedding:
            update["embedding"] = embedding
        if keywords:
            row = con.execute(
                "SELECT keywords FROM agreements WHERE id = ?", (agr_id,)
            ).fetchone()
            existing_kw = row["keywords"] if row else ""
            update["keywords"] = (existing_kw + " " + keywords).strip()
        sets = ", ".join(f"{k} = ?" for k in update)
        con.execute(
            f"UPDATE agreements SET {sets} WHERE id = ?",
            list(update.values()) + [agr_id],
        )
        con.commit()
        con.close()
        return True

    def read_agreement(self, name: str, headmate: str) -> Optional[str]:
        """Read an agreement file by name and headmate."""
        path = self.agreement_path(headmate, name)
        if not path.exists():
            return None
        agr_id = _make_id("agreement", headmate.lower(), _slugify(name))
        con    = self._connect()
        con.execute(
            "UPDATE agreements SET last_accessed = ? WHERE id = ?",
            (time.time(), agr_id)
        )
        con.commit()
        con.close()
        return path.read_text(encoding="utf-8")

    def get_mandatory_agreements(self, headmate: str) -> list[dict]:
        """Get all mandatory agreements for a headmate — loaded every session."""
        con  = self._connect()
        rows = con.execute(
            "SELECT * FROM agreements WHERE headmate = ? AND priority = 'mandatory' "
            "AND active = 1 ORDER BY last_updated DESC",
            (headmate.lower(),)
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def get_voluntary_agreements(self, headmate: str) -> list[dict]:
        """Get all voluntary agreements for a headmate — loaded on invocation."""
        con  = self._connect()
        rows = con.execute(
            "SELECT * FROM agreements WHERE headmate = ? AND priority = 'voluntary' "
            "AND active = 1 ORDER BY last_accessed DESC",
            (headmate.lower(),)
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def match_agreement_trigger(
        self,
        message:  str,
        headmate: str,
    ) -> list[dict]:
        """
        Check if a message invokes any voluntary agreements.
        Returns matching agreements sorted by match strength.
        """
        voluntary = self.get_voluntary_agreements(headmate)
        if not voluntary:
            return []

        msg_lower = message.lower()
        matches   = []

        for agr in voluntary:
            # Direct name match
            if agr["name"].lower() in msg_lower:
                matches.append((agr, 2.0))
                continue
            # Trigger phrase match
            try:
                triggers = json.loads(agr.get("triggers") or "[]")
                for trigger in triggers:
                    if trigger.lower() in msg_lower:
                        matches.append((agr, 1.0))
                        break
            except Exception:
                pass

        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]

    def agreement_exists(self, name: str, headmate: str) -> bool:
        return self.agreement_path(headmate, name).exists()

    def list_agreements(self, headmate: str) -> list[dict]:
        """List all active agreements for a headmate."""
        con  = self._connect()
        rows = con.execute(
            "SELECT id, name, priority, triggers, last_updated FROM agreements "
            "WHERE headmate = ? AND active = 1 ORDER BY priority DESC, name ASC",
            (headmate.lower(),)
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def deactivate_agreement(self, name: str, headmate: str) -> bool:
        """Deactivate an agreement — it's no longer in effect."""
        agr_id = _make_id("agreement", headmate.lower(), _slugify(name))
        con    = self._connect()
        con.execute(
            "UPDATE agreements SET active = 0 WHERE id = ?", (agr_id,)
        )
        con.commit()
        con.close()
        return True
        slug = _slugify(name)
        return self.root / "entities" / f"{slug}.md"

    def place_path(self, name: str, interior: bool = False) -> Path:
        slug    = _slugify(name)
        subdir  = "interior" if interior else "external"
        p       = self.root / "places" / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{slug}.md"

    def resolve_path(self, ref: str) -> Path:
        """Resolve a refs: string to an absolute path."""
        return self.root / ref.lstrip("/")

    # ── Narrative write ───────────────────────────────────────────────────────

    def append_narrative(
        self,
        text:           str,
        headmate:       Optional[str],
        session_id:     str,
        register:       str            = "neutral",
        timestamp:      Optional[float] = None,
        refs:           list[str]      = None,
        embedding:      Optional[bytes] = None,
        memory_subtype: Optional[str]  = None,
        keywords:       str            = "",
        entities:       list[str]      = None,
        intimate:       bool           = False,
        shared_with:    list[str]      = None,
    ) -> str:
        """
        Append a narrative entry to the daily file.
        intimate=True writes to the intimate subdirectory and sets private=1.
        Returns the memory id.
        """
        ts      = timestamp or time.time()
        dt      = datetime.fromtimestamp(ts, tz=timezone.utc)
        anchor  = dt.strftime("%H%M")
        mem_id  = _make_id(session_id, anchor, "i" if intimate else "g")
        path    = self.daily_path(headmate, dt, intimate=intimate)

        # Build the markdown block
        refs_line = ""
        if refs:
            refs_line = f"\nrefs: {', '.join(refs)}"

        block = (
            f"\n## {dt.strftime('%H:%M')} | "
            f"register: {register} | "
            f"session: {session_id[:8]}"
            + (f" | {memory_subtype}" if memory_subtype else "")
            + (" | intimate" if intimate else "")
            + f"\n\n{text.strip()}"
            + refs_line
            + "\n"
        )

        with open(path, "a", encoding="utf-8") as f:
            f.write(block)

        # Index it
        rel_path = str(path.relative_to(self.root))
        self._index(
            mem_id        = mem_id,
            file_path     = rel_path,
            anchor        = anchor,
            memory_type   = "narrative",
            memory_subtype = memory_subtype,
            headmate      = headmate,
            entities      = entities or [],
            keywords      = keywords,
            embedding     = embedding,
            session_id    = session_id,
            created_at    = ts,
            private       = 1 if intimate else 0,
            shared_with   = shared_with or [],
        )

        return mem_id

    # ── Entity write ──────────────────────────────────────────────────────────

    def write_entity(
        self,
        name:          str,
        content:       str,
        headmate:      Optional[str]  = None,
        memory_subtype: Optional[str] = None,
        keywords:      str            = "",
        entities:      list[str]      = None,
        embedding:     Optional[bytes] = None,
        session_id:    str            = "",
        refs:          list[str]      = None,
    ) -> str:
        """
        Write or overwrite an entity document.
        Entities are living documents — call update_entity to enrich.
        """
        path   = self.entity_path(name)
        mem_id = _make_id("entity", _slugify(name))

        refs_section = ""
        if refs:
            refs_section = "\n## References\n" + "\n".join(f"- {r}" for r in refs) + "\n"

        content_block = (
            f"# {name}\n\n"
            f"{content.strip()}\n"
            + refs_section
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(content_block)

        rel_path = str(path.relative_to(self.root))
        self._index(
            mem_id        = mem_id,
            file_path     = rel_path,
            anchor        = None,
            memory_type   = "entity",
            memory_subtype = memory_subtype,
            headmate      = headmate,
            entities      = entities or [name],
            keywords      = keywords or name.lower(),
            embedding     = embedding,
            session_id    = session_id,
            created_at    = time.time(),
        )

        return mem_id

    def update_entity(
        self,
        name:      str,
        additions: str,
        ref:       Optional[str] = None,
        embedding: Optional[bytes] = None,
        keywords:  str = "",
    ) -> bool:
        """
        Append new information to an existing entity document.
        Adds a ref link if provided.
        Returns True if entity existed, False if not found.
        """
        path = self.entity_path(name)
        if not path.exists():
            return False

        content = path.read_text(encoding="utf-8")

        # Append to refs section if it exists, else add it
        if ref:
            if "## References" in content:
                content = content.rstrip() + f"\n- {ref}\n"
            else:
                content = content.rstrip() + f"\n\n## References\n- {ref}\n"

        # Append the new information before refs
        if "## References" in content:
            parts   = content.split("## References", 1)
            content = parts[0].rstrip() + f"\n\n{additions.strip()}\n\n## References" + parts[1]
        else:
            content = content.rstrip() + f"\n\n{additions.strip()}\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        # Update index
        mem_id = _make_id("entity", _slugify(name))
        self._touch(mem_id, bump_access=False)
        if embedding or keywords:
            con = self._connect()
            updates = {}
            if embedding:
                updates["embedding"] = embedding
            if keywords:
                # Append to existing keywords
                row = con.execute(
                    "SELECT keywords FROM memory_index WHERE id = ?", (mem_id,)
                ).fetchone()
                existing = row["keywords"] if row else ""
                updates["keywords"] = (existing + " " + keywords).strip()
            if updates:
                sets = ", ".join(f"{k} = ?" for k in updates)
                con.execute(
                    f"UPDATE memory_index SET {sets} WHERE id = ?",
                    list(updates.values()) + [mem_id],
                )
                con.commit()
            con.close()

        return True

    # ── Place write ───────────────────────────────────────────────────────────

    def write_place(
        self,
        name:          str,
        content:       str,
        interior:      bool           = False,
        headmate:      Optional[str]  = None,
        memory_subtype: Optional[str] = None,
        keywords:      str            = "",
        entities:      list[str]      = None,
        embedding:     Optional[bytes] = None,
        session_id:    str            = "",
    ) -> str:
        path   = self.place_path(name, interior=interior)
        mem_id = _make_id("place", _slugify(name))

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n{content.strip()}\n")

        rel_path = str(path.relative_to(self.root))
        self._index(
            mem_id        = mem_id,
            file_path     = rel_path,
            anchor        = None,
            memory_type   = "place",
            memory_subtype = memory_subtype,
            headmate      = headmate,
            entities      = entities or [],
            keywords      = keywords or name.lower(),
            embedding     = embedding,
            session_id    = session_id,
            created_at    = time.time(),
        )

        return mem_id

    def update_place(self, name: str, additions: str, interior: bool = False) -> bool:
        path = self.place_path(name, interior=interior)
        if not path.exists():
            return False
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{additions.strip()}\n")
        mem_id = _make_id("place", _slugify(name))
        self._touch(mem_id, bump_access=False)
        return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_file(self, path_ref: str) -> Optional[str]:
        """Read a memory file by relative path."""
        path = self.resolve_path(path_ref)
        if not path.exists():
            return None
        self._touch_by_path(str(path.relative_to(self.root)))
        return path.read_text(encoding="utf-8")

    def read_entity(self, name: str) -> Optional[str]:
        path = self.entity_path(name)
        if not path.exists():
            return None
        mem_id = _make_id("entity", _slugify(name))
        self._touch(mem_id)
        return path.read_text(encoding="utf-8")

    def read_place(self, name: str, interior: bool = False) -> Optional[str]:
        path = self.place_path(name, interior=interior)
        if not path.exists():
            return None
        mem_id = _make_id("place", _slugify(name))
        self._touch(mem_id)
        return path.read_text(encoding="utf-8")

    def read_daily(
        self,
        headmate: Optional[str],
        date:     Optional[datetime] = None,
    ) -> Optional[str]:
        path = self.daily_path(headmate, date)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_refs(self, file_path: str) -> list[str]:
        """Extract all refs: lines from a markdown file."""
        path = self.resolve_path(file_path)
        if not path.exists():
            return []
        refs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("refs:"):
                for r in line[5:].split(","):
                    r = r.strip()
                    if r:
                        refs.append(r)
            elif line.startswith("- ") and "/" in line:
                # refs section list items
                refs.append(line[2:].strip())
        return refs

    # ── Links ─────────────────────────────────────────────────────────────────

    def link(
        self,
        from_id:   str,
        to_id:     str,
        link_type: str,
        strength:  float = 1.0,
    ) -> str:
        link_id = _make_id(from_id, to_id)
        con     = self._connect()
        con.execute("""
            INSERT OR REPLACE INTO memory_links
              (id, from_id, to_id, link_type, strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (link_id, from_id, to_id, link_type, strength, time.time()))
        con.commit()
        con.close()
        return link_id

    def get_links(self, mem_id: str, direction: str = "both") -> list[dict]:
        con  = self._connect()
        rows = []
        if direction in ("out", "both"):
            rows += con.execute(
                "SELECT * FROM memory_links WHERE from_id = ? AND active = 1 "
                "ORDER BY strength DESC",
                (mem_id,)
            ).fetchall() if False else con.execute(
                "SELECT * FROM memory_links WHERE from_id = ?  ORDER BY strength DESC",
                (mem_id,)
            ).fetchall()
        if direction in ("in", "both"):
            rows += con.execute(
                "SELECT * FROM memory_links WHERE to_id = ? ORDER BY strength DESC",
                (mem_id,)
            ).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def reinforce_link(self, from_id: str, to_id: str, delta: float = 0.1) -> None:
        link_id = _make_id(from_id, to_id)
        con     = self._connect()
        con.execute("""
            UPDATE memory_links
            SET strength = MIN(1.0, strength + ?)
            WHERE id = ?
        """, (delta, link_id))
        con.commit()
        con.close()

    # ── Index / search ────────────────────────────────────────────────────────

    def search_index(
        self,
        keywords:  str            = "",
        headmate:  Optional[str]  = None,
        memory_type: Optional[str] = None,
        limit:     int            = 10,
        min_confidence: float     = 0.2,
    ) -> list[dict]:
        """
        Keyword search against the index.
        Embedding-based search handled separately by retrieval layer.
        """
        con    = self._connect()
        wheres = ["active = 1", "confidence >= ?"]
        params: list = [min_confidence]

        if headmate:
            wheres.append("(headmate = ? OR headmate IS NULL)")
            params.append(headmate.lower())
        if memory_type:
            wheres.append("memory_type = ?")
            params.append(memory_type)
        if keywords:
            words = [w for w in keywords.lower().split() if len(w) > 2][:6]
            if words:
                kw_clause = " OR ".join("keywords LIKE ?" for _ in words)
                wheres.append(f"({kw_clause})")
                params.extend(f"%{w}%" for w in words)

        sql = (
            f"SELECT * FROM memory_index WHERE {' AND '.join(wheres)} "
            f"ORDER BY last_accessed DESC, access_count DESC "
            f"LIMIT ?"
        )
        params.append(limit)

        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def get_by_id(self, mem_id: str) -> Optional[dict]:
        con = self._connect()
        row = con.execute(
            "SELECT * FROM memory_index WHERE id = ?", (mem_id,)
        ).fetchone()
        con.close()
        return dict(row) if row else None

    def entity_exists(self, name: str) -> bool:
        return self.entity_path(name).exists()

    def place_exists(self, name: str, interior: bool = False) -> bool:
        return self.place_path(name, interior=interior).exists()

    # ── Details ───────────────────────────────────────────────────────────────

    def write_detail(
        self,
        content:    str,
        headmate:   Optional[str],
        session_id: str,
        keywords:   str        = "",
        tags:       list[str]  = None,
        context:    str        = "",
        embedding:  Optional[bytes] = None,
    ) -> str:
        """
        Write a raw detail catch — verbatim or near-verbatim.
        These are the asides, throwaway mentions, offhand details.
        Not interpreted, just caught.
        Returns the detail id.
        """
        det_id = _make_id(session_id, content[:40], str(time.time()))
        con    = self._connect()
        con.execute("""
            INSERT OR IGNORE INTO details
              (id, content, headmate, session_id, keywords, tags,
               context, embedding, created_at, surfaced, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
        """, (
            det_id,
            content,
            headmate.lower() if headmate else None,
            session_id,
            keywords.lower(),
            json.dumps(tags or []),
            context,
            embedding,
            time.time(),
        ))
        con.commit()
        con.close()
        return det_id

    def search_details(
        self,
        keywords:        str           = "",
        headmate:        Optional[str] = None,
        tags:            list[str]     = None,
        limit:           int           = 10,
        unsurfaced_only: bool          = False,
    ) -> list[dict]:
        """
        Search raw details by keyword and/or tag.
        Used by retriever to catch asides that might be relevant.
        """
        con    = self._connect()
        wheres = ["active = 1"]
        params = []

        if headmate:
            wheres.append("(headmate = ? OR headmate IS NULL)")
            params.append(headmate.lower())
        if unsurfaced_only:
            wheres.append("surfaced = 0")
        if tags:
            for tag in tags[:3]:
                wheres.append("tags LIKE ?")
                params.append(f"%{tag}%")
        if keywords:
            words = [w for w in keywords.lower().split() if len(w) > 2][:6]
            if words:
                kw_clause = " OR ".join(
                    "(keywords LIKE ? OR content LIKE ?)" for _ in words
                )
                wheres.append(f"({kw_clause})")
                for w in words:
                    params.extend([f"%{w}%", f"%{w}%"])

        sql = (
            f"SELECT * FROM details WHERE {' AND '.join(wheres)} "
            f"ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def promote_detail(self, det_id: str) -> None:
        """
        Mark a detail as surfaced — promoted to a real memory.
        Keeps the detail for historical reference but flags it as handled.
        """
        con = self._connect()
        con.execute(
            "UPDATE details SET surfaced = 1 WHERE id = ?", (det_id,)
        )
        con.commit()
        con.close()

    def get_recent_details(
        self,
        headmate:  Optional[str]   = None,
        limit:     int             = 20,
        since:     Optional[float] = None,
    ) -> list[dict]:
        """Pull recent unsurfaced details — used by encoding pass."""
        con    = self._connect()
        wheres = ["active = 1", "surfaced = 0"]
        params = []

        if headmate:
            wheres.append("(headmate = ? OR headmate IS NULL)")
            params.append(headmate.lower())
        if since:
            wheres.append("created_at >= ?")
            params.append(since)

        sql = (
            f"SELECT * FROM details WHERE {' AND '.join(wheres)} "
            f"ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]

    # ── Touch / decay ─────────────────────────────────────────────────────────

    def _touch(self, mem_id: str, bump_access: bool = True) -> None:
        con = self._connect()
        if bump_access:
            con.execute("""
                UPDATE memory_index
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (time.time(), mem_id))
        else:
            con.execute("""
                UPDATE memory_index SET last_accessed = ? WHERE id = ?
            """, (time.time(), mem_id))
        con.commit()
        con.close()

    def _touch_by_path(self, rel_path: str) -> None:
        con = self._connect()
        con.execute("""
            UPDATE memory_index
            SET last_accessed = ?, access_count = access_count + 1
            WHERE file_path = ?
        """, (time.time(), rel_path))
        con.commit()
        con.close()

    def touch_memory(self, mem_id: str) -> None:
        """Mark a memory as recently relevant without changing content."""
        self._touch(mem_id, bump_access=True)

    def decay_pass(self, dry_run: bool = False) -> dict:
        """
        Reduce confidence on memories not accessed in 90 days.
        Deactivate anything below 0.2.
        Returns summary of what changed.
        """
        cutoff     = time.time() - (90 * 86400)
        con        = self._connect()
        stale      = con.execute("""
            SELECT id, confidence, access_count FROM memory_index
            WHERE last_accessed < ? AND active = 1 AND access_count < 5
        """, (cutoff,)).fetchall()

        decayed    = 0
        deactivated = 0

        for row in stale:
            new_conf = row["confidence"] * 0.85
            if not dry_run:
                if new_conf < 0.2:
                    con.execute(
                        "UPDATE memory_index SET active = 0 WHERE id = ?",
                        (row["id"],)
                    )
                    deactivated += 1
                else:
                    con.execute(
                        "UPDATE memory_index SET confidence = ? WHERE id = ?",
                        (new_conf, row["id"])
                    )
                    decayed += 1

        if not dry_run:
            con.commit()
        con.close()

        return {"stale": len(stale), "decayed": decayed, "deactivated": deactivated}

    # ── Internal index write ──────────────────────────────────────────────────

    def _index(
        self,
        mem_id:        str,
        file_path:     str,
        anchor:        Optional[str],
        memory_type:   str,
        memory_subtype: Optional[str],
        headmate:      Optional[str],
        entities:      list[str],
        keywords:      str,
        embedding:     Optional[bytes],
        session_id:    str,
        created_at:    float,
        private:       int        = 0,
        shared_with:   list[str]  = None,
    ) -> None:
        con = self._connect()
        con.execute("""
            INSERT OR REPLACE INTO memory_index
              (id, file_path, anchor, memory_type, memory_subtype,
               headmate, entities, keywords, embedding,
               created_at, last_accessed, session_id,
               private, shared_with, active, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1.0)
        """, (
            mem_id, file_path, anchor, memory_type, memory_subtype,
            headmate.lower() if headmate else None,
            json.dumps(entities),
            keywords.lower(),
            embedding,
            created_at, created_at, session_id,
            private,
            json.dumps(shared_with or []),
        ))
        con.commit()
        con.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_-]+", "_", name)
    return name.strip("_")


def _fmt_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _make_id(*parts: str) -> str:
    raw = ":".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ── Singleton ─────────────────────────────────────────────────────────────────

memory_store = MemoryStore()
