"""
core/message_parser.py
Pre-processor for incoming messages.

Extracts structured meaning from inline markup before the message
reaches classification, history, or LLM calls.

Markup conventions:
  *action* or **action**  — stage direction
    Informs Gizmo's tone and emotional response.
    NEVER stored as a fact. NEVER treated as speech.
    Examples: *nervous*, *gets up on the coffee table*, **crying**

  (context)               — lore / background knowledge
    Treated as world-building the speaker wants Gizmo to retain.
    SHOULD be stored as fact — it's intentional context-setting.
    Examples: (she's been awake since 4am), (this is about Tree)

Usage:
    from core.message_parser import parse_message

    parsed = parse_message(raw)
    parsed.clean          # message with all markup stripped — send to LLM
    parsed.spoken         # same as clean (alias)
    parsed.stage          # list of stage direction strings
    parsed.lore           # list of lore strings
    parsed.has_stage       # bool
    parsed.has_lore        # bool
    parsed.stage_block     # formatted string for system prompt injection
    parsed.lore_block      # formatted string for system prompt injection
"""

import re
from dataclasses import dataclass, field


# ── Patterns ──────────────────────────────────────────────────────────────────

# **bold** or *italic* — stage direction
# Non-greedy, allows nested spaces, strips leading/trailing whitespace
_STAGE_RE = re.compile(r'\*{1,2}(.+?)\*{1,2}', re.DOTALL)

# (parenthetical) — lore
# Allows multi-word, stops at closing paren
_LORE_RE = re.compile(r'\(([^)]+)\)', re.DOTALL)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ParsedMessage:
    raw:    str                    # original, unmodified
    clean:  str                    # markup stripped, ready for LLM
    stage:  list[str] = field(default_factory=list)   # stage directions
    lore:   list[str] = field(default_factory=list)   # lore fragments

    @property
    def spoken(self) -> str:
        """Alias for clean — what was actually said."""
        return self.clean

    @property
    def has_stage(self) -> bool:
        return bool(self.stage)

    @property
    def has_lore(self) -> bool:
        return bool(self.lore)

    @property
    def stage_block(self) -> str:
        """
        Formatted block for system prompt injection.
        Tells Gizmo what's happening in the room without it being speech.

        [Stage]
          - gets up on the coffee table
          - nervous
        """
        if not self.stage:
            return ""
        lines = ["[Stage]"]
        for s in self.stage:
            lines.append(f"  - {s}")
        return "\n".join(lines)

    @property
    def lore_block(self) -> str:
        """
        Formatted block for system prompt injection.
        Background context the speaker wants Gizmo to treat as known truth.

        [Context]
          - she's been awake since 4am
          - this is about Tree
        """
        if not self.lore:
            return ""
        lines = ["[Context]"]
        for l in self.lore:
            lines.append(f"  - {l}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        """True if the clean message has no meaningful content."""
        return not self.clean.strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_message(raw: str) -> ParsedMessage:
    """
    Extract stage directions and lore from a raw message.

    Processing order matters:
      1. Extract lore (parentheticals) first — they may contain asterisks
      2. Extract stage directions (asterisks)
      3. Strip both from the clean text
      4. Normalize whitespace in clean text

    Returns a ParsedMessage with all fields populated.
    """
    if not raw:
        return ParsedMessage(raw="", clean="")

    text = raw

    # ── Extract lore ──────────────────────────────────────────────────────────
    lore = []
    def _collect_lore(m: re.Match) -> str:
        content = m.group(1).strip()
        if content:
            lore.append(content)
        return " "   # replace with space to avoid word-merging

    text = _LORE_RE.sub(_collect_lore, text)

    # ── Extract stage directions ──────────────────────────────────────────────
    stage = []
    def _collect_stage(m: re.Match) -> str:
        content = m.group(1).strip()
        if content:
            stage.append(content)
        return " "

    text = _STAGE_RE.sub(_collect_stage, text)

    # ── Normalize clean text ──────────────────────────────────────────────────
    # Collapse multiple spaces, strip leading/trailing whitespace
    clean = re.sub(r'  +', ' ', text).strip()

    return ParsedMessage(
        raw=raw,
        clean=clean,
        stage=stage,
        lore=lore,
    )


def lore_as_facts(lore: list[str], speaker: str) -> list[str]:
    """
    Convert lore fragments into fact sentences attributed to the speaker.
    Used by memory_writer to store lore as extractable facts.

    e.g. "she's been awake since 4am" → "Princess noted: she's been awake since 4am"
    """
    if not lore or not speaker:
        return []
    name = speaker.title()
    return [f"{name} noted: {fragment}" for fragment in lore]
