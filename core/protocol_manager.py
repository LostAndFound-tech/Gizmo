"""
core/protocol_manager.py

Gizmo's self-authored protocol system.

Protocols are documents Gizmo creates himself — instructions, rules,
information, or both. They live on disk and are registered in a central
registry. On each message, relevant protocols are loaded into context
automatically.

Protocol types:
  - "instruction" — tells Gizmo what to do in certain situations
  - "information" — context/facts he wants available
  - "both"        — contains context AND instructions

Registry: /data/personality/protocols/registry.json
Protocol files: /data/personality/protocols/<name>.md (or any path he chooses)

Per-headmate lists: /data/personality/headmates/<name>.json
  "protocols" field: list of protocol paths always loaded for that headmate

Load order in agent:
  1. Personality seed
  2. Per-headmate protocols (always load, no judgement needed)
  3. Global protocols — matched by tags/description against current context
  4. RAG / memory
  5. Response
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_PERSONALITY_DIR  = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_PROTOCOLS_DIR    = _PERSONALITY_DIR / "protocols"
_HEADMATES_DIR    = _PERSONALITY_DIR / "headmates"
_REGISTRY_PATH    = _PROTOCOLS_DIR / "registry.json"

_FAILED_SUMMARY   = {"No transcript content.", "Summary generation failed.", "Summary unavailable."}


# ── Registry helpers ──────────────────────────────────────────────────────────

def _load_registry() -> dict:
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"protocols": []}
    except Exception as e:
        log_error("ProtocolManager", "failed to load registry", exc=e)
        return {"protocols": []}


def _save_registry(registry: dict) -> None:
    try:
        _PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)
        _REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("ProtocolManager", "failed to save registry", exc=e)


# ── Protocol creation ─────────────────────────────────────────────────────────

def create_protocol(
    name: str,
    content: str,
    description: str,
    tags: list[str],
    protocol_type: str = "both",
    headmates: list[str] = None,
) -> dict:
    """
    Create a new protocol file and register it.
    Called by Gizmo (via tool) or by the NLP detection pass.

    Returns: {"success": bool, "path": str, "message": str}
    """
    _PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_name = re.sub(r"[^\w\-]", "_", name.lower().strip())
    if not safe_name:
        return {"success": False, "path": "", "message": "Invalid protocol name"}

    path = _PROTOCOLS_DIR / f"{safe_name}.md"

    # Write file
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = (
            f"# {name}\n"
            f"Created: {timestamp}\n"
            f"Type: {protocol_type}\n"
            f"Tags: {', '.join(tags or [])}\n"
            f"\n---\n\n"
        )
        path.write_text(header + content.strip(), encoding="utf-8")
    except Exception as e:
        log_error("ProtocolManager", f"failed to write protocol file: {name}", exc=e)
        return {"success": False, "path": str(path), "message": str(e)}

    # Register
    registry = _load_registry()
    # Remove any existing entry with same name
    registry["protocols"] = [
        p for p in registry["protocols"]
        if p.get("name") != name
    ]
    registry["protocols"].append({
        "name":        name,
        "path":        str(path),
        "type":        protocol_type,
        "description": description,
        "tags":        tags or [],
        "headmates":   headmates or [],
        "created_at":  datetime.now().isoformat(timespec="seconds"),
        "active":      True,
    })
    _save_registry(registry)

    # Add to per-headmate lists if specified
    if headmates:
        for headmate in headmates:
            _add_to_headmate_list(headmate, str(path))

    log_event("ProtocolManager", "PROTOCOL_CREATED",
        name=name,
        path=str(path),
        type=protocol_type,
        tags=tags,
        headmates=headmates or [],
    )

    return {"success": True, "path": str(path), "message": f"Protocol '{name}' created and registered."}


def _add_to_headmate_list(headmate: str, protocol_path: str) -> None:
    """Add a protocol path to a headmate's always-load list."""
    hm_file = _HEADMATES_DIR / f"{headmate.lower()}.json"
    if not hm_file.exists():
        return
    try:
        data = json.loads(hm_file.read_text(encoding="utf-8"))
        protocols = data.get("protocols", [])
        if protocol_path not in protocols:
            protocols.append(protocol_path)
            data["protocols"] = protocols
            hm_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("ProtocolManager", f"failed to update headmate list for {headmate}", exc=e)


# ── Protocol loading ──────────────────────────────────────────────────────────

def _read_protocol_file(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("ProtocolManager", f"failed to read protocol: {path}", exc=e)
        return None


def _score_protocol(protocol: dict, context: dict, message: str) -> int:
    """
    Score a protocol's relevance to the current context without an LLM call.
    Higher score = more relevant.
    """
    score = 0
    tags     = [t.lower() for t in protocol.get("tags", [])]
    desc     = protocol.get("description", "").lower()
    headmates = [h.lower() for h in protocol.get("headmates", [])]

    current_host = (context.get("current_host") or "").lower()
    fronters     = [f.lower() for f in (context.get("fronters") or [])]
    topics       = [t.lower() for t in (context.get("topics") or [])]
    message_lower = message.lower()

    # Headmate match — strong signal
    if current_host and current_host in headmates:
        score += 10
    for f in fronters:
        if f in headmates:
            score += 5

    # Tag match against topics
    for tag in tags:
        if any(tag in t for t in topics):
            score += 3
        if tag in message_lower:
            score += 2
        if tag in current_host:
            score += 2

    # Description keyword match against message
    desc_words = set(desc.split())
    msg_words  = set(message_lower.split())
    overlap    = desc_words & msg_words
    score += len(overlap)

    return score


def load_protocols_for_context(
    context: dict,
    message: str,
    max_global: int = 3,
) -> str:
    """
    Load relevant protocols for the current context.
    Returns a formatted string block for injection into the system prompt.

    Always loads: per-headmate protocol lists
    Conditionally loads: global protocols scored by relevance
    """
    current_host = (context.get("current_host") or "").lower()
    loaded_paths: set[str] = set()
    blocks: list[str] = []

    registry = _load_registry()
    all_protocols = [p for p in registry.get("protocols", []) if p.get("active", True)]

    # ── Per-headmate always-load ──────────────────────────────────────────────
    if current_host:
        hm_file = _HEADMATES_DIR / f"{current_host}.json"
        if hm_file.exists():
            try:
                data = json.loads(hm_file.read_text(encoding="utf-8"))
                for path in data.get("protocols", []):
                    if path not in loaded_paths:
                        content = _read_protocol_file(path)
                        if content:
                            blocks.append(f"[Protocol — always load for {current_host.title()}]\n{content}")
                            loaded_paths.add(path)
            except Exception as e:
                log_error("ProtocolManager", f"failed to load headmate protocols for {current_host}", exc=e)

    # ── Global protocols — score and pick top N ───────────────────────────────
    scored = []
    for protocol in all_protocols:
        path = protocol.get("path", "")
        if path in loaded_paths:
            continue
        score = _score_protocol(protocol, context, message)
        if score > 0:
            scored.append((score, protocol))

    scored.sort(key=lambda x: x[0], reverse=True)

    for score, protocol in scored[:max_global]:
        path = protocol.get("path", "")
        content = _read_protocol_file(path)
        if content:
            blocks.append(f"[Protocol — {protocol['name']}]\n{content}")
            loaded_paths.add(path)

    if not blocks:
        return ""

    log_event("ProtocolManager", "PROTOCOLS_LOADED",
        host=current_host,
        count=len(blocks),
        paths=list(loaded_paths),
    )

    return "\n\n".join(blocks)


# ── NLP detection pass ────────────────────────────────────────────────────────

# Signals that a rule, boundary, or protocol-worthy moment just happened
_PROTOCOL_SIGNAL_RE = re.compile(
    r"\b("
    r"I('ve| have) decided|I('m| am) deciding|"
    r"from now on|going forward|"
    r"you (must|should|will|are to|need to|have to)|"
    r"(you're|you are) (not allowed|forbidden|required|expected)|"
    r"that'?s (the rule|a rule|our rule|the protocol|how we|how this works)|"
    r"I('ve| have) made a rule|I('m| am) setting a rule|"
    r"I('ve| have) established|"
    r"this is how (we|this|it) works|"
    r"that'?s (a |our |the )?(boundary|limit|line)|"
    r"I (won'?t|will not|refuse to|don'?t) (ever |again )?(do|say|allow|let)|"
    r"I('ve| have) (noticed|realized|learned) that|"
    r"I (expect|require|demand|insist)|"
    r"(that'?s|this is) (non.negotiable|mandatory|required)"
    r")\b",
    re.IGNORECASE,
)


async def detect_and_create_protocol(
    user_message: str,
    gizmo_response: str,
    context: dict,
    llm,
) -> Optional[dict]:
    """
    NLP pass after each exchange. Checks if a protocol-worthy moment just happened.
    If so, drafts and creates the protocol automatically.
    Never raises — errors logged and swallowed.
    """
    combined = f"{gizmo_response}\n{user_message}"

    # Fast heuristic gate — if no signal words, skip LLM call entirely
    if not _PROTOCOL_SIGNAL_RE.search(combined):
        return None

    current_host = (context.get("current_host") or "unknown").lower()
    fronters     = [f.lower() for f in (context.get("fronters") or [])]

    prompt = [{
        "role": "user",
        "content": (
            f"Review this exchange between Gizmo and {current_host.title()}.\n\n"
            f"Gizmo said: {gizmo_response[:800]}\n"
            f"{current_host.title()} said: {user_message[:400]}\n\n"
            f"Did Gizmo establish a rule, boundary, protocol, or persistent behavioral pattern "
            f"that should be remembered and applied in future conversations?\n\n"
            f"If YES, respond with ONLY valid JSON:\n"
            f'{{\n'
            f'  "should_create": true,\n'
            f'  "name": "short descriptive name",\n'
            f'  "description": "one sentence — what this protocol covers",\n'
            f'  "content": "the full protocol text Gizmo should follow",\n'
            f'  "type": "instruction|information|both",\n'
            f'  "tags": ["relevant", "tags"],\n'
            f'  "headmates": ["{current_host}"] or [] if global\n'
            f'}}\n\n'
            f'If NO, respond with ONLY: {{"should_create": false}}'
        )
    }]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You detect when a rule, boundary, or persistent behavioral pattern has been "
                "established in a conversation. Be conservative — only flag clear, intentional "
                "commitments, not passing remarks. JSON only. No preamble."
            ),
            max_new_tokens=300,
            temperature=0.1,
        )

        if not raw or not raw.strip():
            return None

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()

        data = json.loads(raw)

        if not data.get("should_create"):
            return None

        result = create_protocol(
            name          = data.get("name", "unnamed_protocol"),
            content       = data.get("content", ""),
            description   = data.get("description", ""),
            tags          = data.get("tags", []),
            protocol_type = data.get("type", "both"),
            headmates     = data.get("headmates", []),
        )

        if result["success"]:
            log_event("ProtocolManager", "AUTO_CREATED",
                name=data.get("name"),
                host=current_host,
                tags=data.get("tags", []),
            )

        return result

    except Exception as e:
        log_error("ProtocolManager", "detection pass failed", exc=e)
        return None
