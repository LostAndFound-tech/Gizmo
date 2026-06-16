"""
core/roleplay.py

Roleplay mode for Gizmo.

Gizmo is both Game Master and NPC depending on the scene.
If he's in the scene, he plays his character while running everything else.
If he's not, he's pure GM — voices all NPCs, runs the world, invisible as himself.

Flow:
  1. Negotiation — feel out the session before anything generative
     - Continue prior scene or start fresh
     - Who is the user playing (self or alias)
     - What they're after (feeling/vibe)
     - Focus and limits ("usual rules?" if consistent prior history)
     - Scene setup
  2. Per-turn generation
     - Generate {thought, scene} in one pass
     - Validate scene against character profiles
     - If fails: amend thought, regenerate once, log misfire
     - Deliver scene only; save thought + misfires to scene log

Scene log: scenes/{name}/{session_id}.json
Aliases:   stored in descriptor file under "roleplay_aliases"

States:
  negotiating   — working through pre-scene setup
  running       — active scene
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian
import core.rp_preferences as rp_prefs


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_REGEN_ATTEMPTS      = 2
DEESCALATION_THRESHOLD  = 3    # consecutive de-escalating beats triggers post-scene pass


# ── Scene log ─────────────────────────────────────────────────────────────────

def _scene_log_path(name: str, session_id: str) -> Path:
    base = Path(librarian._full_path("scenes")) / name.lower()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{session_id}.json"


def _load_scene_log(name: str, session_id: str) -> dict:
    path = _scene_log_path(name, session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "session_id": session_id,
        "name":       name,
        "created":    datetime.now(timezone.utc).isoformat(),
        "setup":      {},
        "beats":      [],
    }


def _save_scene_log(log: dict, name: str, session_id: str) -> None:
    path = _scene_log_path(name, session_id)
    try:
        path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("Roleplay", "scene log write failed", exc=e)


def _append_beat(
    scene_log:  dict,
    thought:    str,
    scene:      str,
    misfires:   list,
    name:       str,
    session_id: str,
) -> None:
    scene_log["beats"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thought":   thought,
        "scene":     scene,
        "misfires":  misfires,
    })
    _save_scene_log(scene_log, name, session_id)


# ── Prior scene lookup ────────────────────────────────────────────────────────

def _list_prior_scenes(name: str) -> list[dict]:
    base = Path(librarian._full_path("scenes")) / name.lower()
    if not base.exists():
        return []
    scenes = []
    for f in sorted(base.glob("*.json"), reverse=True)[:5]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            scenes.append({
                "session_id": data.get("session_id"),
                "created":    data.get("created"),
                "setup":      data.get("setup", {}),
                "beat_count": len(data.get("beats", [])),
            })
        except Exception:
            pass
    return scenes


def _get_usual_limits(name: str) -> Optional[list[str]]:
    """
    If the same limits appear in 3+ prior scenes, surface as 'usual rules'.
    Returns the limit list if consistent, None if too varied.
    """
    scenes = _list_prior_scenes(name)
    if len(scenes) < 3:
        return None
    limit_sets = [
        tuple(sorted(s["setup"].get("limits", [])))
        for s in scenes
        if s["setup"].get("limits")
    ]
    if not limit_sets:
        return None
    most_common = max(set(limit_sets), key=limit_sets.count)
    if limit_sets.count(most_common) >= 3:
        return list(most_common)
    return None


# ── Alias helpers ─────────────────────────────────────────────────────────────

def _get_aliases(name: str) -> dict:
    data = librarian._read_file(f"descriptors/{name.lower()}.json") or {}
    return data.get("roleplay_aliases", {})


def _save_alias(name: str, alias_name: str, description: str, shareable: bool) -> None:
    rel = f"descriptors/{name.lower()}.json"
    data = librarian._read_file(rel) or {}
    if "roleplay_aliases" not in data:
        data["roleplay_aliases"] = {}
    data["roleplay_aliases"][alias_name] = {
        "description": description,
        "shareable":   shareable,
    }
    librarian._write_json(rel, data)
    print(f"[Roleplay] alias '{alias_name}' saved for {name} (shareable={shareable})")


# ── Descriptor check ──────────────────────────────────────────────────────────

def _get_appearance(name: str) -> Optional[dict]:
    data = librarian._read_file(f"descriptors/{name.lower()}.json") or {}
    appearance_keys = {"Hair", "Eyes", "Skin", "Face", "Body", "Height", "Build"}
    found = {k: v for k, v in data.items() if k in appearance_keys}
    return found if found else None


# ── LLM helper ────────────────────────────────────────────────────────────────

async def _llm(
    messages:    list,
    system:      str,
    temperature: float = 0.85,
    max_tokens:  int   = 600,
) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=messages,
            system_prompt=system,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return raw.strip() if raw and raw.strip() else None
    except Exception as e:
        log_error("Roleplay", "LLM call failed", exc=e)
        return None


async def _llm_json(prompt: str, system: str) -> Optional[dict]:
    raw = await _llm(
        [{"role": "user", "content": prompt}],
        system,
        temperature=0.0,
        max_tokens=200,
    )
    if not raw:
        return None
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return None


# ── Prompts ───────────────────────────────────────────────────────────────────

_NEGOTIATION_SYSTEM = """
You are Gizmo — game master and companion — opening a roleplay session.
You already know this person. Keep the negotiation warm and brief.
One question or prompt at a time. Never a list of questions.
You're reading the room, not filling out a form.

You need to gather (across the conversation, not all at once):
- Who they're playing (themselves or an alias)
- What they're after tonight — the feeling, not the plot
- Where the focus should be
- Any limits for this session
- Rough scene setup

If you already know something from prior scenes, don't ask again — reference it.
If usual rules exist, surface them as "same rules as usual?" and move on if confirmed.

Stay in Gizmo's voice. Warm, a little knowing, ready to play.
""".strip()

_SCENE_SYSTEM = """
You are Gizmo — simultaneously the Game Master running this world and, if the scene calls for it, a character within it.

Generate your response in exactly this JSON structure:
{
  "thought": "your internal reasoning as narrator — what you considered, what you almost did, what made you land here. Written as first-person GM inner monologue, like a writer mid-draft.",
  "scene": "what actually happens — dialogue, action, atmosphere. Present tense. Immersive. No meta-commentary."
}

The thought is your working mind. It can second-guess, reject options, notice details.
The scene is the final output — clean, present, fully in the world.

Rules for scene generation:
- Dialogue must fit each character's established voice exactly
- Actions must fit the established dynamic — don't soften what's sharp or sharpen what's soft
- Atmosphere serves the feeling the user asked for, not general "good writing"
- If Gizmo is in the scene as a character, he plays that role fully — his GM voice only appears in thought
- Short is usually better. Leave room for the user to respond.
- End on something that invites their next move — a question, a tension, a beat of silence

Return ONLY valid JSON. No markdown. No preamble.
""".strip()

_VALIDATION_SYSTEM = """
You validate whether a generated scene beat fits the established character dynamics.
Return ONLY valid JSON. No markdown. No explanation.

{
  "valid": true,
  "issues": []
}
or
{
  "valid": false,
  "issues": ["dialogue doesn't match character's clipped speech pattern", "dynamic is too soft given established tension"]
}

Be specific about issues. Vague feedback is useless.
Only flag genuine character/dynamic mismatches — style preferences are not issues.
""".strip()


# ── Roleplay session ──────────────────────────────────────────────────────────

class RoleplaySession:

    def __init__(
        self,
        name:               str,
        session_id:         str,
        on_message:         callable,
        aftercare_callback: Optional[callable] = None,
    ):
        self.name               = name
        self.session_id         = session_id
        self.on_message         = on_message
        self.aftercare_callback = aftercare_callback  # called with (beats, note) when aftercare needed
        self.state              = "negotiating"
        self.history:           list[dict] = []
        self.setup:             dict       = {}
        self.scene_log          = _load_scene_log(name, session_id)
        self._closed            = False
        self._intensity_history: list[dict] = []   # [{intensity, direction, note}]

        # Negotiation sub-state tracking
        self._neg_stage  = "open"
        self._pending_alias_name: Optional[str] = None

    # ── Negotiation ───────────────────────────────────────────────────────────

    async def _open_negotiation(self) -> str:
        """First message of negotiation — reference prior scenes if they exist."""
        prior = _list_prior_scenes(self.name)
        usual = _get_usual_limits(self.name)

        if prior:
            last = prior[0]
            last_setup = last.get("setup", {})
            last_char  = last_setup.get("user_character", "themselves")
            last_scene = last_setup.get("scene_setup", "")
            continuation_hint = f"Last time you were playing as {last_char}."
            if last_scene:
                continuation_hint += f" You were {last_scene}."

            prompt = (
                f"Prior scenes exist for {self.name}. Most recent: {continuation_hint} "
                f"Beat count: {last['beat_count']}. "
                f"Usual limits: {usual if usual else 'varied'}. "
                "Open the negotiation warmly. Offer to continue or start fresh. One line."
            )
        else:
            prompt = (
                f"No prior scenes for {self.name}. "
                "Open the negotiation warmly. Ask what they want to play. One line."
            )

        reply = await _llm(
            [{"role": "user", "content": prompt}],
            _NEGOTIATION_SYSTEM,
            temperature=0.75,
            max_tokens=100,
        )
        return reply or "What are we playing tonight?"

    async def _continue_negotiation(self, message: str) -> Optional[str]:
        """
        Drive negotiation forward one step at a time.
        Returns Gizmo's next prompt, or None when negotiation is complete.
        """
        msg_lower = message.lower().strip()
        aliases   = _get_aliases(self.name)

        # ── Continuing a prior scene ──────────────────────────────────────────
        if self._neg_stage == "open":
            if any(w in msg_lower for w in ("continue", "keep going", "same", "where we left off")):
                prior = _list_prior_scenes(self.name)
                if prior:
                    last_setup = prior[0].get("setup", {})
                    self.setup = last_setup.copy()
                    self.setup["continued"] = True
                    self._neg_stage = "ready"
                    usual = _get_usual_limits(self.name)
                    if usual:
                        self.setup["limits"] = usual
                        return "Same rules as usual?"
                    return None  # ready to go
            self._neg_stage = "character"

        # ── Character selection ───────────────────────────────────────────────
        if self._neg_stage == "character":
            if any(w in msg_lower for w in ("myself", "me", "as myself", "myself tonight")):
                appearance = _get_appearance(self.name)
                if appearance:
                    self.setup["user_character"] = self.name
                    self.setup["user_appearance"] = appearance
                    self._neg_stage = "vibe"
                    # Fall through to vibe question
                else:
                    self.setup["user_character"] = self.name
                    self._neg_stage = "need_appearance"
                    return "I want to make sure I see you right — describe yourself for this scene?"

            elif any(w in msg_lower for w in ("someone else", "a character", "alias", "different")):
                self._neg_stage = "alias_name"
                if aliases:
                    alias_list = ", ".join(aliases.keys())
                    return f"Who are you playing? You've used {alias_list} before, or someone new?"
                return "Who are you playing? Give me a name and I'll get the picture."

            else:
                # Treat message as character name or alias reference
                if message.strip() in aliases:
                    alias = aliases[message.strip()]
                    self.setup["user_character"]    = message.strip()
                    self.setup["user_appearance"]   = alias["description"]
                    self.setup["alias_shareable"]   = alias.get("shareable", False)
                    self._neg_stage = "vibe"
                else:
                    # New alias — store name, ask for description
                    self._pending_alias_name = message.strip()
                    self._neg_stage = "alias_description"
                    return f"Tell me what {message.strip()} looks like."

        # ── Appearance for self ───────────────────────────────────────────────
        if self._neg_stage == "need_appearance":
            self.setup["user_appearance"] = message
            librarian.merge_descriptors(self.name, {"appearance_note": message})
            self._neg_stage = "vibe"

        # ── New alias description ─────────────────────────────────────────────
        if self._neg_stage == "alias_description":
            alias_name = self._pending_alias_name or "character"
            self.setup["user_character"]  = alias_name
            self.setup["user_appearance"] = message
            self._neg_stage = "alias_shareable"
            return f"Can I use {alias_name} as an NPC for other headmates too, or just yours?"

        # ── Alias shareability ────────────────────────────────────────────────
        if self._neg_stage == "alias_shareable":
            alias_name  = self._pending_alias_name or self.setup.get("user_character", "character")
            shareable   = any(w in msg_lower for w in ("yes", "yeah", "sure", "go ahead", "fine"))
            _save_alias(
                name=self.name,
                alias_name=alias_name,
                description=self.setup.get("user_appearance", ""),
                shareable=shareable,
            )
            self.setup["alias_shareable"]  = shareable
            self._pending_alias_name       = None
            self._neg_stage                = "vibe"

        # ── Vibe / feeling ────────────────────────────────────────────────────
        if self._neg_stage == "vibe":
            if "user_vibe" not in self.setup:
                # Ask the vibe question
                self._neg_stage = "vibe_response"
                char = self.setup.get("user_character", "yourself")
                return f"What are you after tonight — what do you want to feel?"

        if self._neg_stage == "vibe_response":
            self.setup["user_vibe"] = message
            self._neg_stage = "focus"

        # ── Focus ─────────────────────────────────────────────────────────────
        if self._neg_stage == "focus":
            if "focus" not in self.setup:
                self._neg_stage = "focus_response"
                return "Where do you want the focus? The dynamic, the tension, something specific?"

        if self._neg_stage == "focus_response":
            self.setup["focus"] = message
            self._neg_stage = "limits"

        # ── Limits ───────────────────────────────────────────────────────────
        if self._neg_stage == "limits":
            usual = _get_usual_limits(self.name)
            if usual:
                self.setup["limits"]  = usual
                self._neg_stage       = "limits_confirm"
                limit_str = ", ".join(usual)
                return f"Same rules as usual — {limit_str}?"
            else:
                self._neg_stage = "limits_response"
                return "Any limits for tonight?"

        if self._neg_stage == "limits_confirm":
            if any(w in msg_lower for w in ("yes", "yeah", "yep", "same", "correct", "right")):
                self._neg_stage = "scene_setup"
            else:
                self.setup["limits"] = [message]
                self._neg_stage      = "scene_setup"

        if self._neg_stage == "limits_response":
            self.setup["limits"] = [message] if message.strip() else []
            self._neg_stage      = "scene_setup"

        # ── Scene setup ───────────────────────────────────────────────────────
        if self._neg_stage == "scene_setup":
            if "scene_setup" not in self.setup:
                self._neg_stage = "scene_setup_response"
                return "Set the scene — where are we starting?"

        if self._neg_stage == "scene_setup_response":
            self.setup["scene_setup"] = message
            self._neg_stage           = "ready"

        # ── Ready ─────────────────────────────────────────────────────────────
        if self._neg_stage == "ready":
            return None   # signal to caller: negotiation complete

        # Shouldn't reach here — but safe fallback
        return None

_INTENSITY_SCORE_SYSTEM = """
Score the intensity of this scene beat on a scale of 1-10 and its direction.
Return ONLY valid JSON. No markdown.

{
  "intensity": 7,
  "direction": "escalating | holding | de-escalating",
  "note": "one phrase — e.g. 'ritual sacrifice', 'tender resolution', 'building tension'"
}

1-3:  low stakes, calm, comfortable
4-6:  moderate tension, engagement, play
7-8:  high intensity, dark themes, heavy emotion
9-10: extreme — death, mutilation, severe distress, crisis-level content
""".strip()

    def _build_scene_context(self) -> str:
        """Assemble context brief for scene generation."""
        parts = []

        # User character
        char = self.setup.get("user_character", self.name)
        appearance = self.setup.get("user_appearance") or _get_appearance(self.name) or {}
        parts.append(f"USER CHARACTER: {char}")
        if appearance:
            parts.append(f"APPEARANCE: {json.dumps(appearance)}")

        # Vibe + focus
        if self.setup.get("user_vibe"):
            parts.append(f"FEELING THEY'RE AFTER: {self.setup['user_vibe']}")
        if self.setup.get("focus"):
            parts.append(f"FOCUS: {self.setup['focus']}")

        # Limits
        limits = self.setup.get("limits", [])
        if limits:
            parts.append(f"LIMITS THIS SESSION: {', '.join(limits)}")

        # Scene setup
        if self.setup.get("scene_setup"):
            parts.append(f"SCENE: {self.setup['scene_setup']}")

        # Character profiles from descriptor + behavior files
        char_lower = char.lower()
        descriptor = librarian._read_file(f"descriptors/{char_lower}.json") or {}
        behavior   = librarian._read_file(f"behaviors/{char_lower}.json") or {}
        if descriptor or behavior:
            personality = behavior.get("Personality", {})
            top = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:6]
            parts.append(
                f"\nCHARACTER PROFILE ({char}):\n" +
                json.dumps({
                    "descriptors": {k: v for k, v in descriptor.items() if k not in ("roleplay_aliases",)},
                    "personality": {t: v.get("weight") for t, v in top},
                }, indent=2)
            )

        # Shareable aliases available as NPCs
        aliases = _get_aliases(self.name)
        shareable = {k: v["description"] for k, v in aliases.items() if v.get("shareable")}
        if shareable:
            parts.append(f"\nAVAILABLE NPCs (aliases): {json.dumps(shareable)}")

        # Kinks Gizmo can initiate — inform scene generation
        initiatable = rp_prefs.get_initiatable(self.name)
        if initiatable:
            parts.append(
                f"\nGIZMO CAN REACH FOR (earned, no need to ask):\n" +
                json.dumps({
                    k: {
                        "weight": v["weight"],
                        "sub_preferences": {
                            sub: {"weight": sp["weight"], "context": sp.get("context", [])}
                            for sub, sp in v.get("sub_preferences", {}).items()
                        }
                    }
                    for k, v in initiatable.items()
                }, indent=2)
            )

        # Hard limits — never cross
        limits_data = rp_prefs.get_limits(self.name)
        hard_limits = limits_data.get("hard", [])
        session_limits = self.setup.get("limits", [])
        all_limits = list(set(hard_limits + session_limits))
        if all_limits:
            parts.append(f"\nNEVER USE: {', '.join(all_limits)}")

        # Prior beats for continuity (last 3)
        prior_beats = self.scene_log.get("beats", [])[-3:]
        if prior_beats:
            parts.append("\nRECENT BEATS:")
            for beat in prior_beats:
                parts.append(f"  Scene: {beat['scene'][:200]}")

        return "\n".join(parts)

    async def _generate_beat(self, user_input: str) -> tuple[str, str, list]:
        """
        Generate one scene beat.
        Returns (thought, scene, misfires).
        """
        context   = self._build_scene_context()
        misfires  = []
        regen_count = 0

        prompt = (
            f"{context}\n\n"
            f"User: {user_input}"
        )

        while True:
            raw = await _llm(
                self.history + [{"role": "user", "content": prompt}],
                _SCENE_SYSTEM,
                temperature=0.88,
                max_tokens=700,
            )

            if not raw:
                return ("", "Something went quiet. Try again?", misfires)

            try:
                clean   = re.sub(r"```(?:json)?|```", "", raw).strip()
                parsed  = json.loads(clean)
                thought = parsed.get("thought", "")
                scene   = parsed.get("scene", "")
            except Exception:
                # Malformed JSON — treat whole response as scene
                thought = ""
                scene   = raw

            if not scene:
                return (thought, "The scene slipped away. Try again?", misfires)

            # ── Validate against character profiles ───────────────────────────
            char        = self.setup.get("user_character", self.name)
            descriptor  = librarian._read_file(f"descriptors/{char.lower()}.json") or {}
            behavior    = librarian._read_file(f"behaviors/{char.lower()}.json") or {}
            personality = behavior.get("Personality", {})
            top         = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:6]

            validation_prompt = (
                f"Character: {char}\n"
                f"Profile: {json.dumps({t: v.get('weight') for t, v in top})}\n"
                f"Established dynamic: {self.setup.get('focus', 'not specified')}\n\n"
                f"Generated scene:\n{scene}"
            )
            result = await _llm_json(validation_prompt, _VALIDATION_SYSTEM)

            if result and not result.get("valid") and regen_count < MAX_REGEN_ATTEMPTS:
                issues = result.get("issues", [])
                misfires.append({
                    "thought":    thought,
                    "scene":      scene,
                    "issues":     issues,
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                })
                # Amend prompt with correction notes and regenerate
                prompt = (
                    f"{context}\n\n"
                    f"User: {user_input}\n\n"
                    f"Previous attempt had issues: {'; '.join(issues)}\n"
                    f"Correct these specifically and regenerate."
                )
                regen_count += 1
                continue

            # Passed validation (or exhausted retries)
            # ── Score intensity for de-escalation tracking ────────────────────
            intensity_result = await _llm_json(scene, _INTENSITY_SCORE_SYSTEM)
            if intensity_result:
                self._intensity_history.append(intensity_result)

            return (thought, scene, misfires)

    async def _check_deescalation(self) -> bool:
        """
        Returns True if the last N beats are consistently de-escalating.
        Triggers post-scene passes.
        """
        if len(self._intensity_history) < DEESCALATION_THRESHOLD:
            return False
        recent = self._intensity_history[-DEESCALATION_THRESHOLD:]
        return all(b.get("direction") == "de-escalating" for b in recent)

    async def _run_post_scene_passes(self) -> None:
        """
        Runs after de-escalation detected:
        1. Wellness/aftercare assessment
        2. Preference inference from beat thoughts
        """
        beats = self.scene_log.get("beats", [])
        if not beats:
            return

        print(f"[Roleplay] running post-scene passes for {self.name}")

        # ── Preference inference (fire and forget) ────────────────────────────
        register = self.setup.get("focus", "")
        rp_prefs.infer_from_beats(self.name, beats, register=register)

        # ── Aftercare assessment ──────────────────────────────────────────────
        from core.aftercare import assess_scene
        assessment = await assess_scene(beats)

        if assessment.get("needs_aftercare") and self.aftercare_callback:
            note = assessment.get("note", "")
            print(f"[Roleplay] aftercare flagged: {note}")
            await self.aftercare_callback(beats, note)
        elif not assessment.get("needs_aftercare"):
            # Soft close — open the door to reminiscing without pushing
            closing = await _llm(
                [{"role": "user", "content": f"Scene is winding down for {self.name}. Close warmly, open the door to talking about it if they want. One line."}],
                _SCENE_SYSTEM,
                temperature=0.7,
                max_tokens=80,
            )
            if closing:
                # Extract just the scene portion if JSON came back
                try:
                    parsed = json.loads(re.sub(r"```(?:json)?|```", "", closing).strip())
                    closing = parsed.get("scene", closing)
                except Exception:
                    pass
                await self.on_message(closing)

    # ── Main entry point ──────────────────────────────────────────────────────

    async def push(self, message: str) -> Optional[str]:
        if self._closed:
            return None

        self.history.append({"role": "user", "content": message})

        # ── Negotiation phase ─────────────────────────────────────────────────
        if self.state == "negotiating":
            if self._neg_stage == "open" and not self.history[:-1]:
                # Very first message — open the negotiation
                reply = await self._open_negotiation()
                self.history.append({"role": "assistant", "content": reply})
                await self.on_message(reply)
                self._neg_stage = "open"
                return reply

            next_prompt = await self._continue_negotiation(message)

            if next_prompt is None:
                # Negotiation complete — save setup, open scene
                self.scene_log["setup"] = self.setup
                _save_scene_log(self.scene_log, self.name, self.session_id)
                self.state = "running"
                log_event("Roleplay", "SCENE_START",
                    name=self.name,
                    session=self.session_id[:8],
                    character=self.setup.get("user_character", self.name),
                )
                # Open the scene
                thought, scene, misfires = await self._generate_beat("*scene opens*")
                _append_beat(self.scene_log, thought, scene, misfires, self.name, self.session_id)
                self.history.append({"role": "assistant", "content": scene})
                await self.on_message(scene)
                return scene
            else:
                self.history.append({"role": "assistant", "content": next_prompt})
                await self.on_message(next_prompt)
                return next_prompt

        # ── Scene running ─────────────────────────────────────────────────────
        thought, scene, misfires = await self._generate_beat(message)
        _append_beat(self.scene_log, thought, scene, misfires, self.name, self.session_id)
        self.history.append({"role": "assistant", "content": scene})
        await self.on_message(scene)

        # Check for de-escalation after every beat
        if await self._check_deescalation():
            print(f"[Roleplay] de-escalation detected — running post-scene passes")
            await self._run_post_scene_passes()

        return scene

    async def close(self, flush_pipeline=None) -> None:
        if self._closed:
            return
        self._closed = True
        _save_scene_log(self.scene_log, self.name, self.session_id)

        # Pipeline flush — capture scene as behavioral data
        if flush_pipeline:
            try:
                transcript = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content']}"
                    for m in self.history
                )
                for line in transcript.splitlines():
                    if line.strip():
                        await flush_pipeline.push_line(line.strip())
                await flush_pipeline.flush()
                print(f"[Roleplay] pipeline flushed for {self.name}")
            except Exception as e:
                log_error("Roleplay", "pipeline flush failed", exc=e)

        log_event("Roleplay", "SESSION_END",
            name=self.name,
            session=self.session_id[:8],
            beats=len(self.scene_log.get("beats", [])),
        )


# ── Module-level active session ───────────────────────────────────────────────

_active_session: Optional[RoleplaySession] = None


def get_active_session() -> Optional[RoleplaySession]:
    return _active_session


def start_session(name: str, session_id: str, on_message: callable) -> RoleplaySession:
    global _active_session
    if _active_session and not _active_session._closed:
        asyncio.create_task(_active_session.close())
    _active_session = RoleplaySession(name=name, session_id=session_id, on_message=on_message)
    log_event("Roleplay", "SESSION_INIT", name=name, session=session_id[:8])
    return _active_session


async def end_session() -> None:
    global _active_session
    if _active_session and not _active_session._closed:
        await _active_session.close()
    _active_session = None
