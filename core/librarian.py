import os
import json

# ── Tag map ───────────────────────────────────────────────────────────────────
# Storage tags are tight and categorical.
# Query tags are loose and natural — expanded against this map before matching.

TAG_MAP: dict[str, list[str]] = {
    # Appearance
    "appearance":   ["hair", "eyes", "skin", "face", "body", "looks", "color", "height",
                     "weight", "freckles", "tattoo", "piercing", "scar"],
    "fashion":      ["clothing", "dress", "outfit", "style", "wearing", "shoes", "jacket",
                     "shirt", "pants", "skirt", "costume", "accessory", "jewelry"],
    "color":        ["green", "red", "blue", "purple", "black", "white", "pink", "yellow",
                     "brown", "grey", "orange", "gold", "silver"],

    # Identity
    "identity":     ["self", "selfhood", "role", "who i am", "personality", "character"],
    "gender":       ["femininity", "masculinity", "feminine", "masculine", "nonbinary",
                     "gender expression", "pronouns", "trans", "enby"],
    "sexuality":    ["sex", "sexual", "orientation", "attraction", "desire", "intimacy",
                     "kinky", "kink", "bdsm", "queer", "lesbian", "gay", "bi"],

    # Relational
    "relational":   ["relationship", "friendship", "connection", "bond", "trust",
                     "attachment", "social", "interaction", "dynamic"],
    "boundaries":   ["limit", "boundary", "consent", "no", "comfort", "discomfort",
                     "safe", "unsafe", "protective"],
    "care":         ["nurture", "support", "help", "protect", "warmth", "kindness",
                     "empathy", "compassion", "love"],

    # Behavioral
    "behavior":     ["action", "reaction", "habit", "pattern", "tendency", "response"],
    "communication":["speech", "language", "tone", "voice", "words", "phrasing", "dialect"],
    "humor":        ["joke", "funny", "sarcasm", "wit", "playful", "silly", "laugh"],
    "reckless":     ["impulsive", "risky", "dangerous", "thrill", "chaos", "wild",
                     "consequence", "disregard"],

    # Emotional
    "mood":         ["feeling", "emotion", "affect", "energy", "vibe", "state"],
    "anger":        ["rage", "frustration", "irritation", "snapping", "short temper"],
    "warmth":       ["affection", "gentle", "soft", "tender", "loving", "sweet"],
    "anxiety":      ["worry", "fear", "nervous", "panic", "dread", "hypervigilance",
                     "reassurance", "avoidance"],
    "grief":        ["loss", "sadness", "mourning", "missing", "longing"],

    # Clinical
    "adhd":         ["inattention", "distraction", "hyperfocus", "forgetting", "losing things",
                     "impulsivity", "time blindness", "disorganized"],
    "depression":   ["low", "withdrawal", "hopeless", "worthless", "guilt", "fatigue",
                     "anhedonia", "emptiness"],
    "trauma":       ["ptsd", "trigger", "flashback", "hyperarousal", "avoidance",
                     "intrusive", "startle", "numbing"],
    "dissociation": ["derealization", "depersonalization", "switching", "amnesia",
                     "disconnected", "unreal", "foggy"],
    "wellness":     ["health", "symptom", "clinical", "mental", "physical", "medical"],

    # Practical
    "work":         ["job", "task", "project", "commute", "responsibility", "chore"],
    "routine":      ["daily", "habit", "schedule", "morning", "night", "regular"],
    "physical":     ["body", "pain", "tired", "energy", "sleep", "eating", "exercise"],
    "food":         ["eating", "hunger", "appetite", "meal", "snack", "cooking"],

    # System
    "system":       ["plural", "headmate", "alter", "front", "switch", "inside", "outside",
                     "internal", "co-con", "co-fronting"],
    "role":         ["function", "purpose", "protector", "caretaker", "host", "gatekeeper"],
}


def _expand_query_tags(query_tags: list[str]) -> set[str]:
    """
    Expand loose query tags into the full set of storage tags to match against.
    Checks both directions — query tag as storage tag, and query tag as alias.
    """
    expanded = set()
    query_lower = [t.lower() for t in query_tags]

    for qt in query_lower:
        # Direct match — query tag is itself a storage tag
        expanded.add(qt)
        # Forward match — query tag is a storage tag key
        if qt in TAG_MAP:
            expanded.add(qt)
        # Reverse match — query tag appears in a storage tag's alias list
        for storage_tag, aliases in TAG_MAP.items():
            if qt in aliases:
                expanded.add(storage_tag)

    return expanded


# ── Helpers ───────────────────────────────────────────────────────────────────

def _full_path(relative: str) -> str:
    data_dir = os.environ.get("DATA_DIR") or "./data"
    return os.path.join(data_dir, relative)

# ── File I/O ──────────────────────────────────────────────────────────────────

def _write_file(path: str, content: dict, file_name: str, session_id: str) -> None:
    full_path = os.path.join(os.environ.get("DATA_DIR") or "./data", path, session_id, file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

def _append_to_file(_path: str, content: str) -> None:
    try:
        p = _full_path(_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content + "\n")
    except Exception as e:
        print(f"[librarian] append failed: {e}")

def _read_file(_path: str) -> dict | None:
    try:
        p = _full_path(_path)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[librarian] read failed: {e}")
        return None

def _write_json(_path: str, content: dict) -> None:
    try:
        p = _full_path(_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
    except Exception as e:
        print(f"[librarian] write failed: {e}")

# ── Tag query ─────────────────────────────────────────────────────────────────

def get_by_tags(name: str, query_tags: list[str], subfolder: str = "behaviors") -> dict:
    """
    Return a filtered slice of a person's behavior file matching the query tags.
    Query tags are loose — expanded against TAG_MAP before matching.

    Returns:
        {
            "name": "Jess",
            "matched_tags": ["fashion", "appearance"],
            "personality": { matched traits only },
            "episodes": [ matched episodes only ]
        }
    """
    data = _read_file(f"{subfolder}/{name.lower()}.json") or {}
    if not data:
        return {"name": name, "matched_tags": [], "personality": {}, "episodes": []}

    expanded = _expand_query_tags(query_tags)

    # Filter personality traits
    matched_personality = {}
    personality = data.get("Personality", {})
    for trait, entry in personality.items():
        trait_tags = set(entry.get("tags", []))
        if trait_tags & expanded:
            matched_personality[trait] = entry

    # Filter episodes — match if any tag on the episode overlaps
    matched_episodes = []
    for episode in data.get("Episodes", []):
        ep_tags = set(episode.get("tags", []))
        if ep_tags & expanded:
            matched_episodes.append(episode)

    return {
        "name":          name,
        "matched_tags":  list(expanded),
        "personality":   matched_personality,
        "episodes":      matched_episodes,
    }


def get_wellness_by_tags(name: str, query_tags: list[str]) -> dict:
    """
    Return a filtered slice of a person's wellness file matching the query tags.
    """
    data = _read_file(f"wellness/{name.lower()}.json") or {}
    if not data:
        return {"name": name, "matched_tags": [], "signals": []}

    expanded = _expand_query_tags(query_tags)
    matched_signals = []

    for category, signals in data.items():
        if not isinstance(signals, list):
            continue
        for signal in signals:
            signal_tags = set(signal.get("tags", []))
            if signal_tags & expanded:
                matched_signals.append(signal)

    return {
        "name":         name,
        "matched_tags": list(expanded),
        "signals":      matched_signals,
    }


# ── Descriptor merge ──────────────────────────────────────────────────────────

def _safe_dedup(existing_list: list, new_items: list) -> list:
    for item in new_items:
        if item not in existing_list:
            existing_list.append(item)
    return existing_list


def _merge_into(existing: dict, incoming: dict) -> dict:
    for key, value in incoming.items():
        if key not in existing:
            existing[key] = value
        elif isinstance(existing[key], list) and isinstance(value, list):
            existing[key] = _safe_dedup(existing[key], value)
    return existing


def merge_descriptors(name: str, new_data: dict, subfolder: str = "descriptors") -> None:
    """
    Merge descriptor data for a person, object, or location.
    All types get deep merge — people are just as rich as locations.
    Lists dedup. Dicts recurse. Scalars keep existing.
    """
    rel_path = f"{subfolder}/{name.lower()}.json"
    existing = _read_file(rel_path) or {}
    merged   = _deep_merge_descriptor(existing, new_data)
    _write_json(rel_path, merged)
    print(f"[librarian] merged descriptors for {name}")


def _deep_merge_descriptor(existing: dict, incoming: dict) -> dict:
    """
    Deep merge for any descriptor file — people, locations, objects.
    Lists get deduped. Dicts get recursively merged. Scalars keep existing unless missing.
    This means rich nested data accumulates correctly over time:
    - Jess's collar gets details added without overwriting what's known
    - The lobby's tank gets octopus names without losing existing entries
    - Relationships accumulate notes without duplicating
    """
    for key, value in incoming.items():
        if key not in existing:
            existing[key] = value
        elif isinstance(existing[key], dict) and isinstance(value, dict):
            existing[key] = _deep_merge_descriptor(existing[key], value)
        elif isinstance(existing[key], list) and isinstance(value, list):
            existing[key] = _safe_dedup(existing[key], value)
        # Scalars — keep existing (Type, owner, floor_number, color etc don't change)
    return existing

# ── Behavior merge ────────────────────────────────────────────────────────────

def get_known_traits(name: str, subfolder: str = "behaviors") -> dict:
    """
    Return the existing Personality structure for a person.
    Passed into BehaviorCatcher so the LLM avoids duplicating known traits.
    Returns {} if no file exists yet.
    """
    data = _read_file(f"{subfolder}/{name.lower()}.json") or {}
    return data.get("Personality", {})


def merge_behaviors(name: str, new_data: dict, subfolder: str = "behaviors") -> None:
    """
    Merge incoming behavior data for a person.

    Personality structure: Personality[mood][category][trait] = {"count": N}
    - mood:     emotional state when the trait was observed
    - category: type of trait (relational, communication, emotional, etc.)
    - trait:    specific descriptive observation

    Episodes → append action→reaction pairs with tags
    Scalar fields → keep existing
    """
    rel_path = f"{subfolder}/{name.lower()}.json"
    existing = _read_file(rel_path) or {}

    for key, value in new_data.items():

        # ── Nested personality traits: mood → category → trait ────────────────
        if key == "Personality" and isinstance(value, list):
            if "Personality" not in existing:
                existing["Personality"] = {}

            for trait_entry in value:
                if not isinstance(trait_entry, dict):
                    continue

                mood     = trait_entry.get("mood", "unknown").lower().strip()
                category = trait_entry.get("category", "general").lower().strip()
                trait    = trait_entry.get("trait", "").strip()

                if not trait:
                    continue

                # Navigate/create mood → category → trait
                personality = existing["Personality"]
                if mood not in personality:
                    personality[mood] = {}
                if category not in personality[mood]:
                    personality[mood][category] = {}

                if trait in personality[mood][category]:
                    personality[mood][category][trait]["count"] += 1
                else:
                    personality[mood][category][trait] = {"count": 1}

        # ── Action→reaction episode log ───────────────────────────────────────
        elif key == "Episodes" and isinstance(value, list):
            if "Episodes" not in existing:
                existing["Episodes"] = []
            for episode in value:
                if (
                    isinstance(episode, dict)
                    and episode.get("action")
                    and episode.get("reaction")
                ):
                    existing["Episodes"].append(episode)

        # ── Scalar fields — keep existing ─────────────────────────────────────
        elif key not in existing:
            existing[key] = value

    _write_json(rel_path, existing)
    print(f"[librarian] merged behaviors for {name}")
