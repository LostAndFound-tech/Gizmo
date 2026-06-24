import os
import json

# ── Tag map ───────────────────────────────────────────────────────────────────

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

    Three passes:
    1. Direct match - query tag is itself a storage tag
    2. TAG_MAP expansion - clinical/behavioral alias resolution
    3. Vocabulary match - organic tags coined by the knowledge writer
    """
    expanded = set()
    query_lower = [t.lower() for t in query_tags]

    # Load live vocabulary - grows organically as knowledge writer coins new tags
    try:
        vocab_data = _read_file("headmates/system/vocabulary.json")
        vocabulary = set(vocab_data.get("tags", [])) if isinstance(vocab_data, dict) else set()
    except Exception:
        vocabulary = set()

    for qt in query_lower:
        # Direct match
        expanded.add(qt)
        # TAG_MAP forward match
        if qt in TAG_MAP:
            expanded.add(qt)
        # TAG_MAP reverse match - qt appears as an alias
        for storage_tag, aliases in TAG_MAP.items():
            if qt in aliases:
                expanded.add(storage_tag)
        # Vocabulary match - organic tags
        if qt in vocabulary:
            expanded.add(qt)

    return expanded


# ── Path helpers ──────────────────────────────────────────────────────────────

def _data_dir() -> str:
    return os.environ.get("DATA_DIR") or "./data"


def _full_path(relative: str) -> str:
    """Generic path under DATA_DIR. Used for legacy paths and sessions."""
    return os.path.join(_data_dir(), relative)


def headmate_path(name: str, filename: str) -> str:
    """
    Path to a file inside a headmate's folder.
    e.g. headmate_path("jess", "personality.json")
         -> /data/headmates/jess/personality.json
    """
    return os.path.join(_data_dir(), "headmates", name.lower(), filename)


def headmate_knowledge_path(name: str, topic: str) -> str:
    """
    Path to a knowledge file inside a headmate's knowledge folder.
    e.g. headmate_knowledge_path("jess", "preferences.json")
         -> /data/headmates/jess/knowledge/preferences.json
    """
    return os.path.join(_data_dir(), "headmates", name.lower(), "knowledge", topic)


def system_path(filename: str) -> str:
    """
    Path to a system-level file.
    e.g. system_path("vocabulary.json")
         -> /data/headmates/system/vocabulary.json
    """
    return os.path.join(_data_dir(), "headmates", "system", filename)


def system_external_path(topic: str) -> str:
    """
    Path to a system external knowledge file.
    e.g. system_external_path("home.json")
         -> /data/headmates/system/external/home.json
    """
    return os.path.join(_data_dir(), "headmates", "system", "external", topic)


def system_relationship_path(pair: str) -> str:
    """
    Path to a system relationship file.
    e.g. system_relationship_path("jess_princess.json")
         -> /data/headmates/system/relationships/jess_princess.json
    """
    return os.path.join(_data_dir(), "headmates", "system", "relationships", pair)


# ── File I/O ──────────────────────────────────────────────────────────────────

def _read_file(_path: str) -> dict | list | None:
    try:
        with open(_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[librarian] read failed ({_path}): {e}")
        return None


def _write_json(_path: str, content: dict | list) -> None:
    try:
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
    except Exception as e:
        print(f"[librarian] write failed ({_path}): {e}")


def _append_to_file(_path: str, content: str) -> None:
    try:
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
    except Exception as e:
        print(f"[librarian] append failed ({_path}): {e}")


# ── Convenience readers using new paths ───────────────────────────────────────

def read_personality(name: str) -> dict:
    return _read_file(headmate_path(name, "personality.json")) or {}


def read_description(name: str) -> dict:
    return _read_file(headmate_path(name, "description.json")) or {}


def read_wellness(name: str) -> dict:
    return _read_file(headmate_path(name, "wellness.json")) or {}


def read_wellness_classification(name: str) -> dict | None:
    return _read_file(headmate_path(name, "wellness_classification.json"))


def read_relationships(name: str) -> dict:
    return _read_file(headmate_path(name, "relationships.json")) or {}


def read_dynamic(name: str) -> dict:
    return _read_file(headmate_path(name, "dynamic.json")) or {}


def read_history(name: str) -> list:
    return _read_file(headmate_path(name, "history.json")) or []


def read_knowledge_topic(name: str, topic: str) -> list:
    return _read_file(headmate_knowledge_path(name, f"{topic}.json")) or []


def read_gizmo_self() -> dict:
    return _read_file(headmate_path("gizmo", "self.json")) or {}


def read_vocabulary() -> list[str]:
    data = _read_file(system_path("vocabulary.json"))
    return data.get("tags", []) if isinstance(data, dict) else []


def read_index() -> list[dict]:
    data = _read_file(system_path("index.json"))
    return data if isinstance(data, list) else []


# ── Tag query ─────────────────────────────────────────────────────────────────

def get_by_tags(name: str, query_tags: list[str]) -> dict:
    """
    Return a filtered slice of a headmate's personality matching the query tags.
    """
    data        = read_personality(name)
    personality = data.get("Personality", {})
    episodes    = data.get("Episodes", [])

    if not personality and not episodes:
        return {"name": name, "matched_tags": [], "personality": {}, "episodes": []}

    expanded = _expand_query_tags(query_tags)

    matched_personality = {
        trait: entry
        for trait, entry in personality.items()
        if set(entry.get("tags", [])) & expanded
    }

    matched_episodes = [
        ep for ep in episodes
        if set(ep.get("tags", [])) & expanded
    ]

    return {
        "name":         name,
        "matched_tags": list(expanded),
        "personality":  matched_personality,
        "episodes":     matched_episodes,
    }


def get_wellness_by_tags(name: str, query_tags: list[str]) -> dict:
    """
    Return a filtered slice of a headmate's wellness signals matching the query tags.
    """
    data     = read_wellness(name)
    expanded = _expand_query_tags(query_tags)

    matched_signals = [
        signal
        for category, signals in data.items()
        if isinstance(signals, list)
        for signal in signals
        if set(signal.get("tags", [])) & expanded
    ]

    return {
        "name":         name,
        "matched_tags": list(expanded),
        "signals":      matched_signals,
    }


def get_knowledge(
    tags:    list[str],
    speaker: str  = None,
    limit:   int  = 10,
) -> list[dict]:
    """
    Return knowledge entries from the flat index matching the given tags.
    Tags matched directly against knowledge entry tags (no TAG_MAP expansion).
    Entries sourced from or mentioning the speaker float to the top.
    """
    index = read_index()
    if not index or not tags:
        return []

    tag_set       = set(t.lower() for t in tags)
    speaker_lower = speaker.lower() if speaker else ""

    matched = [
        entry for entry in index
        if isinstance(entry, dict)
        and set(t.lower() for t in entry.get("tags", [])) & tag_set
    ]

    if not matched:
        return []

    def _score(entry: dict) -> tuple:
        speaker_match = int(
            entry.get("source", "").lower() == speaker_lower
            or (speaker_lower and speaker_lower in entry.get("fact", "").lower())
        )
        return (speaker_match, entry.get("ts", ""))

    matched.sort(key=_score, reverse=True)
    return matched[:limit]


# ── Descriptor merge ──────────────────────────────────────────────────────────

def _safe_dedup(existing_list: list, new_items: list) -> list:
    for item in new_items:
        if item not in existing_list:
            existing_list.append(item)
    return existing_list


def _deep_merge(existing: dict, incoming: dict) -> dict:
    """
    Recursively merge incoming dict into existing dict.
    list + list  -> deduped union
    dict + dict  -> recurse
    missing key  -> take incoming value
    scalar clash -> keep existing (first write wins)
    """
    for key, value in incoming.items():
        if key not in existing:
            existing[key] = value
        elif isinstance(existing[key], dict) and isinstance(value, dict):
            existing[key] = _deep_merge(existing[key], value)
        elif isinstance(existing[key], list) and isinstance(value, list):
            existing[key] = _safe_dedup(existing[key], value)
    return existing


def merge_descriptors(new_data: dict) -> None:
    """
    Merge descriptor data for one or more entities into headmate description files.

    new_data is a name-keyed dict from the descriptor catcher:
      {"Jess": {"Type": "Person", "physical": {...}, ...},
       "the lobby": {"Type": "Place", "file_key": "lobby", ...}}

    People  -> headmates/{name}/description.json
    Places  -> headmates/system/external/places/{file_key}.json
    Objects -> headmates/system/external/objects/{file_key}.json
    """
    for entity_name, entity_data in new_data.items():
        if not isinstance(entity_data, dict):
            continue

        entity_type = entity_data.get("Type", "Person")
        file_key    = entity_data.get("file_key") or entity_name.lower().replace(" ", "_")

        if entity_type == "Person":
            path = headmate_path(entity_name, "description.json")
        elif entity_type == "Place":
            path = system_external_path(f"places/{file_key}.json")
        else:
            path = system_external_path(f"objects/{file_key}.json")

        existing = _read_file(path) or {}
        merged   = _deep_merge(existing, entity_data)
        _write_json(path, merged)
        print(f"[librarian] merged descriptors for {entity_name} -> {path}")


# ── Behavior merge ────────────────────────────────────────────────────────────

def _normalize_personality(personality: dict) -> dict:
    if not personality:
        return personality
    max_count = max((v.get("count", 0) for v in personality.values()), default=0)
    if max_count == 0:
        return personality
    for trait in personality.values():
        trait["weight"] = round(trait.get("count", 0) / max_count, 4)
    return personality


def merge_behaviors(name: str, new_data: dict) -> None:
    """
    Merge incoming behavior data into headmates/{name}/personality.json.

    Personality -> weighted store with tags, count per trait
    Episodes    -> append action->reaction pairs with tags
    Scalars     -> keep existing
    """
    path     = headmate_path(name, "personality.json")
    existing = _read_file(path) or {}

    for key, value in new_data.items():

        if key == "Personality" and isinstance(value, list):
            if "Personality" not in existing:
                existing["Personality"] = {}
            for trait_entry in value:
                if isinstance(trait_entry, dict):
                    trait = trait_entry.get("trait", "")
                    tags  = trait_entry.get("tags", [])
                else:
                    trait = trait_entry
                    tags  = []

                if not trait:
                    continue

                if trait in existing["Personality"]:
                    existing["Personality"][trait]["count"] += 1
                    existing_tags = existing["Personality"][trait].get("tags", [])
                    existing["Personality"][trait]["tags"] = _safe_dedup(existing_tags, tags)
                else:
                    existing["Personality"][trait] = {
                        "count":  1,
                        "weight": 1.0,
                        "tags":   tags,
                    }
            existing["Personality"] = _normalize_personality(existing["Personality"])

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

        elif key not in existing:
            existing[key] = value

    _write_json(path, existing)
    print(f"[librarian] merged behaviors for {name}")


# ── Wellness helpers ──────────────────────────────────────────────────────────

def append_wellness_signal(name: str, signal: dict) -> None:
    """Append a wellness signal to headmates/{name}/wellness.json."""
    path     = headmate_path(name, "wellness.json")
    existing = _read_file(path) or {}
    category = signal.get("category", "general")
    if category not in existing:
        existing[category] = []
    existing[category].append(signal)
    _write_json(path, existing)


def write_wellness_classification(name: str, classification: dict) -> None:
    """
    Write wellness classification to headmates/{name}/wellness_classification.json.
    Archives the previous one first.
    """
    path     = headmate_path(name, "wellness_classification.json")
    existing = _read_file(path)
    if existing:
        from datetime import datetime, timezone
        ts       = existing.get("last_synthesized", datetime.now(timezone.utc).isoformat())
        ts_clean = ts.replace(":", "-").replace(".", "-")[:19]
        archive  = headmate_path(name, f"wellness_classification_archive/{name}_{ts_clean}.json")
        _write_json(archive, existing)
        print(f"[librarian] archived wellness classification for {name}")
    _write_json(path, classification)
    print(f"[librarian] wrote wellness classification for {name}")


# ── Knowledge helpers ─────────────────────────────────────────────────────────

def append_knowledge_entry(route: str, entry: dict) -> None:
    """
    Append a knowledge entry to the appropriate file.
    route is relative to headmates/ e.g. "jess/knowledge/preferences"
    or "system/external/home"
    """
    path     = os.path.join(_data_dir(), "headmates", f"{route}.json")
    existing = _read_file(path)
    if not isinstance(existing, list):
        existing = []
    existing.append(entry)
    _write_json(path, existing)


def append_index_entry(entries: list[dict]) -> None:
    """Append entries to headmates/system/index.json."""
    path  = system_path("index.json")
    index = _read_file(path)
    if not isinstance(index, list):
        index = []
    index.extend(entries)
    _write_json(path, index)


def write_vocabulary(tags: list[str]) -> None:
    """Write the full vocabulary list to headmates/system/vocabulary.json."""
    _write_json(system_path("vocabulary.json"), {"tags": sorted(tags)})


# ── Relationship helpers ──────────────────────────────────────────────────────

def read_system_relationship(name_a: str, name_b: str) -> dict:
    """Read a system-level relationship file for two headmates."""
    pair = "_".join(sorted([name_a.lower(), name_b.lower()]))
    return _read_file(system_relationship_path(f"{pair}.json")) or {}


def write_system_relationship(name_a: str, name_b: str, data: dict) -> None:
    """Write a system-level relationship file for two headmates."""
    pair = "_".join(sorted([name_a.lower(), name_b.lower()]))
    _write_json(system_relationship_path(f"{pair}.json"), data)
    print(f"[librarian] wrote relationship {pair}")


def update_gizmo_relationship(name: str, updates: dict) -> None:
    """
    Update Gizmo's relationship data with a specific headmate
    inside headmates/{name}/relationships.json.
    """
    path     = headmate_path(name, "relationships.json")
    existing = _read_file(path) or {"with_gizmo": {}, "with_system": {}}
    gizmo    = existing.get("with_gizmo", {})
    gizmo.update(updates)
    existing["with_gizmo"] = gizmo
    _write_json(path, existing)
    print(f"[librarian] updated gizmo relationship for {name}")
