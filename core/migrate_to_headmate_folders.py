"""
migrate_to_headmate_folders.py

One-time migration script. Moves existing data into the new per-headmate folder structure.

OLD:
  data/behaviors/{name}.json
  data/descriptors/{name}.json
  data/wellness/{name}.json
  data/wellness/summaries/{name}/...
  data/wellness/classifications/{name}.json
  data/wellness/classifications/archive/{name}_*.json
  data/knowledge/{name}/...
  data/knowledge/system/...
  data/knowledge/vocabulary.json
  data/knowledge/index.json
  data/behaviors/gizmo_self.json

NEW:
  data/headmates/{name}/personality.json
  data/headmates/{name}/description.json
  data/headmates/{name}/wellness.json
  data/headmates/{name}/wellness_summaries/...
  data/headmates/{name}/wellness_classification.json
  data/headmates/{name}/wellness_classification_archive/...
  data/headmates/{name}/knowledge/...
  data/headmates/system/external/...
  data/headmates/system/vocabulary.json
  data/headmates/system/index.json
  data/headmates/gizmo/self.json

Run:
  python migrate_to_headmate_folders.py
  python migrate_to_headmate_folders.py --dry-run   # preview without moving anything
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate Gizmo data to headmate folder structure")
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving files")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"), help="Path to data directory")
    return parser.parse_args()


def ensure(path: Path, dry_run: bool) -> None:
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def move(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    print(f"  {'[DRY]' if dry_run else ''} {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def move_tree(src: Path, dst: Path, dry_run: bool) -> None:
    """Recursively move a directory tree."""
    if not src.exists() or not src.is_dir():
        return
    print(f"  {'[DRY]' if dry_run else ''} {src}/ -> {dst}/")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def collect_names(data: Path) -> set[str]:
    """Collect all headmate names from existing data files."""
    names = set()

    for folder in ["behaviors", "descriptors", "wellness"]:
        d = data / folder
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix == ".json" and f.stem not in ("classifications", "gizmo_self"):
                names.add(f.stem.lower())

    knowledge = data / "knowledge"
    if knowledge.exists():
        for item in knowledge.iterdir():
            if item.is_dir() and item.name not in ("system", "gizmo", "vocabulary.json", "index.json"):
                names.add(item.name.lower())

    # Remove system-level names
    names.discard("system")
    names.discard("gizmo")

    return names


def migrate(data_dir: str, dry_run: bool) -> None:
    data = Path(data_dir)
    hm   = data / "headmates"

    print(f"\nMigrating data in: {data}")
    print(f"Dry run: {dry_run}\n")

    names = collect_names(data)
    print(f"Found headmates: {sorted(names)}\n")

    # ── Per-headmate files ────────────────────────────────────────────────────
    for name in sorted(names):
        print(f"--- {name} ---")

        # behaviors/{name}.json -> headmates/{name}/personality.json
        move(
            data / "behaviors" / f"{name}.json",
            hm / name / "personality.json",
            dry_run,
        )

        # descriptors/{name}.json -> headmates/{name}/description.json
        move(
            data / "descriptors" / f"{name}.json",
            hm / name / "description.json",
            dry_run,
        )

        # wellness/{name}.json -> headmates/{name}/wellness.json
        move(
            data / "wellness" / f"{name}.json",
            hm / name / "wellness.json",
            dry_run,
        )

        # wellness/classifications/{name}.json -> headmates/{name}/wellness_classification.json
        move(
            data / "wellness" / "classifications" / f"{name}.json",
            hm / name / "wellness_classification.json",
            dry_run,
        )

        # wellness/classifications/archive/{name}_*.json -> headmates/{name}/wellness_classification_archive/
        archive_src = data / "wellness" / "classifications" / "archive"
        if archive_src.exists():
            for f in archive_src.glob(f"{name}_*.json"):
                move(f, hm / name / "wellness_classification_archive" / f.name, dry_run)

        # wellness/summaries/{name}/ -> headmates/{name}/wellness_summaries/
        move_tree(
            data / "wellness" / "summaries" / name,
            hm / name / "wellness_summaries",
            dry_run,
        )

        # knowledge/{name}/ -> headmates/{name}/knowledge/
        move_tree(
            data / "knowledge" / name,
            hm / name / "knowledge",
            dry_run,
        )

        print()

    # ── Gizmo self ────────────────────────────────────────────────────────────
    print("--- gizmo ---")
    move(
        data / "behaviors" / "gizmo_self.json",
        hm / "gizmo" / "self.json",
        dry_run,
    )
    move(
        data / "behaviors" / "gizmo.json",
        hm / "gizmo" / "personality.json",
        dry_run,
    )
    print()

    # ── System / shared ───────────────────────────────────────────────────────
    print("--- system ---")

    # knowledge/system/ -> headmates/system/external/
    move_tree(
        data / "knowledge" / "system" / "external",
        hm / "system" / "external",
        dry_run,
    )

    # knowledge/vocabulary.json -> headmates/system/vocabulary.json
    move(
        data / "knowledge" / "vocabulary.json",
        hm / "system" / "vocabulary.json",
        dry_run,
    )

    # knowledge/index.json -> headmates/system/index.json
    move(
        data / "knowledge" / "index.json",
        hm / "system" / "index.json",
        dry_run,
    )

    print()

    # ── Create new empty files for new concepts ───────────────────────────────
    if not dry_run:
        print("Creating new relationship/dynamic/history stubs...")
        for name in sorted(names):
            # relationships.json
            rel_path = hm / name / "relationships.json"
            if not rel_path.exists():
                rel_path.write_text(json.dumps({
                    "with_gizmo": {
                        "trust_level":  "unknown",
                        "dynamic":      "",
                        "what_works":   [],
                        "what_doesnt":  [],
                        "last_updated": "",
                    },
                    "with_system": {}
                }, indent=2))

            # dynamic.json
            dyn_path = hm / name / "dynamic.json"
            if not dyn_path.exists():
                dyn_path.write_text(json.dumps({
                    "current_register": "neutral",
                    "scene_active":     False,
                    "notes":            "",
                    "last_updated":     "",
                }, indent=2))

            # history.json
            hist_path = hm / name / "history.json"
            if not hist_path.exists():
                hist_path.write_text(json.dumps([], indent=2))

        # system/relationships/ folder
        (hm / "system" / "relationships").mkdir(parents=True, exist_ok=True)

    print("\nMigration complete." if not dry_run else "\nDry run complete — no files moved.")
    print("\nNext steps:")
    print("  1. Verify headmates/ folder looks correct")
    print("  2. Deploy updated librarian.py with new path functions")
    print("  3. Old folders (behaviors/, descriptors/, wellness/, knowledge/) can be")
    print("     archived or deleted once everything is confirmed working")


if __name__ == "__main__":
    args = parse_args()
    migrate(args.data_dir, args.dry_run)
