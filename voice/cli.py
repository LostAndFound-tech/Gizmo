"""
voice/cli.py
Command-line tools for managing the voice ID system.

Usage:
    python -m voice.cli enroll --name alice
    python -m voice.cli enroll --name alice --samples 15
    python -m voice.cli list
    python -m voice.cli test --file path/to/audio.wav
    python -m voice.cli health
    python -m voice.cli delete --name alice

The enroll command runs an interactive recording session.
The test command identifies the speaker in an audio file.
"""

import asyncio
import argparse
import os
import sys


def cmd_enroll(args):
    """Run interactive enrollment for a headmate."""
    from voice.enrollment import ProfileStore, run_enrollment_session

    store = ProfileStore()
    name = args.name.lower().strip()
    samples = getattr(args, "samples", 10)

    print(f"\nEnrolling: {name}")
    existing = store.profiles.get(name)
    if existing:
        print(f"  Existing profile: {existing.sample_count} samples already enrolled")
        print(f"  Adding {samples} more samples...\n")
    else:
        print(f"  New profile — collecting {samples} samples\n")

    asyncio.run(run_enrollment_session(store, name, n_samples=samples))


def cmd_list(args):
    """List all enrolled profiles."""
    from voice.enrollment import ProfileStore, MIN_SAMPLES_FOR_ID

    store = ProfileStore()
    profiles = store.list_profiles()

    if not profiles:
        print("\nNo profiles enrolled yet.")
        print("Run: python -m voice.cli enroll --name <headmate_name>")
        return

    print(f"\n{'Name':<20} {'Samples':>8} {'Status':<12} {'Last Updated'}")
    print("-" * 65)
    for p in sorted(profiles, key=lambda x: x["name"]):
        status = "✓ Ready" if p["ready"] else f"⚠ Need {MIN_SAMPLES_FOR_ID - p['samples']} more"
        updated = p["last_updated"][:16] if p["last_updated"] != "unknown" else "unknown"
        print(f"{p['name']:<20} {p['samples']:>8} {status:<12} {updated}")
    print()


def cmd_test(args):
    """Test speaker identification on an audio file."""
    from voice.enrollment import ProfileStore, compute_embedding
    import wave
    import struct

    store = ProfileStore()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    print(f"\nTesting speaker ID on: {args.file}")

    # Load audio file
    if args.file.endswith(".wav"):
        with wave.open(args.file, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

        if sr != 16000 or channels != 1 or sampwidth != 2:
            print(f"  ⚠ File is {sr}Hz/{channels}ch/{sampwidth*8}bit — expected 16kHz mono 16-bit")
            print(f"  Results may be inaccurate. Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 output.wav")

        audio_bytes = frames
    else:
        print("Only .wav files supported for testing. Convert with ffmpeg first.")
        sys.exit(1)

    embedding = compute_embedding(audio_bytes)
    if embedding is None:
        print("  ✗ Could not compute embedding — audio may be too short or corrupted")
        sys.exit(1)

    name, confidence, status = store.identify(embedding)

    print(f"\n  Result:")
    print(f"    Status:     {status}")
    print(f"    Speaker:    {name or 'unknown'}")
    print(f"    Confidence: {confidence:.3f}")

    # Show scores for all profiles
    if store.profiles:
        print(f"\n  All profile scores:")
        scores = []
        for pname, profile in store.profiles.items():
            score = profile.score(embedding)
            scores.append((pname, score))
        for pname, score in sorted(scores, key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"    {pname:<20} {score:.3f} {bar}")
    print()


def cmd_health(args):
    """Show profile health diagnostics."""
    from voice.enrollment import ProfileStore
    from voice.auto_learn import AutoLearner

    store = ProfileStore()
    learner = AutoLearner(store)
    health = learner.get_profile_health()

    if not health:
        print("\nNo profiles enrolled.")
        return

    print(f"\n{'Name':<20} {'Samples':>8} {'Spread':>8} {'Ready':<8} {'Last Updated'}")
    print("-" * 70)
    for h in health:
        spread = f"{h['embedding_spread']:.4f}" if h["embedding_spread"] is not None else "  n/a "
        ready = "✓" if h["ready"] else "✗"
        updated = h["last_updated"][:16] if h["last_updated"] != "unknown" else "unknown"
        print(f"{h['name']:<20} {h['samples']:>8} {spread:>8} {ready:<8} {updated}")

    print()
    print("  Spread: standard deviation of cosine similarities to centroid.")
    print("  Lower spread = more consistent voice samples = better ID accuracy.")
    print()


def cmd_delete(args):
    """Delete a profile."""
    from voice.enrollment import ProfileStore

    store = ProfileStore()
    name = args.name.lower().strip()

    if name not in store.profiles:
        print(f"\nProfile '{name}' not found.")
        cmd_list(args)
        return

    confirm = input(f"\nDelete profile '{name}' ({store.profiles[name].sample_count} samples)? [y/N] ")
    if confirm.lower() == "y":
        store.delete_profile(name)
        print(f"  ✓ Deleted '{name}'")
    else:
        print("  Cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="Gizmo voice ID management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m voice.cli enroll --name alice
  python -m voice.cli enroll --name alice --samples 15
  python -m voice.cli list
  python -m voice.cli test --file /tmp/test.wav
  python -m voice.cli health
  python -m voice.cli delete --name alice
        """
    )
    sub = parser.add_subparsers(dest="command")

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll a headmate's voice")
    p_enroll.add_argument("--name", required=True, help="Headmate name")
    p_enroll.add_argument("--samples", type=int, default=10, help="Number of samples to record (default: 10)")

    # list
    sub.add_parser("list", help="List all enrolled profiles")

    # test
    p_test = sub.add_parser("test", help="Test speaker ID on a WAV file")
    p_test.add_argument("--file", required=True, help="Path to .wav file (16kHz mono 16-bit)")

    # health
    sub.add_parser("health", help="Show profile health diagnostics")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a profile")
    p_delete.add_argument("--name", required=True, help="Headmate name")

    args = parser.parse_args()

    if args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "delete":
        cmd_delete(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
