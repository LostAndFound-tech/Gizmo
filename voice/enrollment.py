"""
voice/enrollment.py
Voiceprint enrollment and profile management.

Each headmate gets a speaker profile: a collection of embedding vectors
computed from their voice samples. Identification is done by comparing
a new embedding against all profile centroids via cosine similarity.

Auto-learning: when a switch is confirmed (via switch_host or verbal ID),
the utterance embedding is added to that headmate's profile cluster.

Profile storage: JSON + numpy arrays in VOICE_PROFILES_DIR
  profiles/
    alice.npz     ← stacked embedding matrix (N_samples × embedding_dim)
    alice.json    ← metadata (enrolled_at, sample_count, etc.)
    bob.npz
    bob.json
    ...

Requirements:
    pip install numpy scipy pyannote.audio torch

Usage:
    from voice.enrollment import ProfileStore, enroll_from_file, enroll_from_bytes

    store = ProfileStore()
    store.enroll("alice", audio_bytes)          # add samples
    name, confidence = store.identify(embedding) # identify speaker
    store.update("alice", embedding)             # auto-learn from confirmed ID
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

VOICE_PROFILES_DIR = os.getenv("VOICE_PROFILES_DIR", "./data/voice_profiles")
SAMPLE_RATE = 16000

# Identification thresholds
CONFIDENT_THRESHOLD = 0.82    # cosine similarity above this → confident ID
POSSIBLE_THRESHOLD = 0.65     # above this but below confident → ask to confirm
# below POSSIBLE_THRESHOLD → unknown speaker

# Auto-learning limits
MAX_SAMPLES_PER_PROFILE = 200  # cap to avoid unbounded growth
MIN_SAMPLES_FOR_ID = 3         # need at least this many samples before IDing


def _get_model():
    """Load pyannote speaker embedding model. Cached after first load."""
    if not hasattr(_get_model, "_model"):
        try:
            from pyannote.audio import Model, Inference
            import torch

            print("[Enrollment] Loading speaker embedding model...")
            model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=os.getenv("HF_TOKEN"),
            )
            _get_model._inference = Inference(model, window="whole")
            _get_model._model = model
            print("[Enrollment] Speaker embedding model ready.")
        except ImportError:
            raise ImportError(
                "pyannote.audio is required: pip install pyannote.audio torch\n"
                "You also need a HuggingFace token with pyannote model access."
            )
    return _get_model._inference


def _audio_bytes_to_waveform(audio_bytes: bytes) -> "np.ndarray":
    """Convert raw PCM bytes (16kHz mono 16-bit) to float32 numpy array."""
    import numpy as np
    arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


def compute_embedding(audio_bytes: bytes) -> Optional[np.ndarray]:
    """
    Compute a speaker embedding vector from raw PCM audio bytes.
    Returns a 1D numpy array (512-dim for pyannote/embedding).
    Returns None if audio is too short or model fails.
    """
    try:
        import numpy as np
        from pyannote.audio import Inference
        import torch

        waveform = _audio_bytes_to_waveform(audio_bytes)

        # Need at least 1 second of audio for a reliable embedding
        if len(waveform) < SAMPLE_RATE:
            print("[Enrollment] Audio too short for embedding (< 1s)")
            return None

        inference = _get_model()

        # pyannote expects (channels, samples) tensor
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        embedding = inference({"waveform": tensor, "sample_rate": SAMPLE_RATE})

        return np.array(embedding).flatten()

    except Exception as e:
        print(f"[Enrollment] Embedding failed: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerProfile:
    """
    A single headmate's voiceprint profile.
    Stores a matrix of embedding vectors and computes a centroid for ID.
    """

    def __init__(self, name: str, embeddings: Optional[np.ndarray] = None, metadata: Optional[dict] = None):
        self.name = name
        # Shape: (N_samples, embedding_dim)
        self.embeddings = embeddings if embeddings is not None else np.zeros((0, 512))
        self.metadata = metadata or {
            "enrolled_at": datetime.now().isoformat(),
            "sample_count": 0,
            "last_updated": datetime.now().isoformat(),
        }

    @property
    def sample_count(self) -> int:
        return self.embeddings.shape[0]

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Mean embedding vector. None if no samples."""
        if self.sample_count == 0:
            return None
        return self.embeddings.mean(axis=0)

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add a new embedding sample, respecting the cap."""
        if self.sample_count >= MAX_SAMPLES_PER_PROFILE:
            # Drop oldest sample (FIFO)
            self.embeddings = np.vstack([self.embeddings[1:], embedding.reshape(1, -1)])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)]) \
                if self.sample_count > 0 else embedding.reshape(1, -1)
        self.metadata["sample_count"] = self.sample_count
        self.metadata["last_updated"] = datetime.now().isoformat()

    def similarity_to(self, embedding: np.ndarray) -> float:
        """Cosine similarity between embedding and this profile's centroid."""
        c = self.centroid
        if c is None:
            return 0.0
        return cosine_similarity(c, embedding)

    def similarity_to_best(self, embedding: np.ndarray) -> float:
        """
        Max similarity across all stored embeddings (not just centroid).
        More robust for small profile sizes.
        """
        if self.sample_count == 0:
            return 0.0
        sims = [cosine_similarity(e, embedding) for e in self.embeddings]
        return max(sims)

    def score(self, embedding: np.ndarray) -> float:
        """
        Combined score: centroid similarity when we have enough samples,
        best-match similarity when profile is small.
        """
        if self.sample_count < 5:
            return self.similarity_to_best(embedding)
        # Weighted blend: 70% centroid, 30% best-match
        return 0.7 * self.similarity_to(embedding) + 0.3 * self.similarity_to_best(embedding)


class ProfileStore:
    """
    Manages all speaker profiles. Handles save/load and identification.
    """

    def __init__(self, profiles_dir: str = VOICE_PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: dict[str, SpeakerProfile] = {}
        self._load_all()

    def _profile_path(self, name: str) -> tuple[Path, Path]:
        safe = name.lower().strip().replace(" ", "_")
        return (
            self.profiles_dir / f"{safe}.npz",
            self.profiles_dir / f"{safe}.json",
        )

    def _load_all(self) -> None:
        """Load all .npz profile files from disk."""
        for npz_path in self.profiles_dir.glob("*.npz"):
            name = npz_path.stem
            json_path = npz_path.with_suffix(".json")
            try:
                data = np.load(str(npz_path))
                embeddings = data["embeddings"]
                metadata = {}
                if json_path.exists():
                    with open(json_path) as f:
                        metadata = json.load(f)
                self.profiles[name] = SpeakerProfile(name, embeddings, metadata)
                print(f"[Enrollment] Loaded profile '{name}' ({embeddings.shape[0]} samples)")
            except Exception as e:
                print(f"[Enrollment] Failed to load profile '{name}': {e}")

    def save(self, name: str) -> None:
        """Persist a profile to disk."""
        name = name.lower().strip()
        if name not in self.profiles:
            return
        npz_path, json_path = self._profile_path(name)
        profile = self.profiles[name]
        try:
            np.savez(str(npz_path), embeddings=profile.embeddings)
            with open(json_path, "w") as f:
                json.dump(profile.metadata, f, indent=2)
        except Exception as e:
            print(f"[Enrollment] Failed to save profile '{name}': {e}")

    def enroll(self, name: str, audio_bytes: bytes) -> bool:
        """
        Add audio sample to a headmate's profile.
        Creates the profile if it doesn't exist.
        Returns True if embedding was successfully added.
        """
        name = name.lower().strip()
        embedding = compute_embedding(audio_bytes)
        if embedding is None:
            return False

        if name not in self.profiles:
            self.profiles[name] = SpeakerProfile(name)
            print(f"[Enrollment] Created new profile: '{name}'")

        self.profiles[name].add_embedding(embedding)
        self.save(name)
        count = self.profiles[name].sample_count
        print(f"[Enrollment] '{name}' — {count} samples")
        return True

    def enroll_embedding(self, name: str, embedding: np.ndarray) -> None:
        """
        Add a pre-computed embedding to a profile.
        Used by auto-learning path (no need to recompute).
        """
        name = name.lower().strip()
        if name not in self.profiles:
            self.profiles[name] = SpeakerProfile(name)
        self.profiles[name].add_embedding(embedding)
        self.save(name)

    def identify(
        self,
        embedding: np.ndarray,
        exclude: Optional[list[str]] = None,
    ) -> tuple[Optional[str], float, str]:
        """
        Identify speaker from embedding vector.

        Returns:
            (name, confidence, status)
            status: "confident" | "possible" | "unknown"
        """
        if not self.profiles:
            return None, 0.0, "unknown"

        exclude = {e.lower() for e in (exclude or [])}

        best_name = None
        best_score = -1.0

        for name, profile in self.profiles.items():
            if name in exclude:
                continue
            if profile.sample_count < MIN_SAMPLES_FOR_ID:
                continue
            score = profile.score(embedding)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None, 0.0, "unknown"

        if best_score >= CONFIDENT_THRESHOLD:
            return best_name, best_score, "confident"
        elif best_score >= POSSIBLE_THRESHOLD:
            return best_name, best_score, "possible"
        else:
            return None, best_score, "unknown"

    def list_profiles(self) -> list[dict]:
        """Return summary of all profiles."""
        return [
            {
                "name": name,
                "samples": p.sample_count,
                "enrolled_at": p.metadata.get("enrolled_at", "unknown"),
                "last_updated": p.metadata.get("last_updated", "unknown"),
                "ready": p.sample_count >= MIN_SAMPLES_FOR_ID,
            }
            for name, p in self.profiles.items()
        ]

    def delete_profile(self, name: str) -> bool:
        """Remove a profile entirely."""
        name = name.lower().strip()
        npz_path, json_path = self._profile_path(name)
        removed = False
        if name in self.profiles:
            del self.profiles[name]
            removed = True
        for path in (npz_path, json_path):
            if path.exists():
                path.unlink()
                removed = True
        return removed


# ── Interactive enrollment CLI ────────────────────────────────────────────────

async def run_enrollment_session(store: ProfileStore, name: str, n_samples: int = 10) -> None:
    """
    Interactive enrollment: prompt the person to speak N times,
    compute embeddings, add to their profile.

    n_samples: how many audio clips to collect (each ~5 seconds)
    """
    import asyncio

    try:
        import pyaudio
    except ImportError:
        raise ImportError("pyaudio required for enrollment: pip install pyaudio")

    SAMPLE_DURATION = 5  # seconds per sample

    print(f"\n[Enrollment] Starting enrollment for: {name}")
    print(f"  Will collect {n_samples} voice samples (~{SAMPLE_DURATION}s each)")
    print(f"  Speak naturally — read something aloud, describe your day, anything.\n")

    pa = pyaudio.PyAudio()
    loop = asyncio.get_event_loop()
    success_count = 0

    for i in range(n_samples):
        input(f"  Sample {i+1}/{n_samples} — press Enter when ready to speak...")
        print(f"  Recording for {SAMPLE_DURATION} seconds... speak now!")

        frame_count = int(SAMPLE_RATE * SAMPLE_DURATION)
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024,
        )

        frames = []
        for _ in range(0, int(SAMPLE_RATE / 1024 * SAMPLE_DURATION)):
            data = await loop.run_in_executor(None, lambda: stream.read(1024, exception_on_overflow=False))
            frames.append(data)

        stream.stop_stream()
        stream.close()

        audio_bytes = b"".join(frames)
        print(f"  Processing sample {i+1}...")

        ok = store.enroll(name, audio_bytes)
        if ok:
            success_count += 1
            print(f"  ✓ Sample {i+1} added ({store.profiles[name.lower()].sample_count} total)")
        else:
            print(f"  ✗ Sample {i+1} failed — audio may be too short or quiet")

        await asyncio.sleep(0.5)

    pa.terminate()
    print(f"\n[Enrollment] Done! {success_count}/{n_samples} samples added for '{name}'")
    profile = store.profiles.get(name.lower())
    if profile:
        ready = profile.sample_count >= MIN_SAMPLES_FOR_ID
        print(f"  Profile status: {'✓ Ready for identification' if ready else f'⚠ Need {MIN_SAMPLES_FOR_ID - profile.sample_count} more samples'}")


# Singleton
profile_store = ProfileStore()
