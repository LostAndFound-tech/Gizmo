"""
voice/auto_learn.py
Manages the auto-learning loop for speaker profiles.

Two learning paths:
  1. Confident ID — reinforce correct identifications passively
  2. Confirmed ID — verbal or switch_host confirmation adds to profile

Includes outlier protection: if a new embedding is too far from the
existing centroid, it's rejected rather than corrupting the model.

Usage:
    from voice.auto_learn import AutoLearner
    from voice.enrollment import profile_store

    learner = AutoLearner(profile_store)
    learner.learn_confirmed(result, "alice")   # confirmed by human
    learner.learn_passive(result, "alice")     # high-confidence auto-ID
"""

import time
from datetime import datetime
from typing import Optional

import numpy as np

# How far an embedding can be from the centroid before rejection
# (cosine distance = 1 - cosine_similarity)
OUTLIER_DISTANCE_THRESHOLD = 0.45

# Only passively reinforce if confidence is very high
PASSIVE_LEARN_THRESHOLD = 0.88

# Minimum time between passive learns for same speaker (avoid echo-chamber)
PASSIVE_LEARN_COOLDOWN = 30  # seconds

_last_passive_learn: dict[str, float] = {}


class AutoLearner:
    def __init__(self, profile_store):
        self.store = profile_store

    def learn_confirmed(self, result, name: str) -> bool:
        """
        Add a confirmed utterance embedding to the named profile.
        Confirmation comes from verbal ID or switch_host.
        Always runs (no cooldown) but still does outlier check.
        Returns True if embedding was accepted.
        """
        if result.embedding is None:
            return False

        name = name.lower().strip()
        if not name:
            return False

        embedding = result.embedding

        # Outlier check — don't corrupt an established profile with a weird sample
        if name in self.store.profiles:
            profile = self.store.profiles[name]
            if profile.sample_count >= 5:
                centroid = profile.centroid
                if centroid is not None:
                    dist = 1.0 - _cosine_sim(centroid, embedding)
                    if dist > OUTLIER_DISTANCE_THRESHOLD:
                        print(
                            f"[AutoLearn] Rejected outlier for '{name}' "
                            f"(cosine distance={dist:.3f}, threshold={OUTLIER_DISTANCE_THRESHOLD})"
                        )
                        return False

        self.store.enroll_embedding(name, embedding)
        count = self.store.profiles[name].sample_count
        print(f"[AutoLearn] Confirmed learn for '{name}' — {count} samples total")
        return True

    def learn_passive(self, result, name: str) -> bool:
        """
        Passively reinforce a confident automatic identification.
        Subject to cooldown and higher confidence threshold.
        Returns True if embedding was accepted.
        """
        if result.embedding is None:
            return False
        if result.speaker_confidence < PASSIVE_LEARN_THRESHOLD:
            return False

        name = name.lower().strip()
        now = time.time()

        # Cooldown check
        last = _last_passive_learn.get(name, 0)
        if now - last < PASSIVE_LEARN_COOLDOWN:
            return False

        # Outlier check
        if name in self.store.profiles:
            profile = self.store.profiles[name]
            if profile.sample_count >= 5:
                centroid = profile.centroid
                if centroid is not None:
                    dist = 1.0 - _cosine_sim(centroid, result.embedding)
                    if dist > OUTLIER_DISTANCE_THRESHOLD:
                        print(f"[AutoLearn] Passive outlier rejected for '{name}' (dist={dist:.3f})")
                        return False

        _last_passive_learn[name] = now
        self.store.enroll_embedding(name, result.embedding)
        count = self.store.profiles[name].sample_count
        print(f"[AutoLearn] Passive learn for '{name}' — {count} samples")
        return True

    def learn_from_switch(self, audio_bytes: bytes, confirmed_name: str) -> bool:
        """
        Called when switch_host fires with a confirmed headmate name.
        Extracts embedding from the audio and adds to their profile.

        This is the primary hook for integration with the existing
        switch_host tool — call this whenever a confirmed switch happens.
        """
        from voice.enrollment import compute_embedding

        embedding = compute_embedding(audio_bytes)
        if embedding is None:
            print(f"[AutoLearn] switch_host learn failed — couldn't compute embedding for '{confirmed_name}'")
            return False

        # Build a fake result-like object for learn_confirmed
        class _FakeResult:
            pass

        fake = _FakeResult()
        fake.embedding = embedding
        fake.speaker_confidence = 1.0  # confirmed, so max confidence

        return self.learn_confirmed(fake, confirmed_name)

    def get_profile_health(self) -> list[dict]:
        """
        Return per-profile health stats. Useful for diagnostics.
        """
        from voice.enrollment import MIN_SAMPLES_FOR_ID, CONFIDENT_THRESHOLD

        health = []
        for name, profile in self.store.profiles.items():
            centroid = profile.centroid
            spread = None

            if centroid is not None and profile.sample_count >= 2:
                # Average pairwise cosine distance — measures how consistent the voice is
                sims = [
                    _cosine_sim(centroid, e)
                    for e in profile.embeddings
                ]
                spread = float(np.std(sims))

            health.append({
                "name": name,
                "samples": profile.sample_count,
                "ready": profile.sample_count >= MIN_SAMPLES_FOR_ID,
                "embedding_spread": round(spread, 4) if spread is not None else None,
                "last_updated": profile.metadata.get("last_updated", "unknown"),
            })

        return sorted(health, key=lambda x: x["samples"], reverse=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
