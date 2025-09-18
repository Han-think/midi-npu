"""Simple mastering utilities."""

from __future__ import annotations

import numpy as np


def normalize_and_limit(
    audio: np.ndarray, target_db: float = -3.0, drive: float = 1.5
) -> np.ndarray:
    """Apply -3 dB peak normalisation followed by a soft limiter."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak > 0:
        target_amp = 10 ** (target_db / 20.0)
        audio = audio * (target_amp / peak)

    # Soft limiter using tanh to gently compress peaks without introducing harsh artefacts.
    limited = np.tanh(audio * drive)
    limited /= np.max(np.abs(limited)) + 1e-6
    return limited.astype(np.float32)
