"""Render PrettyMIDI objects using a SoundFont (SF2) synthesizer."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pretty_midi as pm


def _sf2() -> Path:
    path = os.environ.get("SF2_PATH")
    if not path or not os.path.exists(path):
        raise RuntimeError("SF2_PATH not set or file missing. Put assets/FluidR3_GM.sf2")
    return Path(path)


def render(midi: pm.PrettyMIDI, sr: int = 32000) -> np.ndarray:
    """Render MIDI to audio, optionally skipping synthesis for CI."""
    if os.environ.get("SKIP_AUDIO") == "1":
        return np.zeros(sr * 2, dtype="float32")

    audio = midi.fluidsynth(fs=sr, sf2_path=str(_sf2()))
    peak = max(1e-9, np.abs(audio).max())
    audio = audio / (peak * 1.2)
    audio = np.tanh(audio * 1.8)
    return audio.astype("float32")
