"""SoundFont based MIDI renderer."""
from __future__ import annotations

import copy
import logging
import os
import time
from typing import Dict

import numpy as np
import pretty_midi

LOGGER = logging.getLogger(__name__)

_DEFAULT_PATCHES: Dict[str, int] = {
    "drums": 0,  # channel 10 drums handled via is_drum flag
    "bass": 33,  # Fingered Bass
    "chords": 0,  # Acoustic Grand Piano
    "lead": 80,  # Lead 1 (square)
    "lead_vocal": 52,  # Choir Aahs
}


def _apply_presets(midi: pretty_midi.PrettyMIDI) -> None:
    for instrument in midi.instruments:
        name = (instrument.name or "").lower()
        if name in _DEFAULT_PATCHES:
            instrument.program = _DEFAULT_PATCHES[name]
            instrument.is_drum = name == "drums"


def render(midi: pretty_midi.PrettyMIDI, sr: int = 32000) -> np.ndarray:
    """Render a MIDI object to audio using the configured SoundFont."""

    sf2_path = os.getenv("SF2_PATH")
    if not sf2_path:
        raise RuntimeError(
            "SF2_PATH environment variable is not set. Please specify a SoundFont (.sf2) file."
        )
    if not os.path.exists(sf2_path):
        raise FileNotFoundError(f"SoundFont file not found at '{sf2_path}'")

    midi_copy = copy.deepcopy(midi)
    _apply_presets(midi_copy)

    start = time.perf_counter()
    audio = midi_copy.fluidsynth(fs=sr, sf2_path=sf2_path)
    duration = time.perf_counter() - start
    LOGGER.info("Rendered MIDI to audio in %.2f ms", duration * 1000.0)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    # pretty_midi returns shape (n, ) or (n, channels). ensure float32
    return audio.astype(np.float32)

