"""Utility functions for simple music theory operations used across the pipeline."""

from __future__ import annotations

import logging
from typing import List, Tuple

# Map of note names to semitone offsets relative to C.
_NOTE_TO_SEMITONE = {
    "c": 0,
    "c#": 1,
    "db": 1,
    "d": 2,
    "d#": 3,
    "eb": 3,
    "e": 4,
    "f": 5,
    "f#": 6,
    "gb": 6,
    "g": 7,
    "g#": 8,
    "ab": 8,
    "a": 9,
    "a#": 10,
    "bb": 10,
    "b": 11,
}

_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_NATURAL_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


def parse_key(key: str) -> Tuple[int, str]:
    """Parse key strings such as "C", "Cmaj", "Am" or "F# minor".

    Returns a tuple of (tonic_midi_number, mode). The tonic is represented as a MIDI
    note number in the 4th octave (C4 = 60). Mode is either "major" or "minor".
    Defaults to C major when parsing fails.
    """

    if not key:
        logging.warning("Empty key provided, defaulting to C major")
        return 60, "major"

    text = key.strip().lower().replace("major", "maj").replace("minor", "min")
    mode = "major"
    if text.endswith("m") and not text.endswith("maj"):
        mode = "minor"
        text = text[:-1]
    elif text.endswith("min"):
        mode = "minor"
        text = text[:-3]
    elif text.endswith("maj"):
        mode = "major"
        text = text[:-3]

    text = text.strip()
    if text.endswith("-flat") or text.endswith("-sharp"):
        text = text.replace("-flat", "b").replace("-sharp", "#")

    semitone = _NOTE_TO_SEMITONE.get(text)
    if semitone is None and len(text) > 1:
        semitone = _NOTE_TO_SEMITONE.get(text[0])

    if semitone is None:
        logging.warning("Failed to parse key '%s', falling back to C", key)
        semitone = 0

    tonic = 60 + semitone  # place tonic in the 4th octave
    return tonic, mode


def build_scale(tonic: int, mode: str) -> List[int]:
    """Return a list with MIDI numbers forming a single octave scale."""
    if mode == "minor":
        intervals = _NATURAL_MINOR_INTERVALS
    else:
        intervals = _MAJOR_INTERVALS
    return [(tonic + interval) for interval in intervals]


def cycle_scale(scale: List[int], length: int) -> List[int]:
    """Repeat scale notes to reach the requested length."""
    if not scale:
        return [60] * max(1, length)
    return [scale[i % len(scale)] + 12 * (i // len(scale)) for i in range(length)]


def clamp_midi_range(note: int, low: int, high: int) -> int:
    """Clamp a MIDI note into the provided range."""
    while note < low:
        note += 12
    while note > high:
        note -= 12
    return note
