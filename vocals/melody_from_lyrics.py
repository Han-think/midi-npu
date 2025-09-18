"""Rule-based lead vocal melody generator."""
from __future__ import annotations

import logging
from typing import List

import pretty_midi

from music_theory import build_scale, clamp_midi_range, cycle_scale, parse_key

LOGGER = logging.getLogger(__name__)

# TODO: Replace the heuristic melody builder with a dedicated melody language model or
# tonic network for improved phrasing and musicality.


def _split_syllables(line: str) -> List[str]:
    tokens = []
    for word in line.split():
        word = word.replace("-", " ")
        tokens.extend(part for part in word.split() if part)
    return tokens


def melody_from_lyrics(
    lines: List[str],
    key: str,
    bpm: int,
    duration_seconds: float,
) -> pretty_midi.PrettyMIDI:
    """Generate a basic lead vocal melody synchronised with the provided lyrics."""

    tonic, mode = parse_key(key)
    scale = build_scale(tonic, mode)
    if not scale:
        scale = [60, 62, 64, 65, 67, 69, 71]

    seconds_per_beat = 60.0 / max(bpm, 1)
    total_beats = duration_seconds / seconds_per_beat

    syllables_per_line = [_split_syllables(line) for line in lines]
    total_syllables = sum(len(syllables) for syllables in syllables_per_line)
    if total_syllables == 0:
        total_syllables = 4
        syllables_per_line = [["la"] * 4]

    rest_weight = 0.5 if len(syllables_per_line) > 1 else 0.0
    denominator = total_syllables + rest_weight * (len(syllables_per_line) - 1)
    if denominator <= 0:
        denominator = total_syllables or 1
    base_length_beats = total_beats / denominator

    melody = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    vocal = pretty_midi.Instrument(program=52, name="lead_vocal")

    melody_pool = cycle_scale(scale, total_syllables + 8)
    current_index = 0
    current_beat = 0.0

    for line_idx, syllables in enumerate(syllables_per_line):
        if not syllables:
            current_beat += base_length_beats * rest_weight
            continue
        for syllable in syllables:
            note_length_beats = base_length_beats
            start_time = current_beat * seconds_per_beat
            end_time = (current_beat + note_length_beats * 0.9) * seconds_per_beat
            pitch = clamp_midi_range(melody_pool[current_index] + 12, 60, 84)
            vocal.notes.append(
                pretty_midi.Note(
                    start=start_time,
                    end=end_time,
                    pitch=pitch,
                    velocity=90,
                )
            )
            current_index = (current_index + 1) % len(melody_pool)
            current_beat += note_length_beats
        if line_idx < len(syllables_per_line) - 1:
            current_beat += base_length_beats * rest_weight

    melody.instruments.append(vocal)
    melody.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    melody.key_signature_changes.append(pretty_midi.KeySignature((tonic % 12) + (12 if mode == "minor" else 0), 0.0))
    LOGGER.info(
        "Generated vocal melody with %d notes spanning %.2f seconds",
        len(vocal.notes),
        duration_seconds,
    )
    return melody

