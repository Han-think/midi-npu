"""Lightweight REMI-style tokenizer utilities."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

import pretty_midi as pm


TIME_SIG_TOKEN = "TIME_SIG_4_4"


def section_prefix(name: str, bpm: int, key: str) -> list[str]:
    """Return the section context tokens that prefix every sample."""
    return [f"<SECTION={name}>", f"<BPM={bpm}>", f"<KEY={key}>", TIME_SIG_TOKEN]


def midi_to_events(midi: pm.PrettyMIDI) -> list[str]:
    """Convert a PrettyMIDI object into a flat list of REMI-like events."""
    events: list[str] = []
    tempos = midi.get_tempo_changes()[1]
    tempo = int(tempos[0]) if len(tempos) > 0 else 120
    events.append(f"TEMPO_{tempo}")

    duration = midi.get_end_time()
    bar_duration = 60.0 / tempo * 4
    current_bar = 0
    current_time = 0.0
    while current_time < duration:
        events.append(f"BAR_{current_bar}")
        current_time += bar_duration
        current_bar += 1

    for instrument in midi.instruments:
        program = 128 if instrument.is_drum else instrument.program
        channel = 9 if instrument.is_drum else 0
        events.extend([f"INST_{program}", f"CH_{channel}"])
        for note in instrument.notes:
            duration_ticks = max(int((note.end - note.start) * 960), 1)
            events.extend(
                [
                    f"NOTE_{note.pitch}",
                    f"DUR_{duration_ticks}",
                    f"VEL_{int(note.velocity)}",
                ]
            )
        events.append("INST_END")

    return events


def build_vocab(samples: Iterable[Sequence[str]]) -> dict[str, int]:
    """Build a simple vocabulary from the tokenized samples."""
    counter: Counter[str] = Counter()
    for sample in samples:
        counter.update(sample)

    tokens: List[str] = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokens.extend(token for token, _ in counter.items())
    return {token: idx for idx, token in enumerate(tokens)}


def events_to_ids(events: Sequence[str], vocab: dict[str, int]) -> list[int]:
    """Map events to vocabulary indices using an <unk> fallback."""
    unk = vocab.get("<unk>", 0)
    return [vocab.get(event, unk) for event in events]
