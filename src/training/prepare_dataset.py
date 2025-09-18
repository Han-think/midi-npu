"""Dataset preparation utilities for section-aware MIDI training."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import pretty_midi as pm

from src.tokenizers.skytnt import (
    build_vocab,
    events_to_ids,
    midi_to_events,
    section_prefix,
)

RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
JSONL_PATH = PROCESSED_ROOT / "jsonl" / "train.jsonl"
VOCAB_PATH = PROCESSED_ROOT / "vocab.json"


@dataclass
class Section:
    name: str
    start: float
    end: float


def _slice_midi(midi: pm.PrettyMIDI, start: float, end: float) -> pm.PrettyMIDI:
    """Return a shallow copy of *midi* containing notes overlapping [start, end]."""
    out = pm.PrettyMIDI(resolution=midi.resolution)
    for instrument in midi.instruments:
        new_inst = pm.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
        )
        for note in instrument.notes:
            if note.start >= end or note.end <= start:
                continue
            new_start = max(note.start, start) - start
            new_end = max(min(note.end, end) - start, new_start + 1e-4)
            new_inst.notes.append(
                pm.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=new_start,
                    end=new_end,
                )
            )
        if new_inst.notes:
            out.instruments.append(new_inst)
    return out


def _iter_sectioned_midis(path: Path) -> Iterator[tuple[list[str], dict[str, str]]]:
    sections_path = path / "sections.json"
    sections: list[Section] | None = None
    if sections_path.exists():
        with sections_path.open("r", encoding="utf-8") as fh:
            raw_sections = json.load(fh)
        sections = [
            Section(name=item["name"], start=float(item["start"]), end=float(item["end"]))
            for item in raw_sections
        ]

    midi_paths = sorted(glob.glob(str(path / "*.mid")))
    bpm = 120
    key = "C"
    for midi_path in midi_paths:
        midi = pm.PrettyMIDI(midi_path)
        if sections:
            for section in sections:
                sliced = _slice_midi(midi, section.start, section.end)
                tokens = section_prefix(section.name, bpm, key) + midi_to_events(sliced)
                meta = {"song": path.name, "section": section.name}
                yield tokens, meta
        else:
            tokens = section_prefix("full", bpm, key) + midi_to_events(midi)
            meta = {"song": path.name, "section": "full"}
            yield tokens, meta


def prepare_dataset() -> None:
    """Create the processed dataset and vocabulary."""
    os.makedirs(JSONL_PATH.parent, exist_ok=True)
    os.makedirs(PROCESSED_ROOT, exist_ok=True)

    token_stream: list[Sequence[str]] = []
    metadata: list[dict[str, str]] = []

    for song_dir in sorted(RAW_ROOT.iterdir()):
        if not song_dir.is_dir():
            continue
        song_samples = list(_iter_sectioned_midis(song_dir))
        if not song_samples:
            continue
        for tokens, meta in song_samples:
            token_stream.append(tokens)
            metadata.append(meta)

    if not token_stream:
        raise RuntimeError(
            "No MIDI files found. Populate data/raw/<song> with .mid files before running prepare."
        )

    vocab = build_vocab(token_stream)
    with VOCAB_PATH.open("w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)

    with JSONL_PATH.open("w", encoding="utf-8") as fh:
        for tokens, meta in zip(token_stream, metadata):
            token_ids = [vocab["<bos>"]] + events_to_ids(tokens, vocab) + [vocab["<eos>"]]
            fh.write(json.dumps({"tokens": token_ids, "meta": meta}, ensure_ascii=False) + "\n")

    print(
        "wrote",
        JSONL_PATH.as_posix(),
        "vocab",
        len(vocab),
        "samples",
        len(token_stream),
    )


if __name__ == "__main__":
    prepare_dataset()
