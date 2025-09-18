import glob
import json
import os
from typing import Iterable

import pretty_midi as pm

from src.tokenizers.skytnt import (
    build_vocab,
    events_to_ids,
    midi_to_events,
    section_prefix,
)

RAW = "data/raw"
OUT = "data/processed"
JSONL = f"{OUT}/jsonl/train.jsonl"
VOCAB = f"{OUT}/vocab.json"


def slice_midi(midi: pm.PrettyMIDI, start: float, end: float) -> pm.PrettyMIDI:
    sliced = pm.PrettyMIDI(resolution=midi.resolution)
    for instrument in midi.instruments:
        new_inst = pm.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
        )
        for note in instrument.notes:
            if note.start >= end or note.end <= start:
                continue
            note_start = max(note.start, start) - start
            note_end = max(min(note.end, end) - start, note_start + 1e-4)
            new_inst.notes.append(
                pm.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note_start,
                    end=note_end,
                )
            )
        if new_inst.notes:
            sliced.instruments.append(new_inst)
    return sliced


def _load_sections(path: str) -> Iterable[dict[str, float]] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    os.makedirs(os.path.dirname(JSONL), exist_ok=True)
    samples: list[list[str]] = []
    metas: list[dict[str, str]] = []

    for song_dir in glob.glob(f"{RAW}/*"):
        if not os.path.isdir(song_dir):
            continue

        midi_paths = glob.glob(f"{song_dir}/*.mid")
        if not midi_paths:
            continue

        sections = _load_sections(f"{song_dir}/sections.json")
        bpm = 120
        key = "C"
        for midi_path in midi_paths:
            midi = pm.PrettyMIDI(midi_path)
            if sections:
                for section in sections:
                    sliced = slice_midi(
                        midi, float(section["start"]), float(section["end"])
                    )
                    events = section_prefix(section["name"], bpm, key) + midi_to_events(
                        sliced
                    )
                    samples.append(events)
                    metas.append(
                        {"song": os.path.basename(song_dir), "section": section["name"]}
                    )
            else:
                events = section_prefix("full", bpm, key) + midi_to_events(midi)
                samples.append(events)
                metas.append({"song": os.path.basename(song_dir), "section": "full"})

    vocab = build_vocab(samples)
    os.makedirs(OUT, exist_ok=True)
    with open(VOCAB, "w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(JSONL), exist_ok=True)
    with open(JSONL, "w", encoding="utf-8") as handle:
        for events, meta in zip(samples, metas):
            payload = {"tokens": events_to_ids(events, vocab), "meta": meta}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print("wrote", JSONL, "vocab", len(vocab), "samples", len(samples))


if __name__ == "__main__":
    main()
