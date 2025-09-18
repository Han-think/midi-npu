from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pretty_midi as pm

from src.tokenizers.skytnt import (
    build_vocab,
    events_to_ids,
    midi_to_events,
    section_prefix,
)


def _scan_raw() -> List[Tuple[Path, Dict]]:
    root = Path("data/raw")
    items: List[Tuple[Path, Dict]] = []
    if not root.exists():
        return items
    for song in root.iterdir():
        if not song.is_dir():
            continue
        mid = song / "track.mid"
        if not mid.exists():
            continue
        sec = song / "sections.json"
        sections = []
        if sec.exists():
            sections = json.load(open(sec, "r", encoding="utf-8"))
        items.append((mid, {"sections": sections}))
    return items


def main() -> None:
    os.makedirs("data/processed/jsonl", exist_ok=True)
    streams: List[List[str]] = []
    records: List[Dict] = []

    for mid_path, meta in _scan_raw():
        m = pm.PrettyMIDI(str(mid_path))
        evs = section_prefix(Path(mid_path).parent.name) + midi_to_events(m)
        streams.append(evs)
        records.append({"tokens": evs, "meta": {"song": mid_path.parent.name, **meta}})

    if not streams:
        # fallback: tiny scale if no data/raw present
        m = pm.PrettyMIDI()
        inst = pm.Instrument(program=0)
        t = 0.0
        for p in [60, 62, 64, 65, 67, 69, 71, 72]:
            inst.notes.append(pm.Note(velocity=90, pitch=p, start=t, end=t + 0.25))
            t += 0.25
        m.instruments.append(inst)
        evs = section_prefix("demo") + midi_to_events(m)
        streams = [evs]
        records = [{"tokens": evs, "meta": {"song": "demo"}}]

    vocab = build_vocab(streams)
    with open("data/processed/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open("data/processed/jsonl/train.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            ids = events_to_ids(r["tokens"], vocab)
            json.dump({"tokens": ids, "meta": r["meta"]}, f, ensure_ascii=False)
            f.write("\n")
    print("prepared: data/processed/vocab.json & data/processed/jsonl/train.jsonl")


if __name__ == "__main__":
    main()
