from __future__ import annotations

from typing import Dict, Iterable, List

import pretty_midi as pm


def section_prefix(name: str) -> List[str]:
    return ["<BOS>", f"<SEC_{name.upper()}>"]


def midi_to_events(m: pm.PrettyMIDI) -> List[str]:
    ev: List[str] = []
    for inst in m.instruments:
        for n in sorted(inst.notes, key=lambda x: x.start):
            ev.append(f"NOTE_{n.pitch}")
    return ev


def build_vocab(event_streams: Iterable[List[str]]) -> Dict[str, int]:
    toks = {"<BOS>": 0}
    for evs in event_streams:
        for e in evs:
            if e not in toks:
                toks[e] = len(toks)
    return toks


def events_to_ids(evs: List[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(e, 0) for e in evs]
