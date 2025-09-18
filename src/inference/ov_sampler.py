from __future__ import annotations

import json
import os
import random
from typing import List, Tuple

import pretty_midi as pm


def ov_generate(
    xml_path: str, vocab_path: str, max_tokens: int = 128
) -> Tuple[List[int], dict]:
    """Greedy-like toy generator (XML 존재 여부만 확인)."""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    # make a simple token stream using NOTE_* if available
    notes = [int(k.split("_")[1]) for k in vocab.keys() if k.startswith("NOTE_")]
    if not notes:
        # fallback to any ids
        ids = list(vocab.values())
        toks = [random.choice(ids) for _ in range(max_tokens)]
        return toks, vocab
    toks: List[int] = []
    base = min(notes)
    for i in range(min(max_tokens, 64)):
        p = base + (i % 8)
        tid = vocab.get(f"NOTE_{p}")
        if tid is None:
            tid = random.choice(list(vocab.values()))
        toks.append(tid)
    return toks, vocab


def tokens_to_midi(tokens: List[int], vocab: dict) -> pm.PrettyMIDI:
    inv = {v: k for k, v in vocab.items()}
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    t = 0.0
    for tid in tokens:
        name = inv.get(tid, "")
        if name.startswith("NOTE_"):
            pitch = int(name.split("_")[1])
            inst.notes.append(pm.Note(velocity=90, pitch=pitch, start=t, end=t + 0.25))
            t += 0.25
    m.instruments.append(inst)
    return m
