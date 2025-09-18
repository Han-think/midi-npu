"""Utilities for sampling tokens from OpenVINO and rendering to MIDI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import openvino as ov
import pretty_midi as pm

from src.render.instrument_map import GM_PROGRAM

PROGRAM_LABELS: Dict[int, str] = {
    GM_PROGRAM["gtr_dist"]: "guitar_distortion",
    GM_PROGRAM["gtr_over"]: "guitar_overdrive",
    GM_PROGRAM["gtr_muted"]: "guitar_muted",
    GM_PROGRAM["bass_finger"]: "bass_finger",
    GM_PROGRAM["bass_pick"]: "bass_pick",
    GM_PROGRAM["bass_slap"]: "bass_slap",
    GM_PROGRAM["koto"]: "koto",
    GM_PROGRAM["shamisen"]: "shamisen",
    128: "drums",
}


def _instrument_for_program(program: int) -> pm.Instrument:
    is_drum = program == 128
    midi_program = 0 if is_drum else max(0, min(program, 127))
    name = PROGRAM_LABELS.get(program, f"program_{midi_program:03d}")
    return pm.Instrument(program=midi_program, is_drum=is_drum, name=name)


def tokens_to_midi(tokens: Iterable[int], vocab: Dict[str, int]) -> pm.PrettyMIDI:
    """Convert a sequence of token ids into a PrettyMIDI object."""
    inverse_vocab = {index: token for token, index in vocab.items()}
    midi = pm.PrettyMIDI()

    instruments: Dict[int, pm.Instrument] = {}
    positions: Dict[int, float] = {}
    current_program: int | None = None

    token_list = list(tokens)
    i = 0
    while i < len(token_list):
        token_str = inverse_vocab.get(token_list[i], "")

        if token_str.startswith("INST_"):
            program = int(token_str.split("_", maxsplit=1)[1])
            if program not in instruments:
                instruments[program] = _instrument_for_program(program)
                positions[program] = 0.0
            current_program = program
            i += 1
            continue

        if token_str.startswith("CH_") or token_str == "INST_END":
            current_program = None if token_str == "INST_END" else current_program
            i += 1
            continue

        if token_str.startswith("NOTE_") and current_program is not None and i + 2 < len(token_list):
            duration_token = inverse_vocab.get(token_list[i + 1], "")
            velocity_token = inverse_vocab.get(token_list[i + 2], "")
            if duration_token.startswith("DUR_") and velocity_token.startswith("VEL_"):
                pitch = int(token_str.split("_", maxsplit=1)[1])
                duration = int(duration_token.split("_", maxsplit=1)[1]) / 960.0
                velocity = int(velocity_token.split("_", maxsplit=1)[1])
                start = positions[current_program]
                end = start + duration
                instruments[current_program].notes.append(
                    pm.Note(pitch=pitch, velocity=velocity, start=start, end=end)
                )
                positions[current_program] = end
                i += 3
                continue

        i += 1

    for instrument in instruments.values():
        if instrument.notes:
            midi.instruments.append(instrument)

    return midi


def _top_p_sample(logits: np.ndarray, top_p: float, rng: np.random.Generator) -> int:
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    sorted_indices = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[sorted_indices])
    cutoff = np.searchsorted(cumulative, top_p, side="right") + 1
    cutoff = max(1, min(cutoff, len(sorted_indices)))
    candidate_indices = sorted_indices[:cutoff]
    candidate_probs = probs[candidate_indices]
    candidate_probs /= candidate_probs.sum()
    return int(rng.choice(candidate_indices, p=candidate_probs))


def ov_generate(
    model_path: Path | str,
    vocab_path: Path | str,
    *,
    max_tokens: int = 512,
    top_p: float = 0.92,
    seed: int | None = None,
) -> Tuple[list[int], Dict[str, int]]:
    """Sample tokens from an OpenVINO-compiled GPT model."""
    core = ov.Core()
    device = os.environ.get("OV_DEVICE", "AUTO")
    compiled_model = core.compile_model(str(model_path), device_name=device)

    with Path(vocab_path).open("r", encoding="utf-8") as fh:
        vocab = json.load(fh)

    bos = vocab.get("<bos>", 1)
    eos = vocab.get("<eos>", 2)

    request = compiled_model.create_infer_request()
    sequence = [bos]
    rng = np.random.default_rng(seed)

    input_port = compiled_model.inputs[0]

    for _ in range(max_tokens):
        outputs = request.infer({input_port: np.array([sequence], dtype=np.int64)})
        logits = next(iter(outputs.values()))[0, -1]
        next_token = _top_p_sample(logits, top_p, rng)
        if next_token == eos:
            break
        sequence.append(next_token)

    return sequence[1:], vocab
