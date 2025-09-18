import json
import os
from collections.abc import Sequence

import numpy as np
import openvino as ov
import pretty_midi as pm

from src.render.instrument_map import GM_PROGRAM


def tokens_to_midi(tokens: Sequence[int], vocab: dict[str, int]) -> pm.PrettyMIDI:
    inverse_vocab = {token_id: token for token, token_id in vocab.items()}
    midi = pm.PrettyMIDI()
    lead = pm.Instrument(program=GM_PROGRAM["gtr_dist"], name="lead")
    time_cursor = 0.0
    index = 0

    while index < len(tokens):
        token = inverse_vocab.get(tokens[index], "")
        if token.startswith("NOTE_") and index + 2 < len(tokens):
            pitch = int(token.split("_")[1])
            duration_token = inverse_vocab.get(tokens[index + 1], "")
            velocity_token = inverse_vocab.get(tokens[index + 2], "")

            if duration_token.startswith("DUR_") and velocity_token.startswith("VEL_"):
                duration = int(duration_token.split("_")[1]) / 960.0
                velocity = int(velocity_token.split("_")[1])
                lead.notes.append(
                    pm.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=time_cursor,
                        end=time_cursor + duration,
                    )
                )
                time_cursor += duration
                index += 3
                continue

        index += 1

    if lead.notes:
        midi.instruments.append(lead)
    return midi


def ov_generate(
    xml_path: str,
    vocab_path: str,
    max_tokens: int = 512,
    top_p: float = 0.92,
) -> tuple[list[int], dict[str, int]]:
    core = ov.Core()
    device = os.environ.get("OV_DEVICE", "AUTO")
    model = core.compile_model(xml_path, device_name=device)

    with open(vocab_path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    bos = vocab.get("<bos>", 1)
    eos = vocab.get("<eos>", 2)
    sequence: list[int] = [bos]
    infer_request = model.create_infer_request()

    for _ in range(max_tokens):
        outputs = infer_request.infer({0: np.array([sequence], dtype=np.int64)})
        logits = list(outputs.values())[0][0, -1]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        sorted_indices = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[sorted_indices])
        nucleus = sorted_indices[cumulative <= top_p]
        pool = nucleus if len(nucleus) > 0 else sorted_indices[:50]
        next_token = int(np.random.choice(pool, p=probs[pool] / probs[pool].sum()))
        if next_token == eos:
            break
        sequence.append(next_token)

    return sequence, vocab
