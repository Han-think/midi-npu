"""FastAPI server exposing composition and demo endpoints."""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import pretty_midi as pm
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.ov_sampler import ov_generate, tokens_to_midi
from src.render import sf2_renderer

APP_VERSION = "0.3.0"
MODEL_XML = Path("exports/gpt_ov/openvino_model.xml")
VOCAB_JSON = Path("data/processed/vocab.json")

app = FastAPI(title="midi-npu (one-pipeline)", version=APP_VERSION)


class Section(BaseModel):
    name: str
    duration: float = Field(..., gt=0.0)


class ComposeRequest(BaseModel):
    base_style: str = "rock"
    bpm: int = 120
    key: str = "C"
    sections: list[Section]
    seed: int | None = 42
    with_vocal: bool = False
    max_tokens: int = 512


class MusicGenRequest(BaseModel):
    prompt: str
    duration: int = 8


@app.get("/health")
def health() -> dict:
    try:
        import openvino as ov

        core = ov.Core()
        return {"status": "ok", "devices": core.available_devices}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"status": "degraded", "error": str(exc)}


def _ensure_paths() -> tuple[Path, Path] | dict[str, str]:
    if not MODEL_XML.exists():
        return {"error": "run scripts/make.ps1 export"}
    if not VOCAB_JSON.exists():
        return {"error": "run scripts/make.ps1 prepare"}
    return MODEL_XML, VOCAB_JSON


def _expand_sections(template: pm.PrettyMIDI, sections: list[Section]) -> tuple[pm.PrettyMIDI, list[dict[str, float]]]:
    if not sections:
        raise ValueError("At least one section is required")

    total_duration = max(template.get_end_time(), 1e-3)
    composed = pm.PrettyMIDI(resolution=template.resolution)
    composed.time_signature_changes = [
        pm.TimeSignature(ts.numerator, ts.denominator, ts.time)
        for ts in template.time_signature_changes
    ]
    composed.key_signature_changes = [
        pm.KeySignature(ks.key_number, ks.time)
        for ks in template.key_signature_changes
    ]
    offsets: list[dict[str, float]] = []
    cursor = 0.0

    for section in sections:
        scale = section.duration / total_duration
        for instrument in template.instruments:
            new_inst = pm.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
            )
            for note in instrument.notes:
                new_inst.notes.append(
                    pm.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start * scale + cursor,
                        end=note.end * scale + cursor,
                    )
                )
            if new_inst.notes:
                composed.instruments.append(new_inst)
        offsets.append({"name": section.name, "start": cursor, "end": cursor + section.duration})
        cursor += section.duration

    return composed, offsets


@app.post("/v1/midi/compose_full")
def compose(request: ComposeRequest) -> dict:
    ensured = _ensure_paths()
    if isinstance(ensured, dict):
        return ensured

    model_path, vocab_path = ensured
    start_time = time.time()

    tokens, vocab = ov_generate(
        model_path,
        vocab_path,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )

    midi = tokens_to_midi(tokens, vocab)
    if not midi.instruments:
        return {"error": "model produced no notes"}

    try:
        expanded, offsets = _expand_sections(midi, request.sections)
    except ValueError as exc:
        return {"error": str(exc)}

    audio = sf2_renderer.render(expanded, sr=32000)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 32000, format="WAV")

    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "format": "wav",
        "sample_rate": 32000,
        "b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "offsets": offsets,
        "elapsed_ms": elapsed_ms,
        "seed": request.seed,
    }


@app.post("/v1/audio/musicgen")
def musicgen(request: MusicGenRequest) -> dict:
    root = Path("models/musicgen_static_ov")
    if not root.exists():
        return {"error": "run scripts/make.ps1 setup (optional demo IR)"}
    return {"message": "MusicGen demo IR installed", "duration": request.duration}
