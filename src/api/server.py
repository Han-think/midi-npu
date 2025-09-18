from __future__ import annotations

import base64
import glob
import io
import os
import time
from typing import List

import pretty_midi as pm
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.ov_sampler import ov_generate, tokens_to_midi
from src.render.sf2_renderer import render as render_sf2

app = FastAPI(title="midi-npu", version="0.4.0")


class Section(BaseModel):
    name: str
    duration: float = Field(..., gt=0)


class ComposeReq(BaseModel):
    base_style: str = "rock"
    bpm: int = 120
    key: str = "C"
    sections: List[Section]
    seed: int | None = 42
    with_vocal: bool = False
    max_tokens: int = 512


class MGReq(BaseModel):
    prompt: str
    duration: int = 8


def _find_xml() -> str:
    xml = os.environ.get("OV_XML_PATH") or "exports/gpt_ov/openvino_model.xml"
    if os.path.exists(xml):
        return xml
    cands = glob.glob("exports/gpt_ov/*.xml") + glob.glob(
        "exports/**/*.xml", recursive=True
    )
    return cands[0] if cands else xml


@app.get("/health")
def health():
    try:
        import openvino as ov

        return {"status": "ok", "devices": ov.Core().available_devices}
    except Exception as e:  # pragma: no cover
        return {"status": "degraded", "error": str(e)}


def _fake_midi(req: ComposeReq) -> pm.PrettyMIDI:
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0, name="fake")
    pattern = [60, 64, 67, 72, 67, 64, 60]
    t = 0.0
    for s in req.sections:
        step = max(0.2, s.duration / max(1, len(pattern)))
        for p in pattern:
            inst.notes.append(
                pm.Note(velocity=80, pitch=p, start=t, end=t + step * 0.9)
            )
            t += step
    m.instruments.append(inst)
    return m


@app.post("/v1/midi/compose_full")
def compose(req: ComposeReq):
    vocab = "data/processed/vocab.json"
    xml = _find_xml()
    use_fake = (not os.path.exists(xml)) and (os.environ.get("ALLOW_FAKE_GEN") == "1")

    if not use_fake and not os.path.exists(vocab):
        return {"error": "missing vocab. run prepare step"}
    if not use_fake and not os.path.exists(xml):
        return {
            "error": f"missing OV model. looked for {xml} and exports/**/*.xml. run export step"
        }

    t0 = time.time()
    if use_fake:
        midi = _fake_midi(req)
    else:
        toks, vc = ov_generate(xml, vocab, max_tokens=req.max_tokens)
        midi = tokens_to_midi(toks, vc)

    total = sum(s.duration for s in req.sections)
    cur = 0.0
    base = pm.PrettyMIDI()
    dur = max(1e-3, midi.get_end_time())
    offsets = []
    for s in req.sections:
        scale = s.duration / dur
        for inst in midi.instruments:
            ni = pm.Instrument(
                program=inst.program, is_drum=inst.is_drum, name=inst.name
            )
            for n in inst.notes:
                ni.notes.append(
                    pm.Note(
                        velocity=n.velocity,
                        pitch=n.pitch,
                        start=n.start * scale + cur,
                        end=n.end * scale + cur,
                    )
                )
            base.instruments.append(ni)
        offsets.append({"name": s.name, "start": cur, "end": cur + s.duration})
        cur += s.duration
    _ = total  # keep lints calm

    audio = render_sf2(base, sr=32000)
    buf = io.BytesIO()
    sf.write(buf, audio, 32000, format="WAV")
    return {
        "format": "wav",
        "sample_rate": 32000,
        "b64": base64.b64encode(buf.getvalue()).decode(),
        "offsets": offsets,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


@app.post("/v1/audio/musicgen")
def musicgen(req: MGReq):
    root = "models/musicgen_static_ov"
    if not os.path.exists(root):
        return {
            "error": "models/musicgen_static_ov not found. run setup (optional demo IR)"
        }
    return {"message": "MusicGen demo IR installed", "duration": req.duration}
