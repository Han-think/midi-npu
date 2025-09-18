"""FastAPI server exposing the full song composition pipeline."""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lyrics.lyric_planner import plan_lyrics
from midi_backend.skytnt_runner import run_section
from mixer.master import normalize_and_limit
from render.sf2_renderer import render
from vocals.melody_from_lyrics import melody_from_lyrics

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

APP_SAMPLE_RATE = 32000


class SectionSpec(BaseModel):
    name: str
    duration: float = Field(..., gt=0, description="Section duration in seconds")


class ComposeRequest(BaseModel):
    base_style: str
    bpm: int = Field(..., gt=0)
    key: str
    sections: List[SectionSpec]
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    with_vocal: bool = True


app = FastAPI(title="MIDI NPU Full Song Composer", version="1.0.0")


def _ensure_length(audio: np.ndarray, duration: float, sample_rate: int) -> np.ndarray:
    target_samples = int(round(duration * sample_rate))
    if audio.ndim == 1:
        current = audio.shape[0]
        if current < target_samples:
            pad = target_samples - current
            audio = np.pad(audio, (0, pad))
        else:
            audio = audio[:target_samples]
        return audio

    current = audio.shape[0]
    if current < target_samples:
        pad = target_samples - current
        audio = np.pad(audio, ((0, pad), (0, 0)))
    else:
        audio = audio[:target_samples, :]
    return audio


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "MIDI NPU composition service. Use /docs for OpenAPI schema.",
        "compose_endpoint": "/v1/audio/compose_full",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/audio/compose_full")
def compose_full(request: ComposeRequest):
    if not request.sections:
        return JSONResponse(
            status_code=400, content={"error": "sections cannot be empty"}
        )

    try:
        lyrics_map = plan_lyrics(
            base_style=request.base_style,
            key=request.key,
            bpm=request.bpm,
            sections=[section.dict() for section in request.sections],
            negative=request.negative_prompt,
            seed=request.seed,
        )
    except Exception as exc:  # pragma: no cover - safety net
        LOGGER.exception("Lyric planning failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})

    offsets = []
    audio_sections: List[np.ndarray] = []
    current_start = 0.0

    for index, section in enumerate(request.sections):
        LOGGER.info("Processing section '%s'", section.name)
        section_start_time = time.perf_counter()
        section_seed = request.seed + index if request.seed is not None else None
        try:
            section_midi = run_section(
                style=request.base_style,
                key=request.key,
                bpm=request.bpm,
                tag=section.name,
                seed=section_seed,
                duration=section.duration,
            )
        except Exception as exc:  # pragma: no cover - failure path
            LOGGER.exception("MIDI generation failed for section '%s'", section.name)
            return JSONResponse(status_code=500, content={"error": str(exc)})

        if request.with_vocal:
            lines = lyrics_map.get(section.name, [])
            if lines:
                try:
                    vocal_midi = melody_from_lyrics(
                        lines=lines,
                        key=request.key,
                        bpm=request.bpm,
                        duration_seconds=section.duration,
                    )
                    section_midi.instruments.extend(vocal_midi.instruments)
                except Exception as exc:  # pragma: no cover - melody failure
                    LOGGER.exception("Vocal melody generation failed")
                    return JSONResponse(status_code=500, content={"error": str(exc)})

        try:
            section_audio = render(section_midi, sr=APP_SAMPLE_RATE)
        except Exception as exc:
            LOGGER.exception("Rendering failed for section '%s'", section.name)
            return JSONResponse(status_code=500, content={"error": str(exc)})

        section_audio = _ensure_length(section_audio, section.duration, APP_SAMPLE_RATE)
        audio_sections.append(section_audio)

        section_end = current_start + section.duration
        offsets.append(
            {"name": section.name, "start": current_start, "end": section_end}
        )
        current_start = section_end
        LOGGER.info(
            "Section '%s' processed in %.2f ms",
            section.name,
            (time.perf_counter() - section_start_time) * 1000.0,
        )

    if not audio_sections:
        return JSONResponse(status_code=500, content={"error": "no audio rendered"})

    master_audio = np.concatenate(audio_sections, axis=0)
    master_audio = normalize_and_limit(master_audio)

    with io.BytesIO() as buffer:
        sf.write(buffer, master_audio, APP_SAMPLE_RATE, format="WAV")
        wav_bytes = buffer.getvalue()
    payload = base64.b64encode(wav_bytes).decode("ascii")

    response = {
        "format": "wav",
        "sample_rate": APP_SAMPLE_RATE,
        "b64": payload,
        "offsets": offsets,
        "lyrics": lyrics_map,
    }
    return response


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9010)
