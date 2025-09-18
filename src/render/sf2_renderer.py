import os
import numpy as np
import pretty_midi as pm


def _sf2_path():
    p = os.environ.get('SF2_PATH')
    if not p or not os.path.exists(p):
        raise RuntimeError('SF2_PATH not set or missing. Put assets/FluidR3_GM.sf2')
    return p


def render(midi: pm.PrettyMIDI, sr=32000) -> np.ndarray:
    # CI에서는 실제 렌더 생략(무음) -> fluidsynth 비의존
    if os.environ.get('SKIP_AUDIO') == '1':
        return np.zeros(sr * 2, dtype='float32')

    audio = midi.fluidsynth(fs=sr, sf2_path=_sf2_path())
    peak = max(1e-9, float(np.abs(audio).max()))
    audio = audio / (peak * 1.2)
    audio = np.tanh(audio * 1.8)  # soft limiter
    return audio.astype('float32')
