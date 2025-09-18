import os, numpy as np, pretty_midi as pm
def _require_sf2():
    sf2=os.environ.get("SF2_PATH")
    if not sf2 or not os.path.exists(sf2):
        raise RuntimeError("SF2_PATH not set or file missing. Place a GM SoundFont, e.g., assets/FluidR3_GM.sf2")
    return sf2

def render(midi: pm.PrettyMIDI, sr=32000) -> np.ndarray:
    import fluidsynth, tempfile, soundfile as sf
    sf2=_require_sf2()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        midi.write(tmp.name.replace(".wav",".mid"))
    synth=fluidsynth.Synth(samplerate=sr)
    synth.start(driver="dsound")  # Windows
    sid=synth.sfload(sf2); synth.program_select(0,sid,0,0)
    # pretty_midi의 fluidsynth 편의 함수를 쓰면 트랙별 프로그램 반영이 간단하지만,
    # 여기서는 간단히 전체 렌더만 수행(개선 TODO)
    audio = midi.fluidsynth(fs=sr, sf2_path=sf2)
    synth.delete()
    # 마스터 노멀라이즈 + 소프트리미터
    peak=max(1e-9, np.abs(audio).max()); audio=audio/(peak*1.2)
    audio=np.tanh(audio*1.8)
    return audio.astype("float32")
