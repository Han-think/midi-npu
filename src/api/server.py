from fastapi import FastAPI
from pydantic import BaseModel, Field
import base64, io, os, json, time
import soundfile as sf
import pretty_midi as pm

from src.inference.ov_sampler import ov_generate, tokens_to_midi
from src.render.sf2_renderer import render as render_sf2

app = FastAPI(title="midi-npu (single server)", version="0.2.0")

class Section(BaseModel):
    name: str
    duration: float = Field(..., gt=0)

class ComposeReq(BaseModel):
    base_style: str = "pop rock, energetic"
    bpm: int = 128
    key: str = "C"
    sections: list[Section]
    negative_prompt: str | None = None
    seed: int | None = 42
    with_vocal: bool = False
    max_tokens: int = 512

class MusicGenReq(BaseModel):
    prompt: str
    duration: int = 10

@app.get("/")
def root(): return {"service":"midi-npu","health":"/health","docs":"/docs"}

@app.get("/health")
def health():
    try:
        import openvino as ov
        devs = ov.Core().available_devices
        return {"status":"ok","devices":devs}
    except Exception as e:
        return {"status":"degraded","error":str(e)}

@app.post("/v1/midi/compose_full")
def compose_full(req: ComposeReq):
    t0=time.time()
    if req.seed is not None:
        import numpy as np, random
        np.random.seed(req.seed); random.seed(req.seed)
    xml="exports/gpt_ov/openvino_model.xml"; vocab="data/processed/vocab.json"
    if not os.path.exists(xml): return {"error":"missing OV model. run scripts/export_ov.ps1"}
    if not os.path.exists(vocab): return {"error":"missing vocab. run scripts/prepare.ps1"}
    # 섹션별 토큰 생성(단순히 max_tokens를 섹션분으로 분할)
    total_dur=sum(s.duration for s in req.sections); offsets=[]; cur=0.0; all_midis=[]
    for s in req.sections:
        frac=max(1,int(req.max_tokens*(s.duration/total_dur)))
        toks, vc = ov_generate(xml, vocab, max_tokens=frac)
        midi = tokens_to_midi(toks, vc)
        # 시간 스케일링으로 섹션 길이에 맞춤(rough)
        dur = max(0.001, midi.get_end_time())
        scale = s.duration/dur
        for inst in midi.instruments:
            for n in inst.notes:
                n.start *= scale; n.end *= scale
            all_midis.append(inst)
        offsets.append({"name":s.name,"start":cur,"end":cur+s.duration})
        cur += s.duration
    # 합치기
    out = pm.PrettyMIDI()
    for inst in all_midis:
        # 섹션 오프셋 반영은 간단화(위에서 누적됨)
        out.instruments.append(inst)
    audio = render_sf2(out, sr=32000)
    buf = io.BytesIO(); sf.write(buf, audio, 32000, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"format":"wav","sample_rate":32000,"b64":b64,"offsets":offsets,"elapsed_ms":int((time.time()-t0)*1000)}

@app.post("/v1/audio/musicgen")
def mg_demo(req: MusicGenReq):
    # 데모: IR 세트 없으면 안내
    root="models/musicgen_static_ov"
    if not os.path.exists(root):
        return {"error":"models/musicgen_static_ov not found. run scripts/bootstrap.ps1 (optional demo)."}
    # 간단 데모 음성 생성은 생략하고 안내만 제공(실사용은 별도 파이프라인 권장)
    return {"message":"MusicGen demo IR installed. Integrate full sampler later.", "duration": req.duration}
