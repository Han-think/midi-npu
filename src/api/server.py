from fastapi import FastAPI
from pydantic import BaseModel, Field
import os, io, base64, time, soundfile as sf, pretty_midi as pm
from src.inference.ov_sampler import ov_generate, tokens_to_midi

app=FastAPI(title='midi-npu (one-pipeline)',version='0.3.0')

class Section(BaseModel):
    name:str; duration:float=Field(...,gt=0)
class ComposeReq(BaseModel):
    base_style:str='rock'; bpm:int=120; key:str='C'
    sections:list[Section]; seed:int|None=42; with_vocal:bool=False; max_tokens:int=512
class MGReq(BaseModel):
    prompt:str; duration:int=8

@app.get('/health')
def health():
    try:
        import openvino as ov; return {'status':'ok','devices':ov.Core().available_devices}
    except Exception as e:
        return {'status':'degraded','error':str(e)}

@app.post('/v1/midi/compose_full')
def compose(req:ComposeReq):
    if not os.path.exists('exports/gpt_ov/openvino_model.xml'): return {'error':'run scripts/make.ps1 export'}
    if not os.path.exists('data/processed/vocab.json'): return {'error':'run scripts/make.ps1 prepare'}
    t0=time.time()
    toks,vocab=ov_generate('exports/gpt_ov/openvino_model.xml','data/processed/vocab.json',max_tokens=req.max_tokens)
    midi=tokens_to_midi(toks,vocab)
    # 섹션 길이에 맞춰 간단히 타임스케일/오프셋
    total=sum(s.duration for s in req.sections); cur=0.0; out=pm.PrettyMIDI(); offsets=[]
    dur=max(1e-3,midi.get_end_time())
    for s in req.sections:
        scale=s.duration/dur
        for inst in midi.instruments:
            ni=pm.Instrument(program=inst.program,is_drum=inst.is_drum,name=inst.name)
            for n in inst.notes:
                ni.notes.append(pm.Note(velocity=n.velocity,pitch=n.pitch,start=n.start*scale+cur,end=n.end*scale+cur))
            out.instruments.append(ni)
        offsets.append({'name':s.name,'start':cur,'end':cur+s.duration}); cur+=s.duration
    import src.render.sf2_renderer as R
    audio=R.render(out, sr=32000); buf=io.BytesIO(); sf.write(buf,audio,32000,format='WAV')
    return {'format':'wav','sample_rate':32000,'b64':base64.b64encode(buf.getvalue()).decode(),'offsets':offsets,'elapsed_ms':int((time.time()-t0)*1000)}

@app.post('/v1/audio/musicgen')
def musicgen(req:MGReq):
    root='models/musicgen_static_ov'
    if not os.path.exists(root): return {'error':'run scripts/make.ps1 setup (optional demo IR)'}
    return {'message':'MusicGen demo IR installed','duration':req.duration}
