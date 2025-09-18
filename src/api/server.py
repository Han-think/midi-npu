from fastapi import FastAPI
from pydantic import BaseModel, Field
import os, io, base64, time, glob, soundfile as sf, pretty_midi as pm
from src.inference.ov_sampler import ov_generate, tokens_to_midi
from src.render.sf2_renderer import render as render_sf2

app = FastAPI(title='midi-npu (one-pipeline)', version='0.3.1')

class Section(BaseModel):
    name: str
    duration: float = Field(..., gt=0)

class ComposeReq(BaseModel):
    base_style: str = 'rock'
    bpm: int = 120
    key: str = 'C'
    sections: list[Section]
    seed: int | None = 42
    with_vocal: bool = False
    max_tokens: int = 512

class MGReq(BaseModel):
    prompt: str
    duration: int = 8

def _find_xml():
    xml = os.environ.get('OV_XML_PATH') or 'exports/gpt_ov/openvino_model.xml'
    if os.path.exists(xml): return xml
    cands = glob.glob('exports/gpt_ov/*.xml') + glob.glob('exports/**/*.xml', recursive=True)
    return cands[0] if cands else xml

@app.get('/health')
def health():
    try:
        import openvino as ov
        return {'status':'ok','devices': ov.Core().available_devices}
    except Exception as e:
        return {'status':'degraded','error': str(e)}

@app.post('/v1/midi/compose_full')
def compose(req: ComposeReq):
    vocab='data/processed/vocab.json'
    xml=_find_xml()
    if not os.path.exists(vocab):
        return {'error':'missing vocab. run prepare step'}
    if not os.path.exists(xml):
        return {'error': f'missing OV model. looked for {xml} and exports/**/*.xml. run export step'}

    t0=time.time()
    toks, vc = ov_generate(xml, vocab, max_tokens=req.max_tokens)
    midi = tokens_to_midi(toks, vc)

    total=sum(s.duration for s in req.sections); cur=0.0
    base = pm.PrettyMIDI()
    dur=max(1e-3, midi.get_end_time())
    offsets=[]
    for s in req.sections:
        scale = s.duration/dur
        for inst in midi.instruments:
            ni = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            for n in inst.notes:
                ni.notes.append(pm.Note(velocity=n.velocity, pitch=n.pitch,
                                        start=n.start*scale+cur, end=n.end*scale+cur))
            base.instruments.append(ni)
        offsets.append({'name':s.name,'start':cur,'end':cur+s.duration})
        cur += s.duration

    audio = render_sf2(base, sr=32000)
    buf = io.BytesIO(); sf.write(buf, audio, 32000, format='WAV')
    return {'format':'wav','sample_rate':32000,'b64':base64.b64encode(buf.getvalue()).decode(),
            'offsets':offsets,'elapsed_ms':int((time.time()-t0)*1000)}

@app.post('/v1/audio/musicgen')
def musicgen(req: MGReq):
    root='models/musicgen_static_ov'
    if not os.path.exists(root):
        return {'error':'models/musicgen_static_ov not found. run setup (optional demo IR)'}
    return {'message':'MusicGen demo IR installed','duration':req.duration}
