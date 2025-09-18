import os, json, numpy as np, pretty_midi as pm, openvino as ov
from typing import Dict, List, Tuple

# 악기/드럼 매핑 (GM)
GM_PROG = {"gtr_muted":28,"gtr_over":29,"gtr_dist":30,"bass_finger":33,"bass_pick":34,"koto":107,"shamisen":106}
GM_DRUM = {"KICK":36,"SNARE":38,"RIM":37,"CLAP":39,"CHH":42,"PHH":44,"OHH":46,"LTOM":45,"MTOM":47,"HTOM":50,"CRASH":49,"RIDE":51}
DRUM_CH=9

RANGE = {"lead":(60,84),"gtr":(40,88),"bass":(28,52),"koto":(55,79),"shamisen":(50,76)}
def clamp(p, lohi): lo,hi=lohi; return max(lo,min(hi,p))

def tokens_to_midi(tokens: List[int], vocab: Dict[str,int]) -> pm.PrettyMIDI:
    inv={v:k for k,v in vocab.items()}
    m=pm.PrettyMIDI(); tracks={"lead": pm.Instrument(program=GM_PROG["gtr_dist"], name="lead"),
                               "bass": pm.Instrument(program=GM_PROG["bass_finger"], name="bass"),
                               "koto": pm.Instrument(program=GM_PROG["koto"], name="koto"),
                               "drum": pm.Instrument(program=0, is_drum=True, name="drums")}
    i=0; t=0.0; cur=tracks["lead"]
    while i<len(tokens):
        tok=inv.get(tokens[i],"")
        if tok.startswith("INST_"):
            pid=int(tok.split("_")[1]); # rough routing
            if pid in (28,29,30): cur=tracks["lead"]
            elif pid in (33,34):  cur=tracks["bass"]
            elif pid==107:        cur=tracks["koto"]
            i+=1; continue
        if tok.startswith("NOTE_") and i+2<len(tokens):
            p=int(tok.split("_")[1]); dur=inv.get(tokens[i+1],""); vel=inv.get(tokens[i+2],"")
            if dur.startswith("DUR_") and vel.startswith("VEL_"):
                d=int(dur.split("_")[1])/960.0; v=int(vel.split("_")[1])
                p=clamp(p, RANGE["lead"])
                cur.notes.append(pm.Note(velocity=v, pitch=p, start=t, end=t+d)); t+=d; i+=3; continue
        i+=1
    for name,inst in tracks.items():
        if inst.notes: m.instruments.append(inst)
    return m

def ov_generate(xml_path:str, vocab_path:str, max_tokens:int=512, temperature:float=0.9, top_p:float=0.92):
    core=ov.Core(); dev=os.environ.get("OV_DEVICE","AUTO")
    model=core.compile_model(xml_path, device_name=dev)
    vocab=json.load(open(vocab_path,"r")); bos=vocab.get("<bos>",1); eos=vocab.get("<eos>",2)
    seq=[bos]; req=model.create_infer_request()
    for _ in range(max_tokens):
        out=req.infer({0: np.array([seq],dtype=np.int64)})
        logits=list(out.values())[0][0,-1]
        # top-p 샘플링 (간단)
        probs=np.exp(logits - logits.max()); probs/=probs.sum()
        idxs=np.argsort(probs)[::-1]; cumsum=np.cumsum(probs[idxs]); cut=idxs[cumsum<=top_p]
        pool=cut if len(cut)>0 else idxs[:50]
        nxt=int(np.random.choice(pool, p=probs[pool]/probs[pool].sum()))
        if nxt==eos: break
        seq.append(nxt)
    return seq, vocab
