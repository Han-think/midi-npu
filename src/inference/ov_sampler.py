import os, json, numpy as np, pretty_midi as pm, openvino as ov
from src.render.instrument_map import GM_PROGRAM


def tokens_to_midi(tokens, vocab):
    inv={v:k for k,v in vocab.items()}
    m=pm.PrettyMIDI()
    lead=pm.Instrument(program=GM_PROGRAM['gtr_dist'], name='lead')
    i=0; t=0.0
    while i<len(tokens):
        tok=inv.get(tokens[i],'')
        if tok.startswith('NOTE_') and i+2<len(tokens):
            p=int(tok.split('_')[1]); dur=inv.get(tokens[i+1],''); vel=inv.get(tokens[i+2],'')
            if dur.startswith('DUR_') and vel.startswith('VEL_'):
                d=int(dur.split('_')[1])/960.0; v=int(vel.split('_')[1])
                lead.notes.append(pm.Note(velocity=v,pitch=p,start=t,end=t+d))
                t+=d; i+=3; continue
        i+=1
    if lead.notes: m.instruments.append(lead)
    return m


def ov_generate(xml_path, vocab_path, max_tokens=512, top_p=0.92):
    core=ov.Core()
    dev=os.environ.get('OV_DEVICE','AUTO')
    model=core.compile_model(xml_path, device_name=dev)
    vocab=json.load(open(vocab_path,'r',encoding='utf-8'))
    bos=vocab.get('<bos>',1); eos=vocab.get('<eos>',2)
    seq=[bos]; req=model.create_infer_request()
    for _ in range(max_tokens):
        out=req.infer({0: np.array([seq], dtype=np.int64)})
        logits=list(out.values())[0][0,-1]
        probs=np.exp(logits - logits.max()); probs /= probs.sum()
        idxs=np.argsort(probs)[::-1]; c=np.cumsum(probs[idxs]); k=idxs[c<=top_p]
        pool=k if len(k)>0 else idxs[:50]
        nxt=int(np.random.choice(pool, p=probs[pool]/probs[pool].sum()))
        if nxt==eos: break
        seq.append(nxt)
    return seq, vocab
