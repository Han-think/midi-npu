import os,json,numpy as np,pretty_midi as pm,openvino as ov
from src.render.instrument_map import GM_PROGRAM

def tokens_to_midi(tokens,vocab):
    inv={v:k for k,v in vocab.items()}; m=pm.PrettyMIDI()
    tracks={'lead': pm.Instrument(program=GM_PROGRAM['gtr_dist'], name='lead'),
            'bass': pm.Instrument(program=GM_PROGRAM['bass_finger'], name='bass'),
            'koto': pm.Instrument(program=GM_PROGRAM['koto'], name='koto')}
    i=0; t=0.0; cur=tracks['lead']
    while i<len(tokens):
        tok=inv.get(tokens[i],'')
        if tok.startswith('INST_'):
            pid=int(tok.split('_')[1])
            cur = tracks['bass'] if pid in (GM_PROGRAM['bass_finger'],GM_PROGRAM['bass_pick']) else (tracks['koto'] if pid==GM_PROGRAM['koto'] else tracks['lead'])
            i+=1; continue
        if tok.startswith('NOTE_') and i+2<len(tokens):
            p=int(tok.split('_')[1]); dur=inv.get(tokens[i+1],''); vel=inv.get(tokens[i+2],'')
            if dur.startswith('DUR_') and vel.startswith('VEL_'):
                d=int(dur.split('_')[1])/960.0; v=int(vel.split('_')[1])
                cur.notes.append(pm.Note(velocity=v,pitch=p,start=t,end=t+d)); t+=d; i+=3; continue
        i+=1
    for tr in tracks.values():
        if tr.notes: m.instruments.append(tr)
    return m

def ov_generate(xml,vocab_path,max_tokens=512,top_p=0.92):
    core=ov.Core(); dev=os.environ.get('OV_DEVICE','AUTO'); model=core.compile_model(xml,device_name=dev)
    vocab=json.load(open(vocab_path,'r')); bos=vocab.get('<bos>',1); eos=vocab.get('<eos>',2)
    seq=[bos]; req=model.create_infer_request()
    for _ in range(max_tokens):
        out=req.infer({0:np.array([seq],dtype=np.int64)}); logits=list(out.values())[0][0,-1]
        probs=np.exp(logits-logits.max()); probs/=probs.sum(); idxs=np.argsort(probs)[::-1]; c=np.cumsum(probs[idxs]); k=idxs[c<=top_p]; pool=k if len(k)>0 else idxs[:50]
        nxt=int(np.random.choice(pool,p=probs[pool]/probs[pool].sum()))
        if nxt==eos: break
        seq.append(nxt)
    return seq, json.load(open(vocab_path,'r'))
