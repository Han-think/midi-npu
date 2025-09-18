import os, json, glob, pretty_midi as pm
from src.tokenizers.skytnt import section_prefix, midi_to_events, build_vocab, events_to_ids
RAW='data/raw'; OUT='data/processed'; JSONL=f'{OUT}/jsonl/train.jsonl'; VOCAB=f'{OUT}/vocab.json'

def slice_midi(m,s,e):
    out=pm.PrettyMIDI(resolution=m.resolution)
    for inst in m.instruments:
        ni=pm.Instrument(program=inst.program,is_drum=inst.is_drum,name=inst.name)
        for n in inst.notes:
            if n.start>=e or n.end<=s: continue
            ns=max(n.start,s)-s; ne=max(min(n.end,e)-s, ns+1e-4)
            ni.notes.append(pm.Note(velocity=n.velocity,pitch=n.pitch,start=ns,end=ne))
        if ni.notes: out.instruments.append(ni)
    return out

def main():
    os.makedirs(os.path.dirname(JSONL),exist_ok=True)
    samples=[]; metas=[]
    for song in glob.glob(f'{RAW}/*'):
        if not os.path.isdir(song): continue
        mids=glob.glob(f'{song}/*.mid'); 
        if not mids: continue
        sections_path=f'{song}/sections.json'
        sections=json.load(open(sections_path,'r',encoding='utf-8')) if os.path.exists(sections_path) else None
        bpm=120; key='C'
        for mp in mids:
            m=pm.PrettyMIDI(mp)
            if sections:
                for sec in sections:
                    sm=slice_midi(m, float(sec['start']), float(sec['end']))
                    ev=section_prefix(sec['name'],bpm,key)+midi_to_events(sm)
                    samples.append(ev); metas.append({'song':os.path.basename(song),'section':sec['name']})
            else:
                ev=section_prefix('full',bpm,key)+midi_to_events(m)
                samples.append(ev); metas.append({'song':os.path.basename(song),'section':'full'})
    vocab=build_vocab(samples); os.makedirs(OUT,exist_ok=True)
    json.dump(vocab, open(VOCAB,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    with open(JSONL,'w',encoding='utf-8') as f:
        for ev,meta in zip(samples,metas):
            f.write(json.dumps({'tokens':events_to_ids(ev,vocab),'meta':meta},ensure_ascii=False)+'\n')
    print('wrote', JSONL, 'vocab', len(vocab), 'samples', len(samples))

if __name__=='__main__': main()
