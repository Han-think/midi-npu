""" data/raw/<song>/*.mid (+ sections.json) -> data/processed/jsonl/train.jsonl, vocab.json """
import os, json, glob
import pretty_midi as pm
from src.tokenizers.skytnt import midi_to_events, build_vocab, events_to_ids, section_prefix

RAW_ROOT = "data/raw"
OUT_DIR  = "data/processed"
JSONL    = os.path.join(OUT_DIR, "jsonl", "train.jsonl")
VOCAB    = os.path.join(OUT_DIR, "vocab.json")

def slice_midi(m: pm.PrettyMIDI, s: float, e: float) -> pm.PrettyMIDI:
    out = pm.PrettyMIDI(resolution=m.resolution)
    for inst in m.instruments:
        ni = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            if n.start>=e or n.end<=s: continue
            ns = max(n.start, s) - s
            ne = max(min(n.end, e)-s, ns+1e-4)
            ni.notes.append(pm.Note(velocity=n.velocity, pitch=n.pitch, start=ns, end=ne))
        if ni.notes: out.instruments.append(ni)
    return out

def main():
    os.makedirs(os.path.dirname(JSONL), exist_ok=True)
    samples, metas = [], []
    for song_dir in glob.glob(os.path.join(RAW_ROOT, "*")):
        if not os.path.isdir(song_dir): continue
        mids = glob.glob(os.path.join(song_dir,"*.mid"))
        if not mids: continue
        sections_path = os.path.join(song_dir,"sections.json")
        sections = json.load(open(sections_path,"r",encoding="utf-8")) if os.path.exists(sections_path) else None
        bpm = 120; key="C"
        for mp in mids:
            m = pm.PrettyMIDI(mp)
            if sections:
                cur=0.0
                for sec in sections:
                    name = sec["name"]; s=float(sec["start"]); e=float(sec["end"])
                    sm = slice_midi(m, s, e)
                    ev = section_prefix(name, bpm, key) + midi_to_events(sm)
                    samples.append(ev); metas.append({"song":os.path.basename(song_dir),"section":name})
            else:
                ev = section_prefix("full", bpm, key) + midi_to_events(m)
                samples.append(ev); metas.append({"song":os.path.basename(song_dir),"section":"full"})
    vocab = build_vocab(samples)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(vocab, open(VOCAB,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(JSONL,"w",encoding="utf-8") as f:
        for ev, meta in zip(samples, metas):
            ids = events_to_ids(ev, vocab)
            f.write(json.dumps({"tokens":ids,"meta":meta}, ensure_ascii=False)+"\n")
    print(f"wrote: {JSONL}, vocab={len(vocab)}, samples={len(samples)}")

if __name__=="__main__": main()
