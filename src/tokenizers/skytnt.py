from __future__ import annotations
import pretty_midi as pm
from typing import List, Dict, Any

DRUM_CH = 9  # MIDI ch10 (0-index)

def section_prefix(section:str, bpm:int, key:str) -> list[str]:
    return [f"<SECTION={section}>", f"<BPM={bpm}>", f"<KEY={key}>", "TIME_SIG_4_4"]

def midi_to_events(m: pm.PrettyMIDI, add_bar=True) -> list[str]:
    events: list[str] = []
    # tempo
    tempi = m.get_tempo_changes()[1]
    tempo = int(tempi[0]) if len(tempi)>0 else 120
    events.append(f"TEMPO_{tempo}")
    # bars (rough)
    if add_bar:
        dur = m.get_end_time()
        bar_len = 60.0/tempo*4
        b=0.0; k=0
        while b < dur:
            events.append(f"BAR_{k}"); k+=1; b += bar_len
    # tracks
    for inst in m.instruments:
        prog = 128 if inst.is_drum else inst.program
        ch   = DRUM_CH if inst.is_drum else 0
        events += [f"INST_{prog}", f"CH_{ch}"]
        for n in inst.notes:
            pitch = n.pitch
            vel   = int(n.velocity)
            dur_t = max(int((n.end - n.start)*960), 1)
            events += [f"NOTE_{pitch}", f"DUR_{dur_t}", f"VEL_{vel}"]
        events.append("INST_END")
    return events

def events_to_ids(events: list[str], vocab: Dict[str,int]) -> list[int]:
    unk = vocab.get("<unk>",0)
    return [vocab.get(t, unk) for t in events]

def build_vocab(samples: list[list[str]], min_freq:int=1) -> Dict[str,int]:
    from collections import Counter
    c = Counter()
    for ev in samples: c.update(ev)
    toks = ["<pad>","<bos>","<eos>","<unk>"] + [t for t,f in c.items() if f>=min_freq]
    return {t:i for i,t in enumerate(toks)}
