import pretty_midi as pm

DRUM_CH = 9  # MIDI ch10 (0-index)


def section_prefix(name: str, bpm: int, key: str):
    return [f'<SECTION={name}>', f'<BPM={bpm}>', f'<KEY={key}>', 'TIME_SIG_4_4']


def midi_to_events(m: pm.PrettyMIDI):
    ev = []
    tempi = m.get_tempo_changes()[1]
    tempo = int(tempi[0]) if len(tempi) > 0 else 120
    ev.append(f'TEMPO_{tempo}')
    # rough bars
    dur = m.get_end_time(); bar = 60.0/tempo*4; t=0.0; b=0
    while t < dur: ev.append(f'BAR_{b}'); t += bar; b += 1
    # tracks
    for inst in m.instruments:
        prog = 128 if inst.is_drum else inst.program
        ch   = DRUM_CH if inst.is_drum else 0
        ev += [f'INST_{prog}', f'CH_{ch}']
        for n in inst.notes:
            ev += [f'NOTE_{n.pitch}', f'DUR_{max(int((n.end-n.start)*960),1)}', f'VEL_{int(n.velocity)}']
        ev.append('INST_END')
    return ev


def build_vocab(samples):
    from collections import Counter
    c = Counter(); [c.update(s) for s in samples]
    toks = ['<pad>','<bos>','<eos>','<unk>'] + [t for t,f in c.items() if f>=1]
    return {t:i for i,t in enumerate(toks)}


def events_to_ids(events, vocab):
    unk = vocab.get('<unk>', 0)
    return [vocab.get(t, unk) for t in events]
