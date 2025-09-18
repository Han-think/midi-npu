import pretty_midi as pm

DRUM_CH = 9  # MIDI ch10 (0-index)


def section_prefix(name: str, bpm: int, key: str) -> list[str]:
    return [f"<SECTION={name}>", f"<BPM={bpm}>", f"<KEY={key}>", "TIME_SIG_4_4"]


def midi_to_events(midi: pm.PrettyMIDI) -> list[str]:
    events: list[str] = []
    tempi = midi.get_tempo_changes()[1]
    tempo = int(tempi[0]) if len(tempi) > 0 else 120
    events.append(f"TEMPO_{tempo}")

    bar_duration = (60.0 / tempo) * 4
    bar_index = 0
    time_cursor = 0.0
    total_duration = midi.get_end_time()
    while time_cursor < total_duration:
        events.append(f"BAR_{bar_index}")
        time_cursor += bar_duration
        bar_index += 1

    for instrument in midi.instruments:
        program = 128 if instrument.is_drum else instrument.program
        channel = DRUM_CH if instrument.is_drum else 0
        events.extend([f"INST_{program}", f"CH_{channel}"])
        for note in instrument.notes:
            duration = max(int((note.end - note.start) * 960), 1)
            events.extend(
                [f"NOTE_{note.pitch}", f"DUR_{duration}", f"VEL_{int(note.velocity)}"]
            )
        events.append("INST_END")
    return events


def build_vocab(samples: list[list[str]]) -> dict[str, int]:
    from collections import Counter

    counter = Counter()
    for sample in samples:
        counter.update(sample)
    tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + [
        token for token, _ in counter.items()
    ]
    return {token: index for index, token in enumerate(tokens)}


def events_to_ids(events: list[str], vocab: dict[str, int]) -> list[int]:
    unknown = vocab.get("<unk>", 0)
    return [vocab.get(token, unknown) for token in events]
