"""Wrapper around the (mocked) skytnt MIDI composition model."""
from __future__ import annotations

import logging
import random
import time
from typing import Optional

import numpy as np
import pretty_midi

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from music_theory import build_scale, clamp_midi_range, cycle_scale, parse_key


LOGGER = logging.getLogger(__name__)


class SkytntRunner:
    """Simplified runner for the skytnt MIDI composer.

    The real model is not bundled with this repository. To keep the pipeline usable
    without the heavy dependency we procedurally generate MIDI phrases that mimic the
    intended behaviour: multi-track output with drums, bass, chords and a lead line.
    """

    def __init__(self) -> None:
        self.device = self._select_device()
        self.model = self._load_model()
        LOGGER.info("Skytnt runner initialised on %s", self.device)

    def _select_device(self):
        if torch is not None:
            if torch.cuda.is_available():  # pragma: no cover - GPU only path
                return torch.device("cuda")
            return torch.device("cpu")
        return "cpu"

    def _load_model(self):
        """Placeholder for the actual model loading logic."""
        if torch is None:
            LOGGER.warning(
                "PyTorch is not available; falling back to procedural composition."
            )
            return None

        try:
            # The actual skytnt weights are not shipped. We keep a minimal module so the
            # API stays compatible and we can still seed the random generator.
            return torch.nn.Identity()
        except Exception as exc:  # pragma: no cover - unexpected failure path
            LOGGER.error("Failed to load skytnt model: %s", exc)
            return None

    # pylint: disable=too-many-locals
    def run_section(
        self,
        style: str,
        key: str,
        bpm: int,
        tag: str,
        seed: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> pretty_midi.PrettyMIDI:
        """Generate a multi-track MIDI section.

        Parameters match the external API. ``duration`` is optional – if omitted the
        resulting MIDI defaults to four bars. When supplied we quantise the duration to
        full bars in 4/4.
        """

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        start_time = time.perf_counter()

        tonic, mode = parse_key(key)
        scale = build_scale(tonic, mode)
        bar_seconds = 4.0 * 60.0 / max(bpm, 1)
        if duration is None:
            bars = 4
        else:
            bars = max(1, int(round(duration / bar_seconds)))

        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        section_tag = tag.lower() if tag else "section"

        drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
        bass = pretty_midi.Instrument(program=33, name="bass")
        chords = pretty_midi.Instrument(program=0, name="chords")
        lead = pretty_midi.Instrument(program=81, name="lead")

        LOGGER.debug(
            "Generating section '%s' with %d bars (style=%s, key=%s, bpm=%s)",
            section_tag,
            bars,
            style,
            key,
            bpm,
        )

        seconds_per_beat = 60.0 / max(bpm, 1)
        hat_velocity = 60 if "soft" in (style or "").lower() else 80
        kick_velocity = 100
        snare_velocity = 90

        for bar in range(bars):
            bar_start = bar * bar_seconds
            # Drums: kick on 1 and 3, snare on 2 and 4, hihat eighths
            for beat in range(4):
                beat_time = bar_start + beat * seconds_per_beat
                if beat % 2 == 0:
                    drums.notes.append(
                        pretty_midi.Note(
                            velocity=kick_velocity, pitch=36, start=beat_time, end=beat_time + 0.2
                        )
                    )
                else:
                    drums.notes.append(
                        pretty_midi.Note(
                            velocity=snare_velocity,
                            pitch=38,
                            start=beat_time,
                            end=beat_time + 0.2,
                        )
                    )
                for eighth in (0.0, 0.5):
                    hat_time = beat_time + eighth * seconds_per_beat
                    drums.notes.append(
                        pretty_midi.Note(
                            velocity=hat_velocity,
                            pitch=42,
                            start=hat_time,
                            end=hat_time + 0.1,
                        )
                    )

            # Bass: root + fifth pattern
            root_pitch = clamp_midi_range(scale[0] - 24, 36, 60)
            fifth_pitch = clamp_midi_range(scale[4 % len(scale)] - 24, 36, 60)
            bass_pattern = [root_pitch, root_pitch, fifth_pitch, root_pitch]
            for beat, pitch in enumerate(bass_pattern):
                start = bar_start + beat * seconds_per_beat
                bass.notes.append(
                    pretty_midi.Note(
                        velocity=70,
                        pitch=pitch,
                        start=start,
                        end=start + seconds_per_beat * 0.95,
                    )
                )

            # Chords: simple triads sustained per bar
            triad = [scale[0], scale[2 % len(scale)], scale[4 % len(scale)]]
            triad = [clamp_midi_range(p, 60, 84) for p in triad]
            chords.notes.extend(
                pretty_midi.Note(
                    velocity=75,
                    pitch=p,
                    start=bar_start,
                    end=bar_start + bar_seconds,
                )
                for p in triad
            )

            # Lead: cycle through scale with slight rhythmic variation
            lead_degrees = cycle_scale(scale, 8)
            for step in range(8):
                start = bar_start + step * (seconds_per_beat / 2.0)
                end = start + seconds_per_beat * 0.45
                pitch = clamp_midi_range(lead_degrees[step] + random.choice([-12, 0, 12]), 60, 96)
                lead.notes.append(
                    pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=start,
                        end=min(end, bar_start + bar_seconds),
                    )
                )

        midi.instruments.extend([drums, bass, chords, lead])
        midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
        key_number = (tonic % 12)
        if mode == "minor":
            key_number += 12
        midi.key_signature_changes.append(pretty_midi.KeySignature(key_number, 0.0))
        midi.estimate_tempo()
        LOGGER.info(
            "Section '%s' generation finished in %.2f ms",
            section_tag,
            (time.perf_counter() - start_time) * 1000.0,
        )
        return midi


RUNNER = SkytntRunner()


def run_section(
    style: str,
    key: str,
    bpm: int,
    tag: str,
    seed: Optional[int] = None,
    duration: Optional[float] = None,
) -> pretty_midi.PrettyMIDI:
    """Convenience wrapper calling the singleton runner."""

    return RUNNER.run_section(style, key, bpm, tag, seed=seed, duration=duration)

