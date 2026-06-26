from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RandomPianoGeneratorConfig:
    seed: int = 260625
    tempo_bpm: int = 92
    scale: list[int] = field(
        default_factory=lambda: [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72]
    )
    chord_roots: list[int] = field(default_factory=lambda: [48, 53, 55, 50])
    steps: int = 32
    description: str = "Deterministic random piano generator used for the saved sample."

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RandomPianoGeneratorConfig":
        fields = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "deterministic_random_piano_generator"
        payload["module"] = "musemotion.music.random_generator"
        return payload


def generate_random_piano(config: RandomPianoGeneratorConfig, output_path: str | Path) -> dict[str, Any]:
    import pretty_midi

    rng = random.Random(config.seed)
    midi = pretty_midi.PrettyMIDI(initial_tempo=config.tempo_bpm)
    piano = pretty_midi.Instrument(program=0, name="Random MUSEmotion Piano")

    current = 0.0
    for step in range(config.steps):
        if step % 8 == 0:
            root = config.chord_roots[(step // 8) % len(config.chord_roots)]
            for interval in [0, 7, 12]:
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=48 + rng.randint(0, 16),
                        pitch=root + interval,
                        start=current,
                        end=current + 1.6,
                    )
                )

        pitch = rng.choice(config.scale) + rng.choice([0, 0, 0, 12])
        duration = rng.choice([0.25, 0.5, 0.75, 1.0])
        velocity = rng.randint(54, 96)
        start = current + rng.choice([0.0, 0.05, 0.1])
        piano.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=start + duration,
            )
        )
        current += rng.choice([0.25, 0.5, 0.5, 0.75])

    midi.instruments.append(piano)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return {
        "midi_path": str(path).replace("\\", "/"),
        "seed": config.seed,
        "tempo_bpm": config.tempo_bpm,
        "note_count": len(piano.notes),
        "duration_seconds": round(midi.get_end_time(), 3),
    }
