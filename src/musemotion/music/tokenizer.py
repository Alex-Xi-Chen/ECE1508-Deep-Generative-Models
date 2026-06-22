from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MidiNote:
    pitch: int
    start: float
    end: float
    velocity: int


@dataclass(frozen=True)
class MusicTokenizerConfig:
    beat_resolution: int = 4
    min_pitch: int = 21
    max_pitch: int = 108
    velocity_bins: int = 8
    max_shift_steps: int = 16
    max_duration_steps: int = 16


class MusicTokenizer:
    special_tokens = ("PAD", "BOS", "EOS")

    def __init__(self, config: MusicTokenizerConfig | None = None, token_to_id: dict[str, int] | None = None):
        self.config = config or MusicTokenizerConfig()
        self.token_to_id = token_to_id or self._build_vocab()
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["BOS"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["EOS"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _build_vocab(self) -> dict[str, int]:
        tokens: list[str] = list(self.special_tokens)
        tokens.extend(f"SHIFT_{step}" for step in self._shift_values())
        tokens.extend(f"PITCH_{pitch}" for pitch in range(self.config.min_pitch, self.config.max_pitch + 1))
        tokens.extend(f"DUR_{step}" for step in self._duration_values())
        tokens.extend(f"VEL_{bucket}" for bucket in range(1, self.config.velocity_bins + 1))
        return {token: idx for idx, token in enumerate(tokens)}

    def _shift_values(self) -> list[int]:
        return _quantized_values(0, self.config.max_shift_steps)

    def _duration_values(self) -> list[int]:
        return _quantized_values(1, self.config.max_duration_steps)

    def encode_notes(self, notes: Iterable[MidiNote]) -> list[int]:
        ordered_notes = sorted(notes, key=lambda note: (note.start, note.pitch))
        token_ids = [self.bos_token_id]
        previous_start_step = 0

        for note in ordered_notes:
            start_step = max(0, round(note.start * self.config.beat_resolution))
            end_step = max(start_step + 1, round(note.end * self.config.beat_resolution))
            shift_step = max(0, start_step - previous_start_step)
            duration_step = max(1, end_step - start_step)
            previous_start_step = start_step

            tokens = [
                f"SHIFT_{self._nearest_shift(shift_step)}",
                f"PITCH_{self._clip_pitch(note.pitch)}",
                f"DUR_{self._nearest_duration(duration_step)}",
                f"VEL_{self._velocity_bucket(note.velocity)}",
            ]
            token_ids.extend(self.token_to_id[token] for token in tokens)

        token_ids.append(self.eos_token_id)
        return token_ids

    def decode_tokens(self, token_ids: Iterable[int]) -> list[MidiNote]:
        tokens = [
            self.id_to_token[int(token_id)]
            for token_id in token_ids
            if int(token_id) in self.id_to_token
            and self.id_to_token[int(token_id)] not in {"PAD", "BOS", "EOS"}
        ]

        notes: list[MidiNote] = []
        current_start_step = 0
        for index in range(0, len(tokens) - 3, 4):
            shift_token, pitch_token, duration_token, velocity_token = tokens[index : index + 4]
            if not (
                shift_token.startswith("SHIFT_")
                and pitch_token.startswith("PITCH_")
                and duration_token.startswith("DUR_")
                and velocity_token.startswith("VEL_")
            ):
                continue

            current_start_step += int(shift_token.split("_", 1)[1])
            duration_step = int(duration_token.split("_", 1)[1])
            pitch = int(pitch_token.split("_", 1)[1])
            velocity = self._bucket_velocity(int(velocity_token.split("_", 1)[1]))
            start = current_start_step / self.config.beat_resolution
            end = (current_start_step + duration_step) / self.config.beat_resolution
            notes.append(MidiNote(pitch=pitch, start=start, end=end, velocity=velocity))
        return notes

    def midi_to_notes(self, midi_path: str | Path) -> list[MidiNote]:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        notes: list[MidiNote] = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                notes.append(MidiNote(note.pitch, note.start, note.end, note.velocity))
        return sorted(notes, key=lambda note: (note.start, note.pitch))

    def notes_to_midi(self, notes: Iterable[MidiNote], output_path: str | Path) -> Path:
        import pretty_midi

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0, name="MUSEmotion Piano")
        for note in notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=max(float(note.end), float(note.start) + 0.05),
                )
            )
        midi.instruments.append(instrument)
        midi.write(str(path))
        return path

    def token_ids_to_midi(self, token_ids: Iterable[int], output_path: str | Path) -> Path:
        return self.notes_to_midi(self.decode_tokens(token_ids), output_path)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "tokens": [self.id_to_token[idx] for idx in range(len(self.id_to_token))],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "MusicTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = MusicTokenizerConfig(**payload["config"])
        token_to_id = {token: idx for idx, token in enumerate(payload["tokens"])}
        return cls(config=config, token_to_id=token_to_id)

    def _nearest_shift(self, value: int) -> int:
        return _nearest(value, self._shift_values())

    def _nearest_duration(self, value: int) -> int:
        return _nearest(value, self._duration_values())

    def _clip_pitch(self, pitch: int) -> int:
        return min(self.config.max_pitch, max(self.config.min_pitch, int(pitch)))

    def _velocity_bucket(self, velocity: int) -> int:
        clipped = min(127, max(1, int(velocity)))
        return min(self.config.velocity_bins, max(1, round(clipped / 127 * self.config.velocity_bins)))

    def _bucket_velocity(self, bucket: int) -> int:
        clipped = min(self.config.velocity_bins, max(1, int(bucket)))
        return round(clipped / self.config.velocity_bins * 127)


def _quantized_values(start: int, stop: int) -> list[int]:
    values = {start, stop}
    step = 1
    while step <= stop:
        if step >= start:
            values.add(step)
        step *= 2
    return sorted(values)


def _nearest(value: int, candidates: list[int]) -> int:
    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))
