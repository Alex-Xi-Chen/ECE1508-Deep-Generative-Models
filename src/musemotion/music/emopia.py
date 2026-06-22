from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from musemotion.config import resolve_path
from musemotion.emotions import quadrant_id
from musemotion.music.tokenizer import MusicTokenizer, MusicTokenizerConfig


EMOPIA_LABEL_PATTERN = re.compile(r"\bQ([1-4])\b|(?<![A-Z0-9])Q([1-4])(?![A-Z0-9])", re.IGNORECASE)


def prepare_emopia_dataset(config: dict[str, Any]) -> None:
    data_config = config.get("data", {})
    raw_dir = resolve_path(data_config.get("raw_dir", "data/raw/emopia"))
    tokenized_dir = resolve_path(data_config.get("tokenized_dir", "artifacts/music/tokenized"))
    tokenizer_dir = resolve_path(data_config.get("tokenizer_dir", "artifacts/music/tokenizer"))
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_config = MusicTokenizerConfig(**config.get("tokenizer", {}))
    tokenizer = MusicTokenizer(tokenizer_config)
    labeled_midis = discover_labeled_midis(raw_dir)
    if not labeled_midis:
        raise FileNotFoundError(
            f"No labeled MIDI files found under {raw_dir}. Expected paths containing Q1, Q2, Q3, or Q4."
        )

    seed = int(data_config.get("seed", 1508))
    rng = random.Random(seed)
    rng.shuffle(labeled_midis)
    splits = split_examples(
        labeled_midis,
        validation_fraction=float(data_config.get("validation_fraction", 0.1)),
        test_fraction=float(data_config.get("test_fraction", 0.1)),
    )

    tokenizer.save(tokenizer_dir / "vocab.json")
    for split_name, examples in splits.items():
        output_file = tokenized_dir / f"{split_name}.jsonl"
        write_tokenized_split(output_file, examples, tokenizer)


def discover_labeled_midis(raw_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"EMOPIA raw directory does not exist: {root}")
    examples: list[dict[str, Any]] = []
    for midi_path in sorted(root.rglob("*")):
        if midi_path.suffix.lower() not in {".mid", ".midi"}:
            continue
        quadrant = infer_quadrant_from_path(midi_path)
        if quadrant is None:
            continue
        examples.append({"path": midi_path, "quadrant": quadrant, "emotion_id": quadrant_id(quadrant)})
    return examples


def infer_quadrant_from_path(path: str | Path) -> str | None:
    text = " ".join(Path(path).parts)
    match = EMOPIA_LABEL_PATTERN.search(text)
    if match is None:
        return None
    value = match.group(1) or match.group(2)
    return f"Q{value}"


def split_examples(
    examples: list[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    total = len(examples)
    test_count = max(1, round(total * test_fraction)) if total >= 3 else 0
    validation_count = max(1, round(total * validation_fraction)) if total >= 3 else 0
    train_count = max(0, total - validation_count - test_count)
    return {
        "train": examples[:train_count],
        "validation": examples[train_count : train_count + validation_count],
        "test": examples[train_count + validation_count :],
    }


def write_tokenized_split(output_file: Path, examples: list[dict[str, Any]], tokenizer: MusicTokenizer) -> None:
    with output_file.open("w", encoding="utf-8") as handle:
        for example in examples:
            notes = tokenizer.midi_to_notes(example["path"])
            if not notes:
                continue
            payload = {
                "source": str(example["path"]),
                "quadrant": example["quadrant"],
                "emotion_id": int(example["emotion_id"]),
                "token_ids": tokenizer.encode_notes(notes),
            }
            handle.write(json.dumps(payload) + "\n")
