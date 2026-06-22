from __future__ import annotations

import json
import random
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from musemotion.config import resolve_path
from musemotion.emotions import quadrant_id
from musemotion.music.tokenizer import MusicTokenizer, MusicTokenizerConfig


EMOPIA_LABEL_PATTERN = re.compile(r"\bQ([1-4])\b|(?<![A-Z0-9])Q([1-4])(?![A-Z0-9])", re.IGNORECASE)
DEFAULT_EMOPIA_URL = "https://zenodo.org/records/5090631/files/EMOPIA_1.0.zip?download=1"


def download_emopia_dataset(
    output_dir: str | Path = "data/raw/emopia",
    url: str = DEFAULT_EMOPIA_URL,
    force: bool = False,
) -> Path:
    output_path = resolve_path(output_dir)
    if output_path.exists() and not force and any(output_path.rglob("*.mid")):
        return output_path
    if force and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="musemotion-emopia-") as temp_name:
        temp_dir = Path(temp_name)
        archive_path = _fetch_archive(url, temp_dir)
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()
        _safe_extract_zip(archive_path, extract_dir)
        source_root = _single_root_directory(extract_dir) or extract_dir
        _copy_directory_contents(source_root, output_path)

    return output_path


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
    labeled_midis = limit_labeled_midis(labeled_midis, data_config)
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


def limit_labeled_midis(examples: list[dict[str, Any]], data_config: dict[str, Any]) -> list[dict[str, Any]]:
    max_examples = data_config.get("max_examples")
    max_examples_per_quadrant = data_config.get("max_examples_per_quadrant")
    if max_examples_per_quadrant is not None:
        per_quadrant_counts: dict[str, int] = {}
        balanced: list[dict[str, Any]] = []
        limit = int(max_examples_per_quadrant)
        for example in examples:
            quadrant = str(example["quadrant"])
            count = per_quadrant_counts.get(quadrant, 0)
            if count >= limit:
                continue
            balanced.append(example)
            per_quadrant_counts[quadrant] = count + 1
        examples = balanced
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    return examples


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


def _fetch_archive(url: str, temp_dir: Path) -> Path:
    local_path = Path(url)
    if local_path.exists():
        return local_path

    archive_path = temp_dir / "EMOPIA_1.0.zip"
    urllib.request.urlretrieve(url, archive_path)
    return archive_path


def _safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (output_dir / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            archive.extract(member, output_dir)


def _single_root_directory(path: Path) -> Path | None:
    children = [child for child in path.iterdir() if child.name not in {"__MACOSX"}]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return None


def _copy_directory_contents(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
