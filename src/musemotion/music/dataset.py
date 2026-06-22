from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class TokenizedMusicDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.examples = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "emotion_id": int(example["emotion_id"]),
            "token_ids": [int(token_id) for token_id in example["token_ids"]],
            "source": example.get("source"),
        }


def collate_music_batch(
    examples: list[dict[str, Any]],
    pad_token_id: int,
    label_pad_token_id: int = -100,
    max_seq_len: int | None = None,
) -> dict[str, torch.Tensor]:
    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive when provided.")

    max_length = max(len(_truncate_token_ids(example["token_ids"], max_seq_len)) - 1 for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    emotion_ids: list[int] = []

    for example in examples:
        token_ids = _truncate_token_ids(example["token_ids"], max_seq_len)
        input_ids = token_ids[:-1]
        labels = token_ids[1:]
        padding = max_length - len(input_ids)
        input_rows.append(input_ids + [pad_token_id] * padding)
        label_rows.append(labels + [label_pad_token_id] * padding)
        attention_rows.append([1] * len(input_ids) + [0] * padding)
        emotion_ids.append(int(example["emotion_id"]))

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        "emotion_ids": torch.tensor(emotion_ids, dtype=torch.long),
    }


def _truncate_token_ids(token_ids: list[int], max_seq_len: int | None) -> list[int]:
    if max_seq_len is None:
        return token_ids
    return token_ids[: max_seq_len + 1]
