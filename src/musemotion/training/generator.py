from __future__ import annotations

import json
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from musemotion.config import resolve_path
from musemotion.models.music_transformer import MusicTransformer, MusicTransformerConfig
from musemotion.music.dataset import TokenizedMusicDataset, collate_music_batch
from musemotion.music.tokenizer import MusicTokenizer


def train_generator(config: dict[str, Any]) -> None:
    data_config = config.get("data", {})
    tokenized_dir = resolve_path(data_config.get("tokenized_dir", "artifacts/music/tokenized"))
    tokenizer_path = resolve_path(data_config.get("tokenizer_dir", "artifacts/music/tokenizer")) / "vocab.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found. Run prepare_emopia first: {tokenizer_path}")

    tokenizer = MusicTokenizer.load(tokenizer_path)
    train_dataset = TokenizedMusicDataset(tokenized_dir / "train.jsonl")
    validation_path = tokenized_dir / "validation.jsonl"
    validation_dataset = TokenizedMusicDataset(validation_path) if validation_path.exists() else None

    model_config = MusicTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **config.get("model", {}),
    )
    model = MusicTransformer(model_config)
    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 1508))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = int(training_config.get("batch_size", 8))
    collate = partial(
        collate_music_batch,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=model_config.max_seq_len,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        if validation_dataset is not None and len(validation_dataset) > 0
        else None
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )
    epochs = int(training_config.get("epochs", 20))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))
    output_dir = resolve_path(training_config.get("output_dir", "artifacts/music/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip_norm)
        validation_loss = evaluate(model, validation_loader, device) if validation_loader is not None else train_loss
        print(f"epoch={epoch} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}")
        if validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
            save_generator_checkpoint(output_dir / "best.pt", model, tokenizer_path, best_validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "best_validation_loss": best_validation_loss,
            }
        )
        write_generator_history(output_dir, history)

    save_generator_checkpoint(output_dir / "last.pt", model, tokenizer_path, best_validation_loss)


def train_one_epoch(
    model: MusicTransformer,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        assert output.loss is not None
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(output.loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model: MusicTransformer, loader: DataLoader | None, device: torch.device) -> float:
    if loader is None:
        return float("inf")
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        assert output.loss is not None
        total_loss += float(output.loss.item())
    return total_loss / max(1, len(loader))


def save_generator_checkpoint(
    path: str | Path,
    model: MusicTransformer,
    tokenizer_path: str | Path,
    validation_loss: float,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "tokenizer_path": str(tokenizer_path),
            "validation_loss": validation_loss,
        },
        checkpoint_path,
    )
    return checkpoint_path


def write_generator_history(output_dir: str | Path, history: list[dict[str, float | int]]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "validation_loss", "best_validation_loss"]
    lines = [",".join(fields)]
    for row in history:
        lines.append(",".join(_history_value(row[field]) for field in fields))
    (output_path / "training_history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_path / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def _history_value(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)
