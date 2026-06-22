import json

import torch

from musemotion.models.music_transformer import MusicTransformer, MusicTransformerConfig
from musemotion.music.dataset import TokenizedMusicDataset, collate_music_batch


def test_music_dataset_and_collate_create_shifted_targets(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        json.dumps({"emotion_id": 1, "token_ids": [1, 10, 11, 2]}) + "\n",
        encoding="utf-8",
    )

    batch = collate_music_batch([TokenizedMusicDataset(data_file)[0]], pad_token_id=0)

    assert batch["input_ids"].tolist() == [[1, 10, 11]]
    assert batch["labels"].tolist() == [[10, 11, 2]]
    assert batch["emotion_ids"].tolist() == [1]


def test_music_collate_truncates_to_model_context_window(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        json.dumps({"emotion_id": 2, "token_ids": [1, 10, 11, 12, 13, 14, 2]}) + "\n",
        encoding="utf-8",
    )

    batch = collate_music_batch(
        [TokenizedMusicDataset(data_file)[0]],
        pad_token_id=0,
        max_seq_len=4,
    )

    assert batch["input_ids"].shape == (1, 4)
    assert batch["input_ids"].tolist() == [[1, 10, 11, 12]]
    assert batch["labels"].tolist() == [[10, 11, 12, 13]]


def test_music_transformer_forward_returns_loss_and_logits():
    model = MusicTransformer(
        MusicTransformerConfig(vocab_size=32, max_seq_len=16, d_model=32, n_heads=4, n_layers=1)
    )
    input_ids = torch.tensor([[1, 4, 5]])
    labels = torch.tensor([[4, 5, 2]])
    emotion_ids = torch.tensor([0])

    output = model(input_ids=input_ids, emotion_ids=emotion_ids, labels=labels)

    assert output.loss is not None
    assert output.logits.shape == (1, 3, 32)
