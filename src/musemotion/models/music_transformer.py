from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MusicTransformerConfig:
    vocab_size: int
    max_seq_len: int = 1024
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    num_emotions: int = 4
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class MusicModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class MusicTransformer(nn.Module):
    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.emotion_embedding = nn.Embedding(config.num_emotions, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        emotion_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> MusicModelOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.emotion_embedding(emotion_ids).unsqueeze(1)
        )
        hidden = self.dropout(hidden)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.eq(0)
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return MusicModelOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def sample(
        self,
        emotion_id: int,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = 32,
        prompt_token_ids: list[int] | None = None,
        device: torch.device | str | None = None,
    ) -> list[int]:
        self.eval()
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        generated = list(prompt_token_ids or [self.config.bos_token_id])
        emotion_ids = torch.tensor([emotion_id], dtype=torch.long, device=target_device)

        for _ in range(max_tokens):
            context = generated[-self.config.max_seq_len :]
            input_ids = torch.tensor([context], dtype=torch.long, device=target_device)
            logits = self(input_ids=input_ids, emotion_ids=emotion_ids).logits[:, -1, :]
            logits[:, self.config.pad_token_id] = -torch.inf
            if temperature <= 0:
                next_token = int(torch.argmax(logits, dim=-1).item())
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    top_values, top_indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                    filtered = torch.full_like(logits, -torch.inf)
                    filtered.scatter_(dim=-1, index=top_indices, src=top_values)
                    logits = filtered
                probs = F.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_token)
            if next_token == self.config.eos_token_id:
                break
        return generated
