from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MusicClassifierConfig:
    """Shape of the MIDI-to-quadrant probe.

    Deliberately much smaller than the generator: it trains on 862 EMOPIA clips, so a
    2-layer, 128-dimensional encoder is already at the edge of what that supports.
    """

    vocab_size: int
    max_seq_len: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 4
    pad_token_id: int = 0


@dataclass
class MusicClassifierOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class MusicClassifier(nn.Module):
    """Bidirectional encoder over music tokens, mean-pooled into a quadrant prediction.

    This is the "backward" half of the round-trip check: the generator turns a quadrant into
    music, and this turns music back into a quadrant. Two differences from the generator are
    deliberate. There is no causal mask, because classifying a finished clip is not a
    next-token problem and the whole sequence is legitimately available. And there is no
    emotion embedding, because the emotion is what gets predicted.

    The encoder scaffolding it shares with ``MusicTransformer`` - the head-divisibility check,
    the two embeddings, the encoder stack, the final norm - is restated here rather than factored
    out, and that is a decision rather than an oversight. Extracting it would change the order in
    which ``MusicTransformer`` constructs its submodules, which changes the random draws each one
    consumes, which changes the weights a fixed seed produces. That would invalidate the recorded
    token sequences in ``tests/test_sample_batch.py`` that prove ``sample`` and ``sample_batch``
    generate identically - the evidence is worth more than the thirty saved lines. The drift risk
    that normally justifies extraction is covered instead by
    ``test_the_two_models_keep_the_same_encoder_core``, which fails if either encoder changes
    without the other.
    """

    def __init__(self, config: MusicClassifierConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
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
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> MusicClassifierOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        # No causal mask: the probe is allowed to see the whole clip at once.
        key_padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)
        pooled = self.final_norm(_masked_mean(hidden, attention_mask))
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            weight = class_weights.to(logits.device) if class_weights is not None else None
            loss = F.cross_entropy(logits, labels, weight=weight)
        return MusicClassifierOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        logits = self(input_ids=input_ids, attention_mask=attention_mask).logits
        return F.softmax(logits, dim=-1)


def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over real positions only, so padding cannot dilute a short clip's representation."""
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    total = (hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return total / count
