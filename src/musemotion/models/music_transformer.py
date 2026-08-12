from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
        # One extra row is the "null"/unconditional embedding used for classifier-free guidance.
        self.null_emotion_id = config.num_emotions
        self.emotion_embedding = nn.Embedding(config.num_emotions + 1, config.d_model)
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

    def _batch_last_token_logits(
        self,
        context: torch.Tensor,
        emotion_ids: torch.Tensor,
        guidance_scale: float,
        use_cfg: bool,
    ) -> torch.Tensor:
        """Last-position logits for a batch of contexts, optionally guided.

        Under guidance the conditioned and unconditioned passes are stacked into one forward
        call of twice the batch size, so guidance costs one wider pass rather than two.
        """
        if use_cfg:
            batch_size = context.shape[0]
            doubled = torch.cat([context, context], dim=0)
            null_ids = torch.full_like(emotion_ids, self.null_emotion_id)
            stacked = torch.cat([emotion_ids, null_ids], dim=0)
            last = self(input_ids=doubled, emotion_ids=stacked).logits[:, -1, :]
            conditioned, unconditioned = last[:batch_size], last[batch_size:]
            return unconditioned + guidance_scale * (conditioned - unconditioned)
        return self(input_ids=context, emotion_ids=emotion_ids).logits[:, -1, :]

    @torch.no_grad()
    def sample_batch(
        self,
        emotion_ids: Sequence[int],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = 32,
        guidance_scale: float = 1.0,
        prompt_token_ids: list[int] | None = None,
        device: torch.device | str | None = None,
    ) -> list[list[int]]:
        """Sample one clip per entry in ``emotion_ids``, stepping every row together.

        Every row starts from BOS and takes one step per iteration, so all contexts share a
        length at every point and no padding is ever needed - each row's logits are exactly
        what it would have received on its own. That makes this a throughput change, not a
        behaviour change.

        Batching is worth it on GPU, where a model this small is dominated by per-call
        overhead at batch size 1. On CPU it is roughly neutral: measured here, per-sequence
        cost went from 0.035 s at batch 1 to 0.047 s at batch 64, because the CPU path
        already saturates its threads.

        One caveat on reproducibility: ``torch.multinomial`` draws the whole batch in a
        single call, so it consumes the RNG stream differently from repeated single-row
        calls. Batched and sequential sampling agree exactly under greedy decoding
        (``temperature <= 0``) but not under sampling, even from the same seed. Callers that
        need to reproduce a run should record the seed they used rather than assume the two
        paths interchange.
        """
        self.eval()
        if not len(emotion_ids):
            return []
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        batch_size = len(emotion_ids)
        emotions = torch.tensor(list(emotion_ids), dtype=torch.long, device=target_device)
        # Every row starts from the same prompt, which is what keeps the contexts equal-length
        # and makes the batched logits exact.
        prompt = list(prompt_token_ids or [self.config.bos_token_id])
        context = torch.tensor(
            [prompt for _ in range(batch_size)], dtype=torch.long, device=target_device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=target_device)
        use_cfg = guidance_scale is not None and abs(guidance_scale - 1.0) > 1e-6

        for _ in range(max_tokens):
            window = context[:, -self.config.max_seq_len :]
            logits = self._batch_last_token_logits(window, emotions, guidance_scale, use_cfg)
            logits[:, self.config.pad_token_id] = -torch.inf
            if temperature <= 0:
                next_tokens = torch.argmax(logits, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    top_values, top_indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                    filtered = torch.full_like(logits, -torch.inf)
                    filtered.scatter_(dim=-1, index=top_indices, src=top_values)
                    logits = filtered
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Rows that already emitted EOS keep emitting it, so their trailing tokens are
            # trivially trimmed afterwards and they cannot wander past their own ending.
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, self.config.eos_token_id), next_tokens
            )
            context = torch.cat([context, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | next_tokens.eq(self.config.eos_token_id)
            if bool(finished.all()):
                break

        return [_trim_at_eos(row, self.config.eos_token_id) for row in context.tolist()]

    @torch.no_grad()
    def sample(
        self,
        emotion_id: int,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = 32,
        guidance_scale: float = 1.0,
        prompt_token_ids: list[int] | None = None,
        device: torch.device | str | None = None,
    ) -> list[int]:
        """Sample one clip. Thin wrapper over :meth:`sample_batch` with a batch of one.

        Keeping one decode loop matters beyond tidiness: the shipped CLI and Gradio app generate
        through this method while the whole evaluation generates through ``sample_batch``. Two
        copies would let the two drift, and the evaluation would quietly stop describing the
        generator that users actually run. A batch of one consumes the RNG stream identically, so
        this produces exactly the tokens the standalone loop did - pinned by the golden-token
        tests in ``tests/test_sample_batch.py``.
        """
        return self.sample_batch(
            emotion_ids=[emotion_id],
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            guidance_scale=guidance_scale,
            prompt_token_ids=prompt_token_ids,
            device=device,
        )[0]


def _trim_at_eos(token_ids: list[int], eos_token_id: int) -> list[int]:
    """Keep everything up to and including the first EOS, dropping the frozen tail after it."""
    for index, token_id in enumerate(token_ids):
        if token_id == eos_token_id:
            return token_ids[: index + 1]
    return token_ids
