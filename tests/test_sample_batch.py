import pytest
import torch

from musemotion.models.music_transformer import MusicTransformer, MusicTransformerConfig


def _model():
    config = MusicTransformerConfig(
        vocab_size=20, max_seq_len=64, d_model=16, n_heads=2, n_layers=1, dropout=0.0
    )
    torch.manual_seed(0)
    model = MusicTransformer(config)
    model.eval()
    return model


def test_greedy_batched_matches_sequential_without_guidance():
    # Greedy decoding consumes no randomness, so batched and sequential sampling must agree
    # token for token. This is the check that batching is a throughput change only.
    model = _model()
    emotions = [0, 1, 2, 3]

    sequential = [
        model.sample(emotion_id=emotion, max_tokens=12, temperature=0.0, guidance_scale=1.0)
        for emotion in emotions
    ]
    batched = model.sample_batch(
        emotion_ids=emotions, max_tokens=12, temperature=0.0, guidance_scale=1.0
    )

    assert batched == sequential


def test_greedy_batched_matches_sequential_with_guidance():
    model = _model()
    emotions = [0, 1, 2, 3]

    sequential = [
        model.sample(emotion_id=emotion, max_tokens=12, temperature=0.0, guidance_scale=3.0)
        for emotion in emotions
    ]
    batched = model.sample_batch(
        emotion_ids=emotions, max_tokens=12, temperature=0.0, guidance_scale=3.0
    )

    assert batched == sequential


def test_every_row_starts_at_bos_and_respects_the_token_cap():
    model = _model()

    rows = model.sample_batch(emotion_ids=[0, 1, 2], max_tokens=10, temperature=1.0, top_k=8)

    assert len(rows) == 3
    for row in rows:
        assert row[0] == model.config.bos_token_id
        assert len(row) <= 11  # the BOS prompt plus at most max_tokens generated


def test_empty_batch_returns_no_rows():
    assert _model().sample_batch(emotion_ids=[], max_tokens=8) == []


def test_padding_token_is_never_generated():
    model = _model()

    rows = model.sample_batch(emotion_ids=[0] * 4, max_tokens=16, temperature=1.0, top_k=None)

    for row in rows:
        assert model.config.pad_token_id not in row[1:]


def test_rows_are_trimmed_at_their_own_eos():
    # Force EOS to dominate so rows terminate, then confirm nothing survives past it.
    model = _model()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[model.config.eos_token_id] = 10.0

    rows = model.sample_batch(emotion_ids=[0, 1], max_tokens=12, temperature=0.0)

    for row in rows:
        assert row[-1] == model.config.eos_token_id
        assert row.count(model.config.eos_token_id) == 1


# Captured from sample() before it was reimplemented on top of sample_batch. Pinning the actual
# tokens, rather than just checking that two calls agree with each other, is the only thing that
# catches a change in what the shipped generator produces - a determinism check passes unchanged
# even if the sampler starts emitting something completely different.
GOLDEN_SEED = 4242
GOLDEN_OUTPUTS = {
    "sampling_g1": ([1, 11, 17, 1, 15, 16, 13, 1, 6, 19, 1, 1, 9], dict(temperature=1.0, top_k=8, guidance_scale=1.0)),
    "sampling_g3": ([1, 17, 1, 1, 1, 17, 13, 1, 1, 19, 1, 1, 9], dict(temperature=1.0, top_k=8, guidance_scale=3.0)),
    "greedy_g3": ([1, 19, 1, 17, 1, 17, 1, 19, 1, 19, 19, 17, 1], dict(temperature=0.0, guidance_scale=3.0)),
}


@pytest.mark.parametrize("name", sorted(GOLDEN_OUTPUTS))
def test_sample_still_produces_its_recorded_tokens(name):
    expected, kwargs = GOLDEN_OUTPUTS[name]
    model = _model()

    torch.manual_seed(GOLDEN_SEED)
    assert model.sample(emotion_id=2, max_tokens=12, **kwargs) == expected


@pytest.mark.parametrize("name", sorted(GOLDEN_OUTPUTS))
def test_batched_sampling_matches_the_same_recorded_tokens(name):
    """The evaluation samples through sample_batch and the shipped CLI through sample.

    If those two ever diverge, the evaluation stops describing the generator users actually run,
    so both are pinned to the same recorded output.
    """
    expected, kwargs = GOLDEN_OUTPUTS[name]
    model = _model()

    torch.manual_seed(GOLDEN_SEED)
    assert model.sample_batch(emotion_ids=[2], max_tokens=12, **kwargs)[0] == expected


def test_sample_is_reproducible_from_a_fixed_seed():
    model = _model()
    torch.manual_seed(1234)
    first = model.sample(emotion_id=2, max_tokens=10, temperature=1.0, top_k=8)
    torch.manual_seed(1234)
    second = model.sample(emotion_id=2, max_tokens=10, temperature=1.0, top_k=8)

    assert first == second
