import pytest
import torch

from musemotion.models.music_classifier import MusicClassifier, MusicClassifierConfig


def _model(**overrides):
    config = MusicClassifierConfig(
        vocab_size=24, max_seq_len=32, d_model=16, n_heads=2, n_layers=1, dropout=0.0, **overrides
    )
    model = MusicClassifier(config)
    model.eval()
    return model


def test_forward_returns_one_logit_row_per_clip():
    model = _model()
    input_ids = torch.randint(3, 24, (5, 12))

    output = model(input_ids=input_ids)

    assert output.logits.shape == (5, 4)
    assert output.loss is None


def test_loss_is_scalar_when_labels_are_supplied():
    model = _model()
    input_ids = torch.randint(3, 24, (4, 10))
    labels = torch.tensor([0, 1, 2, 3])

    output = model(input_ids=input_ids, labels=labels)

    assert output.loss is not None
    assert output.loss.shape == ()
    assert torch.isfinite(output.loss)


def test_class_weights_change_the_loss():
    model = _model()
    input_ids = torch.randint(3, 24, (4, 10))
    labels = torch.tensor([0, 0, 0, 1])
    weights = torch.tensor([0.1, 10.0, 1.0, 1.0])

    unweighted = model(input_ids=input_ids, labels=labels).loss
    weighted = model(input_ids=input_ids, labels=labels, class_weights=weights).loss

    assert not torch.isclose(unweighted, weighted)


def test_padding_does_not_change_a_clips_prediction():
    # The masked mean pool must ignore padded positions, so appending padding to a clip has
    # to leave its logits alone. Without the mask, padding would dilute the pooled vector.
    model = _model()
    clip = torch.randint(3, 24, (1, 8))
    padded = torch.cat([clip, torch.zeros((1, 6), dtype=torch.long)], dim=1)

    unpadded_logits = model(input_ids=clip, attention_mask=torch.ones_like(clip)).logits
    padded_logits = model(
        input_ids=padded,
        attention_mask=torch.cat([torch.ones_like(clip), torch.zeros((1, 6), dtype=torch.long)], dim=1),
    ).logits

    assert torch.allclose(unpadded_logits, padded_logits, atol=1e-5)


def test_prediction_depends_on_later_tokens():
    # The probe is bidirectional: unlike the generator, position 0 must be able to see the
    # end of the clip, so changing the last token has to move the prediction.
    model = _model()
    first = torch.tensor([[3, 4, 5, 6, 7, 8]])
    second = torch.tensor([[3, 4, 5, 6, 7, 23]])

    assert not torch.allclose(model(input_ids=first).logits, model(input_ids=second).logits)


def test_predict_proba_returns_normalised_rows():
    model = _model()
    input_ids = torch.randint(3, 24, (3, 9))

    probabilities = model.predict_proba(input_ids=input_ids)

    assert probabilities.shape == (3, 4)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert bool((probabilities >= 0).all())


def test_sequence_longer_than_max_seq_len_is_rejected():
    model = _model()

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(input_ids=torch.randint(3, 24, (1, 33)))


def _encoder_shape(module):
    """The encoder-core structure both models are expected to share."""
    layer = module.transformer.layers[0]
    return {
        "keys": {
            key
            for key in module.state_dict()
            if key.startswith(("token_embedding", "position_embedding", "transformer", "final_norm"))
        },
        "d_model": layer.linear1.in_features,
        "feedforward": layer.linear1.out_features,
        "heads": layer.self_attn.num_heads,
        "layers": len(module.transformer.layers),
        "norm": tuple(module.final_norm.normalized_shape),
        "activation": getattr(layer.activation, "__name__", type(layer.activation).__name__),
        "batch_first": layer.self_attn.batch_first,
    }


def test_the_two_models_keep_the_same_encoder_core():
    """MusicClassifier deliberately restates MusicTransformer's encoder rather than sharing it.

    Extracting the construction would reorder the submodules created in
    ``MusicTransformer.__init__``, changing which random draws each one consumes and therefore
    the weights a fixed seed produces - which would invalidate the golden token sequences that
    prove ``sample`` and ``sample_batch`` agree. The duplication is accepted for that reason, so
    this test carries the guarantee instead: a change to one encoder core that is not mirrored in
    the other fails here.
    """
    from musemotion.models.music_transformer import MusicTransformer, MusicTransformerConfig

    shared = dict(vocab_size=24, max_seq_len=32, d_model=16, n_heads=2, n_layers=1, dropout=0.0)
    generator = MusicTransformer(MusicTransformerConfig(**shared))
    probe = MusicClassifier(MusicClassifierConfig(**shared))

    assert _encoder_shape(probe) == _encoder_shape(generator)


def test_both_models_reject_a_head_count_that_does_not_divide_the_width():
    """The shared precondition is restated in both, so both are checked."""
    from musemotion.models.music_transformer import MusicTransformer, MusicTransformerConfig

    with pytest.raises(ValueError, match="divisible"):
        MusicTransformer(MusicTransformerConfig(vocab_size=24, d_model=15, n_heads=2))
    with pytest.raises(ValueError, match="divisible"):
        MusicClassifier(MusicClassifierConfig(vocab_size=24, d_model=15, n_heads=2))


def test_the_committed_generator_checkpoint_still_loads():
    """The shipped checkpoint must survive any change to the generator's module layout.

    state_dict keys follow the attribute names, so renaming or nesting a submodule would break
    loading the committed weights - the failure that made extracting a shared encoder unattractive.
    """
    import torch

    from musemotion.config import resolve_path
    from musemotion.models.music_transformer import MusicTransformer, MusicTransformerConfig

    path = resolve_path("models/real_training/music_transformer_fulldata/best.pt")
    if not path.exists():
        pytest.skip("committed generator checkpoint not present")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = MusicTransformer(MusicTransformerConfig(**payload["model_config"]))
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True), None

    assert missing.missing_keys == [] and missing.unexpected_keys == []
    assert unexpected is None
