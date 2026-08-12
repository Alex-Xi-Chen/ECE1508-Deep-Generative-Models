import pytest
import numpy as np
import torch

from musemotion.evaluation.probe import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NOTE_BUDGET,
    FeatureMidiProbe,
    NeuralMidiProbe,
    crop_notes,
    load_probe_metadata,
    load_probes,
    probe_agreement,
)
from musemotion.models.music_classifier import MusicClassifier, MusicClassifierConfig
from musemotion.music.dataset import collate_probe_batch
from musemotion.music.tokenizer import MidiNote, MusicTokenizer


from conftest import build_clip as _clip


def _neural_probe(note_budget=8):
    tokenizer = MusicTokenizer()
    model = MusicClassifier(
        MusicClassifierConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=note_budget * 4 + 2,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    torch.manual_seed(0)
    return NeuralMidiProbe(model=model, tokenizer=tokenizer, device="cpu", note_budget=note_budget)


def _fitted_pipeline(labels_override=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Four synthetic classes separated by note density and pitch, so the fit is trivial.
    from musemotion.evaluation.features import feature_matrix

    clips = []
    labels = []
    for emotion_id, (pitch, step) in enumerate([(72, 0.1), (48, 0.1), (72, 1.0), (48, 1.0)]):
        for offset in range(4):
            clips.append(_clip([pitch + offset, pitch + 2 + offset, pitch + 4 + offset], step=step))
            labels.append(emotion_id)
    matrix = feature_matrix(clips)
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced"))]
    )
    pipeline.fit(matrix, np.asarray(labels_override if labels_override is not None else labels))
    return pipeline, clips


def _feature_probe():
    pipeline, _ = _fitted_pipeline()
    return FeatureMidiProbe.from_pipeline(pipeline, note_budget=DEFAULT_NOTE_BUDGET)


def test_feature_probe_reproduces_sklearn_predict_proba():
    """The stored numbers must score identically to the estimator they came from.

    The probe keeps the scaler statistics and linear coefficients instead of a pickled
    estimator, so scoring is reimplemented in numpy. That is only safe if it agrees with
    scikit-learn to floating-point precision, which is what this pins.
    """
    from musemotion.evaluation.features import feature_matrix

    pipeline, clips = _fitted_pipeline()
    probe = FeatureMidiProbe.from_pipeline(pipeline, note_budget=DEFAULT_NOTE_BUDGET)

    expected = pipeline.predict_proba(feature_matrix(clips, probe.feature_names))
    actual = probe.predict_proba_batch(clips)

    # Columns are remapped onto quadrant ids, so compare per class rather than positionally.
    for column, emotion_id in enumerate(probe.classes):
        assert np.allclose(actual[:, emotion_id], expected[:, column], atol=1e-10)


def test_feature_probe_json_round_trip_is_exact(tmp_path):
    probe = _feature_probe()
    clips = [_clip([60, 62, 64]), _clip([48, 50, 52], step=1.0)]
    probe.save(tmp_path)

    reloaded = FeatureMidiProbe.from_artifacts(tmp_path)

    assert reloaded.feature_names == probe.feature_names
    assert reloaded.classes == probe.classes
    assert np.allclose(reloaded.predict_proba_batch(clips), probe.predict_proba_batch(clips), atol=1e-12)


def test_feature_probe_artifact_is_plain_readable_json(tmp_path):
    """Loading must not execute code, so the artifact is JSON rather than a pickle."""
    import json

    path = _feature_probe().save(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.suffix == ".json"
    assert set(payload) == {
        "feature_names",
        "note_budget",
        "classes",
        "scaler_mean",
        "scaler_scale",
        "coefficients",
        "intercepts",
    }


def test_binary_fit_produces_two_probability_columns():
    """A split with only two quadrants yields one score column; it must still sum to one."""
    pipeline, clips = _fitted_pipeline(labels_override=[0] * 8 + [3] * 8)
    probe = FeatureMidiProbe.from_pipeline(pipeline, note_budget=DEFAULT_NOTE_BUDGET)

    probabilities = probe.predict_proba_batch(clips)

    assert probabilities.shape == (len(clips), 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    expected = pipeline.predict_proba(
        __import__("musemotion.evaluation.features", fromlist=["feature_matrix"]).feature_matrix(
            clips, probe.feature_names
        )
    )
    for column, emotion_id in enumerate(probe.classes):
        assert np.allclose(probabilities[:, emotion_id], expected[:, column], atol=1e-10)


def test_default_sequence_limit_holds_the_default_note_budget_exactly():
    tokenizer = MusicTokenizer()
    notes = _clip(list(range(60, 60 + DEFAULT_NOTE_BUDGET)))

    encoded = tokenizer.encode_notes(notes)

    assert len(encoded) == DEFAULT_MAX_SEQ_LEN


def test_crop_notes_keeps_the_head_in_onset_order():
    notes = _clip([64, 60, 67, 62])

    cropped = crop_notes(notes, 2)

    assert len(cropped) == 2
    assert [note.start for note in cropped] == [0.0, 0.25]
    assert crop_notes(notes, None) == sorted(notes, key=lambda note: (note.start, note.pitch))


def test_neural_probe_returns_a_normalised_four_vector():
    probe = _neural_probe()

    probabilities = probe.predict_proba_notes(_clip([60, 62, 64, 65]))

    assert probabilities.shape == (4,)
    assert np.isclose(probabilities.sum(), 1.0)


def test_neural_probe_prediction_dict_matches_the_classifier_contract():
    probe = _neural_probe()

    prediction = probe.predict_notes(_clip([60, 62, 64]))

    assert set(prediction) == {"quadrant", "emotion_id", "confidence", "probabilities"}
    assert prediction["quadrant"] in {"Q1", "Q2", "Q3", "Q4"}
    assert set(prediction["probabilities"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert 0.0 <= prediction["confidence"] <= 1.0


def test_neural_probe_batches_agree_with_single_clip_scoring():
    probe = _neural_probe()
    clips = [_clip([60, 62, 64]), _clip([48, 50, 52, 53])]

    batched = probe.predict_proba_batch(clips)
    single = np.vstack([probe.predict_proba_notes(clip) for clip in clips])

    assert batched.shape == (2, 4)
    assert np.allclose(batched, single, atol=1e-5)


def test_neural_probe_handles_an_empty_clip():
    probe = _neural_probe()

    probabilities = probe.predict_proba_notes([])

    assert probabilities.shape == (4,)
    assert np.isclose(probabilities.sum(), 1.0)


def test_neural_probe_truncates_clips_beyond_its_budget():
    # A clip longer than the budget must still score rather than overflow the position table.
    probe = _neural_probe(note_budget=8)

    probabilities = probe.predict_proba_notes(_clip(list(range(40, 100))))

    assert probabilities.shape == (4,)


def test_empty_batch_returns_no_rows():
    assert _neural_probe().predict_proba_batch([]).shape == (0, 4)
    assert _feature_probe().predict_proba_batch([]).shape == (0, 4)


def test_feature_probe_returns_normalised_rows_and_exposes_coefficients():
    probe = _feature_probe()

    probabilities = probe.predict_proba_batch([_clip([72, 74, 76], step=0.1)])
    coefficients = probe.coefficients()

    assert probabilities.shape == (1, 4)
    assert np.isclose(probabilities.sum(), 1.0)
    assert set(coefficients) == {"Q1", "Q2", "Q3", "Q4"}
    assert "note_density" in coefficients["Q1"]


def test_feature_probe_columns_align_to_quadrant_ids_not_sklearn_order():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from musemotion.evaluation.features import feature_matrix

    # Only Q2 and Q4 appear in training, so sklearn's two columns are classes [1, 3].
    clips = [_clip([48, 50, 52], step=0.1)] * 3 + [_clip([72, 74, 76], step=1.0)] * 3
    labels = np.asarray([1, 1, 1, 3, 3, 3])
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500))])
    pipeline.fit(feature_matrix(clips), labels)
    probe = FeatureMidiProbe.from_pipeline(pipeline)

    probabilities = probe.predict_proba_batch([_clip([48, 50, 52], step=0.1)])

    assert probabilities.shape == (1, 4)
    # Unseen quadrants must sit at exactly zero rather than absorbing another class's column.
    assert probabilities[0][0] == 0.0
    assert probabilities[0][2] == 0.0
    assert probabilities[0][1] > 0.0


def test_probes_round_trip_through_save_and_load(tmp_path):
    neural = _neural_probe()
    feature = _feature_probe()
    clip = _clip([60, 62, 64, 65])
    neural.save(tmp_path)
    feature.save(tmp_path)

    loaded = load_probes(tmp_path, device="cpu", tokenizer_path=tmp_path / "tokenizer.json")

    assert set(loaded) == {"neural", "feature"}
    assert np.allclose(loaded["neural"].predict_proba_notes(clip), neural.predict_proba_notes(clip), atol=1e-5)
    assert np.allclose(loaded["feature"].predict_proba_notes(clip), feature.predict_proba_notes(clip), atol=1e-6)


def test_load_probes_returns_nothing_when_no_artifacts_exist(tmp_path):
    assert load_probes(tmp_path) == {}
    assert load_probe_metadata(tmp_path) == {}


def test_probe_agreement_counts_matching_predictions():
    assert probe_agreement([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert probe_agreement([0, 1, 2, 3], [0, 1, 2, 0]) == 0.75
    assert probe_agreement([], []) == 0.0
    assert probe_agreement([0, 1], [0]) == 0.0


def test_collate_probe_batch_pads_and_labels_without_shifting():
    tokenizer = MusicTokenizer()
    examples = [
        {"token_ids": [1, 5, 6, 7, 8, 2], "emotion_id": 0},
        {"token_ids": [1, 5, 6, 7, 8, 9, 10, 11, 12, 2], "emotion_id": 3},
    ]

    batch = collate_probe_batch(examples, pad_token_id=tokenizer.pad_token_id)

    assert batch["input_ids"].shape == (2, 10)
    assert batch["attention_mask"].shape == (2, 10)
    assert batch["labels"].tolist() == [0, 3]
    # The shorter clip keeps its own tokens intact and is padded on the right.
    assert batch["input_ids"][0].tolist()[:6] == [1, 5, 6, 7, 8, 2]
    assert batch["attention_mask"][0].tolist() == [1] * 6 + [0] * 4


def test_collate_probe_batch_head_truncates_to_the_limit():
    batch = collate_probe_batch(
        [{"token_ids": list(range(1, 21)), "emotion_id": 1}], pad_token_id=0, max_seq_len=6
    )

    assert batch["input_ids"].shape == (1, 6)
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4, 5, 6]


def test_coefficients_on_a_binary_fit_returns_a_row_per_quadrant():
    """A two-class fit has one coefficient row for two classes, not one row per class.

    Indexing by class position runs off the end of that matrix. This is the path the probability
    test already exercises, so the failure would have surfaced only after a real training run had
    already fitted the probe.
    """
    pipeline, _ = _fitted_pipeline(labels_override=[0] * 8 + [3] * 8)
    probe = FeatureMidiProbe.from_pipeline(pipeline)

    assert probe.coefficients_matrix.shape[0] == 1
    coefficients = probe.coefficients()

    assert set(coefficients) == {"Q1", "Q4"}
    # The single row describes the second class; the first is its negation.
    for name in probe.feature_names:
        assert coefficients["Q1"][name] == pytest.approx(-coefficients["Q4"][name])


def test_coefficients_on_a_four_class_fit_covers_every_quadrant():
    coefficients = _feature_probe().coefficients()

    assert set(coefficients) == {"Q1", "Q2", "Q3", "Q4"}
    for row in coefficients.values():
        assert set(row) == set(_feature_probe().feature_names)
