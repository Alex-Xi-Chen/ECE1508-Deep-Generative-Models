"""Tests for the probe training layer that need no dataset and no GPU.

The shuffled-label control is the project's trust argument, so the property that makes it a
valid control - that nothing it selects on carries real labels - is asserted here rather than
left to be re-derived by reading.
"""
import numpy as np
import pytest

from conftest import build_clip
from musemotion.training.classifier import balanced_class_weights, compute_classifier_metrics
from musemotion.training.probe import _permuted_splits, _majority_class_rate


def _rows(emotion_ids):
    return [
        {"emotion_id": emotion_id, "notes": build_clip([60 + emotion_id, 64, 67]), "token_ids": []}
        for emotion_id in emotion_ids
    ]


def test_control_permutes_validation_as_well_as_train():
    """Selection happens on validation macro-F1.

    Permuting only the training labels would let the control keep choosing the epoch that best
    predicts the very labels it is supposed to have no access to - a contaminated control that
    still looks clean.
    """
    splits = {
        "train": _rows([0, 1, 2, 3] * 4),
        "validation": _rows([0, 1, 2, 3] * 3),
        "test": _rows([0, 1, 2, 3] * 2),
    }

    permuted = _permuted_splits(splits, seed=1508)

    for name in ("train", "validation"):
        original = [row["emotion_id"] for row in splits[name]]
        shuffled = [row["emotion_id"] for row in permuted[name]]
        assert sorted(shuffled) == sorted(original), f"{name} must be a permutation"
        assert shuffled != original, f"{name} labels were not permuted"


def test_control_leaves_test_labels_untouched():
    """The control's question is whether a probe fitted on noise recovers *real* emotion."""
    splits = {"train": _rows([0, 1, 2, 3] * 4), "test": _rows([0, 1, 2, 3] * 2)}

    permuted = _permuted_splits(splits, seed=7)

    assert [row["emotion_id"] for row in permuted["test"]] == [
        row["emotion_id"] for row in splits["test"]
    ]


def test_permutation_does_not_mutate_the_original_rows():
    splits = {"train": _rows([0, 1, 2, 3] * 4)}
    before = [row["emotion_id"] for row in splits["train"]]

    _permuted_splits(splits, seed=3)

    assert [row["emotion_id"] for row in splits["train"]] == before


def test_no_seed_returns_the_splits_unchanged():
    splits = {"train": _rows([0, 1])}

    assert _permuted_splits(splits, seed=None) is splits


def test_majority_class_rate_is_the_bar_a_control_must_clear():
    # The committed test split: always answering Q4 scores 0.315, above uniform 0.25.
    assert _majority_class_rate({"Q1": 20, "Q2": 24, "Q3": 30, "Q4": 34}) == pytest.approx(34 / 108)
    assert _majority_class_rate({}) == 0.0


def test_macro_f1_averages_over_every_quadrant_when_a_label_set_is_given():
    """Two runs must be averaged over the same number of classes to be comparable.

    scikit-learn's macro average spans the union of the true and predicted labels, so a class
    missing only from the predictions is still counted. The averages diverge when a quadrant is
    absent from the *truth* as well - a small or skewed evaluation slice - and then a run scored
    over two classes gets printed beside one scored over four as though they were the same
    statistic. Pinning the label set removes that possibility entirely.
    """
    # Only two quadrants appear in the truth, and the run predicts Q1 for everything.
    logits = np.zeros((4, 4))
    logits[:, 0] = 1.0
    labels = np.array([0, 0, 1, 1])

    observed_only = compute_classifier_metrics((logits, labels))
    all_quadrants = compute_classifier_metrics((logits, labels), label_set=range(4))

    assert all_quadrants["accuracy"] == observed_only["accuracy"] == 0.5
    # Averaged over four quadrants rather than the two present, the score is lower and honest.
    assert all_quadrants["macro_f1"] < observed_only["macro_f1"]
    assert all_quadrants["macro_f1"] == pytest.approx(observed_only["macro_f1"] / 2)


def test_macro_f1_with_a_label_set_never_raises_on_an_absent_class():
    """zero_division=0 accompanies the label set: an unpredicted quadrant scores 0, not NaN."""
    logits = np.zeros((2, 4))
    logits[:, 0] = 1.0

    metrics = compute_classifier_metrics((logits, np.array([0, 0])), label_set=range(4))

    assert np.isfinite(metrics["macro_f1"])
    assert metrics["macro_f1"] == pytest.approx(0.25)


def test_balanced_class_weights_are_inverse_frequency_and_survive_empty_classes():
    weights = balanced_class_weights([0, 0, 0, 1], num_classes=4)

    assert weights[1] > weights[0]  # the rarer class is upweighted
    assert np.all(np.isfinite(weights))  # quadrants with no examples must not divide by zero
