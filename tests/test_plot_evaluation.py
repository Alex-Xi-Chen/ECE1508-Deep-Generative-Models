"""Figure tests.

Nothing imported this module before, which is exactly why renaming a metrics field could leave a
live reader behind and break the stage-attribution figure on every new run while still working on
the committed artifacts.

The metrics payload here is assembled by the real report and harness functions rather than
hand-written, so a schema change breaks these tests at the moment it is made.
"""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from conftest import build_clip
from musemotion.evaluation.distributions import comparable_feature_names
from musemotion.evaluation.features import feature_matrix
from musemotion.evaluation.harness import GeneratedClip, _score_system
from musemotion.evaluation.novelty import SYMBOL_VIEWS, build_corpus_index
from musemotion.evaluation.report import comparison_row, stage_attribution
from musemotion.music.tokenizer import MusicTokenizer
from plot_evaluation import FIGURE_BUILDERS, build_all, stage_attribution_figure


class _Probe:
    """Recovers the conditioning quadrant for every clip except the last, which it misreads.

    Deliberately imperfect. A probe that recovered everything would give every figure a payload
    of 1.0 accuracies with an empty off-diagonal, so the confusion heatmap, the interval whiskers
    and the failure-attribution segments would all be exercised only in their degenerate form.
    """

    def predict_proba_batch(self, clips):
        rows = np.full((len(clips), 4), 0.05)
        for index in range(len(clips)):
            recovered = index % 4
            if index == len(clips) - 1:
                recovered = (recovered + 1) % 4
            rows[index, recovered] = 0.85
        return rows


def _metrics():
    tokenizer = MusicTokenizer()
    emotion_ids = [0, 1, 2, 3] * 3
    clips = [
        GeneratedClip(
            emotion_id=emotion_id,
            notes=build_clip([60 + offset, 64 + offset, 67 + offset, 71 + offset]),
            token_ids=tokenizer.encode_notes(build_clip([60 + offset, 64 + offset])),
            eos_token_id=tokenizer.eos_token_id,
        )
        for offset, emotion_id in enumerate(emotion_ids)
    ]
    corpus = {view: build_corpus_index([c.notes for c in clips], view) for view in SYMBOL_VIEWS}
    reference = feature_matrix([c.notes for c in clips], comparable_feature_names())
    probes = {"feature": _Probe(), "neural": _Probe()}

    systems = {}
    for label in ("real", "guidance=1", "guidance=3", "end_to_end"):
        systems[label] = _score_system(
            clips, probes, {"feature": 0.65, "neural": 0.6}, corpus, reference, 128, {},
            eos_is_learned=(label != "real"),
        )

    # The classifier misses two prompts, and the probe still recovers the intended quadrant for
    # both - the accidental-recovery case. Without it music_only stays 0, and the
    # conditional_product_equals_end_to_end False branch is never exercised by any figure test.
    intended = [c.emotion_id for c in clips]
    classified = list(intended)
    classified[1] = (classified[1] + 1) % 4
    classified[5] = (classified[5] + 2) % 4
    recovered = [index % 4 for index in range(len(clips))]
    return {
        "run": {"config": {"generation": {"guidance_scale": 3.0}}},
        "probe_metadata": {
            "majority_class_accuracy": 0.315,
            "probes": {
                "feature": {"test": {"accuracy": 0.657},
                            "shuffled_label_control": {"test": {"accuracy": 0.231}}},
                "neural": {"test": {"accuracy": 0.602},
                           "shuffled_label_control": {"test": {"accuracy": 0.324}}},
            },
        },
        "systems": systems,
        "comparison_table": [comparison_row(name, payload) for name, payload in systems.items()],
        "end_to_end": {
            "stage_attribution": {
                name: stage_attribution(intended, classified, recovered,
                                        balanced_generator_accuracy=0.79)
                for name in probes
            }
        },
    }


def test_every_figure_builds_from_a_freshly_produced_metrics_payload():
    """strict=True so a schema mismatch raises instead of printing FAILED and scrolling past."""
    figures = build_all(_metrics(), strict=True)

    assert set(figures) == set(FIGURE_BUILDERS)


def test_stage_attribution_figure_reads_the_current_field_names():
    """The rename that broke this read conditional_product where predicted_product used to be."""
    metrics = _metrics()
    payload = metrics["end_to_end"]["stage_attribution"]["feature"]

    assert "conditional_product" in payload
    assert "predicted_product" not in payload
    figure = stage_attribution_figure(metrics)

    # The title is set with loc="left", which get_title() does not return by default.
    title = figure.axes[0].get_title(loc="left")
    assert "conditional product" in title
    assert "quadrant-mix ratio" in title


def test_stage_attribution_figure_still_reads_older_runs():
    """Metrics files written before the rename must keep plotting."""
    metrics = _metrics()
    legacy = dict(metrics["end_to_end"]["stage_attribution"]["feature"])
    legacy["predicted_product"] = legacy.pop("conditional_product")
    legacy["independence_ratio"] = 1.0
    legacy.pop("quadrant_mix_ratio")
    metrics["end_to_end"]["stage_attribution"] = {"feature": legacy}

    title = stage_attribution_figure(metrics).axes[0].get_title(loc="left")

    assert "conditional product" in title
    assert "legacy ratio" in title


def test_a_figure_whose_data_is_absent_is_skipped_not_failed():
    metrics = _metrics()
    metrics["end_to_end"] = {}

    with pytest.raises(ValueError):
        stage_attribution_figure(metrics)
    # build_all treats that as "no data for this run" rather than a defect.
    assert "stage_attribution" not in build_all(metrics, strict=True)


def test_the_fixture_is_not_degenerate():
    """Guards the guard: figures must be exercised on a payload with real failures in it.

    An all-perfect payload would still build every figure while leaving the off-diagonal of the
    confusion matrix, the interval whiskers, and three of the four attribution segments empty.
    """
    metrics = _metrics()

    round_trip = metrics["systems"]["guidance=3"]["round_trip"]
    assert round_trip["overall"]["accuracy"] < 1.0
    assert round_trip["overall"]["ci_low"] < round_trip["overall"]["ci_high"]
    # Something must sit off the diagonal for the heatmap to be meaningfully rendered.
    confusion = round_trip["confusion_matrix"]
    assert any(confusion[i][j] for i in range(4) for j in range(4) if i != j)

    attribution = metrics["end_to_end"]["stage_attribution"]["feature"]
    counts = attribution["counts"]
    assert counts["text_wrong_but_recovered_intended"] > 0
    assert counts["text_correct_and_recovered"] > 0
    # The accidental-recovery branch, which an all-correct classifier never reaches.
    assert attribution["conditional_product_equals_end_to_end"] is False
    assert attribution["lucky_recovery_rate"] > 0.0
