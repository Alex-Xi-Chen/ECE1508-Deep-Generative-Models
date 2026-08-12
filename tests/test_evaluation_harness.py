"""Harness tests that avoid the trained checkpoints entirely.

``run_evaluation`` needs the real generator, classifier, and probes, so what is exercised here
is the scoring layer it delegates to: clip bookkeeping, the reference loader, and
``_score_system`` driven by fake probes whose answers are fixed in advance.
"""
import json

import numpy as np

from musemotion.evaluation.features import feature_matrix
from musemotion.evaluation.harness import GeneratedClip, _score_system
from musemotion.evaluation.probe import load_clip_splits
from musemotion.evaluation.distributions import comparable_feature_names
from musemotion.evaluation.novelty import SYMBOL_VIEWS, build_corpus_index
from musemotion.music.tokenizer import MidiNote, MusicTokenizer


from conftest import build_clip as _clip_notes


class PerfectProbe:
    """Always recovers the quadrant that conditioned the clip."""

    def __init__(self, emotion_ids):
        self._emotion_ids = list(emotion_ids)

    def predict_proba_batch(self, clips):
        rows = np.full((len(clips), 4), 0.05)
        for index, emotion_id in enumerate(self._emotion_ids[: len(clips)]):
            rows[index, max(0, emotion_id)] = 0.85
        return rows


class StubbornProbe:
    """Always answers Q1, whatever it is shown."""

    def predict_proba_batch(self, clips):
        rows = np.full((len(clips), 4), 0.05)
        rows[:, 0] = 0.85
        return rows


def _corpus_and_reference(clips):
    corpus = {view: build_corpus_index(clips, view) for view in SYMBOL_VIEWS}
    reference = feature_matrix(clips, comparable_feature_names())
    return corpus, reference


def _generated(emotion_ids, tokenizer):
    clips = []
    for offset, emotion_id in enumerate(emotion_ids):
        notes = _clip_notes([60 + offset, 63 + offset, 67 + offset, 70 + offset])
        clips.append(
            GeneratedClip(
                emotion_id=emotion_id, notes=notes, token_ids=tokenizer.encode_notes(notes)
            )
        )
    return clips


def test_generated_clip_detects_eos_and_summarises_structure():
    tokenizer = MusicTokenizer()
    notes = _clip_notes([60, 62, 64])
    with_eos = GeneratedClip(
        emotion_id=0,
        notes=notes,
        token_ids=tokenizer.encode_notes(notes),
        eos_token_id=tokenizer.eos_token_id,
    )
    without_eos = GeneratedClip(emotion_id=0, notes=notes, token_ids=[1, 5, 6, 7])

    assert with_eos.emitted_eos is True
    assert without_eos.emitted_eos is False

    record = with_eos.structure_record()
    assert record["note_count"] == 3
    assert record["token_count"] == len(with_eos.token_ids)
    assert "repeated_pitch_fraction" in record


def test_eos_detection_follows_the_configured_token_id():
    # The check must read the tokenizer's EOS id rather than assume the special tokens keep
    # their current order.
    assert GeneratedClip(emotion_id=0, notes=[], token_ids=[1, 5, 9], eos_token_id=9).emitted_eos
    assert not GeneratedClip(emotion_id=0, notes=[], token_ids=[1, 5, 9], eos_token_id=2).emitted_eos
    assert not GeneratedClip(emotion_id=0, notes=[], token_ids=[], eos_token_id=2).emitted_eos


def test_score_system_reports_perfect_round_trip_for_a_perfect_probe():
    tokenizer = MusicTokenizer()
    emotion_ids = [0, 1, 2, 3]
    clips = _generated(emotion_ids, tokenizer)
    corpus, reference = _corpus_and_reference([clip.notes for clip in clips])

    payload = _score_system(
        clips,
        {"feature": PerfectProbe(emotion_ids)},
        {"feature": 0.65},
        corpus,
        reference,
        max_tokens=128,
        config={},
    )

    round_trip = payload["round_trip"]
    assert round_trip["overall"]["accuracy"] == 1.0
    assert round_trip["axes"]["arousal"]["accuracy"] == 1.0
    assert round_trip["axes"]["valence"]["accuracy"] == 1.0
    # Accuracy above the ceiling is expressible, which matters because it really happens.
    assert round_trip["controllability_ratio"] > 1.0
    assert payload["primary_probe"] == "feature"


def test_score_system_reports_chance_level_axes_for_a_single_answer_probe():
    tokenizer = MusicTokenizer()
    emotion_ids = [0, 1, 2, 3]
    clips = _generated(emotion_ids, tokenizer)
    corpus, reference = _corpus_and_reference([clip.notes for clip in clips])

    payload = _score_system(
        clips, {"feature": StubbornProbe()}, {"feature": 0.65}, corpus, reference, 128, {}
    )

    round_trip = payload["round_trip"]
    assert round_trip["overall"]["accuracy"] == 0.25
    # Answering Q1 for everything gets both binary axes right exactly half the time.
    assert round_trip["axes"]["arousal"]["accuracy"] == 0.5
    assert round_trip["axes"]["valence"]["accuracy"] == 0.5
    assert round_trip["overall"]["beats_chance"] is False


def test_score_system_records_probe_agreement_when_two_probes_disagree():
    tokenizer = MusicTokenizer()
    emotion_ids = [0, 1, 2, 3]
    clips = _generated(emotion_ids, tokenizer)
    corpus, reference = _corpus_and_reference([clip.notes for clip in clips])

    payload = _score_system(
        clips,
        {"feature": PerfectProbe(emotion_ids), "neural": StubbornProbe()},
        {"feature": 0.65, "neural": 0.55},
        corpus,
        reference,
        128,
        {},
    )

    # They coincide only on the one clip that really is Q1.
    assert payload["probe_agreement"] == 0.25
    assert set(payload["round_trip_by_probe"]) == {"feature", "neural"}


def test_score_system_skips_round_trip_when_there_is_no_conditioning_quadrant():
    tokenizer = MusicTokenizer()
    clips = _generated([-1, -1, -1], tokenizer)
    corpus, reference = _corpus_and_reference([clip.notes for clip in clips])

    payload = _score_system(
        clips,
        {"feature": StubbornProbe()},
        {"feature": 0.65},
        corpus,
        reference,
        128,
        {},
        score_round_trip=False,
    )

    assert "round_trip" not in payload
    # What the probe assigns is still reportable, as a distribution.
    assert payload["probe_label_distribution"]["feature"]["Q1"] == 1.0


def test_score_system_flags_clips_copied_from_the_corpus():
    tokenizer = MusicTokenizer()
    corpus_notes = [_clip_notes([60, 62, 64, 65, 67, 69]), _clip_notes([48, 50, 52, 53])]
    corpus, reference = _corpus_and_reference(corpus_notes)
    copied = [
        GeneratedClip(
            emotion_id=0, notes=corpus_notes[0], token_ids=tokenizer.encode_notes(corpus_notes[0])
        )
    ]

    payload = _score_system(
        copied, {"feature": StubbornProbe()}, {"feature": 0.65}, corpus, reference, 128,
        {"novelty": {"ngram_sizes": [4], "diversity_sizes": [2]}},
    )

    assert payload["novelty"]["pitch"]["mean_novel_ngram_rate"]["4"] == 0.0
    assert payload["novelty"]["pitch"]["exact_duplicate_rate"] == 1.0
    assert payload["novelty"]["pitch"]["max_longest_copied_run"] == len(corpus_notes[0])


def test_score_system_reports_high_novelty_for_unrelated_clips():
    tokenizer = MusicTokenizer()
    corpus_notes = [_clip_notes([60, 62, 64, 65, 67, 69])]
    corpus, reference = _corpus_and_reference(corpus_notes)
    unrelated = [
        GeneratedClip(
            emotion_id=0,
            notes=_clip_notes([21, 108, 22, 107, 23, 106]),
            token_ids=tokenizer.encode_notes(_clip_notes([21, 108, 22, 107])),
        )
    ]

    payload = _score_system(
        unrelated, {"feature": StubbornProbe()}, {"feature": 0.65}, corpus, reference, 128,
        {"novelty": {"ngram_sizes": [4], "diversity_sizes": [2]}},
    )

    assert payload["novelty"]["pitch"]["mean_novel_ngram_rate"]["4"] == 1.0
    assert payload["novelty"]["pitch"]["exact_duplicate_rate"] == 0.0


def test_score_system_includes_fidelity_and_per_quadrant_features():
    tokenizer = MusicTokenizer()
    emotion_ids = [0, 0, 3, 3]
    clips = _generated(emotion_ids, tokenizer)
    corpus, reference = _corpus_and_reference([clip.notes for clip in clips])

    payload = _score_system(
        clips, {"feature": PerfectProbe(emotion_ids)}, {"feature": 0.65}, corpus, reference, 128, {}
    )

    fidelity = payload["fidelity"]
    assert 0.0 <= fidelity["mean_overlap"] <= 1.0
    assert "note_count" not in fidelity["feature_names"]  # length-dependent, excluded
    assert set(payload["per_quadrant_features"]) == {"Q1", "Q4"}
    assert payload["per_quadrant_features"]["Q1"]["clip_count"] == 2


def test_load_clip_splits_crops_and_reencodes(tmp_path):
    tokenizer = MusicTokenizer()
    notes = _clip_notes(list(range(60, 80)))
    (tmp_path / "test.jsonl").write_text(
        json.dumps({"emotion_id": 2, "quadrant": "Q3", "token_ids": tokenizer.encode_notes(notes)}) + "\n",
        encoding="utf-8",
    )

    splits = load_clip_splits(tmp_path, tokenizer, note_budget=5)

    assert list(splits) == ["test"]
    row = splits["test"][0]
    assert len(row["notes"]) == 5
    # Re-encoded rather than truncated: BOS, five whole note groups, EOS.
    assert len(row["token_ids"]) == 5 * 4 + 2


def test_load_clip_splits_ignores_missing_splits_and_empty_clips(tmp_path):
    tokenizer = MusicTokenizer()
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"emotion_id": 0, "token_ids": [1, 2]}) + "\n\n", encoding="utf-8"
    )

    splits = load_clip_splits(tmp_path, tokenizer, note_budget=8)

    assert splits["train"] == []
    assert "validation" not in splits


def test_headline_novelty_survives_a_configured_ngram_size_that_excludes_eight():
    """ngram_sizes is configurable; the headline column must not silently blank.

    Keying the headline on a hardcoded "8" returned None for any other configuration, which flowed
    into the CSV as an empty cell, the markdown as a dash, and dropped every bar from the figure -
    with nothing raising.
    """
    tokenizer = MusicTokenizer()
    clips = _generated([0, 1], tokenizer)
    corpus, reference = _corpus_and_reference([clip.notes for clip in clips])

    payload = _score_system(
        clips, {"feature": StubbornProbe()}, {"feature": 0.65}, corpus, reference, 128,
        {"novelty": {"ngram_sizes": [4, 16], "diversity_sizes": [1]}},
    )

    novelty = payload["novelty"]
    assert novelty["mean_novel_8gram_rate"] is not None
    # 4 and 16 are equidistant from 8; the smaller is chosen deterministically.
    assert novelty["headline_ngram_size"] == 4
    assert novelty["mean_novel_8gram_rate"] == novelty["pitch"]["mean_novel_ngram_rate"]["4"]


def test_headline_diversity_falls_back_to_a_computed_size():
    tokenizer = MusicTokenizer()
    clips = _generated([0, 1, 2], tokenizer)
    corpus, reference = _corpus_and_reference([clip.notes for clip in clips])

    payload = _score_system(
        clips, {"feature": StubbornProbe()}, {"feature": 0.65}, corpus, reference, 128,
        # headline_diversity_n defaults to 2, which is not among the configured sizes.
        {"novelty": {"ngram_sizes": [4], "diversity_sizes": [1]}},
    )

    assert payload["diversity"]["mean_pairwise_cosine_diversity"] is not None
    assert payload["diversity"]["headline_n"] == 1
