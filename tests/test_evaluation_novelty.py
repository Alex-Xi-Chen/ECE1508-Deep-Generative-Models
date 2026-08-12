import pytest

from musemotion.evaluation.novelty import (
    build_corpus_index,
    distinct_ngram_ratio,
    interval_symbols,
    pitch_duration_symbols,
    set_diversity,
)
from musemotion.music.tokenizer import MidiNote


from conftest import build_clip as _clip


CORPUS_PITCHES = [
    [60, 62, 64, 65, 67, 69, 71, 72],
    [48, 50, 52, 53, 55, 57, 59, 60],
    [72, 71, 69, 67, 65, 64, 62, 60],
]
CORPUS = [_clip(pitches) for pitches in CORPUS_PITCHES]


def test_verbatim_copy_scores_zero_novelty_and_full_run_length():
    index = build_corpus_index(CORPUS, "pitch")
    symbols = pitch_duration_symbols(CORPUS[0])

    assert index.novel_ngram_rate(symbols, 4) == 0.0
    assert index.longest_copied_run(symbols) == len(symbols)
    assert index.contains_exact(symbols) is True


def test_transposed_copy_looks_novel_by_pitch_but_is_caught_by_intervals():
    transposed = [MidiNote(note.pitch + 5, note.start, note.end, note.velocity) for note in CORPUS[0]]
    pitch_index = build_corpus_index(CORPUS, "pitch")
    interval_index = build_corpus_index(CORPUS, "interval")

    # Absolute pitches all changed, so the pitch view sees nothing it recognises.
    assert pitch_index.novel_ngram_rate(pitch_duration_symbols(transposed), 4) == 1.0
    assert pitch_index.contains_exact(pitch_duration_symbols(transposed)) is False

    # The interval view is transposition-invariant, so the copy is exposed.
    interval_query = interval_symbols(transposed)
    assert interval_index.novel_ngram_rate(interval_query, 4) == 0.0
    assert interval_index.contains_exact(interval_query) is True


def test_unrelated_clip_is_fully_novel():
    index = build_corpus_index(CORPUS, "pitch")
    unseen = pitch_duration_symbols(_clip([21, 108, 22, 107, 23, 106]))

    assert index.novel_ngram_rate(unseen, 4) == 1.0
    assert index.longest_copied_run(unseen) == 0


def test_partial_overlap_is_reported_between_the_extremes():
    index = build_corpus_index(CORPUS, "pitch")
    # First four notes come from a corpus clip, the rest do not occur anywhere.
    hybrid = pitch_duration_symbols(_clip([60, 62, 64, 65, 21, 108, 22, 107]))

    rate = index.novel_ngram_rate(hybrid, 4)

    assert 0.0 < rate < 1.0
    assert index.longest_copied_run(hybrid) == 4


def test_novel_rate_is_one_when_the_clip_is_shorter_than_n():
    index = build_corpus_index(CORPUS, "pitch")
    short = pitch_duration_symbols(_clip([60, 62]))

    assert index.novel_ngram_rate(short, 8) == 1.0


def test_longest_run_respects_the_cap():
    index = build_corpus_index(CORPUS, "pitch")
    symbols = pitch_duration_symbols(CORPUS[0])

    assert index.longest_copied_run(symbols, max_run=3) == 3


def test_runs_do_not_match_across_clip_boundaries():
    # Concatenating two corpus clips must not read as a single copied run: the sentinel
    # between them blocks any window that would straddle the boundary.
    index = build_corpus_index(CORPUS, "pitch")
    joined = pitch_duration_symbols(_clip(CORPUS_PITCHES[0] + CORPUS_PITCHES[1]))

    assert index.longest_copied_run(joined) == len(CORPUS_PITCHES[0])


def test_identical_clips_have_zero_diversity():
    symbols = [pitch_duration_symbols(CORPUS[0])] * 4

    report = set_diversity(symbols, "pitch", n=2)

    assert report.mean_pairwise_diversity == 0.0
    assert report.mean_pairwise_cosine_diversity == pytest.approx(0.0, abs=1e-12)
    assert report.exact_duplicate_pairs == 6


def test_distinct_clips_have_high_diversity():
    symbols = [pitch_duration_symbols(clip) for clip in CORPUS]

    report = set_diversity(symbols, "pitch", n=2)

    assert report.mean_pairwise_diversity > 0.5
    assert report.exact_duplicate_pairs == 0


def test_diversity_handles_sets_too_small_to_pair():
    assert set_diversity([], "pitch", 2).clip_count == 0
    assert set_diversity([pitch_duration_symbols(CORPUS[0])], "pitch", 2).mean_pairwise_diversity == 0.0


def test_cosine_diversity_separates_clips_that_jaccard_calls_identical():
    # Same two symbols in both clips, but very different proportions. Jaccard sees one
    # identical set; cosine sees the imbalance.
    heavy = pitch_duration_symbols(_clip([60] * 9 + [62]))
    balanced = pitch_duration_symbols(_clip([60, 62] * 5))

    report = set_diversity([heavy, balanced], "pitch", n=1)

    assert report.mean_pairwise_diversity == 0.0
    assert report.mean_pairwise_cosine_diversity > 0.0


def test_distinct_ngram_ratio_counts_unique_windows():
    repeated = pitch_duration_symbols(_clip([60, 60, 60, 60]))

    # Three identical 2-grams collapse to one distinct window.
    assert distinct_ngram_ratio([repeated], 2) == 1 / 3


def test_empty_corpus_reports_everything_as_novel():
    index = build_corpus_index([], "pitch")
    symbols = pitch_duration_symbols(CORPUS[0])

    assert index.clip_count == 0
    assert index.novel_ngram_rate(symbols, 4) == 1.0
    assert index.longest_copied_run(symbols) == 0
