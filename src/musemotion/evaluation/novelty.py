"""Novelty against the training corpus, and diversity within a generated set.

These are two different questions that the word "unique" tends to blur together:

* **Novelty** asks whether a clip is copied from EMOPIA. Measured as the share of a clip's
  note n-grams that never occur in the training corpus, plus the longest run of notes it
  reproduces exactly.
* **Diversity** asks whether the generated clips differ from *each other*, which is what
  catches mode collapse.

Both are reported alongside the identical measurement taken on real held-out EMOPIA clips.
A novelty rate on its own has no scale; the same number next to the real-data reference does.

Two symbol views are used. The pitch view keys on absolute pitch, so a transposed copy of a
training phrase still counts as novel. The interval view keys on pitch *deltas*, so a
transposed copy is correctly flagged as a copy. The interval view is the stricter test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from musemotion.music.tokenizer import MidiNote


DEFAULT_NGRAM_SIZES: tuple[int, ...] = (4, 8, 16)
DEFAULT_MAX_RUN = 64

# Jaccard diversity saturates fast. Measured on real held-out EMOPIA clips, mean pairwise
# Jaccard diversity is 0.873 at n=1, 0.996 at n=2, and exactly 1.000 from n=4 up: no two
# real clips share even one 4-note pattern. So n=1 and n=2 are the sizes with any grading
# left in them, and n=4 works only as a collapse tripwire - a generated set scoring below
# 1.0 there is repeating whole phrases, which real music never does.
DEFAULT_DIVERSITY_SIZES: tuple[int, ...] = (1, 2, 4)

# Odd 64-bit multiplier for the polynomial rolling hash. numpy's uint64 arithmetic wraps
# silently, which is exactly the modular arithmetic this needs.
_HASH_BASE = 1099511628211
_UINT64_MASK = (1 << 64) - 1

# Interned symbols occupy ids from 1 upward. Clip separators and query-only symbols live in
# disjoint high ranges so every id is non-negative and globally consistent: the hash of a
# symbol must not depend on which array it happens to sit in, or corpus and query hashes
# would never agree.
_SENTINEL_BASE = 1 << 40
_UNSEEN_BASE = 1 << 41

Symbol = tuple[int, int]


def pitch_duration_symbols(notes: Iterable[MidiNote], beat_resolution: int = 4) -> list[Symbol]:
    """One symbol per note: absolute pitch paired with its quantised duration."""
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    return [(int(note.pitch), _duration_step(note, beat_resolution)) for note in ordered]


def interval_symbols(notes: Iterable[MidiNote], beat_resolution: int = 4) -> list[Symbol]:
    """One symbol per note transition: pitch delta paired with the arriving note's duration.

    Transposition-invariant, so the same phrase played in a different key produces the same
    symbol sequence. A clip of ``n`` notes yields ``n - 1`` symbols.
    """
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    return [
        (int(ordered[index + 1].pitch) - int(ordered[index].pitch), _duration_step(ordered[index + 1], beat_resolution))
        for index in range(len(ordered) - 1)
    ]


SYMBOL_VIEWS: dict[str, object] = {
    "pitch": pitch_duration_symbols,
    "interval": interval_symbols,
}


class CorpusIndex:
    """Membership index over every note n-gram in a corpus, for one symbol view.

    Symbols are interned to integer ids, the corpus is concatenated with a unique sentinel
    between clips, and window hashes are derived from one prefix-hash pass. Sentinels are
    distinct and never appear in a query, so no window straddling a clip boundary can match.

    Hash arrays are cached per window length. The longest-run search walks lengths upward
    from 1 and stops at the first miss, which is valid because a matching window of length
    ``L + 1`` contains a matching window of length ``L``. In practice it stops after a
    handful of probes, so only a few short lengths are ever cached.
    """

    def __init__(self, clips: Iterable[Sequence[Symbol]]):
        self._symbol_ids: dict[Symbol, int] = {}
        self._sequences: list[np.ndarray] = []
        self._exact: set[tuple[int, ...]] = set()

        stream: list[int] = []
        sentinel_offset = 0
        for clip in clips:
            ids = [self._intern(symbol) for symbol in clip]
            if not ids:
                continue
            self._sequences.append(np.asarray(ids, dtype="int64"))
            self._exact.add(tuple(ids))
            stream.extend(ids)
            stream.append(_SENTINEL_BASE + sentinel_offset)
            sentinel_offset += 1

        self._stream = np.asarray(stream, dtype="int64") if stream else np.zeros(0, dtype="int64")
        self._prefix = _prefix_hashes(self._stream)
        self._window_cache: dict[int, np.ndarray] = {}

    @property
    def symbol_count(self) -> int:
        return int(self._stream.size)

    @property
    def clip_count(self) -> int:
        return len(self._sequences)

    def novel_ngram_rate(self, clip: Sequence[Symbol], n: int) -> float:
        """Share of the clip's length-``n`` windows absent from the corpus.

        Returns 1.0 when the clip is shorter than ``n``: nothing it contains was copied,
        because it contains no window of that length at all.
        """
        ids = self._query_ids(clip)
        if ids.size < n or n <= 0:
            return 1.0
        present = self._windows_present(ids, n)
        return float(1.0 - present.mean())

    def longest_copied_run(self, clip: Sequence[Symbol], max_run: int = DEFAULT_MAX_RUN) -> int:
        """Length of the longest window of the clip that occurs verbatim in the corpus."""
        ids = self._query_ids(clip)
        limit = min(max_run, int(ids.size))
        length = 0
        while length < limit:
            if not self._windows_present(ids, length + 1).any():
                break
            length += 1
        return length

    def contains_exact(self, clip: Sequence[Symbol]) -> bool:
        """Whether the clip's whole symbol sequence appears as a corpus clip."""
        ids = self._query_ids(clip)
        return tuple(int(value) for value in ids) in self._exact

    def _intern(self, symbol: Symbol) -> int:
        identifier = self._symbol_ids.get(symbol)
        if identifier is None:
            identifier = len(self._symbol_ids) + 1
            self._symbol_ids[symbol] = identifier
        return identifier

    def _query_ids(self, clip: Sequence[Symbol]) -> np.ndarray:
        """Map query symbols to corpus ids without mutating the corpus vocabulary.

        Symbols the corpus never contained get distinct ids in the reserved unseen range, so
        they cannot match anything and cannot collide with each other.
        """
        ids: list[int] = []
        local: dict[Symbol, int] = {}
        for symbol in clip:
            identifier = self._symbol_ids.get(symbol)
            if identifier is None:
                identifier = local.get(symbol)
                if identifier is None:
                    identifier = _UNSEEN_BASE + len(local)
                    local[symbol] = identifier
            ids.append(identifier)
        return np.asarray(ids, dtype="int64")

    def _windows_present(self, ids: np.ndarray, n: int) -> np.ndarray:
        """Boolean mask over the query's length-``n`` windows: present in the corpus?"""
        if ids.size < n or self._stream.size < n:
            return np.zeros(max(0, ids.size - n + 1), dtype=bool)
        corpus_hashes = self._corpus_window_hashes(n)
        query_hashes = _window_hashes(_prefix_hashes(ids), ids.size, n)
        if corpus_hashes.size == 0:
            return np.zeros(query_hashes.size, dtype=bool)
        # corpus_hashes is sorted, so this is an explicit binary search per query window.
        positions = np.searchsorted(corpus_hashes, query_hashes)
        positions = np.clip(positions, 0, corpus_hashes.size - 1)
        return corpus_hashes[positions] == query_hashes

    def _corpus_window_hashes(self, n: int) -> np.ndarray:
        cached = self._window_cache.get(n)
        if cached is None:
            # Sorted so np.isin can use a binary search rather than building a hash table
            # of the whole corpus on every query.
            cached = np.sort(_window_hashes(self._prefix, self._stream.size, n))
            self._window_cache[n] = cached
        return cached


def build_corpus_index(
    clips: Iterable[Iterable[MidiNote]],
    view: str,
    beat_resolution: int = 4,
) -> CorpusIndex:
    """Build an index over note lists, converting them through the named symbol view."""
    to_symbols = SYMBOL_VIEWS[view]
    return CorpusIndex(to_symbols(notes, beat_resolution) for notes in clips)  # type: ignore[operator]


@dataclass
class DiversityReport:
    """Pairwise dissimilarity within one set of clips, under one symbol view.

    Two measures, because they fail differently. ``mean_pairwise_diversity`` is
    ``1 - Jaccard`` over n-gram *sets*: exact, but it saturates at 1.0 for n >= 4 even on
    real music. ``mean_pairwise_cosine_diversity`` is ``1 - cosine`` over n-gram *count*
    vectors, so two clips that hammer the same motif at different rates still score as
    similar. The cosine measure stays graded at every n.
    """

    view: str
    n: int
    clip_count: int
    mean_pairwise_diversity: float = 0.0
    min_pairwise_diversity: float = 0.0
    mean_pairwise_cosine_diversity: float = 0.0
    min_pairwise_cosine_diversity: float = 0.0
    distinct_ngram_ratio: float = 0.0
    exact_duplicate_pairs: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "view": self.view,
            "n": self.n,
            "clip_count": self.clip_count,
            "mean_pairwise_diversity": self.mean_pairwise_diversity,
            "min_pairwise_diversity": self.min_pairwise_diversity,
            "mean_pairwise_cosine_diversity": self.mean_pairwise_cosine_diversity,
            "min_pairwise_cosine_diversity": self.min_pairwise_cosine_diversity,
            "distinct_ngram_ratio": self.distinct_ngram_ratio,
            "exact_duplicate_pairs": self.exact_duplicate_pairs,
        }


def set_diversity(clips: Sequence[Sequence[Symbol]], view: str, n: int = 2) -> DiversityReport:
    """Mean pairwise n-gram dissimilarity across a set of clips.

    A Jaccard diversity of 1.0 means no two clips share a single length-``n`` note pattern;
    0.0 means every clip carries exactly the same set of patterns.
    """
    counters = [_ngram_counts(clip, n) for clip in clips]
    usable = [counts for counts in counters if counts]
    report = DiversityReport(view=view, n=n, clip_count=len(clips))
    if len(usable) < 2:
        return report

    # Each clip's norm is computed once here rather than inside the pair loop, where it would be
    # recomputed once per partner - about 1.9 million redundant sums over a 200-clip system.
    norms = [float(np.sqrt(sum(value * value for value in counts.values()))) for counts in usable]

    jaccard_diversities: list[float] = []
    cosine_diversities: list[float] = []
    duplicate_pairs = 0
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            first, second = usable[left], usable[right]
            shared, dot = _overlap(first, second)
            union = len(first) + len(second) - shared
            jaccard_diversities.append(1.0 - (shared / union if union else 0.0))
            cosine_diversities.append(1.0 - _cosine_from_parts(dot, norms[left], norms[right]))
            # Compared with counts, not just keys. Two clips built from the same three notes,
            # one four notes long and one forty, share a key set but are not duplicates - and
            # short mode-collapsed clips are exactly where a false duplicate would matter most.
            if first == second:
                duplicate_pairs += 1

    report.mean_pairwise_diversity = float(np.mean(jaccard_diversities))
    report.min_pairwise_diversity = float(np.min(jaccard_diversities))
    report.mean_pairwise_cosine_diversity = float(np.mean(cosine_diversities))
    report.min_pairwise_cosine_diversity = float(np.min(cosine_diversities))
    report.exact_duplicate_pairs = duplicate_pairs
    report.distinct_ngram_ratio = distinct_ngram_ratio(clips, n)
    return report


def _ngram_counts(clip: Sequence[Symbol], n: int) -> dict[tuple[Symbol, ...], int]:
    counts: dict[tuple[Symbol, ...], int] = {}
    for window in _ngram_list(clip, n):
        counts[window] = counts.get(window, 0) + 1
    return counts


def _overlap(
    first: dict[tuple[Symbol, ...], int],
    second: dict[tuple[Symbol, ...], int],
) -> tuple[int, float]:
    """Shared key count and count-weighted dot product, in one pass.

    Iterates the smaller counter and probes the larger, so the cost is the smaller length rather
    than building an intersection set per pair.
    """
    smaller, larger = (first, second) if len(first) <= len(second) else (second, first)
    shared = 0
    dot = 0.0
    for key, count in smaller.items():
        other = larger.get(key)
        if other is not None:
            shared += 1
            dot += count * other
    return shared, dot


def _cosine_from_parts(dot: float, left_norm: float, right_norm: float) -> float:
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    # Counts are non-negative, so the true cosine is in [0, 1]. The clamp absorbs the rounding
    # that would otherwise report a diversity of 1e-16 for two identical clips.
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def distinct_ngram_ratio(clips: Sequence[Sequence[Symbol]], n: int = 8) -> float:
    """Unique length-``n`` windows over total windows, pooled across the whole set."""
    total = 0
    unique: set[tuple[Symbol, ...]] = set()
    for clip in clips:
        windows = _ngram_list(clip, n)
        total += len(windows)
        unique.update(windows)
    return float(len(unique) / total) if total else 0.0


def _ngram_list(clip: Sequence[Symbol], n: int) -> list[tuple[Symbol, ...]]:
    if n <= 0 or len(clip) < n:
        return []
    return [tuple(clip[index : index + n]) for index in range(len(clip) - n + 1)]


def _duration_step(note: MidiNote, beat_resolution: int) -> int:
    """Quantise a note's duration onto the tokenizer grid, so symbols compare exactly."""
    return max(1, int(round((float(note.end) - float(note.start)) * beat_resolution)))


def _prefix_hashes(ids: np.ndarray) -> np.ndarray:
    """Polynomial prefix hashes, offset by one so index 0 is the empty prefix.

    ``prefix[i + 1] = prefix[i] * BASE + ids[i]``, modulo 2**64. Accumulated with Python
    ints and an explicit mask (much faster than numpy scalar arithmetic in a tight loop),
    then handed back as uint64 where the same wraparound holds for the vectorised
    window-hash subtraction.
    """
    if ids.size == 0:
        return np.zeros(1, dtype=np.uint64)
    running = 0
    accumulated = [0]
    for value in ids.tolist():
        running = (running * _HASH_BASE + value) & _UINT64_MASK
        accumulated.append(running)
    return np.asarray(accumulated, dtype=np.uint64)


def _window_hashes(prefix: np.ndarray, length: int, n: int) -> np.ndarray:
    """Hashes of every length-``n`` window, from the prefix-hash array."""
    if length < n or n <= 0:
        return np.zeros(0, dtype=np.uint64)
    power = np.uint64(pow(_HASH_BASE, n, 1 << 64) & _UINT64_MASK)
    ends = prefix[n : length + 1]
    starts = prefix[0 : length - n + 1]
    return ends - starts * power
