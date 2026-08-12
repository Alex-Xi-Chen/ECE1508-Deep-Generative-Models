"""Distributional comparison between generated clips and real EMOPIA clips.

This is the part that turns a two-clip anecdote into a measurement. The presentation
observed that one Q2 clip had roughly four times the note density of one Q1 clip; these
functions ask whether that holds across hundreds of clips, and whether the generated
per-quadrant distributions actually look like EMOPIA's.

The framing follows Yang & Lerch (2020): a single distance between two sets is not
interpretable on its own, so every inter-set distance is reported next to the intra-set
distance of the real data. Real EMOPIA clips already differ from each other by some amount;
that spread is the yardstick for whether generated clips are unusually far away.

Overlapping area is the headline per-feature number - the shared area under two histograms,
1.0 for identical distributions and 0.0 for disjoint ones. It is bounded and unitless, so
features on wildly different scales stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from musemotion.evaluation.features import FEATURE_NAMES, LENGTH_DEPENDENT_FEATURES


DEFAULT_HISTOGRAM_BINS = 20


@dataclass(frozen=True)
class SetDistances:
    """Intra-set spread of each set, and the distance between them."""

    intra_reference: float
    intra_candidate: float
    inter: float
    reference_count: int
    candidate_count: int

    @property
    def inter_over_intra(self) -> float:
        """Inter-set distance relative to the reference set's own spread.

        Near 1.0 means the generated clips sit no further from real clips than real clips sit
        from each other. Well above 1.0 means they occupy a different region of feature space.

        This must not be read alone, because a mean distance rewards collapse. A set clustered
        tightly near the centre of the reference distribution has a small mean distance to every
        reference point - smaller than the reference's own mean pairwise distance, which includes
        its far-apart pairs - so it scores *better* than real data while being obviously worse
        music. That is not hypothetical: the random-piano baseline scores 0.975 here, nominally
        closer to real EMOPIA than real held-out clips at 1.006, while its marginal overlap of
        0.324 correctly ranks it last. Pair this with ``per_feature_overlap``, which catches the
        degenerate case, and with the diversity measures, which catch the collapse directly.
        """
        if self.intra_reference <= 0:
            return 0.0
        return float(self.inter / self.intra_reference)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "intra_reference": self.intra_reference,
            "intra_candidate": self.intra_candidate,
            "inter": self.inter,
            "inter_over_intra": self.inter_over_intra,
            "reference_count": self.reference_count,
            "candidate_count": self.candidate_count,
        }


def comparable_feature_names(names: Sequence[str] = FEATURE_NAMES) -> list[str]:
    """Feature names with the length-dependent ones removed.

    Most generated clips run to the token cap instead of emitting an ending (measured: 67 to 92
    per cent hit the cap, depending on guidance), so note count and span mostly reflect a
    sampling hyperparameter rather than the model. Comparing them against real clips would
    largely measure the cap, not the music.
    """
    return [name for name in names if name not in LENGTH_DEPENDENT_FEATURES]


def standardise(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both matrices using the *reference* mean and standard deviation.

    Standardising on the reference alone keeps the transform independent of whatever the
    candidate set happens to contain, so distances from different candidate sets remain
    comparable to each other.
    """
    if reference.size == 0:
        return reference, candidate
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (reference - mean) / scale, (candidate - mean) / scale


def set_distances(
    reference: np.ndarray,
    candidate: np.ndarray,
    intra_reference: float | None = None,
) -> tuple[SetDistances, dict[str, Any]]:
    """Mean pairwise Euclidean distances within and between two feature matrices.

    Returns the distances plus a cache of values that depend only on the reference set.
    Standardisation uses the reference statistics alone, so the reference's own intra-set
    distance is identical for every candidate it is compared against - and computing it means an
    n x n broadcast over the whole training corpus, which is the single most expensive operation
    in a run. ``intra_reference`` lets a caller pass the previous result back in.

    The inter-set distance matrix is computed once and reused for the nearest-neighbour summary
    rather than being rebuilt by a second call.
    """
    scaled_reference, scaled_candidate = standardise(reference, candidate)
    if intra_reference is None:
        intra_reference = _mean_intra_distance(scaled_reference)

    inter_matrix = (
        _pairwise_distances(scaled_candidate, scaled_reference)
        if scaled_reference.size and scaled_candidate.size
        else np.zeros((0, 0))
    )
    distances = SetDistances(
        intra_reference=intra_reference,
        intra_candidate=_mean_intra_distance(scaled_candidate),
        inter=float(inter_matrix.mean()) if inter_matrix.size else 0.0,
        reference_count=int(reference.shape[0]) if reference.ndim == 2 else 0,
        candidate_count=int(candidate.shape[0]) if candidate.ndim == 2 else 0,
    )
    return distances, {"intra_reference": intra_reference, "inter_matrix": inter_matrix}


def nearest_neighbour_from_matrix(inter_matrix: np.ndarray) -> dict[str, float]:
    """Nearest-reference distance per candidate, from an already-computed distance matrix."""
    if inter_matrix.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "p05": 0.0}
    distances = inter_matrix.min(axis=1)
    return {
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "min": float(distances.min()),
        "p05": float(np.percentile(distances, 5)),
    }


def overlapping_area(
    reference_values: Iterable[float],
    candidate_values: Iterable[float],
    bins: int = DEFAULT_HISTOGRAM_BINS,
    tail_percentile: float = 1.0,
) -> float:
    """Shared area under two normalised histograms, over a shared bin range.

    1.0 means the two distributions coincide; 0.0 means they never overlap.

    The bin range comes from a trimmed percentile span rather than the raw union min and max,
    and this matters more than it looks. With raw extremes, one degenerate clip stretches the
    range until every real value collapses into a single bin and the score jumps to nearly 1.0 -
    so a generator that emits one pathological clip would score as *more* faithful than one that
    does not. That failure is reachable: 128 notes sharing an onset gives a note density of 512
    against a real range of roughly 1.6 to 21.3. Values outside the trimmed span are clipped
    into the end bins, so they still count as mass, just not as range.
    """
    reference = np.asarray(list(reference_values), dtype="float64")
    candidate = np.asarray(list(candidate_values), dtype="float64")
    if reference.size == 0 or candidate.size == 0:
        return 0.0

    pooled = np.concatenate([reference, candidate])
    low = float(np.percentile(pooled, tail_percentile))
    high = float(np.percentile(pooled, 100.0 - tail_percentile))
    if high <= low:
        # A trimmed span can collapse when most of the mass sits on one value; fall back to the
        # full range, and treat a genuinely constant pooled set as complete overlap.
        low, high = float(pooled.min()), float(pooled.max())
        if high <= low:
            return 1.0

    edges = np.linspace(low, high, bins + 1)
    reference_counts, _ = np.histogram(np.clip(reference, low, high), bins=edges)
    candidate_counts, _ = np.histogram(np.clip(candidate, low, high), bins=edges)
    reference_share = reference_counts / reference_counts.sum()
    candidate_share = candidate_counts / candidate_counts.sum()
    return float(np.minimum(reference_share, candidate_share).sum())


def per_feature_overlap(
    reference: np.ndarray,
    candidate: np.ndarray,
    names: Sequence[str],
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict[str, float]:
    """Overlapping area for every feature column, keyed by feature name."""
    if reference.size == 0 or candidate.size == 0:
        return dict.fromkeys(names, 0.0)
    return {
        name: overlapping_area(reference[:, column], candidate[:, column], bins)
        for column, name in enumerate(names)
    }


def feature_means(matrix: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    """Column means, keyed by feature name."""
    if matrix.size == 0:
        return dict.fromkeys(names, 0.0)
    return {name: float(value) for name, value in zip(names, matrix.mean(axis=0))}


def nearest_neighbour_distances(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Distance from each candidate clip to its closest reference clip, summarised.

    A generated set that memorised the corpus would show near-zero minimum distances. The
    same statistic computed for real held-out clips is the reference frame - real music also
    has near neighbours in a training corpus, so a small distance only means something when
    it is smaller than that.
    """
    scaled_reference, scaled_candidate = standardise(reference, candidate)
    if scaled_reference.size == 0 or scaled_candidate.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "p05": 0.0}
    distances = _pairwise_distances(scaled_candidate, scaled_reference).min(axis=1)
    return {
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "min": float(distances.min()),
        "p05": float(np.percentile(distances, 5)),
    }


def _pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix of shape ``(len(left), len(right))``."""
    difference = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.maximum((difference**2).sum(axis=-1), 0.0))


def _mean_intra_distance(matrix: np.ndarray) -> float:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    distances = _pairwise_distances(matrix, matrix)
    upper = np.triu_indices(distances.shape[0], k=1)
    return float(distances[upper].mean())


def _mean_inter_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] == 0 or right.shape[0] == 0:
        return 0.0
    return float(_pairwise_distances(left, right).mean())
