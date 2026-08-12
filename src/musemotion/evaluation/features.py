"""Symbolic features for a list of notes.

Every feature is computed from ``MidiNote`` objects, so the same code path serves real
EMOPIA clips (via ``MusicTokenizer.midi_to_notes``) and generated clips (via
``MusicTokenizer.decode_tokens``). The feature vector backs three things: the feature
probe, the per-quadrant distribution comparison against real EMOPIA, and the
nearest-neighbour novelty check.

The harmonic features matter more than they look. Measured on EMOPIA, note density,
velocity, and note duration separate the two arousal levels almost perfectly, while mean
pitch spans only about three semitones across all four quadrants. Valence has to come
from harmony, so the pitch-class histogram and the Krumhansl-Schmuckler key correlations
are the only features here with a chance of carrying it.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from musemotion.music.tokenizer import MidiNote


# Krumhansl-Schmuckler key profiles: the perceived stability of each scale degree,
# indexed from the tonic. Correlating a duration-weighted pitch-class histogram against
# all 12 rotations of each profile estimates key and mode.
KRUMHANSL_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KRUMHANSL_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    "note_count",
    "span_seconds",
    "note_density",
    "mean_pitch",
    "std_pitch",
    "pitch_range",
    "mean_ioi",
    "std_ioi",
    "mean_duration",
    "std_duration",
    "mean_velocity",
    "std_velocity",
    "mean_abs_interval",
    "repeated_pitch_fraction",
    "polyphony",
    "key_major_corr",
    "key_minor_corr",
    "key_mode_margin",
)

PITCH_CLASS_FEATURE_NAMES: tuple[str, ...] = tuple(f"pitch_class_{index}" for index in range(12))

FEATURE_NAMES: tuple[str, ...] = SCALAR_FEATURE_NAMES + PITCH_CLASS_FEATURE_NAMES

# Features whose value scales with how long a clip is. Most generated clips run to the
# ``max_tokens`` cap instead of emitting an ending, so clip length here is largely a sampling
# hyperparameter rather than a model output. Comparisons that mix lengths should drop these.
LENGTH_DEPENDENT_FEATURES: frozenset[str] = frozenset({"note_count", "span_seconds"})


def symbolic_features(notes: Iterable[MidiNote]) -> dict[str, float]:
    """Return the full feature dictionary for one clip.

    An empty or single-note clip yields zeros rather than NaNs, so degenerate generations
    stay comparable instead of poisoning any aggregate computed over them.
    """
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    if not ordered:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    # MIDI pitches are integers; keeping an integer array means the repeated-note test is an
    # exact comparison rather than a float equality check.
    pitch_numbers = np.asarray([int(note.pitch) for note in ordered], dtype="int64")
    pitches = pitch_numbers.astype("float64")
    starts = np.asarray([note.start for note in ordered], dtype="float64")
    ends = np.asarray([note.end for note in ordered], dtype="float64")
    velocities = np.asarray([note.velocity for note in ordered], dtype="float64")
    durations = np.maximum(ends - starts, 0.0)

    span = float(ends.max() - starts.min())
    # Inter-onset intervals are taken over *unique* onsets. EMOPIA is polyphonic piano, so
    # consecutive notes frequently share an onset and would otherwise contribute zeros.
    unique_onsets = np.unique(starts)
    iois = np.diff(unique_onsets) if unique_onsets.size >= 2 else np.zeros(0)
    intervals = np.abs(np.diff(pitches)) if pitches.size >= 2 else np.zeros(0)
    repeated = (
        float(np.mean(np.diff(pitch_numbers) == 0)) if pitch_numbers.size >= 2 else 0.0
    )

    histogram = pitch_class_histogram(ordered)
    major_corr, minor_corr = key_correlations(histogram)

    features = {
        "note_count": float(len(ordered)),
        "span_seconds": span,
        "note_density": float(len(ordered) / span) if span > 0 else 0.0,
        "mean_pitch": float(pitches.mean()),
        "std_pitch": float(pitches.std()),
        "pitch_range": float(pitches.max() - pitches.min()),
        "mean_ioi": float(iois.mean()) if iois.size else 0.0,
        "std_ioi": float(iois.std()) if iois.size else 0.0,
        "mean_duration": float(durations.mean()),
        "std_duration": float(durations.std()),
        "mean_velocity": float(velocities.mean()),
        "std_velocity": float(velocities.std()),
        "mean_abs_interval": float(intervals.mean()) if intervals.size else 0.0,
        "repeated_pitch_fraction": repeated,
        # Mean simultaneous voices: total sounding time divided by wall-clock span.
        "polyphony": float(durations.sum() / span) if span > 0 else 0.0,
        "key_major_corr": major_corr,
        "key_minor_corr": minor_corr,
        "key_mode_margin": major_corr - minor_corr,
    }
    for index, value in enumerate(histogram):
        features[f"pitch_class_{index}"] = float(value)
    return features


def pitch_class_histogram(notes: Iterable[MidiNote]) -> np.ndarray:
    """Duration-weighted, sum-normalised histogram over the 12 pitch classes.

    Weighting by duration rather than by note count means a held whole note counts for
    more than a passing sixteenth, which is what makes the key estimate track what a
    listener would hear as tonic.
    """
    histogram = np.zeros(12, dtype="float64")
    for note in notes:
        weight = max(float(note.end) - float(note.start), 0.0)
        histogram[int(note.pitch) % 12] += weight
    total = histogram.sum()
    if total <= 0:
        # Every note had zero duration; fall back to unweighted counts.
        for note in notes:
            histogram[int(note.pitch) % 12] += 1.0
        total = histogram.sum()
    return histogram / total if total > 0 else histogram


def key_correlations(histogram: Sequence[float]) -> tuple[float, float]:
    """Best major and best minor Krumhansl correlation over all 12 rotations."""
    values = np.asarray(histogram, dtype="float64")
    if values.size != 12 or not np.any(values):
        return 0.0, 0.0
    best_major = max(_correlation(values, np.roll(KRUMHANSL_MAJOR, shift)) for shift in range(12))
    best_minor = max(_correlation(values, np.roll(KRUMHANSL_MINOR, shift)) for shift in range(12))
    return float(best_major), float(best_minor)


def estimated_key(histogram: Sequence[float]) -> tuple[int, str]:
    """Return the ``(tonic_pitch_class, "major" | "minor")`` that correlates best."""
    values = np.asarray(histogram, dtype="float64")
    if values.size != 12 or not np.any(values):
        return 0, "major"
    candidates = [
        (_correlation(values, np.roll(profile, shift)), shift, mode)
        for profile, mode in ((KRUMHANSL_MAJOR, "major"), (KRUMHANSL_MINOR, "minor"))
        for shift in range(12)
    ]
    _, tonic, mode = max(candidates, key=lambda candidate: candidate[0])
    return int(tonic), str(mode)


def feature_vector(notes: Iterable[MidiNote], names: Sequence[str] | None = None) -> np.ndarray:
    """Feature dictionary flattened into a vector with a stable column order."""
    features = symbolic_features(notes)
    selected = names if names is not None else FEATURE_NAMES
    return np.asarray([features[name] for name in selected], dtype="float64")


def feature_matrix(clips: Iterable[Iterable[MidiNote]], names: Sequence[str] | None = None) -> np.ndarray:
    """Stack per-clip feature vectors into a ``(clips, features)`` matrix."""
    selected = list(names) if names is not None else list(FEATURE_NAMES)
    rows = [feature_vector(notes, selected) for notes in clips]
    if not rows:
        return np.zeros((0, len(selected)), dtype="float64")
    return np.vstack(rows)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 when either input has no variance."""
    left_centred = left - left.mean()
    right_centred = right - right.mean()
    denominator = float(np.linalg.norm(left_centred) * np.linalg.norm(right_centred))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left_centred, right_centred) / denominator)
