import numpy as np

from musemotion.evaluation.features import (
    FEATURE_NAMES,
    estimated_key,
    feature_matrix,
    feature_vector,
    pitch_class_histogram,
    symbolic_features,
)
from musemotion.music.tokenizer import MidiNote


def _scale(root, steps, step_seconds=0.5, velocity=80):
    return [
        MidiNote(pitch=root + offset, start=index * step_seconds, end=(index + 1) * step_seconds, velocity=velocity)
        for index, offset in enumerate(steps)
    ]


MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11, 12]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10, 12]


def test_key_correlation_separates_major_from_minor():
    major = symbolic_features(_scale(60, MAJOR_STEPS))
    minor = symbolic_features(_scale(60, MINOR_STEPS))

    assert major["key_mode_margin"] > 0
    assert minor["key_mode_margin"] < 0


def test_estimated_key_recovers_tonic_and_mode():
    assert estimated_key(pitch_class_histogram(_scale(60, MAJOR_STEPS))) == (0, "major")
    assert estimated_key(pitch_class_histogram(_scale(60, MINOR_STEPS))) == (0, "minor")
    # A minor shares C major's pitch classes but weights A as tonic.
    assert estimated_key(pitch_class_histogram(_scale(57, MINOR_STEPS))) == (9, "minor")


def test_density_duration_and_ioi_on_a_fixed_rhythm():
    # Eight notes, each 0.5s long, one starting every 0.5s: 4 seconds of music at 2 notes/s.
    features = symbolic_features(_scale(60, MAJOR_STEPS))

    assert features["note_count"] == 8
    assert features["span_seconds"] == 4.0
    assert features["note_density"] == 2.0
    assert features["mean_duration"] == 0.5
    assert features["mean_ioi"] == 0.5
    assert features["polyphony"] == 1.0


def test_inter_onset_intervals_ignore_simultaneous_notes():
    # A three-note chord followed by one note: only one distinct onset gap exists.
    chord = [MidiNote(pitch=pitch, start=0.0, end=1.0, velocity=80) for pitch in (60, 64, 67)]
    chord.append(MidiNote(pitch=72, start=2.0, end=3.0, velocity=80))

    features = symbolic_features(chord)

    assert features["mean_ioi"] == 2.0
    assert features["polyphony"] > 1.0


def test_pitch_class_histogram_is_duration_weighted_and_normalised():
    notes = [
        MidiNote(pitch=60, start=0.0, end=3.0, velocity=80),
        MidiNote(pitch=61, start=3.0, end=4.0, velocity=80),
    ]

    histogram = pitch_class_histogram(notes)

    assert histogram.sum() == 1.0
    # The held C accounts for three of the four sounding seconds.
    assert histogram[0] == 0.75
    assert histogram[1] == 0.25


def test_repeated_pitch_fraction_flags_a_single_repeated_note():
    repeated = [MidiNote(pitch=60, start=i * 0.25, end=i * 0.25 + 0.25, velocity=80) for i in range(10)]

    assert symbolic_features(repeated)["repeated_pitch_fraction"] == 1.0
    assert symbolic_features(_scale(60, MAJOR_STEPS))["repeated_pitch_fraction"] == 0.0


def test_degenerate_clips_return_zeros_not_nans():
    empty = symbolic_features([])
    single = symbolic_features([MidiNote(pitch=60, start=0.0, end=1.0, velocity=80)])

    assert set(empty) == set(FEATURE_NAMES)
    assert all(value == 0.0 for value in empty.values())
    assert not any(np.isnan(value) for value in single.values())
    assert single["note_count"] == 1.0


def test_feature_vector_and_matrix_shapes_follow_the_name_order():
    clips = [_scale(60, MAJOR_STEPS), _scale(60, MINOR_STEPS)]

    vector = feature_vector(clips[0])
    matrix = feature_matrix(clips)

    assert vector.shape == (len(FEATURE_NAMES),)
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert feature_matrix([]).shape == (0, len(FEATURE_NAMES))


def test_feature_vector_respects_a_restricted_name_list():
    names = ["mean_pitch", "note_density"]

    vector = feature_vector(_scale(60, MAJOR_STEPS), names)

    assert vector.shape == (2,)
    assert vector[1] == 2.0
