import numpy as np

from musemotion.evaluation.distributions import (
    comparable_feature_names,
    feature_means,
    nearest_neighbour_distances,
    nearest_neighbour_from_matrix,
    overlapping_area,
    per_feature_overlap,
    set_distances,
    standardise,
)
from musemotion.evaluation.features import FEATURE_NAMES


def test_length_dependent_features_are_excluded_from_comparison():
    names = comparable_feature_names()

    # Most generated clips stop at the token cap, so comparing absolute length would largely
    # measure the sampling budget, not the music.
    assert "note_count" not in names
    assert "span_seconds" not in names
    assert "note_density" in names
    assert len(names) == len(FEATURE_NAMES) - 2


def test_the_feature_probe_trains_on_exactly_the_comparable_features():
    """One definition, two consumers.

    The probe and the fidelity comparison must agree on which features are usable. Two separate
    lists would let a newly added feature reach one and not the other, silently desyncing the
    judge from the distribution it is judged against.
    """
    from musemotion.training.probe import PROBE_FEATURE_NAMES

    assert list(PROBE_FEATURE_NAMES) == comparable_feature_names()


def test_identical_distributions_overlap_completely():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert overlapping_area(values, values) == 1.0


def test_disjoint_distributions_do_not_overlap():
    assert overlapping_area([0.0, 0.1, 0.2], [10.0, 10.1, 10.2]) == 0.0


def test_partial_overlap_falls_between_the_extremes():
    overlap = overlapping_area([0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0])

    assert 0.0 < overlap < 1.0


def test_overlap_handles_constant_and_empty_inputs():
    # Two sets that are the same single value coincide exactly.
    assert overlapping_area([3.0, 3.0], [3.0, 3.0]) == 1.0
    assert overlapping_area([], [1.0]) == 0.0
    assert overlapping_area([1.0], []) == 0.0


def test_standardise_uses_the_reference_statistics_only():
    reference = np.array([[0.0], [10.0]])
    candidate = np.array([[5.0]])

    scaled_reference, scaled_candidate = standardise(reference, candidate)

    assert np.isclose(scaled_reference.mean(), 0.0)
    # The candidate sits at the reference mean, so it maps to zero rather than to its own mean.
    assert np.isclose(scaled_candidate[0, 0], 0.0)


def test_standardise_survives_a_constant_reference_column():
    reference = np.array([[2.0], [2.0]])
    candidate = np.array([[5.0]])

    scaled_reference, scaled_candidate = standardise(reference, candidate)

    assert np.all(np.isfinite(scaled_reference))
    assert np.all(np.isfinite(scaled_candidate))


def test_a_candidate_drawn_from_the_reference_sits_within_its_own_spread():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(60, 4))
    candidate = rng.normal(size=(60, 4))

    distances, _ = set_distances(reference, candidate)

    assert distances.reference_count == 60
    assert distances.candidate_count == 60
    # Same underlying distribution, so the inter-set distance should look like the intra-set one.
    assert 0.8 < distances.inter_over_intra < 1.25


def test_a_shifted_candidate_reads_as_further_than_the_reference_spread():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(60, 4))
    candidate = rng.normal(size=(60, 4)) + 8.0

    distances, _ = set_distances(reference, candidate)

    assert distances.inter_over_intra > 3.0


def test_set_distances_on_degenerate_inputs():
    empty = np.zeros((0, 3))
    single = np.zeros((1, 3))

    distances, _ = set_distances(single, empty)

    assert distances.intra_reference == 0.0
    assert distances.inter == 0.0
    assert distances.inter_over_intra == 0.0


def test_nearest_neighbour_distance_is_zero_for_a_memorised_candidate():
    reference = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    copied = reference[:2].copy()

    result = nearest_neighbour_distances(reference, copied)

    assert np.isclose(result["min"], 0.0)
    assert np.isclose(result["mean"], 0.0)


def test_nearest_neighbour_distance_grows_for_unseen_candidates():
    reference = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    far = np.array([[20.0, 20.0]])

    memorised = nearest_neighbour_distances(reference, reference.copy())
    unseen = nearest_neighbour_distances(reference, far)

    assert unseen["mean"] > memorised["mean"]


def test_per_feature_overlap_and_means_are_keyed_by_name():
    names = ["a", "b"]
    reference = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    candidate = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])

    overlaps = per_feature_overlap(reference, candidate, names)
    means = feature_means(reference, names)

    assert set(overlaps) == {"a", "b"}
    assert overlaps["a"] == 1.0
    assert means["a"] == 1.0
    assert means["b"] == 11.0


def test_per_feature_overlap_on_empty_input_returns_zeros_for_every_name():
    overlaps = per_feature_overlap(np.zeros((0, 2)), np.zeros((0, 2)), ["a", "b"])

    assert overlaps == {"a": 0.0, "b": 0.0}
    assert feature_means(np.zeros((0, 2)), ["a", "b"]) == {"a": 0.0, "b": 0.0}


def test_set_distances_reuses_a_supplied_reference_spread():
    """The reference's own spread is identical for every candidate, so it is computed once.

    Passing it back in must produce the same result as recomputing it - that equivalence is what
    makes the caching safe.
    """
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(40, 3))
    candidate = rng.normal(size=(15, 3))

    fresh, cache = set_distances(reference, candidate)
    reused, _ = set_distances(reference, candidate, intra_reference=cache["intra_reference"])

    assert reused.intra_reference == fresh.intra_reference
    assert reused.inter == fresh.inter
    assert reused.inter_over_intra == fresh.inter_over_intra


def test_nearest_neighbour_from_matrix_matches_the_standalone_computation():
    """The inter-set matrix is reused rather than rebuilt; both paths must agree."""
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(30, 3))
    candidate = rng.normal(size=(12, 3))

    _, cache = set_distances(reference, candidate)
    reused = nearest_neighbour_from_matrix(cache["inter_matrix"])
    standalone = nearest_neighbour_distances(reference, candidate)

    for key in ("mean", "median", "min", "p05"):
        assert np.isclose(reused[key], standalone[key]), key


def test_nearest_neighbour_from_an_empty_matrix_is_zero():
    assert nearest_neighbour_from_matrix(np.zeros((0, 0)))["mean"] == 0.0


def test_one_outlier_cannot_inflate_overlap_between_disjoint_sets():
    """The regression that a one-line revert to raw min/max would reintroduce.

    With bin edges taken from the raw union extremes, a single degenerate clip stretches the
    range until both histograms collapse into one bin and the score jumps to near 1.0 - so a
    generator that emits one pathological clip would read as *more* faithful than one that does
    not. Reachable in practice: 128 notes on a single onset gives a note density of 512 against
    a real range of roughly 1.6 to 21.3.
    """
    rng = np.random.default_rng(0)
    reference = rng.normal(8.0, 1.0, 200)
    candidate = rng.normal(20.0, 1.0, 200)

    clean = overlapping_area(reference, candidate)
    with_outlier = overlapping_area(reference, np.append(candidate, 512.0))

    assert clean == 0.0
    # The outlier must not buy the candidate any credit at all.
    assert with_outlier < 0.05, f"one outlier inflated overlap to {with_outlier}"


def test_out_of_range_values_still_count_as_mass():
    """Trimming sets the range, it does not discard data: clipped values land in the end bins."""
    inside = [1.0] * 50
    with_tail = [1.0] * 50 + [900.0] * 50

    # Half the candidate's mass sits far outside the reference, so overlap must drop accordingly.
    assert overlapping_area(inside, with_tail) < 0.6


def test_a_collapsed_set_at_the_centroid_beats_real_data_on_distance_alone():
    """Pins the weakness that the random-piano baseline exposed in a real run.

    A mean distance rewards collapse: a set clustered at the centre of the reference distribution
    sits closer to every reference point than the reference's own far-apart pairs sit to each
    other, so inter_over_intra drops below 1.0 for output that is obviously worse. Measured, the
    random-piano row scored 0.975 against real held-out clips at 1.006. The marginal overlap is
    what catches it, which is why the two are always reported together.
    """
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(200, 6))
    collapsed = np.full((40, 6), 0.001)          # degenerate, parked at the centroid
    faithful = rng.normal(size=(40, 6))          # genuinely drawn from the reference

    collapsed_distance, _ = set_distances(reference, collapsed)
    faithful_distance, _ = set_distances(reference, faithful)

    # The distance metric prefers the degenerate set - this is the failure being pinned.
    assert collapsed_distance.inter_over_intra < faithful_distance.inter_over_intra
    assert collapsed_distance.inter_over_intra < 1.0

    # Overlap is not fooled: it ranks the degenerate set far below the faithful one.
    names = [f"f{i}" for i in range(6)]
    collapsed_overlap = np.mean(list(per_feature_overlap(reference, collapsed, names).values()))
    faithful_overlap = np.mean(list(per_feature_overlap(reference, faithful, names).values()))
    assert collapsed_overlap < 0.3 < faithful_overlap

    # And intra-candidate spread names the collapse outright.
    assert collapsed_distance.intra_candidate < 0.01
    assert faithful_distance.intra_candidate > 1.0
