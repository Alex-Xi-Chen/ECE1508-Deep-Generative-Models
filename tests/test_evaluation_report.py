import numpy as np

from musemotion.evaluation.report import (
    COMPARISON_COLUMNS,
    accuracy_from_labels,
    accuracy_result,
    axis_accuracies,
    comparison_row,
    comparison_table_csv,
    comparison_table_markdown,
    confusion_matrix,
    per_quadrant_accuracy,
    round_trip_result,
    stage_attribution,
    well_formedness,
    wilson_interval,
)


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(30, 60)

    assert low < 0.5 < high
    assert 0.0 <= low and high <= 1.0


def test_wilson_interval_stays_inside_the_unit_range_at_the_extremes():
    assert wilson_interval(0, 20)[0] == 0.0
    assert wilson_interval(20, 20)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_as_the_sample_grows():
    small_low, small_high = wilson_interval(5, 10)
    large_low, large_high = wilson_interval(500, 1000)

    assert (small_high - small_low) > (large_high - large_low)


def test_beats_chance_uses_the_interval_not_the_point_estimate():
    # 3/10 is above 0.25, but the interval still covers chance, so this must not be called a win.
    marginal = accuracy_result(3, 10)
    assert marginal.accuracy > marginal.chance
    assert marginal.beats_chance is False

    convincing = accuracy_result(300, 1000)
    assert convincing.beats_chance is True


def test_confusion_matrix_rows_are_the_conditioning_quadrant():
    matrix = confusion_matrix([0, 0, 1], [0, 1, 1])

    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1
    assert matrix[2] == [0, 0, 0, 0]


def test_axis_accuracies_separate_arousal_from_valence():
    # Q1(0) mistaken for Q2(1): arousal is preserved (both high), valence is not.
    axes = axis_accuracies([0], [1])

    assert axes["arousal"].accuracy == 1.0
    assert axes["valence"].accuracy == 0.0
    assert axes["arousal"].chance == 0.5


def test_axis_accuracies_treat_a_q1_q4_confusion_as_valence_preserved():
    # Q1(0) mistaken for Q4(3): both are high valence, arousal differs.
    axes = axis_accuracies([0], [3])

    assert axes["valence"].accuracy == 1.0
    assert axes["arousal"].accuracy == 0.0


def test_per_quadrant_accuracy_reports_every_quadrant():
    result = per_quadrant_accuracy([0, 0, 1, 1], [0, 1, 1, 1])

    assert result["Q1"]["accuracy"] == 0.5
    assert result["Q2"]["accuracy"] == 1.0
    assert result["Q3"]["count"] == 0


def test_round_trip_result_computes_the_controllability_ratio():
    probabilities = np.array([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0]])

    result = round_trip_result("feature", [0, 1], probabilities, ceiling=0.5)

    assert result.overall.accuracy == 1.0
    assert result.controllability_ratio == 2.0
    assert result.mean_confidence > 0.8
    assert result.to_dict()["confusion_labels"] == ["Q1", "Q2", "Q3", "Q4"]


def test_round_trip_result_handles_no_clips():
    result = round_trip_result("feature", [], np.zeros((0, 4)), ceiling=0.5)

    assert result.overall.count == 0
    assert result.controllability_ratio == 0.0


def test_round_trip_result_without_a_ceiling_reports_no_ratio():
    result = round_trip_result("neural", [0], np.array([[1.0, 0.0, 0.0, 0.0]]))

    assert result.controllability_ratio is None


def test_stage_attribution_splits_blame_between_the_two_stages():
    intended = [0, 1, 2, 3]
    classified = [0, 1, 2, 0]  # the text stage misreads the last prompt
    recovered = [0, 9 % 4, 2, 3]  # generation recovers three of them

    attribution = stage_attribution(intended, classified, recovered)
    counts = attribution["counts"]

    assert counts["total"] == 4
    assert counts["text_correct_and_recovered"] == 3
    assert counts["text_wrong_but_recovered_intended"] == 1
    assert attribution["text_accuracy"]["accuracy"] == 0.75


def test_stage_attribution_measures_generation_only_where_text_was_right():
    intended = [0, 1]
    classified = [0, 0]  # second prompt misclassified
    recovered = [0, 0]  # clip matches what conditioned it, not what was intended

    attribution = stage_attribution(intended, classified, recovered)

    # Generation is judged on the one prompt the text stage got right, and it succeeded there.
    assert attribution["generation_accuracy_given_correct_text"]["count"] == 1
    assert attribution["generation_accuracy_given_correct_text"]["accuracy"] == 1.0
    assert attribution["end_to_end_accuracy"]["accuracy"] == 0.5


def test_independence_ratio_is_one_when_the_product_predicts_the_outcome():
    intended = [0, 1, 2, 3]
    attribution = stage_attribution(intended, intended, intended)

    assert attribution["end_to_end_accuracy"]["accuracy"] == 1.0
    assert attribution["predicted_product"] == 1.0
    assert attribution["independence_ratio"] == 1.0


def test_lucky_recovery_is_counted_separately_not_credited_as_competence():
    # The text stage picks Q2 for a prompt meant as Q1, the clip is generated from Q2, yet the
    # probe reads it back as Q1. End-to-end counts it correct, but it is luck: the generation
    # stage was handed the wrong quadrant and did not reproduce it. This is the one case where
    # end-to-end can exceed the product of the stage accuracies, so it must stay visible.
    attribution = stage_attribution(intended=[0], classified=[1], recovered=[0])

    counts = attribution["counts"]
    assert counts["text_wrong_but_recovered_intended"] == 1
    assert counts["text_correct_and_recovered"] == 0
    assert attribution["end_to_end_accuracy"]["accuracy"] == 1.0
    # No prompt had correct text, so there is nothing to credit generation with.
    assert attribution["generation_accuracy_given_correct_text"]["count"] == 0
    assert attribution["predicted_product"] == 0.0
    assert attribution["independence_ratio"] is None


def test_well_formedness_reports_eos_rate_and_decode_yield():
    clips = [
        {"emitted_eos": False, "note_count": 32, "token_count": 129, "repeated_pitch_fraction": 0.1},
        {"emitted_eos": True, "note_count": 16, "token_count": 65, "repeated_pitch_fraction": 0.9},
    ]

    result = well_formedness(clips, max_tokens=128)

    assert result["clip_count"] == 2
    assert result["eos_rate"] == 0.5
    assert result["mean_decode_yield"] == 1.0
    assert result["degenerate_repeat_rate"] == 0.5
    assert result["empty_clip_rate"] == 0.0


def test_well_formedness_flags_empty_clips():
    result = well_formedness([{"emitted_eos": False, "note_count": 0, "token_count": 129}], max_tokens=128)

    assert result["empty_clip_rate"] == 1.0
    assert result["mean_decode_yield"] == 0.0


def test_well_formedness_on_no_clips_is_empty():
    assert well_formedness([], max_tokens=128) == {}


def test_comparison_row_and_table_render_every_column():
    payload = {
        "round_trip": {
            "overall": {"accuracy": 0.6, "count": 50, "ci_low": 0.45, "ci_high": 0.73},
            "axes": {"arousal": {"accuracy": 0.9}, "valence": {"accuracy": 0.55}},
            "controllability_ratio": 0.91,
        },
        "novelty": {"mean_novel_8gram_rate": 0.99, "mean_longest_copied_run": 4.2, "max_longest_copied_run": 9},
        "diversity": {"mean_pairwise_diversity": 0.99, "mean_pairwise_cosine_diversity": 0.98},
        "fidelity": {"mean_overlap": 0.62, "inter_over_intra": 1.3},
        "well_formedness": {"eos_rate": 0.0, "mean_note_count": 128.0, "clip_count": 50},
        "probe_agreement": 0.8,
    }

    row = comparison_row("guidance=3", payload)
    csv_text = comparison_table_csv([row])
    markdown = comparison_table_markdown([row])

    assert row["system"] == "guidance=3"
    assert row["round_trip_accuracy"] == 0.6
    assert row["arousal_accuracy"] == 0.9
    assert row["max_copied_run"] == 9
    assert csv_text.splitlines()[0] == ",".join(COMPARISON_COLUMNS)
    assert "guidance=3" in markdown


def test_comparison_columns_cover_every_key_comparison_row_emits():
    """The column list is written by hand, so it can drift from what the row actually contains.

    A key added to comparison_row but not to COMPARISON_COLUMNS is silently dropped from the CSV.
    """
    row = comparison_row("x", {})

    assert set(row) == set(COMPARISON_COLUMNS)


def test_comparison_row_tolerates_a_system_with_no_round_trip():
    # The unconditional and random-piano floors have no conditioning quadrant to score.
    row = comparison_row("random_piano", {"well_formedness": {"clip_count": 4, "eos_rate": 0.0}})

    assert row["clips"] == 4
    assert row["round_trip_accuracy"] is None
    assert "random_piano" in comparison_table_markdown([row])


def test_token_cap_and_length_stats_are_omitted_for_re_encoded_rows():
    """Re-encoded reference clips exceed the cap by construction, so those columns are not theirs.

    A note-budget clip re-encodes to 4N + 2 tokens against a 4N cap, which would report a ~94%
    cap-hit rate and a mean token count larger than the cap itself, printed beside the generated
    rows and reading higher than any of them.
    """
    reference = [
        {"emitted_eos": True, "note_count": 128, "token_count": 514, "repeated_pitch_fraction": 0.1}
    ] * 10

    result = well_formedness(reference, max_tokens=512, eos_is_learned=False)

    assert result["hit_token_cap_rate"] is None
    assert result["mean_token_count"] is None
    assert result["eos_rate"] is None
    assert result["eos_rate_applicable"] is False
    # Length itself is still reported: it is a property of the clips, not of the sampler.
    assert result["mean_note_count"] == 128.0


def test_a_clip_that_stopped_early_on_eos_does_not_count_as_hitting_the_cap():
    clips = [
        # Ran to the cap: BOS plus max_tokens generated, no ending emitted.
        {"emitted_eos": False, "note_count": 32, "token_count": 129, "repeated_pitch_fraction": 0.0},
        # Emitted EOS as its final token, so it stopped rather than being cut off.
        {"emitted_eos": True, "note_count": 30, "token_count": 129, "repeated_pitch_fraction": 0.0},
    ]

    result = well_formedness(clips, max_tokens=128)

    assert result["hit_token_cap_rate"] == 0.5
    assert result["eos_rate"] == 0.5
