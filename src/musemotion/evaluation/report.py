"""Turning predictions into reportable numbers.

Three ideas do most of the work here.

**Confidence intervals.** With 50 clips per quadrant, an accuracy of 0.55 carries roughly
+/-7 points of sampling error. Reporting the point estimate alone invites reading noise as a
result, so every accuracy comes with a Wilson interval.

**The ceiling.** Round-trip accuracy has no meaning on its own. The probe scores about 0.66
on *real* held-out EMOPIA clips, so that is the achievable maximum, and the controllability
ratio expresses round-trip accuracy as a fraction of it.

**Decomposition.** The four quadrants are two binary axes. Q1/Q2 share high arousal and
Q1/Q4 share high valence, so splitting accuracy along those axes says *which* half of the
emotional signal survives generation. Measured on EMOPIA, note density, velocity, and note
duration separate arousal almost perfectly while mean pitch spans only about three semitones
across all four quadrants - so a large gap between the two axis accuracies is the expected
result, not a bug.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from musemotion.emotions import EMOPIA_QUADRANTS, quadrant_name


# Derived from the quadrant table rather than restated as literals, so the two axes cannot drift
# out of step with the definitions in emotions.py.
HIGH_AROUSAL_IDS = frozenset(
    quadrant.id for quadrant in EMOPIA_QUADRANTS if quadrant.arousal == "high"
)
HIGH_VALENCE_IDS = frozenset(
    quadrant.id for quadrant in EMOPIA_QUADRANTS if quadrant.valence == "high"
)

CHANCE_ACCURACY = 1.0 / len(EMOPIA_QUADRANTS)
BINARY_CHANCE_ACCURACY = 0.5


@dataclass
class AccuracyResult:
    """One accuracy, with the interval and the counts it came from."""

    accuracy: float
    count: int
    correct: int
    ci_low: float
    ci_high: float
    chance: float = CHANCE_ACCURACY

    @property
    def beats_chance(self) -> bool:
        """Whether the interval clears chance, rather than just the point estimate."""
        return self.ci_low > self.chance

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "count": self.count,
            "correct": self.correct,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "chance": self.chance,
            "beats_chance": self.beats_chance,
        }


def wilson_interval(correct: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval, default 95%.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves
    sensibly for small samples and for proportions near 0 or 1 - both of which occur here.
    """
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1.0 + (z**2) / total
    centre = proportion + (z**2) / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + (z**2) / (4 * total**2))
    return max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)


def accuracy_result(correct: int, total: int, chance: float = CHANCE_ACCURACY) -> AccuracyResult:
    low, high = wilson_interval(correct, total)
    return AccuracyResult(
        accuracy=float(correct / total) if total else 0.0,
        count=int(total),
        correct=int(correct),
        ci_low=low,
        ci_high=high,
        chance=chance,
    )


def accuracy_from_labels(
    expected: Sequence[int],
    predicted: Sequence[int],
    chance: float = CHANCE_ACCURACY,
) -> AccuracyResult:
    # Materialised before use: a generator would be drained by the zip and then measured as
    # length zero, quietly reporting an accuracy of 0.0 over "no clips" instead of failing.
    expected_ids = [int(value) for value in expected]
    predicted_ids = [int(value) for value in predicted]
    correct = sum(1 for left, right in zip(expected_ids, predicted_ids) if left == right)
    return accuracy_result(correct, len(expected_ids), chance)


def confusion_matrix(expected: Sequence[int], predicted: Sequence[int]) -> list[list[int]]:
    """Rows are the conditioning quadrant, columns are what the probe recovered."""
    size = len(EMOPIA_QUADRANTS)
    matrix = np.zeros((size, size), dtype="int64")
    for actual, guess in zip(expected, predicted):
        if 0 <= int(actual) < size and 0 <= int(guess) < size:
            matrix[int(actual), int(guess)] += 1
    return matrix.tolist()


def axis_accuracies(
    expected: Sequence[int],
    predicted: Sequence[int],
) -> dict[str, AccuracyResult]:
    """Accuracy along the arousal and valence axes separately."""
    expected_ids = [int(value) for value in expected]
    predicted_ids = [int(value) for value in predicted]
    arousal_correct = sum(
        1
        for actual, guess in zip(expected_ids, predicted_ids)
        if (actual in HIGH_AROUSAL_IDS) == (guess in HIGH_AROUSAL_IDS)
    )
    valence_correct = sum(
        1
        for actual, guess in zip(expected_ids, predicted_ids)
        if (actual in HIGH_VALENCE_IDS) == (guess in HIGH_VALENCE_IDS)
    )
    total = len(expected_ids)
    return {
        "arousal": accuracy_result(arousal_correct, total, BINARY_CHANCE_ACCURACY),
        "valence": accuracy_result(valence_correct, total, BINARY_CHANCE_ACCURACY),
    }


def per_quadrant_accuracy(
    expected: Sequence[int],
    predicted: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Accuracy restricted to each conditioning quadrant in turn."""
    results: dict[str, dict[str, Any]] = {}
    expected_ids = [int(value) for value in expected]
    predicted_ids = [int(value) for value in predicted]
    for quadrant in EMOPIA_QUADRANTS:
        pairs = [
            (actual, guess)
            for actual, guess in zip(expected_ids, predicted_ids)
            if actual == quadrant.id
        ]
        correct = sum(1 for actual, guess in pairs if actual == guess)
        results[quadrant.name] = accuracy_result(correct, len(pairs)).to_dict()
    return results


@dataclass
class RoundTripResult:
    """Everything one probe reports about one set of conditioned generations."""

    probe: str
    overall: AccuracyResult
    axes: dict[str, AccuracyResult]
    per_quadrant: dict[str, dict[str, Any]]
    confusion: list[list[int]]
    ceiling: float | None = None
    mean_confidence: float = 0.0
    predictions: list[int] = field(default_factory=list, repr=False)

    @property
    def controllability_ratio(self) -> float | None:
        """Round-trip accuracy as a fraction of what the probe achieves on real clips."""
        if self.ceiling is None or self.ceiling <= 0:
            return None
        return float(self.overall.accuracy / self.ceiling)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "overall": self.overall.to_dict(),
            "axes": {name: value.to_dict() for name, value in self.axes.items()},
            "per_quadrant": self.per_quadrant,
            "confusion_matrix": self.confusion,
            "confusion_labels": [quadrant.name for quadrant in EMOPIA_QUADRANTS],
            "ceiling": self.ceiling,
            "controllability_ratio": self.controllability_ratio,
            "mean_confidence": self.mean_confidence,
        }


def round_trip_result(
    probe_name: str,
    expected: Sequence[int],
    probabilities: np.ndarray,
    ceiling: float | None = None,
) -> RoundTripResult:
    """Score one probe's predictions against the quadrants that conditioned generation."""
    if probabilities.size == 0:
        empty = accuracy_result(0, 0)
        return RoundTripResult(
            probe=probe_name,
            overall=empty,
            axes={"arousal": empty, "valence": empty},
            per_quadrant={},
            confusion=confusion_matrix([], []),
            ceiling=ceiling,
        )
    predicted = [int(value) for value in np.argmax(probabilities, axis=-1)]
    confidences = probabilities[np.arange(probabilities.shape[0]), predicted]
    return RoundTripResult(
        probe=probe_name,
        overall=accuracy_from_labels(expected, predicted),
        axes=axis_accuracies(expected, predicted),
        per_quadrant=per_quadrant_accuracy(expected, predicted),
        confusion=confusion_matrix(expected, predicted),
        ceiling=ceiling,
        mean_confidence=float(np.mean(confidences)),
        predictions=predicted,
    )


def stage_attribution(
    intended: Sequence[int],
    classified: Sequence[int],
    recovered: Sequence[int],
) -> dict[str, Any]:
    """Localise end-to-end failures between the text stage and the generation stage.

    Adding the text stage can only cost accuracy in expectation, so the useful question is not
    which number is higher but where the loss happens and whether the two stages fail
    independently.

    One exception keeps this from being a strict bound, and it is counted rather than ignored:
    a clip generated from the *wrong* quadrant can still be recovered as the intended one, when
    the two quadrants are close enough that the probe cannot separate them. That is the
    ``text_wrong_but_recovered_intended`` bucket. It is luck, not competence, which is why it is
    reported separately instead of being folded into the success count.

    ``independence_ratio`` divides observed end-to-end accuracy by the product of the two
    stage accuracies. Near 1.0 the stages fail independently. Well below 1.0 the errors are
    correlated - the text stage is misreading exactly the quadrants generation renders least
    legibly - which would mean one fix pays off in both places.
    """
    intended_ids = [int(value) for value in intended]
    classified_ids = [int(value) for value in classified]
    recovered_ids = [int(value) for value in recovered]
    total = len(intended_ids)

    both = text_only = music_only = neither = 0
    for want, text, music in zip(intended_ids, classified_ids, recovered_ids):
        text_ok = text == want
        music_ok = music == want
        if text_ok and music_ok:
            both += 1
        elif text_ok:
            text_only += 1
        elif music_ok:
            # The classifier picked the wrong quadrant, yet the clip still reads as the
            # intended one. Usually means the two quadrants are near-indistinguishable here.
            music_only += 1
        else:
            neither += 1

    text_accuracy = accuracy_from_labels(intended_ids, classified_ids)
    end_to_end = accuracy_from_labels(intended_ids, recovered_ids)
    # Generation measured only where the text stage got it right, so it is not penalised for
    # being handed the wrong quadrant.
    conditioned_pairs = [
        (text, music)
        for want, text, music in zip(intended_ids, classified_ids, recovered_ids)
        if text == want
    ]
    generation_given_text = accuracy_result(
        sum(1 for text, music in conditioned_pairs if text == music), len(conditioned_pairs)
    )
    predicted_product = text_accuracy.accuracy * generation_given_text.accuracy

    return {
        "counts": {
            "total": total,
            "text_correct_and_recovered": both,
            "text_correct_but_not_recovered": text_only,
            "text_wrong_but_recovered_intended": music_only,
            "both_wrong": neither,
        },
        "text_accuracy": text_accuracy.to_dict(),
        "generation_accuracy_given_correct_text": generation_given_text.to_dict(),
        "end_to_end_accuracy": end_to_end.to_dict(),
        "predicted_product": predicted_product,
        "independence_ratio": (
            float(end_to_end.accuracy / predicted_product) if predicted_product > 0 else None
        ),
    }


def well_formedness(
    clips: Sequence[dict[str, Any]],
    max_tokens: int,
    tokens_per_note: int = 4,
    eos_is_learned: bool = True,
) -> dict[str, Any]:
    """Structural health of a batch of generations.

    Decode yield comes back around 0.997 or better, so malformed token groups are not a real
    problem in practice. The EOS rate is the interesting column: most clips run to the token
    cap, but a minority do emit an ending, and the share that does rises sharply with guidance
    (measured: 0.08 at guidance 1.0, 0.33 at guidance 3.0). Length is therefore mostly, but not
    entirely, a function of the sampling cap — which is why length-dependent features stay out
    of the distributional comparison.

    ``eos_is_learned`` must be False for the real-EMOPIA reference row. Those clips are decoded
    and re-encoded through the tokenizer, which appends EOS unconditionally, so their EOS rate
    would be a trivial 1.0 that measures this project's encoder rather than the music. It is
    omitted instead of reported, so nobody reads it as "real music ends properly and generated
    music does not".
    """
    if not clips:
        return {}
    total = len(clips)
    note_counts = [int(clip.get("note_count", 0)) for clip in clips]
    token_counts = [int(clip.get("token_count", 0)) for clip in clips]
    empty = sum(1 for count in note_counts if count == 0)
    # Ratio of notes actually decoded to notes the token budget could have carried.
    possible = [max(1, (tokens - 1) // tokens_per_note) for tokens in token_counts]
    yields = [
        min(1.0, notes / capacity) for notes, capacity in zip(note_counts, possible) if capacity > 0
    ]
    degenerate = sum(1 for clip in clips if float(clip.get("repeated_pitch_fraction", 0.0)) > 0.5)

    # Both of these describe the sampler, so both are meaningless for a re-encoded reference row.
    # A real clip cropped to the note budget re-encodes to 4N + 2 tokens, which exceeds a 4N token
    # cap by construction - it would report a ~94% cap-hit rate and a mean token count larger than
    # the cap itself, printed beside the generated rows and reading higher than any of them.
    if eos_is_learned:
        eos_rate = float(sum(1 for clip in clips if clip.get("emitted_eos")) / total)
        # A clip hit the cap only if it never chose to stop. Length alone cannot tell the two
        # apart: a run that exhausts the budget and a run whose final token is EOS both come back
        # as the BOS prompt plus max_tokens, so the EOS flag is what separates "cut off" from
        # "ended". The prompt is why this compares against max_tokens rather than equalling it.
        capped = sum(
            1
            for clip, count in zip(clips, token_counts)
            if not clip.get("emitted_eos") and count > max_tokens
        )
        hit_cap_rate = float(capped / total)
    else:
        eos_rate = None
        hit_cap_rate = None

    return {
        "clip_count": total,
        "eos_rate": eos_rate,
        "eos_rate_applicable": eos_is_learned,
        "empty_clip_rate": float(empty / total),
        "mean_decode_yield": float(np.mean(yields)) if yields else 0.0,
        "mean_note_count": float(np.mean(note_counts)),
        "mean_token_count": float(np.mean(token_counts)) if eos_is_learned else None,
        "hit_token_cap_rate": hit_cap_rate,
        "degenerate_repeat_rate": float(degenerate / total),
    }


def comparison_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten one system's metrics into a single row of the comparison table.

    Every row is measured the same way, so columns can be read straight down.
    """
    round_trip = payload.get("round_trip", {})
    novelty = payload.get("novelty", {})
    diversity = payload.get("diversity", {})
    fidelity = payload.get("fidelity", {})
    structure = payload.get("well_formedness", {})
    overall = round_trip.get("overall", {})
    axes = round_trip.get("axes", {})
    return {
        "system": label,
        "clips": overall.get("count", structure.get("clip_count")),
        # Which probe produced the round-trip column. It is selected by availability, so a run
        # missing one probe silently reports the other one's number; without this the two runs
        # would look comparable when they are not.
        "probe": payload.get("primary_probe"),
        "round_trip_accuracy": overall.get("accuracy"),
        "round_trip_ci_low": overall.get("ci_low"),
        "round_trip_ci_high": overall.get("ci_high"),
        "arousal_accuracy": axes.get("arousal", {}).get("accuracy"),
        "valence_accuracy": axes.get("valence", {}).get("accuracy"),
        "controllability_ratio": round_trip.get("controllability_ratio"),
        "probe_agreement": payload.get("probe_agreement"),
        "novel_8gram_rate": novelty.get("mean_novel_8gram_rate"),
        "longest_copied_run": novelty.get("mean_longest_copied_run"),
        "max_copied_run": novelty.get("max_longest_copied_run"),
        "diversity_n2": diversity.get("mean_pairwise_diversity"),
        "cosine_diversity_n2": diversity.get("mean_pairwise_cosine_diversity"),
        "mean_feature_overlap": fidelity.get("mean_overlap"),
        "inter_over_intra": fidelity.get("inter_over_intra"),
        "eos_rate": structure.get("eos_rate"),
        "mean_note_count": structure.get("mean_note_count"),
    }


COMPARISON_COLUMNS: tuple[str, ...] = (
    "system",
    "clips",
    "probe",
    "round_trip_accuracy",
    "round_trip_ci_low",
    "round_trip_ci_high",
    "arousal_accuracy",
    "valence_accuracy",
    "controllability_ratio",
    "probe_agreement",
    "novel_8gram_rate",
    "longest_copied_run",
    "max_copied_run",
    "diversity_n2",
    "cosine_diversity_n2",
    "mean_feature_overlap",
    "inter_over_intra",
    "eos_rate",
    "mean_note_count",
)


def comparison_table_csv(rows: Sequence[dict[str, Any]]) -> str:
    """Render the comparison rows as CSV text."""
    lines = [",".join(COMPARISON_COLUMNS)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(column)) for column in COMPARISON_COLUMNS))
    return "\n".join(lines) + "\n"


def comparison_table_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Render a trimmed version of the comparison table as markdown."""
    columns = [
        ("system", "system"),
        ("clips", "n"),
        ("round_trip_accuracy", "round-trip"),
        ("arousal_accuracy", "arousal"),
        ("valence_accuracy", "valence"),
        ("controllability_ratio", "vs ceiling"),
        ("novel_8gram_rate", "novel 8-gram"),
        ("max_copied_run", "max copy"),
        ("cosine_diversity_n2", "diversity"),
        ("mean_feature_overlap", "fidelity"),
    ]
    lines = [
        "| " + " | ".join(header for _, header in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        cells = [_markdown_cell(row.get(key)) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


__all__ = [
    "AccuracyResult",
    "BINARY_CHANCE_ACCURACY",
    "CHANCE_ACCURACY",
    "COMPARISON_COLUMNS",
    "RoundTripResult",
    "accuracy_from_labels",
    "accuracy_result",
    "axis_accuracies",
    "comparison_row",
    "comparison_table_csv",
    "comparison_table_markdown",
    "confusion_matrix",
    "per_quadrant_accuracy",
    "quadrant_name",
    "round_trip_result",
    "stage_attribution",
    "well_formedness",
    "wilson_interval",
]
