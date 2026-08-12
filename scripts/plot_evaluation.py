"""Figures for an evaluation run.

Every function takes the parsed ``metrics.json`` and **returns** a matplotlib figure, so the
Colab notebook can display them inline. Running the module as a script instead writes PNGs:

    python scripts/plot_evaluation.py --metrics artifacts/evaluation/metrics.json --save

Palette and axis styling come from ``scripts/plot_style.py``, shared with the training charts so
the two sets of figures sit together without clashing.

Two conventions worth stating, because they are easy to get wrong:

* **One y-axis per plot, always.** Where two measures share a panel they are both unitless
  proportions in [0, 1]. Measures on different scales get their own panel instead, never a
  second axis - a twin axis invents whatever correlation its arbitrary alignment implies.
* **The table is the accessible twin.** The harness writes ``comparison_table.csv`` next to
  ``metrics.json``; every value plotted here is readable there without relying on colour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Sibling module, resolved via the sys.path entry above.
from plot_style import (  # noqa: E402
    BLUE,
    FIGURES,
    GREEN,
    INK,
    MUTED_INK,
    NEUTRAL_MARK,
    ORANGE,
    PURPLE,
    REFERENCE,
    REPO_ROOT,
    SURFACE,
    style_axis,
)

DEFAULT_METRICS = REPO_ROOT / "artifacts/evaluation/metrics.json"

# Single hue, light to dark: magnitude, so one hue only.
BLUE_RAMP = LinearSegmentedColormap.from_list("musemotion_blue", ["#f7fafc", BLUE, "#1a365d"])

# The harness names each swept system "guidance=<scale>"; parsing and relabelling both key off
# this prefix, so it lives in one place.
GUIDANCE_PREFIX = "guidance="

QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
QUADRANT_CHARACTER = {
    "Q1": "positive\nenergetic",
    "Q2": "negative\nenergetic",
    "Q3": "negative\nsubdued",
    "Q4": "positive\ncalm",
}


def _guidance_systems(systems: Any) -> list[str]:
    """The guidance-sweep system keys, ordered by their numeric scale."""
    return sorted(
        (name for name in systems if name.startswith(GUIDANCE_PREFIX)),
        key=lambda name: float(name.split("=", 1)[1]),
    )


def _system_label(name: str, short: bool = False) -> str:
    """Axis label for a system key: ``guidance=3`` reads as ``guidance 3``, or ``g 3`` short."""
    return name.replace(GUIDANCE_PREFIX, "g " if short else "guidance ")


def _generated_systems(metrics: dict[str, Any]) -> list[str]:
    """Systems with a round-trip score, real clips first, then the guidance sweep."""
    systems = metrics.get("systems", {})
    ordered = [name for name in ("real",) if name in systems]
    ordered.extend(_guidance_systems(systems))
    ordered.extend(name for name in ("end_to_end",) if name in systems)
    return [name for name in ordered if "round_trip" in systems.get(name, {})]


def _round_trip(metrics: dict[str, Any], system: str, probe: str | None = None) -> dict[str, Any]:
    payload = metrics.get("systems", {}).get(system, {})
    if probe is None:
        return payload.get("round_trip", {})
    return payload.get("round_trip_by_probe", {}).get(probe, {})


def _ceiling(metrics: dict[str, Any]) -> float | None:
    real = _round_trip(metrics, "real").get("overall", {})
    return real.get("accuracy")


def round_trip_figure(metrics: dict[str, Any]) -> Figure:
    """Round-trip accuracy per system, with intervals, against chance and the ceiling.

    One series, so no legend: the title names it. The shipped configuration is the only bar
    given the series colour and the rest are neutral, because the comparison is "how does the
    shipped setting do", not "here are eight equal categories".
    """
    systems = [name for name in _generated_systems(metrics) if name != "real"]
    ceiling = _ceiling(metrics)
    shipped = _shipped_system(metrics)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    values, low_error, high_error, labels, colours = [], [], [], [], []
    for name in systems:
        overall = _round_trip(metrics, name).get("overall", {})
        accuracy = overall.get("accuracy")
        if accuracy is None:
            continue
        values.append(accuracy)
        low_error.append(max(0.0, accuracy - overall.get("ci_low", accuracy)))
        high_error.append(max(0.0, overall.get("ci_high", accuracy) - accuracy))
        labels.append(_system_label(name))
        colours.append(BLUE if name == shipped else NEUTRAL_MARK)

    positions = range(len(values))
    ax.bar(
        positions,
        values,
        color=colours,
        width=0.6,
        yerr=[low_error, high_error],
        capsize=4,
        error_kw={"ecolor": MUTED_INK, "elinewidth": 1.0, "capthick": 1.0},
    )
    # Above the interval cap, not the bar top, so the value never sits on the whisker.
    for position, value, high in zip(positions, values, high_error):
        ax.text(position, value + high + 0.03, f"{value:.3f}", ha="center", fontsize=9, color=INK)

    # Reference lines are labelled in a gutter to the right of the last bar. Writing them over
    # the plot put the ceiling caption on top of a bar, where it was unreadable.
    gutter = len(values) - 0.5
    ax.set_xlim(-0.6, gutter + 1.5)
    ax.axhline(0.25, color=REFERENCE, linewidth=1.0)
    ax.text(gutter + 0.15, 0.25, "chance\n(4-class)", fontsize=8, color=REFERENCE, va="center")
    if ceiling is not None:
        ax.axhline(ceiling, color=ORANGE, linewidth=1.2)
        ax.text(
            gutter + 0.15,
            ceiling,
            f"ceiling {ceiling:.3f}\n(probe on real clips)",
            fontsize=8,
            color=ORANGE,
            va="center",
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    style_axis(
        ax,
        "Round-trip quadrant recovery — does the conditioning survive generation?",
        "",
        "accuracy (95% Wilson interval)",
        legend=False,
    )
    fig.tight_layout()
    return fig


def axis_decomposition_figure(metrics: dict[str, Any]) -> Figure:
    """Arousal against valence accuracy, per system.

    The four quadrants are two binary axes, and separating them says which half of the
    emotional signal actually survives. Chance for a binary split is 0.5, not 0.25.
    """
    systems = _generated_systems(metrics)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    width = 0.36
    positions = list(range(len(systems)))
    for offset, (axis_name, colour) in enumerate((("arousal", BLUE), ("valence", ORANGE))):
        values = [
            _round_trip(metrics, name).get("axes", {}).get(axis_name, {}).get("accuracy", 0.0)
            for name in systems
        ]
        shifted = [position + (offset - 0.5) * width for position in positions]
        ax.bar(shifted, values, width=width, color=colour, label=axis_name)
        for x, value in zip(shifted, values):
            ax.text(x, value + 0.02, f"{value:.2f}", ha="center", fontsize=8, color=INK)

    gutter = len(systems) - 0.5
    ax.set_xlim(-0.7, gutter + 1.3)
    ax.axhline(0.5, color=REFERENCE, linewidth=1.0)
    ax.text(gutter + 0.15, 0.5, "chance\n(binary axis)", fontsize=8, color=REFERENCE, va="center")
    ax.set_xticks(positions)
    ax.set_xticklabels([_system_label(name) for name in systems], fontsize=9)
    ax.set_ylim(0, 1.05)
    style_axis(
        ax,
        "Which half of the signal survives — arousal vs valence",
        "",
        "binary axis accuracy",
    )
    fig.tight_layout()
    return fig


def guidance_sweep_figure(metrics: dict[str, Any]) -> Figure:
    """Controllability against distributional fidelity across the guidance sweep.

    Both series are unitless proportions in [0, 1], so they legitimately share one axis.
    Crossing curves are the finding: guidance buys recoverable emotion and pays for it in
    resemblance to real EMOPIA.
    """
    systems = _guidance_systems(metrics.get("systems", {}))
    scales = [float(name.split("=", 1)[1]) for name in systems]
    accuracy = [_round_trip(metrics, name).get("overall", {}).get("accuracy") for name in systems]
    overlap = [
        metrics["systems"][name].get("fidelity", {}).get("mean_overlap") for name in systems
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(scales, accuracy, "-o", color=BLUE, linewidth=2, markersize=8, label="round-trip accuracy")
    ax.plot(
        scales,
        overlap,
        "-s",
        color=ORANGE,
        linewidth=2,
        markersize=8,
        label="feature overlap with real EMOPIA",
    )
    # Endpoints only: a value on every marker would be noise.
    for series, colour in ((accuracy, BLUE), (overlap, ORANGE)):
        if series and series[-1] is not None:
            ax.annotate(
                f"{series[-1]:.3f}",
                (scales[-1], series[-1]),
                textcoords="offset points",
                xytext=(8, -3),
                fontsize=9,
                color=colour,
            )

    # Reference-line captions live in a gutter left of the first data point; written at the
    # first x they collided with the guidance=1 markers.
    span = (max(scales) - min(scales)) if len(scales) > 1 else 1.0
    gutter = min(scales) - span * 0.16
    ax.set_xlim(gutter - span * 0.02, max(scales) + span * 0.1)

    ceiling = _ceiling(metrics)
    if ceiling is not None:
        ax.axhline(ceiling, color=REFERENCE, linewidth=1.0)
        ax.text(gutter, ceiling, "probe\nceiling", fontsize=8, color=REFERENCE, va="center", ha="center")
    ax.axhline(0.25, color=REFERENCE, linewidth=1.0, linestyle="-", alpha=0.5)
    ax.text(gutter, 0.25, "chance", fontsize=8, color=REFERENCE, alpha=0.9, va="center", ha="center")

    ax.set_xticks(scales)
    ax.set_ylim(0, 1.05)
    style_axis(
        ax,
        "Classifier-free guidance: controllability against fidelity",
        "guidance scale (1.0 = guidance off)",
        "proportion",
    )
    fig.tight_layout()
    return fig


def confusion_figure(
    metrics: dict[str, Any], system: str | None = None, probe: str | None = None
) -> Figure:
    """Confusion matrix as a single-hue heatmap, rows normalised.

    Rows are the conditioning quadrant, columns what the probe recovered. Counts are printed
    in every cell, so the figure never depends on colour alone to be read.
    """
    system = system or _shipped_system(metrics)
    payload = _round_trip(metrics, system, probe)
    matrix = payload.get("confusion_matrix")
    if not matrix:
        raise ValueError(f"No confusion matrix recorded for system {system!r}")

    row_totals = [max(1, sum(row)) for row in matrix]
    shares = [[value / total for value in row] for row, total in zip(matrix, row_totals)]

    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(shares, cmap=BLUE_RAMP, vmin=0.0, vmax=1.0)

    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            share = shares[row][column]
            ax.text(
                column,
                row,
                f"{matrix[row][column]}\n{share:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                # Ink flips on dark cells for contrast; it is never the series colour.
                color=SURFACE if share > 0.55 else INK,
            )

    labels = [f"{name}\n{QUADRANT_CHARACTER[name]}" for name in QUADRANTS]
    ax.set_xticks(range(4), labels=labels, fontsize=8)
    ax.set_yticks(range(4), labels=QUADRANTS, fontsize=9)
    ax.set_xlabel("recovered by the probe", fontsize=9, color=MUTED_INK)
    ax.set_ylabel("conditioning quadrant", fontsize=9, color=MUTED_INK)
    ax.set_title(
        f"Round-trip confusion — {system}", fontsize=11, fontweight="bold", color=INK, loc="left"
    )
    ax.tick_params(colors=MUTED_INK, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("share of the conditioning quadrant", fontsize=8, color=MUTED_INK)
    bar.ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def _draw_reference_panel(
    ax,
    metrics: dict[str, Any],
    systems: Sequence[str],
    title: str,
    extract: Any,
    limits: tuple[float, float] | None,
) -> None:
    """One small-multiple panel: a bar per system, with real EMOPIA carrying the accent colour.

    Real clips are the yardstick every panel is read against, so they keep the accent hue while
    the generated rows stay neutral. Colour follows the entity, identically in every panel.
    """
    ax.set_facecolor(SURFACE)
    values, colours, labels = [], [], []
    for name in systems:
        value = extract(metrics.get("systems", {}).get(name, {}))
        if value is None:
            continue
        values.append(value)
        colours.append(ORANGE if name == "real" else BLUE)
        labels.append(_system_label(name, short=True))

    positions = range(len(values))
    ax.bar(positions, values, color=colours, width=0.62)
    for position, value in zip(positions, values):
        text = f"{value:.3f}" if isinstance(value, float) else str(value)
        ax.text(position, value, text, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    if limits:
        ax.set_ylim(*limits)
    style_axis(ax, title, "", "", legend=False)


def novelty_figure(metrics: dict[str, Any]) -> Figure:
    """Novelty and diversity against the real-data reference.

    Three panels rather than one, because the measures carry different units - two
    proportions and a count of notes. Small multiples keep each on its own honest scale
    instead of forcing a second y-axis.
    """
    systems = _generated_systems(metrics)
    panels = (
        ("mean novel 8-gram rate", lambda payload: payload.get("novelty", {}).get("mean_novel_8gram_rate"), (0, 1.05)),
        (
            "longest copied run (notes)",
            lambda payload: payload.get("novelty", {}).get("max_longest_copied_run"),
            None,
        ),
        (
            "pairwise cosine diversity",
            lambda payload: payload.get("diversity", {}).get("mean_pairwise_cosine_diversity"),
            (0, 1.05),
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    for ax, (title, extract, limits) in zip(axes, panels):
        _draw_reference_panel(ax, metrics, systems, title, extract, limits)

    fig.suptitle(
        "Novelty and diversity — orange is real held-out EMOPIA, the reference frame",
        fontsize=11,
        fontweight="bold",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def feature_distribution_figure(
    metrics: dict[str, Any],
    system: str | None = None,
    features: Sequence[str] = ("note_density", "mean_velocity", "mean_duration", "mean_pitch"),
) -> Figure:
    """Per-quadrant feature means, generated against real, as small multiples.

    This is the measured version of the claim that the quadrants sound different. Each panel
    holds one feature on its own scale.
    """
    system = system or _shipped_system(metrics)
    generated = metrics.get("systems", {}).get(system, {}).get("per_quadrant_features", {})
    real = metrics.get("systems", {}).get("real", {}).get("per_quadrant_features", {})
    if not generated:
        raise ValueError(f"No per-quadrant features recorded for system {system!r}")

    columns = 2
    rows = (len(features) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.4 * rows), facecolor=SURFACE)
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    width = 0.36
    for ax, feature in zip(flat, features):
        ax.set_facecolor(SURFACE)
        positions = list(range(len(QUADRANTS)))
        for offset, (label, source, colour) in enumerate(
            (("real EMOPIA", real, ORANGE), (system, generated, BLUE))
        ):
            values = [source.get(quadrant, {}).get(feature, 0.0) for quadrant in QUADRANTS]
            shifted = [position + (offset - 0.5) * width for position in positions]
            ax.bar(shifted, values, width=width, color=colour, label=label)
        ax.set_xticks(positions)
        ax.set_xticklabels(QUADRANTS, fontsize=9)
        style_axis(ax, feature.replace("_", " "), "", "", legend=(ax is flat[0]))
    for ax in flat[len(features) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Per-quadrant features, generated against real EMOPIA",
        fontsize=11,
        fontweight="bold",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def stage_attribution_figure(metrics: dict[str, Any], probe: str | None = None) -> Figure:
    """Where end-to-end failures happen, as one part-to-whole bar of four outcomes.

    End-to-end accuracy cannot exceed generator-only accuracy, so the question this answers is
    not which stage is better but where the loss lands.
    """
    attribution = metrics.get("end_to_end", {}).get("stage_attribution", {})
    if not attribution:
        raise ValueError("No end-to-end stage attribution recorded in this run")
    probe = probe or ("feature" if "feature" in attribution else next(iter(attribution)))
    counts = attribution[probe]["counts"]

    segments = (
        ("text right, clip recovered", counts.get("text_correct_and_recovered", 0), GREEN),
        ("text right, clip not recovered", counts.get("text_correct_but_not_recovered", 0), ORANGE),
        ("text wrong, clip still read as intended", counts.get("text_wrong_but_recovered_intended", 0), PURPLE),
        ("both wrong", counts.get("both_wrong", 0), NEUTRAL_MARK),
    )
    total = max(1, counts.get("total", sum(value for _, value, _ in segments)))

    fig, ax = plt.subplots(figsize=(10, 2.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    left = 0.0
    for label, value, colour in segments:
        share = value / total
        # 2px surface gap between adjacent segments rather than a stroke around them.
        ax.barh([0], [share], left=left, color=colour, height=0.5, label=f"{label} ({value})")
        if share > 0.08:
            ax.text(
                left + share / 2,
                0,
                f"{share:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color=SURFACE,
                fontweight="bold",
            )
        left += share
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)

    payload = attribution[probe]
    ratio = payload.get("independence_ratio")
    subtitle = (
        f"end-to-end {payload['end_to_end_accuracy']['accuracy']:.3f}  ·  "
        f"predicted product {payload['predicted_product']:.3f}"
    )
    if ratio is not None:
        subtitle += f"  ·  independence ratio {ratio:.2f}"
    ax.set_title(
        f"End-to-end failure attribution ({probe} probe, n={total})\n{subtitle}",
        fontsize=11,
        fontweight="bold",
        color=INK,
        loc="left",
    )
    ax.tick_params(colors=MUTED_INK, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    for text in legend.get_texts():
        text.set_color(MUTED_INK)
    fig.tight_layout()
    return fig


def _dig(payload: Any, path: Sequence[str]) -> float | None:
    """Walk a nested dict by key path, returning None the moment anything is missing."""
    node = payload
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node


def _draw_optional_bars(
    ax,
    positions: Sequence[float],
    values: Sequence[float | None],
    width: float,
    colour: str,
    label: str,
) -> None:
    """Bar series where a missing value is annotated rather than drawn as zero.

    A measurement that was never taken must not render as a 0.000 bar: in the probe-quality
    figure that would assert a collapse far below chance on the strength of a missing run, which
    is the opposite of what the figure exists to show.
    """
    present = [(x, value) for x, value in zip(positions, values) if value is not None]
    if present:
        ax.bar(
            [x for x, _ in present],
            [value for _, value in present],
            width=width,
            color=colour,
            label=label,
        )
    for x, value in zip(positions, values):
        if value is None:
            ax.text(x, 0.03, "not run", ha="center", fontsize=8, color=MUTED_INK, rotation=90)
        else:
            ax.text(x, value + 0.015, f"{value:.3f}", ha="center", fontsize=9, color=INK)


def probe_quality_figure(metrics: dict[str, Any]) -> Figure:
    """Each probe's accuracy on real clips, beside its shuffled-label control.

    The control is the reason to trust anything downstream: a probe trained on permuted labels
    has to collapse to chance. If it does not, the probe is reading something other than
    emotion and no round-trip number means anything.
    """
    probes = metrics.get("probe_metadata", {}).get("probes", {})
    if not probes:
        raise ValueError("No probe metadata recorded in this run")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = list(probes)
    width = 0.36
    positions = list(range(len(names)))
    for offset, (label, colour, path) in enumerate(
        (
            ("trained on real labels", BLUE, ("test", "accuracy")),
            ("shuffled-label control", NEUTRAL_MARK, ("shuffled_label_control", "test", "accuracy")),
        )
    ):
        values = [_dig(probes[name], path) for name in names]
        shifted = [position + (offset - 0.5) * width for position in positions]
        _draw_optional_bars(ax, shifted, values, width, colour, label)

    # Reference captions sit in a gutter right of the bars; written over the plot they landed on
    # the control bar's own value label.
    gutter = len(names) - 0.5
    ax.set_xlim(-0.7, gutter + 1.35)
    ax.axhline(0.25, color=REFERENCE, linewidth=1.0)
    ax.text(gutter + 0.12, 0.25, "uniform\nchance", fontsize=8, color=REFERENCE, va="center")
    # The bar a control actually has to clear. The test split is imbalanced, so always answering
    # the most common quadrant already scores above uniform chance without learning anything.
    majority = metrics.get("probe_metadata", {}).get("majority_class_accuracy")
    if majority:
        ax.axhline(majority, color=ORANGE, linewidth=1.0, linestyle="-")
        ax.text(
            gutter + 0.12,
            majority + 0.075,
            f"always\nmajority-class\n({majority:.3f})",
            fontsize=8,
            color=ORANGE,
            va="center",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{name} probe" for name in names], fontsize=9)
    ax.set_ylim(0, 1.0)
    style_axis(ax, "Probe quality on real held-out EMOPIA clips", "", "accuracy")
    fig.tight_layout()
    return fig


def _shipped_system(metrics: dict[str, Any]) -> str:
    """The guidance row matching the configured inference default, else the last one."""
    generation = metrics.get("run", {}).get("config", {}).get("generation", {})
    configured = generation.get("guidance_scale")
    systems = metrics.get("systems", {})
    if configured is not None:
        candidate = f"{GUIDANCE_PREFIX}{float(configured):g}"
        if candidate in systems:
            return candidate
    guidance = _guidance_systems(systems)
    return guidance[-1] if guidance else next(iter(systems), "")


FIGURE_BUILDERS = {
    "probe_quality": probe_quality_figure,
    "round_trip": round_trip_figure,
    "axis_decomposition": axis_decomposition_figure,
    "guidance_sweep": guidance_sweep_figure,
    "confusion": confusion_figure,
    "novelty": novelty_figure,
    "feature_distributions": feature_distribution_figure,
    "stage_attribution": stage_attribution_figure,
}


def build_all(metrics: dict[str, Any], strict: bool = False) -> dict[str, Figure]:
    """Build every figure the run supports, skipping any whose inputs are absent.

    Only ``ValueError`` is treated as "this run has no data for that figure" - that is what the
    three deliberate guards raise. Anything else is a defect in the plotting code or a change in
    the metrics schema, so it propagates by default rather than scrolling past as a "skipped"
    line above eight rendered plots.
    """
    figures: dict[str, Figure] = {}
    for name, builder in FIGURE_BUILDERS.items():
        try:
            figures[name] = builder(metrics)
        except ValueError as error:
            print(f"skipped {name}: {error}")
        except Exception as error:
            if strict:
                raise
            # Surfaced as a failure rather than a skip: the figure is missing because the code
            # broke, not because the run lacks that measurement.
            print(f"FAILED {name}: {type(error).__name__}: {error}")
    return figures


def load_metrics(path: str | Path = DEFAULT_METRICS) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    # Selecting a headless backend belongs here, not at import. This module is imported by the
    # Colab notebook to render figures inline; switching the backend at import time would
    # replace the already-initialised inline backend and make every plt.show() in the session
    # silently draw nothing.
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Render figures for an evaluation run.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS), help="Path to metrics.json.")
    parser.add_argument("--save", action="store_true", help="Write PNGs instead of only building them.")
    parser.add_argument("--output-dir", default=str(FIGURES), help="Where to write PNGs.")
    args = parser.parse_args(argv)

    metrics = load_metrics(args.metrics)
    figures = build_all(metrics)
    if not args.save:
        print(f"built {len(figures)} figures: {', '.join(figures)}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        destination = output_dir / f"evaluation_{name}.png"
        figure.savefig(destination, dpi=130, facecolor=SURFACE)
        plt.close(figure)
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
