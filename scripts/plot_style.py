"""Shared chart styling for the figure scripts.

Both ``plot_training.py`` and ``plot_evaluation.py`` draw into the same README and the same
notebook, so their palette and axis treatment live here rather than being copied into each. One
repo-root definition too, instead of one per script.

The colour order is not arbitrary. These four hues pass the categorical checks - lightness band,
chroma floor, contrast against the surface - in this order, but with green adjacent to orange the
worst colour-vision separation is ΔE 6.5 for protanopia, which is borderline; ordering them
blue, orange, purple, green lifts it to 21.3. Hues are assigned per series and never cycled.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "figures"

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
PURPLE = "#6b46c1"
GREEN = "#2f855a"
SERIES = (BLUE, ORANGE, PURPLE, GREEN)

SURFACE = "#ffffff"
INK = "#1a202c"
MUTED_INK = "#4a5568"
GRID = "#cbd5e0"
REFERENCE = "#718096"
NEUTRAL_MARK = "#a0aec0"


def style_axis(ax, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    """Recessive grid and axes, left-aligned bold title, text in ink rather than series colour."""
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED_INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED_INK)
    # Hairline and solid: dashes read as a threshold when this is only a grid.
    ax.grid(True, alpha=0.35, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
    if legend and ax.get_legend_handles_labels()[0]:
        frame = ax.legend(fontsize=8, frameon=False, loc="best")
        for text in frame.get_texts():
            text.set_color(MUTED_INK)
