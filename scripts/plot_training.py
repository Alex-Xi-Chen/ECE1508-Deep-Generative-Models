"""Regenerate the real-training figures and summary CSV from the committed histories.

Reads the per-epoch CSV histories under ``models/real_training/`` and writes the
charts + ``figures/real_training_summary.csv`` used by the README. Rerun after a new
training run:

    python scripts/plot_training.py

Requires matplotlib + pandas (already in the project dev environment).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "figures"
CLASSIFIER_HISTORY = REPO_ROOT / "models/real_training/classifier_bert_base/training_history.csv"
MUSIC_HISTORY = REPO_ROOT / "models/real_training/music_transformer_fulldata/training_history.csv"

# Dataset sizes for this run (GoEmotions full after neutral-drop + tie filtering;
# EMOPIA full 80/10/10 split). Baked in so the summary is reproducible without the
# gitignored artifacts/ tokenized files.
CLS_TRAIN, CLS_VAL, CLS_TEST = 25873, 3249, 3270
MUS_TRAIN, MUS_VAL, MUS_TEST = 862, 108, 108

TRAIN_C = "#2b6cb0"   # blue
VAL_C = "#dd6b20"     # orange
ACC_C = "#2f855a"     # green
F1_C = "#6b46c1"      # purple


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def classifier_curve(cls: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(cls.epoch, cls.train_loss, "-o", color=TRAIN_C, label="train loss")
    ax1.plot(cls.epoch, cls.validation_loss, "-o", color=VAL_C, label="val loss")
    _style(ax1, "BERT classifier: loss", "epoch", "cross-entropy")

    ax2.plot(cls.epoch, cls.validation_accuracy, "-o", color=ACC_C, label="val accuracy")
    ax2.plot(cls.epoch, cls.validation_macro_f1, "-o", color=F1_C, label="val macro-F1")
    best = cls.loc[cls.validation_macro_f1.idxmax()]
    ax2.axvline(best.epoch, ls="--", color="gray", alpha=0.7)
    ax2.annotate(f"best (ep {int(best.epoch)})", (best.epoch, best.validation_macro_f1),
                 textcoords="offset points", xytext=(6, -12), fontsize=9)
    _style(ax2, "BERT classifier: accuracy & macro-F1", "epoch", "score")
    fig.suptitle("Emotion classifier — full GoEmotions, bert-base-uncased", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "classifier_training_curve.png", dpi=130)
    plt.close(fig)


def music_curve(mus: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mus.epoch, mus.train_loss, "-", color=TRAIN_C, label="train loss")
    ax.plot(mus.epoch, mus.validation_loss, "-", color=VAL_C, label="val loss")
    best = mus.loc[mus.validation_loss.idxmin()]
    ax.scatter([best.epoch], [best.validation_loss], color="red", zorder=5,
               label=f"best val {best.validation_loss:.3f} (ep {int(best.epoch)})")
    _style(ax, "Music generator — full EMOPIA, emotion-conditioned Transformer",
           "epoch", "cross-entropy")
    fig.tight_layout()
    fig.savefig(FIGURES / "music_generator_training_curve.png", dpi=130)
    plt.close(fig)


def learning_curves(cls: pd.DataFrame, mus: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(cls.epoch, cls.train_loss, "-o", color=TRAIN_C, label="train")
    ax1.plot(cls.epoch, cls.validation_loss, "-o", color=VAL_C, label="validation")
    _style(ax1, "Classifier loss", "epoch", "cross-entropy")
    ax2.plot(mus.epoch, mus.train_loss, "-", color=TRAIN_C, label="train")
    ax2.plot(mus.epoch, mus.validation_loss, "-", color=VAL_C, label="validation")
    _style(ax2, "Music generator loss", "epoch", "cross-entropy")
    fig.suptitle("Learning curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=130)
    plt.close(fig)


def _gap_axis(ax, epoch, gap, title):
    ax.plot(epoch, gap, "-o", color="#c53030", markersize=3)
    ax.axhline(0, ls="--", color="gray", alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss - train loss")
    ax.grid(True, alpha=0.3)


def overfitting(cls: pd.DataFrame, mus: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _gap_axis(ax1, cls.epoch, cls.validation_loss - cls.train_loss, "Classifier val-train loss gap")
    _gap_axis(ax2, mus.epoch, mus.validation_loss - mus.train_loss, "Music generator val-train loss gap")
    fig.suptitle("Overfitting analysis (gap between validation and training loss)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "overfitting_analysis.png", dpi=130)
    plt.close(fig)


def best_metrics(cls_best, mus_best_loss, test_acc, test_f1) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["val acc", "val macro-F1", "test acc", "test macro-F1"]
    values = [cls_best.validation_accuracy, cls_best.validation_macro_f1, test_acc, test_f1]
    bars = ax1.bar(labels, values, color=[ACC_C, F1_C, ACC_C, F1_C], alpha=0.85)
    ax1.axhline(0.25, ls="--", color="gray", label="chance (4-class)")
    ax1.set_ylim(0, 1)
    ax1.set_title("Classifier best/test scores", fontsize=12, fontweight="bold")
    ax1.bar_label(bars, fmt="%.3f", padding=2)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(["best val loss"], [mus_best_loss], color=VAL_C, alpha=0.85, width=0.4)
    ax2.set_title("Music generator best val loss", fontsize=12, fontweight="bold")
    ax2.text(0, mus_best_loss, f"{mus_best_loss:.3f}", ha="center", va="bottom")
    ax2.set_ylim(0, max(2.0, mus_best_loss * 1.3))
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "best_metrics.png", dpi=130)
    plt.close(fig)


def summary_table(rows: list[list[str]], header: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=header)
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.axis("off")
    table = ax.table(cellText=df.to_numpy(), colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#2b6cb0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "performance_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    cls = pd.read_csv(CLASSIFIER_HISTORY)
    mus = pd.read_csv(MUSIC_HISTORY)

    cls_best = cls.loc[cls.validation_macro_f1.idxmax()]  # trainer selects on macro-F1
    mus_best = mus.loc[mus.validation_loss.idxmin()]
    # Test-set eval of the selected model (from metrics.json), reported in README.
    test_acc, test_f1 = 0.8275, 0.7885

    classifier_curve(cls)
    music_curve(mus)
    learning_curves(cls, mus)
    overfitting(cls, mus)
    best_metrics(cls_best, mus_best.validation_loss, test_acc, test_f1)

    header = ["Model", "Dataset", "Train Examples", "Validation Examples", "Epochs",
              "Initial Val Loss", "Final Val Loss", "Best Val Loss", "Best Epoch",
              "Best Val Acc (%)", "Best Macro F1 (%)", "Final Train Loss", "Final Loss Gap"]
    rows = [
        ["BERT Emotion Classifier", "GoEmotions (full)", CLS_TRAIN, CLS_VAL, int(cls.epoch.max()),
         round(cls.validation_loss.iloc[0], 4), round(cls.validation_loss.iloc[-1], 4),
         round(cls.validation_loss.min(), 4), int(cls_best.epoch),
         round(cls_best.validation_accuracy * 100, 2), round(cls_best.validation_macro_f1 * 100, 2),
         round(cls.train_loss.iloc[-1], 4),
         round(cls.validation_loss.iloc[-1] - cls.train_loss.iloc[-1], 4)],
        ["Music Transformer Generator", "EMOPIA (full)", MUS_TRAIN, MUS_VAL, int(mus.epoch.max()),
         round(mus.validation_loss.iloc[0], 4), round(mus.validation_loss.iloc[-1], 4),
         round(mus.validation_loss.min(), 4), int(mus_best.epoch),
         "", "",
         round(mus.train_loss.iloc[-1], 4),
         round(mus.validation_loss.iloc[-1] - mus.train_loss.iloc[-1], 4)],
    ]
    df = summary_table(rows, header)
    df.to_csv(FIGURES / "real_training_summary.csv", index=False)
    print("wrote figures + real_training_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
