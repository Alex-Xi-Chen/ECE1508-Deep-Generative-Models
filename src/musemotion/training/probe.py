"""Train the MIDI-to-quadrant probes used to score generated clips.

The probes are judges, so their own accuracy is the ceiling every round-trip number gets
read against. This module therefore reports three things per probe, not one:

* accuracy and macro-F1 on **real held-out EMOPIA test clips** - the ceiling,
* the same metrics on the validation split, used only for checkpoint selection,
* accuracy after retraining on **permuted labels** - a control that must collapse to chance.

If the control does not collapse, the probe is reading something other than emotion and no
downstream metric means anything. That check is the reason this module trains twice.

Both probes see exactly the same notes. Clips are decoded, head-cropped to ``note_budget``,
then re-encoded, so the neural probe's token stream and the feature probe's feature vector
describe an identical 128-note window rather than two differently-truncated ones.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from musemotion.config import resolve_path
from musemotion.emotions import EMOPIA_QUADRANTS, quadrant_name
from musemotion.evaluation.distributions import comparable_feature_names
from musemotion.evaluation.features import feature_matrix
from musemotion.evaluation.probe import (
    DEFAULT_NOTE_BUDGET,
    PROBE_METADATA_FILENAME,
    FeatureMidiProbe,
    NeuralMidiProbe,
    load_clip_splits,
)
from musemotion.models.music_classifier import MusicClassifier, MusicClassifierConfig
from musemotion.music.dataset import collate_probe_batch
from musemotion.music.tokenizer import MusicTokenizer
from musemotion.training.classifier import balanced_class_weights, compute_classifier_metrics
from musemotion.training.generator import write_generator_history


PROBE_HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "best_validation_macro_f1",
]

# The probe trains on the same features the fidelity comparison uses, from one shared
# definition. The excluded ones are the length-dependent features: most generated clips stop at
# the token cap, so absolute length carries no emotional information here - it only tells the
# probe which generator setting produced the clip, which would be leakage. Deriving this from
# `comparable_feature_names` rather than restating the list keeps the probe and the distribution
# comparison from drifting apart if a feature is added later.
PROBE_FEATURE_NAMES: tuple[str, ...] = tuple(comparable_feature_names())

# Macro-F1 is averaged over all four quadrants everywhere in this module, so a run that
# collapses onto a subset (the shuffled-label control, typically) stays comparable with one
# that does not.
_ALL_QUADRANT_IDS: tuple[int, ...] = tuple(quadrant.id for quadrant in EMOPIA_QUADRANTS)


def train_music_probe(config: dict[str, Any]) -> dict[str, Any]:
    """Train both probes, write them to disk, and return the metadata that was recorded."""
    data_config = config.get("data", {})
    tokenized_dir = resolve_path(data_config.get("tokenized_dir", "artifacts/music/tokenized"))
    tokenizer_path = resolve_path(data_config.get("tokenizer", "artifacts/music/tokenizer/vocab.json"))
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found. Run prepare_emopia first: {tokenizer_path}")

    note_budget = int(data_config.get("note_budget", DEFAULT_NOTE_BUDGET))
    tokenizer = MusicTokenizer.load(tokenizer_path)
    splits = load_clip_splits(tokenized_dir, tokenizer, note_budget)
    if not splits.get("train"):
        raise FileNotFoundError(f"No tokenized training clips found under {tokenized_dir}")

    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 1508))
    _seed_everything(seed)

    output_dir = resolve_path(training_config.get("output_dir", "artifacts/probe"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_controls = bool(config.get("controls", {}).get("shuffled_labels", True))

    quadrant_counts = {name: _quadrant_counts(rows) for name, rows in splits.items()}
    metadata: dict[str, Any] = {
        "note_budget": note_budget,
        "seed": seed,
        "chance_accuracy": 1.0 / len(EMOPIA_QUADRANTS),
        # Uniform chance is the wrong bar for *accuracy* on an imbalanced split. This test split
        # runs 20/24/30/34, so a model that always answers Q4 scores 0.315 while learning nothing.
        # A control has to be read against this, not against 0.25 - macro-F1 is the statistic that
        # is not fooled by it, which is why both are reported.
        "majority_class_accuracy": _majority_class_rate(quadrant_counts.get("test", {})),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "split_quadrant_counts": quadrant_counts,
        "probes": {},
    }

    feature_probe, feature_metrics = train_feature_probe(splits, config)
    feature_probe.save(output_dir)
    metadata["probes"]["feature"] = feature_metrics

    neural_probe, neural_metrics = train_neural_probe(splits, tokenizer, config, output_dir)
    neural_probe.save(output_dir)
    metadata["probes"]["neural"] = neural_metrics

    if run_controls:
        metadata["probes"]["feature"]["shuffled_label_control"] = _shuffled_feature_control(
            splits, config, seed
        )
        metadata["probes"]["neural"]["shuffled_label_control"] = _shuffled_neural_control(
            splits, tokenizer, config, seed
        )

    (output_dir / PROBE_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _print_summary(metadata)
    return metadata


def train_feature_probe(
    splits: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    permutation_seed: int | None = None,
) -> tuple[FeatureMidiProbe, dict[str, Any]]:
    """Fit standardised logistic regression over symbolic features.

    ``permutation_seed`` runs this as the shuffled-label control: labels are permuted within
    each split, so nothing the probe could learn survives. Test metrics are still scored against
    the *true* test labels, which is the question being asked - can a probe fitted on noise
    predict real emotion?
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_config = config.get("feature_probe", {})
    note_budget = int(config.get("data", {}).get("note_budget", DEFAULT_NOTE_BUDGET))
    names = list(PROBE_FEATURE_NAMES)
    splits = _permuted_splits(splits, permutation_seed)

    train_rows = splits["train"]
    train_x = feature_matrix((row["notes"] for row in train_rows), names)
    train_y = np.asarray([row["emotion_id"] for row in train_rows], dtype="int64")

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(feature_config.get("C", 1.0)),
                    max_iter=int(feature_config.get("max_iter", 2000)),
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)
    # The probe keeps the fitted numbers rather than the estimator object, so the saved artifact
    # loads without executing code and survives a scikit-learn version change.
    probe = FeatureMidiProbe.from_pipeline(pipeline, feature_names=names, note_budget=note_budget)

    metrics = {
        "feature_names": names,
        "train": _score_feature_probe(probe, train_rows),
        "validation": _score_feature_probe(probe, splits.get("validation", [])),
        "test": _score_feature_probe(probe, splits.get("test", [])),
        "coefficients": probe.coefficients(),
    }
    return probe, metrics


def train_neural_probe(
    splits: dict[str, list[dict[str, Any]]],
    tokenizer: MusicTokenizer,
    config: dict[str, Any],
    output_dir: Path | None,
    permutation_seed: int | None = None,
    write_history: bool = True,
) -> tuple[NeuralMidiProbe, dict[str, Any]]:
    """Train the bidirectional encoder probe, selecting the checkpoint on validation macro-F1.

    ``permutation_seed`` runs this as the shuffled-label control. The permutation covers the
    validation split as well as the training split, which matters: the checkpoint is chosen by
    validation macro-F1, so permuting only the training labels would leave the control free to
    pick whichever epoch best predicts the very labels it is supposed to have no access to. Test
    metrics are still scored against true labels, since that is the question being asked.
    """
    model_config_values = dict(config.get("model", {}))
    training_config = config.get("training", {})
    note_budget = int(config.get("data", {}).get("note_budget", DEFAULT_NOTE_BUDGET))
    splits = _permuted_splits(splits, permutation_seed)

    train_rows = list(splits["train"])

    # A clip of N notes encodes to 4N + 2 tokens; size the position table to fit exactly.
    model_config_values.setdefault("max_seq_len", note_budget * 4 + 2)
    model_config = MusicClassifierConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        num_classes=len(EMOPIA_QUADRANTS),
        **model_config_values,
    )
    model = MusicClassifier(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = int(training_config.get("batch_size", 16))
    collate = partial(
        collate_probe_batch,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=model_config.max_seq_len,
    )
    train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True, collate_fn=collate)
    validation_rows = splits.get("validation", [])
    validation_loader = (
        DataLoader(validation_rows, batch_size=batch_size, shuffle=False, collate_fn=collate)
        if validation_rows
        else None
    )

    class_weights = _class_weights([int(row["emotion_id"]) for row in train_rows])
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )
    epochs = int(training_config.get("epochs", 20))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))

    # Without a validation split there is nothing to select on. Falling through would score
    # every epoch 0.0, overwrite the checkpoint each time (0.0 >= 0.0), and then record
    # best_validation_macro_f1: 0.0 as though it had been measured - and that number is the
    # ceiling every round-trip accuracy is divided by. Selecting the last epoch is the honest
    # behaviour, and it is stated rather than inferred.
    can_select = validation_loader is not None
    if not can_select:
        print(
            "probe: no validation split found, so the final epoch is kept rather than the best; "
            "no validation ceiling will be recorded"
        )

    best_macro_f1: float | None = None
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_probe_epoch(model, train_loader, optimizer, device, grad_clip_norm, class_weights)
        evaluation = _evaluate_probe(model, validation_loader, device, class_weights)
        macro_f1 = evaluation.get("macro_f1") if can_select else None
        if macro_f1 is not None and (best_macro_f1 is None or macro_f1 >= best_macro_f1):
            best_macro_f1 = macro_f1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif not can_select:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(
            f"probe epoch={epoch} train_loss={train_loss:.4f} "
            + (
                f"validation_loss={evaluation['loss']:.4f} "
                f"validation_accuracy={evaluation['accuracy']:.4f} "
                f"validation_macro_f1={macro_f1:.4f}"
                if macro_f1 is not None
                else "(no validation split)"
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                # None rather than NaN: json.dumps writes a bare NaN literal that strict JSON
                # parsers such as jq and JSON.parse reject outright.
                "validation_loss": evaluation.get("loss"),
                "validation_accuracy": evaluation.get("accuracy"),
                "validation_macro_f1": macro_f1,
                "best_validation_macro_f1": best_macro_f1,
            }
        )
        if write_history and output_dir is not None:
            write_generator_history(output_dir, history, fields=PROBE_HISTORY_FIELDS)

    model.load_state_dict(best_state)
    probe = NeuralMidiProbe(model=model, tokenizer=tokenizer, device=device, note_budget=note_budget)

    metrics = {
        "model_config": asdict(model_config),
        "epochs": epochs,
        "best_validation_macro_f1": best_macro_f1,
        "train": _score_neural_probe(probe, train_rows),
        "validation": _score_neural_probe(probe, validation_rows),
        "test": _score_neural_probe(probe, splits.get("test", [])),
    }
    return probe, metrics


def _train_probe_epoch(
    model: MusicClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    grad_clip_norm: float,
    class_weights: torch.Tensor | None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch, class_weights=class_weights)
        assert output.loss is not None
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(output.loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def _evaluate_probe(
    model: MusicClassifier,
    loader: DataLoader | None,
    device: torch.device,
    class_weights: torch.Tensor | None,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    total_loss = 0.0
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch, class_weights=class_weights)
        assert output.loss is not None
        total_loss += float(output.loss.item())
        logits.append(output.logits.detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy())
    if not logits:
        return {}
    metrics = compute_classifier_metrics(
        (np.concatenate(logits), np.concatenate(labels)), label_set=_ALL_QUADRANT_IDS
    )
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def _score_feature_probe(
    probe: FeatureMidiProbe,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {}
    probabilities = probe.predict_proba_batch([row["notes"] for row in rows])
    truth = np.asarray([row["emotion_id"] for row in rows], dtype="int64")
    return _classification_summary(probabilities, truth)


def _score_neural_probe(probe: NeuralMidiProbe, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    probabilities = probe.predict_proba_batch([row["notes"] for row in rows])
    truth = np.asarray([row["emotion_id"] for row in rows], dtype="int64")
    return _classification_summary(probabilities, truth)


def _classification_summary(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro-F1, and the confusion matrix, reusing the classifier's metric helper."""
    metrics = compute_classifier_metrics((probabilities, truth), label_set=_ALL_QUADRANT_IDS)
    predictions = np.argmax(probabilities, axis=-1)
    size = len(EMOPIA_QUADRANTS)
    confusion = np.zeros((size, size), dtype="int64")
    for actual, predicted in zip(truth, predictions):
        confusion[int(actual), int(predicted)] += 1
    return {
        "count": int(truth.size),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": confusion.tolist(),
    }


def _shuffled_feature_control(
    splits: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    _, metrics = train_feature_probe(splits, config, permutation_seed=seed)
    return {"test": metrics.get("test", {}), "validation": metrics.get("validation", {})}


def _shuffled_neural_control(
    splits: dict[str, list[dict[str, Any]]],
    tokenizer: MusicTokenizer,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed + 1)
    _, metrics = train_neural_probe(
        splits, tokenizer, config, output_dir=None, permutation_seed=seed, write_history=False
    )
    return {"test": metrics.get("test", {}), "validation": metrics.get("validation", {})}


def _permuted_splits(
    splits: dict[str, list[dict[str, Any]]],
    seed: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Permute labels within train and validation, leaving test untouched.

    Validation has to be permuted too. The neural probe selects its checkpoint on validation
    macro-F1, so a control with true validation labels would hand-pick the epoch that best
    predicts labels it is meant not to have seen - a contaminated control that still looks like
    a clean one. Test labels stay true because the control's question is whether a probe fitted
    on noise can recover real emotion.
    """
    if seed is None:
        return splits
    permuted = dict(splits)
    for offset, name in enumerate(("train", "validation")):
        rows = splits.get(name)
        if not rows:
            continue
        labels = [int(row["emotion_id"]) for row in rows]
        random.Random(seed + offset).shuffle(labels)
        permuted[name] = [
            {**row, "emotion_id": label} for row, label in zip(rows, labels)
        ]
    return permuted


def _class_weights(labels: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency weights, from the same helper the text classifier uses."""
    return torch.tensor(balanced_class_weights(labels), dtype=torch.float)


def _quadrant_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {quadrant.name: 0 for quadrant in EMOPIA_QUADRANTS}
    for row in rows:
        counts[quadrant_name(int(row["emotion_id"]))] += 1
    return counts


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _majority_class_rate(counts: dict[str, int]) -> float:
    """Accuracy a model reaches by always predicting the most common class."""
    total = sum(counts.values())
    return float(max(counts.values()) / total) if total else 0.0


def _print_summary(metadata: dict[str, Any]) -> None:
    chance = metadata["chance_accuracy"]
    majority = metadata.get("majority_class_accuracy", chance)
    print(
        "\nprobe ceilings on real held-out EMOPIA test clips "
        "(uniform chance = %.3f, always-majority-class = %.3f)" % (chance, majority)
    )
    for name, values in metadata["probes"].items():
        test = values.get("test", {})
        control = values.get("shuffled_label_control", {}).get("test", {})
        print(
            "  %-8s accuracy=%.4f macro_f1=%.4f" % (
                name, test.get("accuracy", float("nan")), test.get("macro_f1", float("nan"))
            )
        )
        if control:
            # Macro-F1 is the honest one for the control: a run that collapses onto the frequent
            # classes can match the majority-class accuracy while its macro-F1 stays near chance.
            print(
                "           shuffled-label control accuracy=%.4f macro_f1=%.4f%s"
                % (
                    control.get("accuracy", float("nan")),
                    control.get("macro_f1", float("nan")),
                    "  <- at or below the majority-class rate, so it learned nothing"
                    if control.get("accuracy", 1.0) <= majority + 0.02
                    else "  <- ABOVE the majority-class rate: investigate before trusting this probe",
                )
            )
