"""MIDI-to-quadrant probes: the "backward" model in the round-trip check.

Two independent probes, one interface. Both read *music* - a MIDI file or a decoded note
list - never the generator's internal token stream. Scoring the artifact a listener would
actually receive keeps the judge honest and means any decoding bug shows up in the metric
instead of hiding behind it.

``NeuralMidiProbe`` is a small bidirectional Transformer over the tokenised clip.
``FeatureMidiProbe`` is logistic regression over hand-built symbolic features. They share
the tokenizer's note representation but nothing else, so their agreement rate is real
evidence: a finding both of them confirm is far harder to dismiss than one probe's opinion.

Both crop to the same ``note_budget`` used at training time. Most generated clips run to the
token cap instead of emitting an ending, so without a matched budget the probe would be
trained on ~225-note real clips and asked about 128-note generated ones, and that domain
shift alone would sink accuracy for reasons unrelated to how well conditioning works.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from musemotion.emotions import EMOPIA_QUADRANTS, quadrant_id, quadrant_name
from musemotion.evaluation.features import FEATURE_NAMES, feature_vector
from musemotion.music.tokenizer import MidiNote, MusicTokenizer


DEFAULT_NOTE_BUDGET = 128

# A clip of N notes encodes to 1 (BOS) + 4N + 1 (EOS) tokens, so the default sequence limit
# is sized to hold exactly DEFAULT_NOTE_BUDGET notes without truncation.
DEFAULT_MAX_SEQ_LEN = DEFAULT_NOTE_BUDGET * 4 + 2

NEURAL_PROBE_FILENAME = "neural_probe.pt"
FEATURE_PROBE_FILENAME = "feature_probe.json"
PROBE_METADATA_FILENAME = "probe_metadata.json"


class MidiProbe(Protocol):
    """What every probe must offer, so callers never branch on which one they hold."""

    name: str

    def predict_proba_notes(self, notes: Iterable[MidiNote]) -> np.ndarray:
        ...

    def predict_notes(self, notes: Iterable[MidiNote]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ProbePrediction:
    quadrant: str
    emotion_id: int
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "quadrant": self.quadrant,
            "emotion_id": self.emotion_id,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


def _warn_on_tokenizer_mismatch(saved: MusicTokenizer, supplied: MusicTokenizer) -> None:
    """Warn when a probe is asked to read a vocabulary other than the one it was trained on.

    Compares the actual token-to-id mapping rather than just its size, because the sizes agree
    in exactly the case that matters: a tokenizer rebuilt over a different pitch range produces
    the same number of tokens with different meanings.
    """
    if saved.token_to_id == supplied.token_to_id:
        return
    differing = sum(
        1
        for token, index in saved.token_to_id.items()
        if supplied.token_to_id.get(token) != index
    )
    print(
        "warning: the probe was trained on a different tokenizer vocabulary than the one it is "
        f"being given ({differing} of {len(saved.token_to_id)} tokens map differently, "
        f"sizes {saved.vocab_size} and {supplied.vocab_size}). Its predictions describe a "
        "different note mapping than the clips it is scoring."
    )


def notes_from_midi(midi_path: str | Path) -> list[MidiNote]:
    """Read a MIDI file into notes. The tokenizer vocabulary is irrelevant here."""
    return MusicTokenizer().midi_to_notes(midi_path)


def crop_notes(notes: Iterable[MidiNote], note_budget: int | None) -> list[MidiNote]:
    """Head-crop to ``note_budget`` notes, in onset order."""
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    if note_budget is None:
        return ordered
    return ordered[:note_budget]


def load_clip_splits(
    tokenized_dir: str | Path,
    tokenizer: MusicTokenizer,
    note_budget: int,
) -> dict[str, list[dict[str, Any]]]:
    """Read the tokenized splits and normalise every clip to the same note window.

    Decode, head-crop, re-encode. Re-encoding rather than truncating the raw token stream keeps
    every sequence well formed - a clean BOS, whole four-token note groups, and EOS - instead of
    ending mid-note wherever the cap happened to land. Clips that decode to nothing are dropped.

    Probe training and the evaluation harness both load through this one function on purpose.
    The contract that makes the round-trip metric valid is that the probe is trained on exactly
    the window it is later asked about; two separate loaders could drift apart and break that
    silently, with no test failing and every number still looking plausible.
    """
    root = Path(tokenized_dir)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        path = root / f"{split_name}.jsonl"
        if not path.exists():
            continue
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            notes = crop_notes(tokenizer.decode_tokens(payload["token_ids"]), note_budget)
            if not notes:
                continue
            rows.append(
                {
                    "emotion_id": int(payload["emotion_id"]),
                    "notes": notes,
                    "token_ids": tokenizer.encode_notes(notes),
                    "source": payload.get("source"),
                }
            )
        splits[split_name] = rows
    return splits


class _ProbeMixin:
    """Shared plumbing: the note-list path is canonical, everything else routes through it."""

    name: str
    note_budget: int | None

    def predict_proba_notes(self, notes: Iterable[MidiNote]) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict_notes(self, notes: Iterable[MidiNote]) -> dict[str, Any]:
        probabilities = np.asarray(self.predict_proba_notes(notes), dtype="float64")
        emotion_id = int(np.argmax(probabilities))
        return ProbePrediction(
            quadrant=quadrant_name(emotion_id),
            emotion_id=emotion_id,
            confidence=float(probabilities[emotion_id]),
            probabilities={
                quadrant.name: float(probabilities[quadrant.id]) for quadrant in EMOPIA_QUADRANTS
            },
        ).to_dict()

    def predict_from_midi(self, midi_path: str | Path) -> dict[str, Any]:
        return self.predict_notes(notes_from_midi(midi_path))

    def predict_many_notes(self, clips: Sequence[Iterable[MidiNote]]) -> list[dict[str, Any]]:
        return [self.predict_notes(notes) for notes in clips]


class NeuralMidiProbe(_ProbeMixin):
    """Bidirectional Transformer encoder over the tokenised clip."""

    name = "neural"

    def __init__(
        self,
        model: Any,
        tokenizer: MusicTokenizer,
        device: Any = None,
        note_budget: int | None = DEFAULT_NOTE_BUDGET,
    ):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device or "cpu")
        self.note_budget = note_budget
        self.model.to(self.device)
        self.model.eval()

    def predict_proba_notes(self, notes: Iterable[MidiNote]) -> np.ndarray:
        return self.predict_proba_batch([notes])[0]

    def predict_proba_batch(
        self, clips: Sequence[Iterable[MidiNote]], chunk_size: int = 64
    ) -> np.ndarray:
        """Score many clips, in chunks. Returns a ``(clips, 4)`` array.

        Chunking is not an optimisation, it is what keeps this from running out of memory.
        Attention is quadratic in sequence length and linear in batch, so scoring all 862
        training clips at 514 tokens in one pass asks for several gigabytes in a single
        allocation. That call happens during probe training, before the checkpoint is written,
        so an out-of-memory error there would discard the whole run.
        """
        if not clips:
            return np.zeros((0, len(EMOPIA_QUADRANTS)), dtype="float64")
        if len(clips) > chunk_size:
            return np.vstack(
                [
                    self.predict_proba_batch(clips[start : start + chunk_size], chunk_size)
                    for start in range(0, len(clips), chunk_size)
                ]
            )

        import torch

        limit = self.model.config.max_seq_len
        sequences = [
            self.tokenizer.encode_notes(crop_notes(notes, self.note_budget))[:limit] for notes in clips
        ]
        # A clip that decoded to nothing still needs a row, so fall back to a bare BOS.
        sequences = [sequence or [self.tokenizer.bos_token_id] for sequence in sequences]
        width = max(len(sequence) for sequence in sequences)
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.tensor(
            [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            [[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        probabilities = self.model.predict_proba(input_ids=input_ids, attention_mask=attention_mask)
        return probabilities.detach().cpu().numpy().astype("float64")

    def save(self, directory: str | Path) -> Path:
        import torch

        from dataclasses import asdict

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_config": asdict(self.model.config),
                "state_dict": self.model.state_dict(),
                "note_budget": self.note_budget,
            },
            output_dir / NEURAL_PROBE_FILENAME,
        )
        self.tokenizer.save(output_dir / "tokenizer.json")
        return output_dir / NEURAL_PROBE_FILENAME

    @classmethod
    def from_artifacts(
        cls,
        directory: str | Path,
        device: str | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> "NeuralMidiProbe":
        import torch

        from musemotion.models.music_classifier import MusicClassifier, MusicClassifierConfig

        probe_dir = Path(directory)
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # weights_only=True keeps loading to tensors and plain containers instead of running the
        # pickle machinery. The payload is a config dict, a state dict, and an int, so nothing
        # here needs the unrestricted loader.
        payload = torch.load(
            probe_dir / NEURAL_PROBE_FILENAME, map_location=target_device, weights_only=True
        )
        model = MusicClassifier(MusicClassifierConfig(**payload["model_config"]))
        model.load_state_dict(payload["state_dict"])

        saved_tokenizer_path = probe_dir / "tokenizer.json"
        tokenizer = MusicTokenizer.load(tokenizer_path or saved_tokenizer_path)
        # Callers pass the generator's tokenizer so probe and generator share one vocabulary.
        # That is only safe if it is the vocabulary the probe's embeddings were trained on, and
        # matching vocab_size is not enough to establish it: re-running prepare_emopia with a
        # different min_pitch yields the same 88 pitch tokens bound to different pitches, and the
        # probe would then read every clip transposed with nothing raising.
        if tokenizer_path is not None and saved_tokenizer_path.exists():
            _warn_on_tokenizer_mismatch(MusicTokenizer.load(saved_tokenizer_path), tokenizer)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=target_device,
            note_budget=payload.get("note_budget", DEFAULT_NOTE_BUDGET),
        )


class FeatureMidiProbe(_ProbeMixin):
    """Standardised multinomial logistic regression over hand-built symbolic features.

    Representation-independent from the generator, trains in seconds on CPU, and exposes
    per-feature coefficients - which is what turns "valence is near chance" from an
    assertion into something attributable to specific features.

    The fitted model is stored as its numbers rather than as a pickled estimator: the
    standardiser's mean and scale, the coefficient matrix, the intercepts, and the class order.
    Three things follow from that. Loading executes no code. The artifact stays readable and
    diffable. And it does not break when scikit-learn changes version, which pickled estimators
    routinely do - scoring needs only numpy, so the evaluation runs even where sklearn is a
    different version than the one that trained it.
    """

    name = "feature"

    def __init__(
        self,
        coefficients: np.ndarray,
        intercepts: np.ndarray,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        classes: Sequence[int],
        feature_names: Sequence[str] = FEATURE_NAMES,
        note_budget: int | None = DEFAULT_NOTE_BUDGET,
    ):
        self.coefficients_matrix = np.asarray(coefficients, dtype="float64")
        self.intercepts = np.asarray(intercepts, dtype="float64")
        self.scaler_mean = np.asarray(scaler_mean, dtype="float64")
        # A zero-variance column would divide by zero; the training-time scaler substitutes 1.0
        # for those, and this mirrors it so a loaded probe behaves identically.
        scale = np.asarray(scaler_scale, dtype="float64")
        self.scaler_scale = np.where(scale > 0, scale, 1.0)
        self.classes = [int(value) for value in classes]
        self.feature_names = list(feature_names)
        self.note_budget = note_budget

    @classmethod
    def from_pipeline(
        cls,
        pipeline: Any,
        feature_names: Sequence[str] = FEATURE_NAMES,
        note_budget: int | None = DEFAULT_NOTE_BUDGET,
    ) -> "FeatureMidiProbe":
        """Extract the numbers from a fitted ``StandardScaler`` + ``LogisticRegression``."""
        scaler = pipeline.named_steps["scale"]
        model = pipeline.named_steps["model"]
        return cls(
            coefficients=model.coef_,
            intercepts=model.intercept_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            classes=model.classes_,
            feature_names=feature_names,
            note_budget=note_budget,
        )

    def predict_proba_notes(self, notes: Iterable[MidiNote]) -> np.ndarray:
        return self.predict_proba_batch([notes])[0]

    def predict_proba_batch(self, clips: Sequence[Iterable[MidiNote]]) -> np.ndarray:
        if not clips:
            return np.zeros((0, len(EMOPIA_QUADRANTS)), dtype="float64")
        matrix = np.vstack(
            [feature_vector(crop_notes(notes, self.note_budget), self.feature_names) for notes in clips]
        )
        standardised = (matrix - self.scaler_mean) / self.scaler_scale
        scores = standardised @ self.coefficients_matrix.T + self.intercepts
        return self._align_columns(_softmax_or_sigmoid(scores))

    def _align_columns(self, probabilities: np.ndarray) -> np.ndarray:
        """Reorder the model's class columns onto canonical quadrant ids.

        Classes are ordered by sorted label, which happens to match quadrant ids for a fully
        populated training set. A split missing a quadrant would silently shift every column,
        so the mapping is made explicit rather than assumed.
        """
        aligned = np.zeros((probabilities.shape[0], len(EMOPIA_QUADRANTS)), dtype="float64")
        for column, emotion_id in enumerate(self.classes):
            if 0 <= emotion_id < len(EMOPIA_QUADRANTS):
                aligned[:, emotion_id] = probabilities[:, column]
        return aligned

    def coefficients(self) -> dict[str, dict[str, float]]:
        """Per-quadrant linear coefficients, keyed by feature name.

        A binary fit produces a single coefficient row for two classes, where the row describes
        the second class and the first is its negation. Indexing by class position would run off
        the end of that matrix, so the two-class case is handled explicitly.
        """
        rows = self.coefficients_matrix
        if rows.shape[0] == 1 and len(self.classes) == 2:
            paired = [(self.classes[0], -rows[0]), (self.classes[1], rows[0])]
        else:
            paired = list(zip(self.classes, rows))
        return {
            quadrant_name(emotion_id): {
                name: float(value) for name, value in zip(self.feature_names, row)
            }
            for emotion_id, row in paired
            if 0 <= emotion_id < len(EMOPIA_QUADRANTS)
        }

    def save(self, directory: str | Path) -> Path:
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / FEATURE_PROBE_FILENAME
        destination.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "note_budget": self.note_budget,
                    "classes": self.classes,
                    "scaler_mean": self.scaler_mean.tolist(),
                    "scaler_scale": self.scaler_scale.tolist(),
                    "coefficients": self.coefficients_matrix.tolist(),
                    "intercepts": self.intercepts.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_artifacts(cls, directory: str | Path) -> "FeatureMidiProbe":
        payload = json.loads(
            (Path(directory) / FEATURE_PROBE_FILENAME).read_text(encoding="utf-8")
        )
        return cls(
            coefficients=np.asarray(payload["coefficients"], dtype="float64"),
            intercepts=np.asarray(payload["intercepts"], dtype="float64"),
            scaler_mean=np.asarray(payload["scaler_mean"], dtype="float64"),
            scaler_scale=np.asarray(payload["scaler_scale"], dtype="float64"),
            classes=payload["classes"],
            feature_names=payload.get("feature_names", FEATURE_NAMES),
            note_budget=payload.get("note_budget", DEFAULT_NOTE_BUDGET),
        )


def _softmax_or_sigmoid(scores: np.ndarray) -> np.ndarray:
    """Turn decision scores into probabilities the way logistic regression does.

    A single score column means a binary fit, where the probability is the sigmoid and the
    negative class takes the complement. More columns mean a multinomial fit, where the
    probabilities are the softmax. Both are computed shifted by the row maximum so a large
    score cannot overflow the exponential.
    """
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if scores.shape[1] == 1:
        positive = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        return np.column_stack([1.0 - positive, positive])
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def load_probes(
    directory: str | Path,
    device: str | None = None,
    tokenizer_path: str | Path | None = None,
) -> dict[str, MidiProbe]:
    """Load whichever probes are present, keyed by name. Missing ones are simply absent."""
    probe_dir = Path(directory)
    probes: dict[str, MidiProbe] = {}
    if (probe_dir / NEURAL_PROBE_FILENAME).exists():
        probes["neural"] = NeuralMidiProbe.from_artifacts(
            probe_dir, device=device, tokenizer_path=tokenizer_path
        )
    if (probe_dir / FEATURE_PROBE_FILENAME).exists():
        probes["feature"] = FeatureMidiProbe.from_artifacts(probe_dir)
    return probes


def load_probe_metadata(directory: str | Path) -> dict[str, Any]:
    """Probe accuracy on real held-out clips, plus the controls, if they were recorded.

    Round-trip accuracy is uninterpretable without this: the ceiling it should be read
    against is whatever the probe scores on real music.
    """
    path = Path(directory) / PROBE_METADATA_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def probe_agreement(first: Sequence[int], second: Sequence[int]) -> float:
    """Fraction of clips on which two probes predict the same quadrant."""
    if not first or len(first) != len(second):
        return 0.0
    matches = sum(1 for left, right in zip(first, second) if int(left) == int(right))
    return float(matches / len(first))


__all__ = [
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_NOTE_BUDGET",
    "FeatureMidiProbe",
    "MidiProbe",
    "NeuralMidiProbe",
    "ProbePrediction",
    "crop_notes",
    "load_probe_metadata",
    "load_probes",
    "notes_from_midi",
    "probe_agreement",
    "quadrant_id",
]
