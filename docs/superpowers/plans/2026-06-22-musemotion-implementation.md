# MUSEmotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU/Colab-ready MUSEmotion pipeline that fine-tunes BERT on GoEmotions quadrants, trains an emotion-conditioned autoregressive MIDI Transformer on EMOPIA, and serves text-to-MIDI generation through CLI and Gradio.

**Architecture:** Core code lives under `src/musemotion/` with separate modules for config, emotion labels, classifier training, MIDI tokenization, music modeling, inference, and frontend. Large data and checkpoints stay out of git. Local tests use synthetic data so `pytest` can run without GoEmotions, EMOPIA, or a GPU.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers/Datasets/Evaluate, scikit-learn, pretty_midi, PyYAML, Gradio, pytest.

---

## File Structure

- Create `pyproject.toml` for editable installs, pytest config, package metadata, and dependencies.
- Create `requirements.txt` for Colab/simple pip installs.
- Create `configs/classifier.yaml`, `configs/music.yaml`, and `configs/inference.yaml`.
- Replace `README.md` with setup, training, inference, dataset placement, and frontend instructions.
- Create `src/musemotion/config.py` for YAML loading and path expansion.
- Create `src/musemotion/emotions.py` for quadrant constants and GoEmotions mapping.
- Create `src/musemotion/data/goemotions.py` for GoEmotions dataset mapping.
- Create `src/musemotion/music/tokenizer.py` for note-event tokenization and MIDI decoding.
- Create `src/musemotion/music/dataset.py` for tokenized JSONL datasets and collation.
- Create `src/musemotion/models/music_transformer.py` for the emotion-conditioned Transformer decoder.
- Create `src/musemotion/training/classifier.py` and `src/musemotion/training/generator.py` for training loops.
- Create CLI modules under `src/musemotion/cli/`.
- Create `src/musemotion/inference/pipeline.py` and `src/musemotion/frontend/app.py`.
- Create `notebooks/musemotion_colab.ipynb`.
- Create tests under `tests/`.

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `configs/classifier.yaml`
- Create: `configs/music.yaml`
- Create: `configs/inference.yaml`
- Create: `src/musemotion/__init__.py`
- Create: `src/musemotion/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

```python
from pathlib import Path

from musemotion.config import load_yaml_config, resolve_path


def test_load_yaml_config_returns_nested_dictionary(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: bert-base-uncased\n", encoding="utf-8")

    config = load_yaml_config(config_file)

    assert config["model"]["name"] == "bert-base-uncased"


def test_resolve_path_expands_relative_paths_from_repo_root():
    path = resolve_path("artifacts/demo")

    assert isinstance(path, Path)
    assert path.name == "demo"
    assert path.parent.name == "artifacts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -q`

Expected: import failure because `musemotion.config` does not exist.

- [ ] **Step 3: Implement project scaffold and config loader**

Implement `load_yaml_config(path)` with `yaml.safe_load` and `resolve_path(path)` relative to the repo root. Add package metadata and dependencies.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -q`

Expected: `2 passed`.

---

### Task 2: Emotion Quadrant Mapping

**Files:**
- Create: `src/musemotion/emotions.py`
- Create: `src/musemotion/data/__init__.py`
- Create: `src/musemotion/data/goemotions.py`
- Test: `tests/test_emotions.py`

- [ ] **Step 1: Write the failing mapping tests**

```python
from musemotion.emotions import (
    EMOPIA_QUADRANTS,
    map_goemotion_labels_to_quadrant,
    quadrant_id,
)


def test_quadrant_ids_are_stable():
    assert [q.name for q in EMOPIA_QUADRANTS] == ["Q1", "Q2", "Q3", "Q4"]
    assert quadrant_id("Q3") == 2


def test_single_goemotion_label_maps_to_expected_quadrant():
    result = map_goemotion_labels_to_quadrant(["joy"])

    assert result is not None
    assert result.name == "Q1"


def test_conflicting_goemotion_labels_are_dropped():
    assert map_goemotion_labels_to_quadrant(["joy", "grief"]) is None


def test_majority_goemotion_labels_pick_unique_quadrant():
    result = map_goemotion_labels_to_quadrant(["joy", "excitement", "sadness"])

    assert result is not None
    assert result.name == "Q1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_emotions.py -q`

Expected: import failure because `musemotion.emotions` does not exist.

- [ ] **Step 3: Implement labels and mapper**

Create frozen `EmotionQuadrant` dataclass, stable `EMOPIA_QUADRANTS`, `GOEMOTIONS_TO_QUADRANT`, `quadrant_id(name)`, and `map_goemotion_labels_to_quadrant(labels)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_emotions.py -q`

Expected: `4 passed`.

---

### Task 3: MIDI Tokenizer

**Files:**
- Create: `src/musemotion/music/__init__.py`
- Create: `src/musemotion/music/tokenizer.py`
- Test: `tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing tokenizer tests**

```python
from musemotion.music.tokenizer import MidiNote, MusicTokenizer


def test_tokenizer_round_trips_synthetic_notes():
    tokenizer = MusicTokenizer()
    notes = [
        MidiNote(pitch=60, start=0.0, end=0.5, velocity=64),
        MidiNote(pitch=64, start=0.5, end=1.0, velocity=80),
    ]

    token_ids = tokenizer.encode_notes(notes)
    decoded = tokenizer.decode_tokens(token_ids)

    assert [note.pitch for note in decoded] == [60, 64]
    assert decoded[0].start == 0.0
    assert decoded[1].start >= decoded[0].end


def test_tokenizer_can_save_and_load_vocab(tmp_path):
    tokenizer = MusicTokenizer()
    vocab_path = tmp_path / "vocab.json"

    tokenizer.save(vocab_path)
    loaded = MusicTokenizer.load(vocab_path)

    assert loaded.token_to_id == tokenizer.token_to_id
    assert loaded.id_to_token == tokenizer.id_to_token
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tokenizer.py -q`

Expected: import failure because `musemotion.music.tokenizer` does not exist.

- [ ] **Step 3: Implement tokenizer**

Represent notes as `MidiNote`. Implement vocabulary creation, quantization, `encode_notes`, `decode_tokens`, `midi_to_notes`, `notes_to_midi`, `save`, and `load`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tokenizer.py -q`

Expected: `2 passed`.

---

### Task 4: Music Dataset And Generator Model

**Files:**
- Create: `src/musemotion/music/dataset.py`
- Create: `src/musemotion/models/__init__.py`
- Create: `src/musemotion/models/music_transformer.py`
- Test: `tests/test_music_model.py`

- [ ] **Step 1: Write failing dataset/model tests**

```python
import json

import torch

from musemotion.models.music_transformer import MusicTransformerConfig, MusicTransformer
from musemotion.music.dataset import TokenizedMusicDataset, collate_music_batch


def test_music_dataset_and_collate_create_shifted_targets(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text(
        json.dumps({"emotion_id": 1, "token_ids": [1, 10, 11, 2]}) + "\n",
        encoding="utf-8",
    )

    batch = collate_music_batch([TokenizedMusicDataset(data_file)[0]], pad_token_id=0)

    assert batch["input_ids"].tolist() == [[1, 10, 11]]
    assert batch["labels"].tolist() == [[10, 11, 2]]
    assert batch["emotion_ids"].tolist() == [1]


def test_music_transformer_forward_returns_loss_and_logits():
    model = MusicTransformer(MusicTransformerConfig(vocab_size=32, max_seq_len=16, d_model=32, n_heads=4, n_layers=1))
    input_ids = torch.tensor([[1, 4, 5]])
    labels = torch.tensor([[4, 5, 2]])
    emotion_ids = torch.tensor([0])

    output = model(input_ids=input_ids, emotion_ids=emotion_ids, labels=labels)

    assert output.loss is not None
    assert output.logits.shape == (1, 3, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_music_model.py -q`

Expected: import failure because dataset/model modules do not exist.

- [ ] **Step 3: Implement dataset and Transformer**

Implement JSONL loading, padding collator, config dataclass, causal Transformer encoder stack with emotion and positional embeddings, cross-entropy loss ignoring pad labels, and a `sample` method.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_music_model.py -q`

Expected: `2 passed`.

---

### Task 5: Training And Preprocessing CLIs

**Files:**
- Create: `src/musemotion/cli/__init__.py`
- Create: `src/musemotion/cli/train_classifier.py`
- Create: `src/musemotion/cli/prepare_emopia.py`
- Create: `src/musemotion/cli/train_generator.py`
- Create: `src/musemotion/training/__init__.py`
- Create: `src/musemotion/training/classifier.py`
- Create: `src/musemotion/training/generator.py`
- Test: `tests/test_cli_imports.py`

- [ ] **Step 1: Write failing CLI import tests**

```python
import importlib


def test_cli_modules_import_without_side_effects():
    for module_name in [
        "musemotion.cli.train_classifier",
        "musemotion.cli.prepare_emopia",
        "musemotion.cli.train_generator",
    ]:
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_imports.py -q`

Expected: import failure because CLI modules do not exist.

- [ ] **Step 3: Implement real CLI modules**

Implement GoEmotions fine-tuning with Transformers `Trainer`, EMOPIA preprocessing with local MIDI discovery and label parsing, and generator training with PyTorch DataLoader.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_imports.py -q`

Expected: `1 passed`.

---

### Task 6: End-To-End Inference And Frontend

**Files:**
- Create: `src/musemotion/inference/__init__.py`
- Create: `src/musemotion/inference/pipeline.py`
- Create: `src/musemotion/cli/generate.py`
- Create: `src/musemotion/frontend/__init__.py`
- Create: `src/musemotion/frontend/app.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write failing inference test**

```python
from pathlib import Path

from musemotion.inference.pipeline import GenerationResult, generate_with_components


class FakeClassifier:
    def predict(self, text):
        return {"quadrant": "Q1", "emotion_id": 0, "confidence": 0.9, "probabilities": {"Q1": 0.9}}


class FakeGenerator:
    def generate_midi(self, emotion_id, output_path, **kwargs):
        Path(output_path).write_bytes(b"MThd")
        return {"token_count": 4, "path": str(output_path)}


def test_generate_with_components_returns_metadata_and_file(tmp_path):
    output = tmp_path / "clip.mid"

    result = generate_with_components("I feel excited", FakeClassifier(), FakeGenerator(), output)

    assert isinstance(result, GenerationResult)
    assert result.quadrant == "Q1"
    assert output.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference.py -q`

Expected: import failure because inference modules do not exist.

- [ ] **Step 3: Implement inference and Gradio app**

Implement component loading, generation orchestration, CLI generation, and Gradio interface that calls the same pipeline.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference.py -q`

Expected: `1 passed`.

---

### Task 7: Documentation And Final Verification

**Files:**
- Modify: `README.md`
- Create: `notebooks/musemotion_colab.ipynb`

- [ ] **Step 1: Update README**

Document install, dataset placement, training commands, inference commands, frontend launch, artifact layout, and local test command.

- [ ] **Step 2: Add Colab notebook skeleton**

Create a notebook that installs requirements, mounts Drive, checks EMOPIA placement, trains classifier, preprocesses EMOPIA, trains generator, and launches Gradio.

- [ ] **Step 3: Run full local verification**

Run: `python -m pytest -q`

Expected: all tests pass locally without large datasets.

- [ ] **Step 4: Inspect git status**

Run: `git status --short --branch`

Expected: implementation files modified/created; generated folders ignored.
