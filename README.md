# MUSEmotion

MUSEmotion generates a short piano MIDI clip from a user's emotional text.

The project has two trained stages:

1. **Emotion classification:** fine-tune BERT on GoEmotions after mapping fine-grained labels to EMOPIA's four valence-arousal quadrants.
2. **Conditional music generation:** train an emotion-conditioned autoregressive Transformer on tokenized EMOPIA piano MIDI clips.

## Repository Layout

```text
configs/                 Training and inference configs
src/musemotion/          Python package
tests/                   Local unit tests with synthetic fixtures
notebooks/               Colab entrypoint
data/raw/emopia/         Expected local EMOPIA dataset path, ignored by git
artifacts/               Generated tokenized data, checkpoints, metrics, samples
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
```

On Colab, install with:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Dataset Preparation

GoEmotions is downloaded automatically by Hugging Face Datasets:

```python
from datasets import load_dataset
load_dataset("go_emotions")
```

EMOPIA is not stored in this repo. Place the extracted MIDI dataset under:

```text
data/raw/emopia/
```

The preprocessing script recursively scans for `.mid` and `.midi` files whose path contains `Q1`, `Q2`, `Q3`, or `Q4`.

## Train The Emotion Classifier

```bash
python -m musemotion.cli.train_classifier --config configs/classifier.yaml
```

Outputs are written to `artifacts/classifier/`:

- fine-tuned BERT weights
- tokenizer files
- `metrics.json`
- `label_mapping.json`

## Preprocess EMOPIA

```bash
python -m musemotion.cli.prepare_emopia --config configs/music.yaml
```

Outputs are written to:

- `artifacts/music/tokenizer/vocab.json`
- `artifacts/music/tokenized/train.jsonl`
- `artifacts/music/tokenized/validation.jsonl`
- `artifacts/music/tokenized/test.jsonl`

## Train The Music Generator

```bash
python -m musemotion.cli.train_generator --config configs/music.yaml
```

The generator is a decoder-only Transformer trained with next-token maximum likelihood and emotion embeddings. Checkpoints are written to `artifacts/music/checkpoints/`.

## Generate A MIDI Clip

```bash
python -m musemotion.cli.generate ^
  --text "I feel hopeful but calm today" ^
  --output artifacts/samples/demo.mid
```

The command prints JSON metadata with the predicted quadrant, confidence, token count, and MIDI path.

## Launch The Frontend

```bash
python -m musemotion.frontend.app --config configs/inference.yaml
```

The Gradio interface lets a user type emotional text, generate a conditioned MIDI clip, and download the result.

## Local Verification

The local tests do not require GoEmotions, EMOPIA, GPU access, or trained checkpoints.

```bash
pytest -q
```

## Emotion Quadrants

- `Q1`: high valence, high arousal, positive and energetic
- `Q2`: low valence, high arousal, negative and energetic
- `Q3`: low valence, low arousal, negative and subdued
- `Q4`: high valence, low arousal, positive and calm
