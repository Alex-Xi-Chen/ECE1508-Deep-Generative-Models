# MUSEmotion Design

## Goal

MUSEmotion is a full training and inference pipeline that takes one line of emotion-laden text and produces a short piano MIDI clip whose mood matches the text. The project has two learned stages:

1. Fine-tune BERT on GoEmotions after mapping fine-grained text emotions into EMOPIA's four valence-arousal quadrants.
2. Train an emotion-conditioned autoregressive Transformer decoder on EMOPIA piano MIDI token sequences.

A simple frontend lets a user type text, run the classifier, generate a MIDI file, and play or download the result.

## Scope

The repo will provide a real GPU/Colab-ready pipeline, not a toy demo. It will not vendor large datasets or checkpoints. It will include deterministic local unit tests for preprocessing, tokenization, model shape contracts, sampling, and MIDI decoding.

Large external assets are expected at these paths:

- `data/raw/emopia/` - local EMOPIA dataset checkout or extracted archive.
- Hugging Face GoEmotions - loaded at runtime via `datasets.load_dataset("go_emotions")`.
- `artifacts/` - generated preprocessed data, checkpoints, metrics, and sample MIDI files.

## Shared Label Space

Both models use the same four EMOPIA quadrant labels:

- `Q1` - high valence, high arousal: joy, amusement, excitement, pride, gratitude, optimism, relief, admiration, approval.
- `Q2` - low valence, high arousal: anger, annoyance, disapproval, fear, nervousness, disgust, embarrassment.
- `Q3` - low valence, low arousal: sadness, grief, disappointment, remorse.
- `Q4` - high valence, low arousal: love, caring, desire, curiosity, realization, calm neutral, surprise when not negative.

GoEmotions examples can contain multiple labels. The preparation script maps each fine-grained label to one quadrant. If an example maps to multiple quadrants, it is kept only when there is a unique majority quadrant. Otherwise it is dropped as conflicting. Original labels are preserved in metadata for reporting.

## Text Classifier

The classifier fine-tunes `bert-base-uncased` by default using Hugging Face Transformers.

Responsibilities:

- Load GoEmotions through Hugging Face datasets.
- Convert fine-grained labels into four quadrant IDs.
- Tokenize text with the BERT tokenizer.
- Train a sequence classifier using cross-entropy.
- Save model weights, tokenizer files, label mapping, and metrics.
- Provide a lightweight prediction API that returns the quadrant, confidence, and probability distribution.

Primary command:

```bash
python -m musemotion.cli.train_classifier --config configs/classifier.yaml
```

## MIDI Tokenizer

EMOPIA MIDI files are converted into symbolic event tokens for autoregressive modeling.

Token vocabulary:

- Special tokens: `PAD`, `BOS`, `EOS`.
- Pitch tokens: `PITCH_21` through `PITCH_108`.
- Duration tokens: quantized beat durations such as `DUR_1`, `DUR_2`, `DUR_4`, `DUR_8`, `DUR_16`.
- Velocity tokens: bucketed velocity tokens such as `VEL_1` through `VEL_8`.
- Shift tokens: quantized onset gaps such as `SHIFT_0`, `SHIFT_1`, `SHIFT_2`, `SHIFT_4`, `SHIFT_8`, `SHIFT_16`.

Each note is represented as `SHIFT`, `PITCH`, `DUR`, `VEL`. This is compact enough for a course project and easy to decode back into MIDI.

The tokenizer writes:

- `artifacts/music/tokenizer/vocab.json`
- `artifacts/music/tokenized/train.jsonl`
- `artifacts/music/tokenized/validation.jsonl`
- `artifacts/music/tokenized/test.jsonl`

Primary command:

```bash
python -m musemotion.cli.prepare_emopia --config configs/music.yaml
```

## Music Generator

The generator is a decoder-only Transformer trained with teacher-forced maximum likelihood. It conditions generation on emotion by adding an emotion embedding to each token embedding before self-attention.

Responsibilities:

- Load tokenized EMOPIA examples.
- Train next-token prediction with causal masking.
- Track train and validation cross-entropy.
- Save checkpoints and generated validation samples.
- Generate short piano MIDI clips from an emotion quadrant.

Default architecture:

- 4 Transformer decoder layers.
- 8 attention heads.
- 256 embedding dimensions.
- 1024 maximum context tokens.
- Emotion embedding size equal to token embedding size.

Primary command:

```bash
python -m musemotion.cli.train_generator --config configs/music.yaml
```

## End-To-End Inference

Inference composes the trained classifier and generator:

1. Normalize user text.
2. Predict the EMOPIA quadrant with BERT.
3. Sample a token sequence from the conditional music generator.
4. Decode tokens into a piano MIDI clip.
5. Return paths and metadata: predicted quadrant, confidence, token count, and MIDI path.

Primary command:

```bash
python -m musemotion.cli.generate --text "I feel hopeful but calm today" --output artifacts/samples/demo.mid
```

## Frontend

The frontend will be a Gradio app because it runs naturally in Colab and locally. It includes:

- Textbox for emotional state.
- Generate button.
- Predicted quadrant and confidence display.
- MIDI file output for playback/download.
- Optional controls for temperature, top-k, maximum notes, and random seed.

Primary command:

```bash
python -m musemotion.frontend.app --config configs/inference.yaml
```

## Colab Workflow

The repo will include a notebook-oriented path without duplicating implementation logic:

1. Install requirements.
2. Mount or upload EMOPIA to `data/raw/emopia/`.
3. Run classifier training.
4. Run EMOPIA preprocessing.
5. Run generator training.
6. Launch Gradio.

The notebook can call CLI modules directly; core code stays in `src/musemotion/`.

## Error Handling

- EMOPIA preprocessing fails early with a clear message if MIDI files or labels cannot be found.
- Training commands validate config paths before allocating GPU memory.
- Inference fails clearly if checkpoints or tokenizer artifacts are missing.
- MIDI decoding skips incomplete trailing event groups instead of crashing.
- Sampling always terminates on `EOS` or `max_tokens`.

## Testing

Local tests will not require GoEmotions, EMOPIA, or a GPU. They will cover:

- GoEmotions fine-label to quadrant mapping, including conflict handling.
- MIDI token encode/decode behavior on synthetic notes.
- Music dataset padding and causal-target construction.
- Generator forward pass shape and loss behavior.
- Sampling termination and token constraints.
- End-to-end inference with fake classifier/generator components.

## Repository Structure

```text
configs/
  classifier.yaml
  music.yaml
  inference.yaml
notebooks/
  musemotion_colab.ipynb
src/musemotion/
  cli/
  data/
  frontend/
  inference/
  models/
  music/
  training/
tests/
```

Generated folders are ignored by git:

- `data/`
- `artifacts/`
- `checkpoints/`
- `.superpowers/`
- `tmp/`

## Done Criteria

- The repo contains runnable CLI entrypoints for classifier training, EMOPIA preprocessing, generator training, end-to-end generation, and the frontend.
- `pytest` passes locally without downloading large datasets.
- The README explains setup, dataset placement, Colab/GPU workflow, and demo generation.
- Configuration files expose practical GPU training defaults.
- The code can train real models when the user provides EMOPIA and has access to a GPU/Colab runtime.
