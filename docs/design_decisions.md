# Design Decisions

This document records the non-obvious choices in MUSEmotion and the reasoning behind
them, so the mapping, training, and generation behaviour are reproducible and reviewable.

## Emotion label mapping (GoEmotions -> EMOPIA quadrants)

Both models share EMOPIA's four valence-arousal quadrants:

| Quadrant | Valence | Arousal | Character |
|----------|---------|---------|-----------|
| Q1 | high | high | positive and energetic |
| Q2 | low | high | negative and energetic |
| Q3 | low | low | negative and subdued |
| Q4 | high | low | positive and calm |

GoEmotions has 27 emotions plus `neutral`. The mapping is a fixed rule-based table in
[`src/musemotion/emotions.py`](../src/musemotion/emotions.py):

- **Q1**: admiration, amusement, approval, curiosity, desire, excitement, joy, optimism, pride
- **Q2**: anger, annoyance, disapproval, disgust, embarrassment, fear, nervousness
- **Q3**: disappointment, grief, remorse, sadness
- **Q4**: caring, gratitude, love, relief

### Dropped labels (not mapped -> their examples are dropped)

`neutral`, `surprise`, `confusion`, `realization`.

Reason: these labels do not carry a clear valence, so forcing them into a valence-based
quadrant injects label noise. `surprise` is bivalent (a pleasant surprise vs a shock),
`confusion` and `realization` are cognitive states without a fixed valence, and `neutral`
is by definition not an emotion. `neutral` is also the single most frequent GoEmotions
label (~28% of examples); mapping it to a quadrant made that quadrant dominate and pushed
the classifier toward predicting one class.

### Reassignments from the first draft of the mapping

- `gratitude`: Q1 -> **Q4**, and `relief`: Q1 -> **Q4**. Both are warm, low-arousal
  positives, so they fit calm-positive (Q4) better than energetic-positive (Q1).
- `desire`: Q4 -> **Q1**. Positive but high arousal (craving/wanting is activating).
- `curiosity`: Q4 -> **Q1**. A positive approach/interest emotion that is more activating
  than calm.

### Multi-label handling

GoEmotions is multi-label. For each example we map every label to a quadrant, take the
**majority quadrant**, and **drop the example when there is no unique majority** (a tie
across quadrants), since those examples send a conflicting emotional signal.

### Validation (planned)

Manually check representative examples per quadrant and report the final label
distribution after mapping.

## Classifier training

- **Full GoEmotions, no sample cap.** An earlier run capped training to 177 examples,
  which is far too few; the full corpus is ~43k examples.
- **`bert-base-uncased`.** A tiny base model (bert-tiny) underfits 4-way emotion.
- **Class-weighted cross-entropy** (balanced, inverse frequency). The quadrants are
  imbalanced; without weighting the model collapses to the majority quadrant.
- **Checkpoint selected on macro-F1 with early stopping.** Validation loss rises after a
  couple of epochs while accuracy and macro-F1 stay high, so we select the best checkpoint
  by macro-F1 rather than by loss and stop once it plateaus.
- **Evaluation**: accuracy and macro-F1 on the mapped GoEmotions test split.

## Music generation

- **Full EMOPIA (~1078 clips).** An earlier run used only 76 clips.
- **Distinct seed per generated clip.** A shared seed makes the sampler draw the same
  random sequence for every clip, so clips come out artificially similar regardless of the
  emotion. Different seeds give a fair comparison across quadrants.
- **Classifier-free guidance (CFG).** The emotion is added as an embedding to the token
  embeddings, which alone is a weak signal. During training the emotion is randomly dropped
  (~10%, `training.condition_dropout`) and replaced with a learned null embedding, so the
  model also learns an unconditional distribution. At generation the logits are extrapolated
  away from that unconditional prediction by `generation.guidance_scale` (>1) to make the
  target emotion more pronounced. Without it, the four quadrants tend to produce nearly
  identical clips.
- **`--quadrant` override.** `python -m musemotion.cli.generate --quadrant Q1 ...`
  conditions the generator directly on a quadrant and skips the classifier, which lets us
  test the generator in isolation and produce controlled clips for the human evaluation.
- **Evaluation**: training and validation loss, plus a small human evaluation where
  listeners rate musical quality and how well a clip matches its intended quadrant.

### Remaining limitation

EMOPIA is small (~1078 clips), so overall musical quality is bounded by the available data
even with classifier-free guidance improving how distinct the emotions sound.

## Model hosting

The `bert-base-uncased` classifier weights (438 MB) exceed GitHub's 100 MB per-file limit,
so they are hosted on Google Drive and fetched with
[`scripts/download_classifier.py`](../scripts/download_classifier.py). The music generator
checkpoint (14 MB), configs, tokenizers, and metrics are committed to the repository.
