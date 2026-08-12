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

### Validation

The mapping is exercised end to end by the 40-prompt set in
[`src/musemotion/evaluation/prompts.py`](../src/musemotion/evaluation/prompts.py), ten
sentences per quadrant with the quadrant each is meant to express, fixed before any results
were inspected. The classifier scores **0.850** on it, consistent with its 0.828 on the
mapped GoEmotions test split.

The disagreements are informative rather than simply wrong. Five of the six are Q4 read as
Q1 — "I feel settled and gently hopeful" is called Q1 with full confidence, because
`hopeful` maps through `optimism` to Q1 in the table above. That is the mapping and the
intent disagreeing about arousal, not the model failing, and it is worth keeping in mind
that Q4 is also the quadrant the music generator renders least reliably.

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
- **Evaluation**: training and validation loss, plus the objective evaluation below. A
  listening study is still outstanding and is the only thing that can speak to quality.

## Evaluating the generated music

Loss and classifier accuracy measure prediction, not perception. Neither says whether a
listener would hear the intended emotion, so the output itself is measured two ways, which
are complementary rather than alternatives: a generator that had simply memorised EMOPIA
would score near-perfectly on the first and only the second would catch it.

- **Round-trip recovery.** A MIDI-to-quadrant probe reads generated clips back and predicts
  the conditioning quadrant. Two probes are trained, one neural and one over hand-built
  symbolic features, sharing the note representation and nothing else; their agreement is
  evidence in its own right.
- **Novelty.** Note n-gram overlap with the training corpus, plus the longest exactly-copied
  run, on both an absolute-pitch and a transposition-invariant view.

Three decisions make those numbers readable.

- **Everything is reported beside real held-out EMOPIA clips scored identically.** The probes
  reach 0.657 on real music, so that is the ceiling, and round-trip accuracy is also expressed
  as a fraction of it. A novelty rate has no scale on its own; the same rate next to the
  real-data reference does.
- **The probes are retrained on permuted labels as a control**, with the permutation covering
  the validation split too — otherwise the control would still select its checkpoint on true
  labels. If the control does not collapse, the probe is reading something other than emotion
  and nothing downstream means anything.
- **Accuracy is read against the majority-class rate, not uniform chance.** The test split runs
  20/24/30/34, so always answering Q4 scores 0.315 without learning anything.

### What the evaluation found

- Classifier-free guidance is what makes the conditioning recoverable: round-trip accuracy
  0.550 with guidance off against 0.790 at guidance 3.0, with non-overlapping intervals over
  200 clips per setting.
- The recoverable signal is almost entirely **arousal**. One cross-arousal error in 200 clips,
  against 41 valence errors. With guidance off, valence sits at 0.590 against a 0.5 baseline;
  guidance adds +0.205 to valence and only +0.11 to arousal.
- Valence errors are **directional**: 36 positive-to-negative against 5 the other way. Q4 is
  the weak quadrant at 0.54 recall, with 23 of 50 clips read as Q3.
- No memorisation. Not one generated clip shares an 8-note pattern with the training corpus;
  the longest copied run is 6-7 notes against 40 for real held-out clips.
- Guidance 3.0 is the shipped setting because 5.0 buys no further accuracy while costing
  fidelity, diversity and clip length. Probe agreement independently peaks at 3.0.

Full numbers are in the README, and any run writes them to `artifacts/evaluation/metrics.json`.

### Remaining limitations

EMOPIA is small (~1078 clips), so overall musical quality is bounded by the available data
even with classifier-free guidance improving how distinct the emotions sound.

Valence is the weaker axis by a wide margin, and the tokenizer is a plausible reason: arousal
is written directly into note density, velocity and duration, which the `SHIFT, PITCH, DUR,
VEL` vocabulary represents explicitly, while valence lives in harmony and mode, which it
encodes only implicitly. Adding an explicit key or mode token is the obvious next experiment.

No metric here measures musical quality. Distributional similarity to real EMOPIA is the
closest available proxy and it is only a proxy; the listening study remains the gap.

## Model hosting

The `bert-base-uncased` classifier weights (438 MB) exceed GitHub's 100 MB per-file limit,
so they are hosted on Google Drive and fetched with
[`scripts/download_classifier.py`](../scripts/download_classifier.py). The music generator
checkpoint (14 MB), configs, tokenizers, and metrics are committed to the repository.
