# Real Training Model Artifacts

This folder keeps the trained artifacts from the full-dataset runs used to generate the committed training charts.

## Classifier

`classifier_bert_base/` contains the GoEmotions-to-EMOPIA quadrant classifier:

- base model: `bert-base-uncased`
- training data: full GoEmotions mapped to EMOPIA quadrants (25,873 train / 3,249 validation / 3,270 test), after dropping the no-clear-valence labels neutral/surprise/confusion/realization and cross-quadrant ties
- selection: best validation macro-F1 (epoch 4), class-weighted loss, early stopping
- test set: accuracy 0.828, macro-F1 0.788
- committed files: `config.json`, `vocab.txt`, tokenizer configs, `label_mapping.json`, `metrics.json`, `training_history.csv`
- weights: `pytorch_model.bin` (438 MB) is hosted on Google Drive and fetched with `python scripts/download_classifier.py`

## Music Generator

`music_transformer_fulldata/` contains the emotion-conditioned Transformer:

- training data: full EMOPIA (862 train / 108 validation / 108 test tokenized clips)
- epochs: 50, best validation loss 1.519 at epoch 38
- key files: `best.pt`, `tokenizer/vocab.json`, `training_history.csv`, `training_history.json`

## MIDI-to-quadrant probes

`music_probe/` contains the two models that read generated MIDI back and predict its quadrant.
They are the judges behind the round-trip metric, so their own accuracy on real held-out clips
is the ceiling every round-trip number is read against.

- training data: the committed `emopia_tokenized/` split, head-cropped to 128 notes so the
  probes are asked about the same clip length they were trained on
- `feature_probe.json` — standardised multinomial logistic regression over 28 symbolic
  features. Test accuracy 0.657, macro-F1 0.660. Stored as scaler statistics and coefficients
  rather than a pickled estimator, so loading executes no code and a scikit-learn version change
  cannot break it.
- `neural_probe.pt` — 2-layer bidirectional Transformer encoder over the token stream
  (d_model 128), mean-pooled. Test accuracy 0.602, macro-F1 0.595, selected on validation
  macro-F1 at epoch 10; it overfits after that.
- `probe_metadata.json` — both probes' train/validation/test metrics, confusion matrices, the
  feature coefficients, and the shuffled-label controls.

**The control is the reason to trust the rest.** Retrained on labels permuted within the training
*and validation* splits — permuting only training labels would leave the control free to select
its checkpoint on the true labels it is supposed to have no access to — the probes drop to 0.231
and 0.324 accuracy.

Read those against **0.315, not 0.25**. The test split runs 20/24/30/34, so a model that always
answers Q4 scores 0.315 while learning nothing; uniform chance is the wrong bar for accuracy on
an imbalanced split. The feature control lands below even uniform chance. The neural control sits
at the majority-class rate with a macro-F1 of 0.271, which is what collapsing onto the frequent
classes looks like. Neither retains usable signal.

Four-class symbolic emotion recognition on EMOPIA is genuinely hard; roughly 0.66 is the
achievable range, not 1.0.

## Tokenized EMOPIA

`emopia_tokenized/` holds the exact split that trained the generator, tokenized, so the
evaluation and probe training both run without downloading the 1 GB EMOPIA archive. See that
folder's README for the format and the CC BY 4.0 attribution.

Large datasets and per-epoch optimizer checkpoints are excluded from git.
