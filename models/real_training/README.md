# Real Training Model Artifacts

This folder keeps the trained artifacts from the full-dataset runs used to generate the committed training charts.

## Classifier

`classifier_bert_base/` contains the GoEmotions-to-EMOPIA quadrant classifier:

- base model: `bert-base-uncased`
- training data: full GoEmotions mapped to EMOPIA quadrants (27,906 train / 3,514 validation / 3,526 test), after dropping `neutral` and cross-quadrant ties
- selection: best validation macro-F1 (epoch 2), class-weighted loss, early stopping
- test set: accuracy 0.806, macro-F1 0.770
- committed files: `config.json`, `vocab.txt`, tokenizer configs, `label_mapping.json`, `metrics.json`, `training_history.csv`
- weights: `pytorch_model.bin` (438 MB) is hosted on Google Drive and fetched with `python scripts/download_classifier.py`

## Music Generator

`music_transformer_fulldata/` contains the emotion-conditioned Transformer:

- training data: full EMOPIA (862 train / 108 validation / 108 test tokenized clips)
- epochs: 50, best validation loss 1.515 at epoch 41
- key files: `best.pt`, `tokenizer/vocab.json`, `training_history.csv`, `training_history.json`

Large datasets and per-epoch optimizer checkpoints are excluded from git.
