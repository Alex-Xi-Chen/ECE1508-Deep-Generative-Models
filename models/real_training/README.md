# Real Training Model Artifacts

This folder keeps the lightweight trained artifacts from the real 42-epoch runs used to generate the committed training visualizations.

## Classifier

`classifier_tiny_bert_42epoch/` contains the final Hugging Face checkpoint for the GoEmotions-to-EMOPIA quadrant classifier:

- base model: `prajjwal1/bert-tiny`
- training data: real GoEmotions subset mapped to EMOPIA quadrants
- epochs: 42
- key files: `pytorch_model.bin`, `config.json`, `vocab.txt`, tokenizer configs, `label_mapping.json`, `training_history.csv`

## Music Generator

`music_transformer_42epoch/` contains the trained symbolic music generator checkpoints:

- model: emotion-conditioned Transformer
- training data: real EMOPIA MIDI subset
- epochs: 42
- key files: `best.pt`, `last.pt`, `tokenizer/vocab.json`, `training_history.csv`

Large datasets and per-epoch optimizer checkpoints are still excluded from git.
