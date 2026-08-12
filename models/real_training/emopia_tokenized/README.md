# Tokenized EMOPIA splits

The exact 862 / 108 / 108 split that trained the committed music generator, tokenized into the
`SHIFT, PITCH, DUR, VEL` event vocabulary in
[`src/musemotion/music/tokenizer.py`](../../../src/musemotion/music/tokenizer.py).

These files are committed so the evaluation in
[`configs/evaluation.yaml`](../../../configs/evaluation.yaml) runs without downloading the
1 GB EMOPIA archive. The evaluation needs them for two things:

- **the novelty reference** — measuring how much of a generated clip is copied requires the
  training corpus to compare against, and the longest-copied-run search needs the actual note
  sequences rather than a summary,
- **the ceiling row** — the `test` split supplies the real held-out clips whose probe accuracy
  every round-trip number is read against.

Regenerate them from the original dataset with:

```bash
python -m musemotion.cli.download_emopia --output data/raw/emopia
python -m musemotion.cli.prepare_emopia --config configs/music.yaml
```

## Format

One JSON object per line:

| field | meaning |
|---|---|
| `token_ids` | the clip as event-vocabulary token ids, including BOS and EOS |
| `emotion_id` | 0-3, matching `EMOPIA_QUADRANTS` in `src/musemotion/emotions.py` |
| `quadrant` | `Q1`-`Q4` |
| `source` | the originating MIDI file, relative to the EMOPIA dataset root |

`source` is stored relative to the dataset root rather than as the absolute path the
preprocessing run produced, so the provenance of each clip is preserved without embedding the
directory layout of the machine that generated it.

## Attribution

Derived from **EMOPIA v1.0**, distributed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
via [Zenodo record 5090631](https://zenodo.org/records/5090631). CC BY 4.0 permits
redistribution of the dataset and of derived works, including commercially, provided the
creators are credited; it carries no non-commercial or share-alike restriction.

> Hung, H.-T., Ching, J., Doh, S., Kim, N., Nam, J., & Yang, Y.-H. (2021). *EMOPIA: A
> Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation.*
> Proceedings of the 22nd International Society for Music Information Retrieval Conference
> (ISMIR).

The content here is a quantized, lossy re-encoding of note events from those MIDI files, not
the original audio or the original MIDI.
