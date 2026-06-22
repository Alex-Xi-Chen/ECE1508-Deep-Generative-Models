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
