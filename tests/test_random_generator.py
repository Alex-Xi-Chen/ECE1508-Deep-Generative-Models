import json

import pretty_midi

from musemotion.music.random_generator import RandomPianoGeneratorConfig, generate_random_piano


def test_saved_random_piano_model_generates_non_empty_midi(tmp_path):
    model_config = RandomPianoGeneratorConfig.from_dict(
        json.loads(open("samples/random_piano_seed_260625_model.json", encoding="utf-8").read())
    )
    midi_path = tmp_path / "random.mid"

    metadata = generate_random_piano(model_config, midi_path)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    note_count = sum(len(instrument.notes) for instrument in midi.instruments)

    assert metadata["note_count"] == 44
    assert note_count == 44
    assert round(midi.get_end_time(), 1) == 16.8
