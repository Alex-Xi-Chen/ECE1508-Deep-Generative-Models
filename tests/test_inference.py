from pathlib import Path

from musemotion.inference.pipeline import GenerationResult, generate_with_components


class FakeClassifier:
    def predict(self, text):
        return {"quadrant": "Q1", "emotion_id": 0, "confidence": 0.9, "probabilities": {"Q1": 0.9}}


class FakeGenerator:
    def generate_midi(self, emotion_id, output_path, **kwargs):
        Path(output_path).write_bytes(b"MThd")
        return {"token_count": 4, "path": str(output_path)}


def test_generate_with_components_returns_metadata_and_file(tmp_path):
    output = tmp_path / "clip.mid"

    result = generate_with_components("I feel excited", FakeClassifier(), FakeGenerator(), output)

    assert isinstance(result, GenerationResult)
    assert result.quadrant == "Q1"
    assert output.exists()
