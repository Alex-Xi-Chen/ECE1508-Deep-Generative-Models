from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from musemotion.config import load_yaml_config, resolve_path
from musemotion.emotions import quadrant_id, quadrant_name


class EmotionClassifier(Protocol):
    def predict(self, text: str) -> dict[str, Any]:
        ...


class MidiGenerator(Protocol):
    def generate_midi(self, emotion_id: int, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class GenerationResult:
    text: str
    quadrant: str
    emotion_id: int
    confidence: float
    probabilities: dict[str, float]
    midi_path: str
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_with_components(
    text: str,
    classifier: EmotionClassifier,
    generator: MidiGenerator,
    output_path: str | Path,
    **generation_kwargs: Any,
) -> GenerationResult:
    prediction = classifier.predict(text)
    generated = generator.generate_midi(
        emotion_id=int(prediction["emotion_id"]),
        output_path=output_path,
        **generation_kwargs,
    )
    return GenerationResult(
        text=text,
        quadrant=str(prediction["quadrant"]),
        emotion_id=int(prediction["emotion_id"]),
        confidence=float(prediction["confidence"]),
        probabilities=dict(prediction.get("probabilities", {})),
        midi_path=str(generated["path"]),
        token_count=int(generated.get("token_count", 0)),
    )


class BertEmotionClassifier:
    def __init__(self, model: Any, tokenizer: Any, device: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device: str | None = None) -> "BertEmotionClassifier":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        return cls(model=model, tokenizer=tokenizer, device=target_device)

    def predict(self, text: str) -> dict[str, Any]:
        import torch

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu()
        emotion_id = int(torch.argmax(probs).item())
        label = self.model.config.id2label.get(emotion_id, quadrant_name(emotion_id))
        if label.startswith("LABEL_"):
            label = quadrant_name(emotion_id)
        return {
            "quadrant": label,
            "emotion_id": quadrant_id(label),
            "confidence": float(probs[emotion_id].item()),
            "probabilities": {
                self.model.config.id2label.get(index, quadrant_name(index)): float(value.item())
                for index, value in enumerate(probs)
            },
        }


class MusicGeneratorComponent:
    def __init__(self, model: Any, tokenizer: Any, device: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
    ) -> "MusicGeneratorComponent":
        import torch

        from musemotion.models.music_transformer import MusicTransformer, MusicTransformerConfig
        from musemotion.music.tokenizer import MusicTokenizer

        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(str(checkpoint_path), map_location=target_device)
        resolved_tokenizer_path = tokenizer_path or checkpoint.get("tokenizer_path")
        if resolved_tokenizer_path is None:
            raise ValueError("Tokenizer path is required when checkpoint does not include it.")
        tokenizer = MusicTokenizer.load(resolve_path(resolved_tokenizer_path))
        model = MusicTransformer(MusicTransformerConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model=model, tokenizer=tokenizer, device=target_device)

    def generate_midi(
        self,
        emotion_id: int,
        output_path: str | Path,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = 32,
        seed: int | None = None,
    ) -> dict[str, Any]:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        token_ids = self.model.sample(
            emotion_id=emotion_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=self.device,
        )
        path = self.tokenizer.token_ids_to_midi(token_ids, output_path)
        return {"path": str(path), "token_count": len(token_ids)}


def generate_from_config(
    text: str,
    config_path: str | Path = "configs/inference.yaml",
    output_path: str | Path | None = None,
    **generation_kwargs: Any,
) -> GenerationResult:
    config = load_yaml_config(config_path)
    sample_dir = resolve_path(config.get("output", {}).get("sample_dir", "artifacts/samples"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(output_path) if output_path is not None else sample_dir / "musemotion_sample.mid"
    classifier = BertEmotionClassifier.from_pretrained(resolve_path(config["classifier"]["model_dir"]))
    generator = MusicGeneratorComponent.from_checkpoint(
        resolve_path(config["generator"]["checkpoint"]),
        tokenizer_path=resolve_path(config["generator"]["tokenizer"]),
    )
    defaults = config.get("generation", {})
    defaults.update(generation_kwargs)
    return generate_with_components(text, classifier, generator, destination, **defaults)
