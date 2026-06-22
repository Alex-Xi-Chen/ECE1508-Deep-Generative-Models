from __future__ import annotations

import argparse
import json
from pathlib import Path

from musemotion.config import load_yaml_config, resolve_path
from musemotion.inference.pipeline import (
    BertEmotionClassifier,
    MusicGeneratorComponent,
    generate_with_components,
)


def build_app(config_path: str | Path = "configs/inference.yaml"):
    import gradio as gr

    config = load_yaml_config(config_path)
    sample_dir = resolve_path(config.get("output", {}).get("sample_dir", "artifacts/samples"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    classifier = BertEmotionClassifier.from_pretrained(resolve_path(config["classifier"]["model_dir"]))
    generator = MusicGeneratorComponent.from_checkpoint(
        resolve_path(config["generator"]["checkpoint"]),
        tokenizer_path=resolve_path(config["generator"]["tokenizer"]),
    )

    def generate(text: str, temperature: float, top_k: int, max_tokens: int, seed: int | None):
        output_path = sample_dir / "musemotion_frontend.mid"
        result = generate_with_components(
            text,
            classifier,
            generator,
            output_path,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed if seed is not None and seed >= 0 else None,
        )
        metadata = json.dumps(result.to_dict(), indent=2)
        return result.midi_path, metadata

    with gr.Blocks(title="MUSEmotion") as demo:
        gr.Markdown("# MUSEmotion")
        gr.Markdown("Type how you feel, then generate a short emotion-conditioned piano MIDI clip.")
        text = gr.Textbox(label="How are you feeling?", lines=3)
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=float(config.get("generation", {}).get("temperature", 1.0)), label="Temperature")
            top_k = gr.Slider(1, 128, value=int(config.get("generation", {}).get("top_k", 32)), step=1, label="Top-k")
        with gr.Row():
            max_tokens = gr.Slider(64, 1024, value=int(config.get("generation", {}).get("max_tokens", 512)), step=32, label="Max tokens")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
        button = gr.Button("Generate MIDI")
        midi_file = gr.File(label="Generated MIDI")
        metadata = gr.Code(label="Prediction metadata", language="json")
        button.click(generate, inputs=[text, temperature, top_k, max_tokens, seed], outputs=[midi_file, metadata])
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the MUSEmotion Gradio frontend.")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference YAML config.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_app(args.config).launch(share=args.share)


if __name__ == "__main__":
    main()
