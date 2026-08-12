from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from musemotion.config import load_yaml_config, resolve_path
from musemotion.inference.pipeline import (
    BertEmotionClassifier,
    MusicGeneratorComponent,
    generate_with_components,
)

NO_PROBE_MESSAGE = (
    "_Round-trip check unavailable: no probe artifacts found. Train them with_ "
    "`python -m musemotion.cli.train_probe`."
)


def _load_probes_if_available(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the probes and their metadata, or return nothing if they are not present.

    The panel is an addition to the app, so anything wrong here has to leave the rest of the app
    working rather than stop it from starting. The whole lookup sits inside the try for that
    reason: a `probes:` key written with no value under it parses as None, and reading through
    it would raise before the guard could catch it.
    """
    from musemotion.evaluation.probe import load_probe_metadata, load_probes

    try:
        probes_config = config.get("probes") or {}
        probe_dir = resolve_path(
            probes_config.get("directory") or "models/real_training/music_probe"
        )
        probes = load_probes(
            probe_dir, tokenizer_path=resolve_path(config["generator"]["tokenizer"])
        )
        return probes, load_probe_metadata(probe_dir)
    except Exception as error:
        print(f"probe panel disabled: {error}")
        return {}, {}


def _ceiling_caveat(metadata: dict[str, Any]) -> str:
    """Describe each probe's measured ceiling, read from the artifacts rather than hardcoded.

    The probes can be retrained, and their accuracies differ from each other, so quoting one
    frozen number in prose would go stale and misstate both.
    """
    accuracies = {
        name: values.get("test", {}).get("accuracy")
        for name, values in metadata.get("probes", {}).items()
    }
    measured = {name: value for name, value in accuracies.items() if value is not None}
    if not measured:
        return (
            "_Worth reading against the probes' accuracy on real EMOPIA clips rather than "
            "against 100%: four-way symbolic emotion recognition is hard, and valence is much "
            "weaker than arousal._"
        )
    # Phrasing follows the number of probes actually loaded. "for both" was left over from a
    # two-probe assumption and would describe a single probe as a pair.
    parts = ", ".join(f"{name} {value:.3f}" for name, value in measured.items())
    subject = "this probe reaches" if len(measured) == 1 else "these probes reach"
    scope = "it" if len(measured) == 1 else "both"
    return (
        f"_Worth reading against the measured ceiling rather than as a failure: on real EMOPIA "
        f"clips {subject} {parts} (chance is 0.25), and valence is far weaker than arousal "
        f"for {scope}._"
    )


def _round_trip_markdown(
    text_quadrant: str,
    midi_path: str,
    probes: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Report what each probe hears in the generated clip, against what the text predicted.

    The probes read the MIDI file that was just written rather than the generator's token
    stream, so this reports on the same artifact the listener downloads.
    """
    if not probes:
        return NO_PROBE_MESSAGE

    from musemotion.evaluation.probe import notes_from_midi

    try:
        notes = notes_from_midi(midi_path)
    except Exception as error:
        return f"_Round-trip check failed to read the generated clip: {error}_"
    if not notes:
        return "_Round-trip check skipped: the generated clip decoded to no notes._"

    lines = [
        f"**Text stage (BERT):** `{text_quadrant}`",
        "",
        "| probe | hears | confidence | agrees with text |",
        "|---|---|---|---|",
    ]
    verdicts = []
    for name, probe in probes.items():
        prediction = probe.predict_notes(notes)
        agrees = prediction["quadrant"] == text_quadrant
        verdicts.append(agrees)
        lines.append(
            f"| {name} | `{prediction['quadrant']}` | {prediction['confidence']:.2f} | "
            f"{'yes' if agrees else 'no'} |"
        )

    # The wording has to match how many probes actually answered. Claiming agreement between
    # "both" probes from a single opinion would overstate the evidence, and only two independent
    # probes agreeing is worth calling out in the first place.
    agreeing = sum(1 for verdict in verdicts if verdict)
    if len(verdicts) == 1:
        only = "agrees with" if agreeing else "does not agree with"
        lines.extend(
            ["", f"**Round trip: the one available probe {only} the text stage.**"]
        )
    elif agreeing == len(verdicts):
        lines.extend(["", f"**Round trip: all {len(verdicts)} probes agree with the text stage.**"])
    elif agreeing:
        lines.extend(["", "**Round trip: the probes disagree with each other.**"])
    else:
        lines.extend(["", "**Round trip: no probe hears the predicted quadrant.**"])

    if agreeing < len(verdicts):
        lines.extend(["", _ceiling_caveat(metadata or {})])
    return "\n".join(lines)


def build_app(config_path: str | Path = "configs/inference.yaml"):
    import gradio as gr

    config = load_yaml_config(config_path)
    generation_defaults = config.get("generation", {})
    sample_dir = resolve_path(config.get("output", {}).get("sample_dir", "artifacts/samples"))
    sample_dir.mkdir(parents=True, exist_ok=True)
    classifier = BertEmotionClassifier.from_pretrained(resolve_path(config["classifier"]["model_dir"]))
    generator = MusicGeneratorComponent.from_checkpoint(
        resolve_path(config["generator"]["checkpoint"]),
        tokenizer_path=resolve_path(config["generator"]["tokenizer"]),
    )
    probes, probe_metadata = _load_probes_if_available(config)

    def generate(
        text: str,
        temperature: float,
        top_k: int,
        max_tokens: int,
        guidance_scale: float,
        seed: int | None,
    ):
        output_path = sample_dir / "musemotion_frontend.mid"
        result = generate_with_components(
            text,
            classifier,
            generator,
            output_path,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            guidance_scale=guidance_scale,
            seed=seed if seed is not None and seed >= 0 else None,
        )
        metadata = json.dumps(result.to_dict(), indent=2)
        verdict = _round_trip_markdown(
            result.quadrant, result.midi_path, probes, probe_metadata
        )
        return result.midi_path, metadata, verdict

    with gr.Blocks(title="MUSEmotion") as demo:
        gr.Markdown("# MUSEmotion")
        gr.Markdown("Type how you feel, then generate a short emotion-conditioned piano MIDI clip.")
        text = gr.Textbox(label="How are you feeling?", lines=3)
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=float(generation_defaults.get("temperature", 1.0)), label="Temperature")
            top_k = gr.Slider(1, 128, value=int(generation_defaults.get("top_k", 32)), step=1, label="Top-k")
        with gr.Row():
            max_tokens = gr.Slider(64, 1024, value=int(generation_defaults.get("max_tokens", 512)), step=32, label="Max tokens")
            guidance_scale = gr.Slider(
                1.0,
                8.0,
                value=float(generation_defaults.get("guidance_scale", 1.0)),
                step=0.5,
                label="Guidance scale (1.0 = off, higher = stronger emotion)",
            )
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
        button = gr.Button("Generate MIDI")
        midi_file = gr.File(label="Generated MIDI")
        gr.Markdown("### Round-trip check")
        gr.Markdown(
            "A second model reads the generated MIDI back and predicts its quadrant. "
            "Agreement means the emotion survived generation."
            if probes
            else "Not available in this session."
        )
        round_trip = gr.Markdown(NO_PROBE_MESSAGE if not probes else "")
        metadata = gr.Code(label="Prediction metadata", language="json")
        button.click(
            generate,
            inputs=[text, temperature, top_k, max_tokens, guidance_scale, seed],
            outputs=[midi_file, metadata, round_trip],
        )
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
