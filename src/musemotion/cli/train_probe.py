from __future__ import annotations

import argparse

from musemotion.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the MIDI-to-quadrant probes that score generated clips."
    )
    parser.add_argument("--config", default="configs/probe.yaml", help="Path to probe YAML config.")
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip the shuffled-label control (faster, but drops the check that the probe reads emotion).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from musemotion.training.probe import train_music_probe

    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    if args.skip_controls:
        config.setdefault("controls", {})["shuffled_labels"] = False
    train_music_probe(config)


if __name__ == "__main__":
    main()
