from __future__ import annotations

import argparse

from musemotion.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the emotion-conditioned MIDI Transformer.")
    parser.add_argument("--config", default="configs/music.yaml", help="Path to music YAML config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    from musemotion.training.generator import train_generator

    args = build_parser().parse_args(argv)
    train_generator(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
