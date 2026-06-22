from __future__ import annotations

import argparse

from musemotion.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on GoEmotions quadrant labels.")
    parser.add_argument("--config", default="configs/classifier.yaml", help="Path to classifier YAML config.")
    return parser


def main(argv: list[str] | None = None) -> None:
    from musemotion.training.classifier import train_classifier

    args = build_parser().parse_args(argv)
    train_classifier(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
