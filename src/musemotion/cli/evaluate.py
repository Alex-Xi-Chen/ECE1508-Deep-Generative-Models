from __future__ import annotations

import argparse

from musemotion.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score generated clips: round-trip quadrant recovery, novelty, and fidelity."
    )
    parser.add_argument(
        "--config", default="configs/evaluation.yaml", help="Path to evaluation YAML config."
    )
    parser.add_argument(
        "--clips-per-quadrant",
        type=int,
        default=None,
        help="Override how many clips to generate per quadrant, per guidance scale.",
    )
    parser.add_argument(
        "--guidance-scales",
        type=float,
        nargs="+",
        default=None,
        help="Override the guidance sweep, e.g. --guidance-scales 1 3.",
    )
    parser.add_argument(
        "--skip-end-to-end",
        action="store_true",
        help="Skip the text-to-music row, which needs the separately downloaded BERT weights.",
    )
    parser.add_argument("--output", default=None, help="Override the output directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    from musemotion.evaluation.harness import run_evaluation

    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    if args.clips_per_quadrant is not None:
        config.setdefault("generation", {})["clips_per_quadrant"] = args.clips_per_quadrant
    if args.guidance_scales is not None:
        config.setdefault("generation", {})["guidance_scales"] = list(args.guidance_scales)
    if args.skip_end_to_end:
        config.setdefault("end_to_end", {})["enabled"] = False
    if args.output is not None:
        config.setdefault("output", {})["directory"] = args.output
    run_evaluation(config)


if __name__ == "__main__":
    main()
