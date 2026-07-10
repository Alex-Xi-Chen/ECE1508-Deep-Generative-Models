from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an emotion-conditioned piano MIDI clip.")
    parser.add_argument("--text", default=None, help="Emotion-laden user text (classified into a quadrant).")
    parser.add_argument(
        "--quadrant",
        default=None,
        choices=["Q1", "Q2", "Q3", "Q4"],
        help="Force an EMOPIA quadrant and skip the classifier (for testing the generator in isolation).",
    )
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference YAML config.")
    parser.add_argument("--output", default=None, help="Destination MIDI path.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling override.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum generated token count.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser


def main(argv: list[str] | None = None) -> None:
    from musemotion.inference.pipeline import generate_from_config

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.text and not args.quadrant:
        parser.error("provide --text (classifier) or --quadrant (forced emotion).")
    overrides = {
        key: value
        for key, value in {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        }.items()
        if value is not None
    }
    result = generate_from_config(
        args.text or "",
        config_path=args.config,
        output_path=args.output,
        quadrant=args.quadrant,
        **overrides,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
