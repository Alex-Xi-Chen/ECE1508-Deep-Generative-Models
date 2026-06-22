from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an emotion-conditioned piano MIDI clip from text.")
    parser.add_argument("--text", required=True, help="Emotion-laden user text.")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference YAML config.")
    parser.add_argument("--output", default=None, help="Destination MIDI path.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling override.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum generated token count.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser


def main(argv: list[str] | None = None) -> None:
    from musemotion.inference.pipeline import generate_from_config

    args = build_parser().parse_args(argv)
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
    result = generate_from_config(args.text, config_path=args.config, output_path=args.output, **overrides)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
