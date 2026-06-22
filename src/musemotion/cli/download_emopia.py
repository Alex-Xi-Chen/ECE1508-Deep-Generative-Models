from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and extract the official EMOPIA dataset.")
    parser.add_argument("--output", default="data/raw/emopia", help="Directory where EMOPIA will be extracted.")
    parser.add_argument("--url", default=None, help="Dataset archive URL or local zip path.")
    parser.add_argument("--force", action="store_true", help="Replace an existing extracted dataset.")
    return parser


def main(argv: list[str] | None = None) -> None:
    from musemotion.music.emopia import DEFAULT_EMOPIA_URL, download_emopia_dataset

    args = build_parser().parse_args(argv)
    output_dir = download_emopia_dataset(
        output_dir=args.output,
        url=args.url or DEFAULT_EMOPIA_URL,
        force=args.force,
    )
    print(f"EMOPIA dataset ready at {output_dir}")


if __name__ == "__main__":
    main()
