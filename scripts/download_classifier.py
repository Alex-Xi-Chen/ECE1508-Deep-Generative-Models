"""Download the bert-base emotion classifier weights from Google Drive.

The 438 MB ``pytorch_model.bin`` is too large for git, so it is hosted on Google
Drive and fetched into ``models/real_training/classifier_bert_base/`` on demand.
The rest of that directory (config, tokenizer, label mapping, metrics) is committed.

    python scripts/download_classifier.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE_ID = "1RSKe82_WaX7hD5yP17nbdKCI1S8W_-jP"
DEFAULT_OUTPUT = REPO_ROOT / "models/real_training/classifier_bert_base/pytorch_model.bin"


def download(file_id: str, output: Path) -> Path:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("gdown is required: pip install gdown") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(output), quiet=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID, help="Google Drive file id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination path")
    args = parser.parse_args()
    path = download(args.file_id, args.output)
    print(f"downloaded classifier weights to {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
