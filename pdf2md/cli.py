"""Command-line entry point.

Backward-compatible with the old script: `--pdf_path` still works. Adds `--out`
and `--config`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .converter import PdfToMarkdown


def _load_config(path: str | None) -> dict:
    candidate = Path(path) if path else Path("config/config.yaml")
    if candidate.is_file():
        with open(candidate, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert PDF files to Markdown (MIT-only, no AGPL/ML deps).",
    )
    parser.add_argument(
        "--pdf_path",
        required=True,
        nargs="+",
        help="One or more input PDF paths",
    )
    parser.add_argument("--out", help="Output directory (overrides config OUTPUT_DIR)")
    parser.add_argument("--config", help="Path to a YAML config file")
    args = parser.parse_args(argv)

    pdf_paths = [Path(path) for path in args.pdf_path]
    missing = [path for path in pdf_paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    config = _load_config(args.config)
    if args.out:
        config["OUTPUT_DIR"] = args.out

    extractor = PdfToMarkdown(config)
    out_dir = extractor.config["OUTPUT_DIR"]
    for pdf_path in pdf_paths:
        extractor.extract(str(pdf_path))
        print(f"Wrote {out_dir}/{pdf_path.stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
