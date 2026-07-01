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
        description="Convert a PDF file to Markdown (MIT-only, no AGPL/ML deps).",
    )
    parser.add_argument("--pdf_path", required=True, help="Path to the input PDF file")
    parser.add_argument("--out", help="Output directory (overrides config OUTPUT_DIR)")
    parser.add_argument("--config", help="Path to a YAML config file")
    args = parser.parse_args(argv)

    if not Path(args.pdf_path).is_file():
        print(f"error: file not found: {args.pdf_path}", file=sys.stderr)
        return 2

    config = _load_config(args.config)
    if args.out:
        config["OUTPUT_DIR"] = args.out

    extractor = PdfToMarkdown(config)
    extractor.extract(args.pdf_path)
    out_dir = extractor.config["OUTPUT_DIR"]
    name = Path(args.pdf_path).stem
    print(f"Wrote {out_dir}/{name}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
