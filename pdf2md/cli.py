"""Command-line entry point.

Backward-compatible with the old script: `--pdf_path` still works. Adds `--out`,
`--config`, and `--progress` / `--no-progress`.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml

from .converter import PdfToMarkdown
from .progress import TqdmProgress, progress_enabled


def _load_config(path: str | None) -> dict:
    candidate = Path(path) if path else Path("config/config.yaml")
    if candidate.is_file():
        with open(candidate, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


#: Toggles worth announcing at startup: config key -> label used in the summary.
_FEATURES = (
    ("extract_images", "images"),
    ("detect_borderless_tables", "borderless tables"),
    ("detect_blockquote", "blockquotes"),
    ("detect_footnotes", "footnotes"),
    ("bold_as_heading", "bold headings"),
)


def _feature_summary(config: dict) -> str:
    """One line naming the toggles that are on, so a run is never a black box."""
    on = [label for key, label in _FEATURES if config.get(key)]
    off = [label for key, label in _FEATURES if not config.get(key)]
    text = "features on: " + (", ".join(on) if on else "none")
    if off:
        text += f" | off: {', '.join(off)}"
    return text


@contextmanager
def _route_logging_through_progress(enabled: bool):
    """Keep log records from shredding the progress bar.

    pdfminer warns freely on real-world PDFs (invalid MediaBox, bad font
    descriptors, ...). Those records reach stderr even with no handler
    configured, via `logging.lastResort`, and land in the middle of the bar - so
    the redirect has to engage even though the root logger looks empty. Routing
    them through `tqdm.write` puts each on its own line above the bar, and since
    the root logger keeps its own level, nothing previously silent starts being
    emitted.
    """
    if not enabled:
        yield
        return

    from tqdm.contrib.logging import logging_redirect_tqdm

    with logging_redirect_tqdm():
        yield


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert PDF files to Markdown (MIT-only, no AGPL/ML deps).",
        epilog=(
            "Progress bars need tqdm (`uv sync --extra progress`) and are only "
            "drawn when stderr is a terminal; use --progress to force one on."
        ),
    )
    parser.add_argument(
        "--pdf_path",
        required=True,
        nargs="+",
        help="One or more input PDF paths",
    )
    parser.add_argument("--out", help="Output directory (overrides config OUTPUT_DIR)")
    parser.add_argument("--config", help="Path to a YAML config file")
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Force the progress bar on, even when stderr is not a terminal.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=None,
        help="Never draw a progress bar (the default when stderr is redirected).",
    )
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

    show_progress = progress_enabled(force=args.progress)
    progress = TqdmProgress(enabled=show_progress)
    extractor = PdfToMarkdown(config, progress=progress if show_progress else None)
    out_dir = extractor.config["OUTPUT_DIR"]
    with progress, _route_logging_through_progress(show_progress):
        progress.status(_feature_summary(extractor.config))
        for pdf_path in pdf_paths:
            extractor.extract(str(pdf_path))
            print(f"Wrote {out_dir}/{pdf_path.stem}.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
