"""Optional progress reporting for the CLI.

The pipeline itself never imports ``tqdm``: :class:`pdf2md.converter.PdfToMarkdown`
only calls an opaque ``progress(stage, done, total)`` callback, so library users
pay nothing and the MIT-only core stays free of UI dependencies.

This module supplies the tqdm-backed implementation used by the CLI. It degrades
to silence (no bar, no error) when tqdm is not installed - install it with
``uv sync --extra progress`` - or when stderr is not a terminal.

Every step of the pipeline reports itself through one callback::

    progress(stage, done, total)

``total > 0`` is a per-page step (``done`` of ``total`` pages finished) and gets
a bar. ``total == ONE_SHOT`` is a one-off step with nothing to count, so it gets
a single status line instead. Either way no step runs invisibly.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

#: Signature of the callback handed to ``PdfToMarkdown``.
ProgressCallback = Callable[[str, int, int], None]

#: Pass as ``total`` for a step that has no per-page progress to report.
ONE_SHOT = 0

#: Embedded-image export - its own full pdfminer pass, before the pipeline.
STAGE_IMAGES = "images"
#: Pass A - reading pdfplumber primitives per page (usually the slowest step).
STAGE_LOAD = "load"
#: Pass A - grouping characters into lines and detecting tables, per page.
STAGE_ANALYZE = "analyze"
#: Document-wide statistics (font sizes, page width) the later stages classify by.
STAGE_STATS = "stats"
#: Repeating header/footer detection across all pages.
STAGE_HEADERS = "headers"
#: Pass B - cleaning, reading order and block classification, per page.
STAGE_RENDER = "render"
#: Markdown assembly and post-processing.
STAGE_SERIALIZE = "serialize"

_LABEL = {
    STAGE_IMAGES: "extracting images",
    STAGE_LOAD: "reading pages",
    STAGE_ANALYZE: "analyzing layout",
    STAGE_STATS: "measuring document",
    STAGE_HEADERS: "detecting headers/footers",
    STAGE_RENDER: "rendering pages",
    STAGE_SERIALIZE: "writing markdown",
}

#: Prefix for status lines, so they stand apart from anything else on stderr.
_PREFIX = "[pdf2md]"


def progress_available() -> bool:
    """True when tqdm can be imported, i.e. a bar is actually possible."""
    try:
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


def progress_enabled(stream=None, force: bool | None = None) -> bool:
    """Decide whether to draw a bar.

    ``force=False`` (``--no-progress``) always wins, ``force=True``
    (``--progress``) forces the bar on. With ``force=None`` the
    ``PDF2MD_NO_PROGRESS`` / ``PDF2MD_PROGRESS`` environment variables are
    consulted, and finally "stderr is a TTY" - so redirected output and CI logs
    stay clean by default.
    """
    if force is False or not progress_available():
        return False
    if force is True:
        return True
    if os.environ.get("PDF2MD_NO_PROGRESS"):
        return False
    if os.environ.get("PDF2MD_PROGRESS"):
        return True
    stream = sys.stderr if stream is None else stream
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


class TqdmProgress:
    """A :data:`ProgressCallback` that draws one tqdm bar per stage.

    A stage change closes the finished bar (leaving it on screen at 100%) and
    starts a fresh one, so every stage stays visible instead of overwriting one
    line. Steps reported with ``total=ONE_SHOT`` print a ``[pdf2md] ...`` line
    instead of a bar. Use it as a context manager so a bar left open by an
    exception is still closed.

    ``enabled=False`` (or a missing tqdm) turns every call into a no-op, which
    keeps callers from having to branch. That is also what ``--no-progress``
    selects, so a disabled reporter stays completely silent.
    """

    def __init__(self, enabled: bool = True, stream=None):
        self.enabled = bool(enabled) and progress_available()
        self.stream = sys.stderr if stream is None else stream
        self._bar = None
        self._stage = None

    def __call__(self, stage: str, done: int, total: int) -> None:
        if not self.enabled:
            return
        if total > 0:
            self._advance(stage, done, total)
        else:
            self.status(_LABEL.get(stage, stage))

    def status(self, message: str) -> None:
        """Print one ``[pdf2md] ...`` line; used for steps without a bar."""
        if not self.enabled:
            return
        from tqdm import tqdm

        tqdm.write(f"{_PREFIX} {message}", file=self.stream)

    def __enter__(self) -> TqdmProgress:
        return self

    def __exit__(self, *exc_info) -> bool:
        self._close()
        return False

    def _advance(self, stage: str, done: int, total: int) -> None:
        if stage != self._stage or self._bar is None:
            self._close()
            self._stage = stage
            from tqdm import tqdm

            self._bar = tqdm(
                total=total,
                desc=_LABEL.get(stage, stage),
                unit="page",
                file=self.stream,
                dynamic_ncols=True,
            )
        # Callers report an absolute position; tqdm counts deltas, and feeding it
        # the delta (rather than assigning .n) keeps its smoothed rate and ETA.
        self._bar.update(max(0, min(done, total) - self._bar.n))
        if done >= total:
            self._close()

    def _close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
