"""Progress reporting: callback contract, silent mode, CLI flags.

The pipeline must stay usable without a callback and without tqdm installed, so
these tests pin both the "reports every page of every stage" contract and the
degradation paths.
"""

from __future__ import annotations

import io
import logging
import os

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from pdf2md import PdfToMarkdown
from pdf2md.cli import _feature_summary
from pdf2md.cli import main as cli_main
from pdf2md.progress import (
    ONE_SHOT,
    STAGE_ANALYZE,
    STAGE_HEADERS,
    STAGE_IMAGES,
    STAGE_LOAD,
    STAGE_RENDER,
    STAGE_SERIALIZE,
    STAGE_STATS,
    TqdmProgress,
    progress_available,
    progress_enabled,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
#: Steps with a page counter - these get a bar.
PAGE_STAGES = (STAGE_IMAGES, STAGE_LOAD, STAGE_ANALYZE, STAGE_RENDER)
#: One-off steps - these get a "[pdf2md] ..." line instead, in pipeline order.
STEP_STAGES = (STAGE_STATS, STAGE_HEADERS, STAGE_SERIALIZE)


def _fx(name):
    path = os.path.join(FIXTURES, name)
    if not os.path.isfile(path):
        pytest.skip(f"fixture {name} not generated; run tests/fixtures/generate.py")
    return path


def _multipage_pdf(tmp_path, pages=3):
    """A plain N-page PDF; every committed fixture is single-page."""
    path = tmp_path / "multipage.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    for i in range(pages):
        c.drawString(72, 720, f"Heading on page {i + 1}")
        c.drawString(72, 700, "Body text on this page.")
        c.showPage()
    c.save()
    return str(path)


# ---- the callback contract ------------------------------------------------
def test_every_step_reports_itself(tmp_path):
    """No step may run invisibly: each one is a bar or an announcement."""
    calls = []
    extractor = PdfToMarkdown(
        {"OUTPUT_DIR": str(tmp_path)},
        progress=lambda stage, done, total: calls.append((stage, done, total)),
    )
    extractor.convert_to_markdown(_multipage_pdf(tmp_path, pages=3))

    for stage in PAGE_STAGES:
        per_stage = [(done, total) for s, done, total in calls if s == stage]
        # One call per page, counting 1..N, all against the same total.
        assert [done for done, _ in per_stage] == [1, 2, 3], stage
        assert {total for _, total in per_stage} == {3}, stage

    # Every other step announces itself exactly once, in pipeline order.
    stepped = [stage for stage, _, total in calls if total == ONE_SHOT]
    assert stepped == list(STEP_STAGES)


def test_stages_run_in_pipeline_order(tmp_path):
    """Bars must not interleave: a stage finishes before the next one starts."""
    calls = []
    PdfToMarkdown(
        {"OUTPUT_DIR": str(tmp_path)},
        progress=lambda stage, done, total: calls.append(stage),
    ).convert_to_markdown(_multipage_pdf(tmp_path, pages=3))

    first_seen = []
    for stage in calls:
        if stage not in first_seen:
            first_seen.append(stage)
    assert first_seen == [
        STAGE_IMAGES,
        STAGE_LOAD,
        STAGE_ANALYZE,
        STAGE_STATS,
        STAGE_HEADERS,
        STAGE_RENDER,
        STAGE_SERIALIZE,
    ]
    # Each stage's calls form one contiguous run.
    runs = [s for i, s in enumerate(calls) if i == 0 or calls[i - 1] != s]
    assert len(runs) == len(set(runs))


def test_no_callback_costs_nothing(monkeypatch):
    """The default stays callback-free, and skips the page count it would need."""
    import pdf2md.converter as conv

    calls = []
    monkeypatch.setattr(conv, "_page_count", lambda p: calls.append(p) or 0)
    doc = PdfToMarkdown({"extract_images": False}).convert(_fx("basic.pdf"))
    assert doc.pages
    assert calls == []  # never opened the PDF just to count its pages


def test_convert_pdf_accepts_a_callback(tmp_path):
    """The module-level convenience wrapper forwards the callback too."""
    from pdf2md import convert_pdf

    calls = []
    md = convert_pdf(
        _fx("basic.pdf"),
        {"OUTPUT_DIR": str(tmp_path), "extract_images": False},
        progress=lambda stage, done, total: calls.append(stage),
    )
    assert md.strip()
    assert STAGE_LOAD in calls


# ---- enabling / disabling -------------------------------------------------
def test_progress_disabled_without_tty(monkeypatch):
    monkeypatch.delenv("PDF2MD_PROGRESS", raising=False)
    monkeypatch.delenv("PDF2MD_NO_PROGRESS", raising=False)
    assert progress_enabled(force=False) is False
    assert progress_enabled(stream=io.StringIO()) is False


@pytest.mark.skipif(
    not progress_available(), reason="tqdm not installed; no bar is ever enabled"
)
def test_env_vars_override_tty_detection(monkeypatch):
    monkeypatch.delenv("PDF2MD_NO_PROGRESS", raising=False)
    monkeypatch.setenv("PDF2MD_PROGRESS", "1")
    assert progress_enabled(stream=io.StringIO()) is True

    monkeypatch.setenv("PDF2MD_NO_PROGRESS", "1")
    assert progress_enabled(stream=io.StringIO()) is False
    # An explicit flag still beats the environment.
    assert progress_enabled(stream=io.StringIO(), force=True) is True


def test_disabled_progress_is_silent():
    buf = io.StringIO()
    bar = TqdmProgress(enabled=False, stream=buf)
    with bar:
        bar(STAGE_LOAD, 1, 3)
        bar(STAGE_STATS, 0, ONE_SHOT)
        bar.status("features on: images")
    assert buf.getvalue() == ""


# ---- what actually lands on the terminal ----------------------------------
def test_progress_draws_bars_and_announces_steps():
    buf = io.StringIO()
    bar = TqdmProgress(stream=buf)
    if not bar.enabled:
        pytest.skip("tqdm not installed; run `uv sync --extra progress`")

    with bar:
        bar(STAGE_IMAGES, 1, 2)
        bar(STAGE_IMAGES, 2, 2)  # closes the "extracting images" bar
        bar(STAGE_STATS, 0, ONE_SHOT)  # announced, never drawn
        bar(STAGE_ANALYZE, 1, 1)
        bar.status("features on: images")

    out = buf.getvalue()
    assert "extracting images: 100%" in out
    assert "analyzing layout: 100%" in out
    assert "[pdf2md] measuring document" in out
    assert "[pdf2md] features on: images" in out
    assert out.endswith("\n")  # nothing left dangling


def test_bar_is_closed_when_a_stage_raises():
    """A bar abandoned mid-stage is still closed by the context manager."""
    buf = io.StringIO()
    bar = TqdmProgress(stream=buf)
    if not bar.enabled:
        pytest.skip("tqdm not installed; run `uv sync --extra progress`")

    with pytest.raises(RuntimeError):
        with bar:
            bar(STAGE_LOAD, 1, 10)  # bar left open at 10%
            raise RuntimeError("boom")
    assert bar._bar is None
    assert buf.getvalue().endswith("\n")


def test_feature_summary_names_both_sides():
    text = _feature_summary({"extract_images": True, "detect_footnotes": False})
    assert text.startswith("features on: images")
    assert "off: " in text and "footnotes" in text.split("off: ")[1]


# ---- the CLI --------------------------------------------------------------
def test_cli_no_progress_is_silent_and_still_writes(tmp_path, capsys):
    rc = cli_main(
        ["--pdf_path", _fx("basic.pdf"), "--out", str(tmp_path), "--no-progress"]
    )
    assert rc == 0
    assert (tmp_path / "basic.md").is_file()
    captured = capsys.readouterr()
    assert captured.err == ""  # not one byte of progress output
    assert "Wrote" in captured.out


def test_cli_progress_forced_on_without_a_tty(tmp_path, capsys):
    if not TqdmProgress().enabled:
        pytest.skip("tqdm not installed; run `uv sync --extra progress`")
    rc = cli_main(
        ["--pdf_path", _fx("basic.pdf"), "--out", str(tmp_path), "--progress"]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "[pdf2md] features on:" in err
    assert "reading pages" in err


def test_log_records_do_not_shred_the_bar(tmp_path, monkeypatch, capsys):
    """A warning raised mid-conversion must not land inside the bar line.

    pdfminer warns freely on real-world PDFs, and with no handler configured
    those records still reach stderr through `logging.lastResort` - so the
    redirect has to engage even though the root logger looks empty.
    """
    if not TqdmProgress().enabled:
        pytest.skip("tqdm not installed; run `uv sync --extra progress`")
    # pytest installs its own root handlers; a real CLI run has none, which is
    # the case the redirect has to cope with.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    import pdf2md.converter as conv

    real_load = conv.load_page

    def noisy_load(page):
        log = logging.getLogger("pdfminer.pdfpage")
        log.warning("Invalid MediaBox in /Page, defaulting to US Letter")
        log.info("info stays below the root level")
        return real_load(page)

    monkeypatch.setattr(conv, "load_page", noisy_load)
    # Multi-page, so the bar actually redraws while the warnings are arriving.
    pdf = _multipage_pdf(tmp_path, pages=5)
    rc = cli_main(["--pdf_path", pdf, "--out", str(tmp_path), "--progress"])
    assert rc == 0

    err = capsys.readouterr().err
    assert "Invalid MediaBox" in err  # the warning still gets through
    assert "info stays below" not in err  # but nothing previously silent does

    # No line may carry both a bar and a log record: that is what "shredded" is.
    for line in err.replace("\r", "\n").split("\n"):
        assert not ("page/s" in line and "Invalid MediaBox" in line), line
