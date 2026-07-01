"""Golden end-to-end tests over committed fixture PDFs."""

from __future__ import annotations

import os

import pytest

from pdf2md import PdfToMarkdown, convert_pdf

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fx(name):
    path = os.path.join(FIXTURES, name)
    if not os.path.isfile(path):
        pytest.skip(f"fixture {name} not generated; run tests/fixtures/generate.py")
    return path


def test_basic_elements():
    md = convert_pdf(_fx("basic.pdf"))
    assert "# Document Title" in md
    assert "## Section One" in md
    assert "**bold**" in md
    assert "*italic*" in md
    assert "[Visit example.com](https://example.com)" in md
    assert "- First bullet" in md
    assert "  - Nested bullet" in md  # nesting via indentation
    assert "1. First step" in md
    assert "```python" in md
    assert "def greet(name):" in md
    # Heading text must not carry stray bold markers.
    assert "# **Document Title**" not in md


def test_table_is_gfm():
    md = convert_pdf(_fx("table.pdf"))
    assert "| Name" in md
    assert "| Ada" in md and "Engineer" in md
    sep = [ln for ln in md.splitlines() if set(ln) <= set("|- ") and "-" in ln]
    assert sep, "expected a GFM separator row"


def test_two_column_reading_order():
    md = convert_pdf(_fx("twocol.pdf"))
    assert "# Two Column Paper" in md
    # Title precedes body; entire left column precedes the right column.
    assert md.index("Two Column Paper") < md.index("Left column line 1")
    assert md.index("Left column line 12") < md.index("Right column line 1")
    # The two-column body must NOT be misread as a table.
    assert "|" not in md


def test_extract_writes_file(tmp_path):
    extractor = PdfToMarkdown({"OUTPUT_DIR": str(tmp_path)})
    full, pages = extractor.extract(_fx("basic.pdf"))
    out = tmp_path / "basic.md"
    assert out.is_file()
    assert out.read_text(encoding="utf-8").strip() == full.strip()
    assert len(pages) >= 1
