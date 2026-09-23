"""Embedded images: the Markdown link must resolve relative to the .md file."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("reportlab")

from PIL import Image  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from pdf2md import PdfToMarkdown  # noqa: E402


def test_image_link_is_relative_to_markdown_file(tmp_path, monkeypatch):
    png = tmp_path / "pic.png"
    Image.new("RGB", (120, 80), (200, 40, 40)).save(png)
    pdf = str(tmp_path / "withimg.pdf")
    c = canvas.Canvas(pdf, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "Text above the image.")
    c.drawImage(str(png), 72, 500, width=200, height=140)
    c.showPage()
    c.save()

    # Mirror CLI usage: a relative --out, resolved against the working directory.
    monkeypatch.chdir(tmp_path)
    full, _ = PdfToMarkdown({"OUTPUT_DIR": "out"}).extract(pdf)

    md_file = tmp_path / "out" / "withimg.md"
    assert md_file.is_file()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", full)
    assert links, "expected an image reference in the Markdown"
    for link in links:
        # Issue #8: the link must resolve from the directory holding the .md file.
        assert (md_file.parent / link).is_file(), link
