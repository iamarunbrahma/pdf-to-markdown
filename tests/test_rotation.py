"""Rotated-page handling: text must be recovered in correct reading order."""

from __future__ import annotations

import pytest

pytest.importorskip("reportlab")

from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from pdf2md import convert_pdf  # noqa: E402


def test_drawn_rotated_page_reads_in_order(tmp_path):
    path = str(tmp_path / "rot.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    c.saveState()
    c.translate(500, 120)
    c.rotate(90)  # draw content sideways (no /Rotate flag)
    c.setFont("Helvetica", 12)
    c.drawString(0, 0, "First line of rotated text.")
    c.drawString(0, -16, "Second line follows below.")
    c.restoreState()
    c.showPage()
    c.save()

    md = convert_pdf(path)
    assert "First line of rotated text." in md
    assert "Second line follows below." in md
    assert md.index("First line") < md.index("Second line")  # correct order
    assert "```" not in md  # rotated prose must not be mistaken for code
