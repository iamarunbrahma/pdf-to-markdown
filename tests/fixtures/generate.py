"""Generate the committed fixture PDFs used by the test suite.

Dev-only: requires reportlab (BSD). Run once and commit the resulting PDFs:
    python tests/fixtures/generate.py
"""

from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = letter


def _path(name):
    return os.path.join(HERE, name)


def basic():
    c = canvas.Canvas(_path("basic.pdf"), pagesize=letter)
    y = H - inch

    c.setFont("Helvetica-Bold", 24)
    c.drawString(inch, y, "Document Title")
    y -= 36

    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, y, "Section One")
    y -= 24

    c.setFont("Helvetica", 11)
    c.drawString(inch, y, "This is a normal paragraph of body text that wraps")
    y -= 14
    c.drawString(inch, y, "across two physical lines in the PDF source.")
    y -= 24

    # Mixed inline styles on one baseline.
    x = inch
    c.setFont("Helvetica", 11)
    c.drawString(x, y, "Here is ")
    x += c.stringWidth("Here is ", "Helvetica", 11)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "bold")
    x += c.stringWidth("bold", "Helvetica-Bold", 11)
    c.setFont("Helvetica", 11)
    c.drawString(x, y, " and ")
    x += c.stringWidth(" and ", "Helvetica", 11)
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(x, y, "italic")
    x += c.stringWidth("italic", "Helvetica-Oblique", 11)
    c.setFont("Helvetica", 11)
    c.drawString(x, y, " text.")
    y -= 24

    # Hyperlink.
    c.setFont("Helvetica", 11)
    link_text = "Visit example.com"
    c.drawString(inch, y, link_text)
    tw = c.stringWidth(link_text, "Helvetica", 11)
    c.linkURL("https://example.com", (inch, y - 2, inch + tw, y + 11), relative=0)
    y -= 30

    # Unordered list.
    c.setFont("Helvetica", 11)
    for item in ("- First bullet", "- Second bullet"):
        c.drawString(inch, y, item)
        y -= 14
    # Nested bullet.
    c.drawString(inch + 24, y, "- Nested bullet")
    y -= 20

    # Ordered list.
    for item in ("1. First step", "2. Second step"):
        c.drawString(inch, y, item)
        y -= 14
    y -= 16

    # Horizontal rule.
    c.line(inch, y, W - inch, y)
    y -= 24

    # Code block (monospace).
    c.setFont("Courier", 10)
    for code in ("def greet(name):", "    return f'hi {name}'"):
        c.drawString(inch, y, code)
        y -= 12

    c.showPage()
    c.save()


def tables():
    doc = SimpleDocTemplate(_path("table.pdf"), pagesize=letter)
    data = [
        ["Name", "Role", "Years"],
        ["Ada", "Engineer", "10"],
        ["Bob", "Designer", "4"],
    ]
    t = Table(data)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    doc.build([t])


def twocol():
    c = canvas.Canvas(_path("twocol.pdf"), pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W / 2, H - inch, "Two Column Paper")

    left_x, right_x = inch, W / 2 + 0.25 * inch
    top = H - 1.6 * inch
    c.setFont("Helvetica", 10)
    left = [f"Left column line {i}." for i in range(1, 13)]
    right = [f"Right column line {i}." for i in range(1, 13)]
    y = top
    for line in left:
        c.drawString(left_x, y, line)
        y -= 14
    y = top
    for line in right:
        c.drawString(right_x, y, line)
        y -= 14
    c.showPage()
    c.save()


if __name__ == "__main__":
    basic()
    tables()
    twocol()
    print("Fixtures written to", HERE)
