"""OPTIONAL high-accuracy math via pix2tex (LaTeX-OCR).

This module is deliberately NOT imported by the MIT-only core. It is the opt-in
upgrade path for math: pix2tex itself is MIT-licensed but depends on BSD `torch`,
so it lives behind an extra (`pip install -r requirements-math.txt`).

Usage:
    from pdf2md.converter import PdfToMarkdown
    from pdf2md.plugins.math_ml import enhance_math_blocks
    md = enhance_math_blocks("paper.pdf", PdfToMarkdown())

It re-OCRs every detected display-math region with pix2tex and replaces the
heuristic LaTeX, then serializes. Inline math is left to the core heuristic.
"""

from __future__ import annotations

from ..ir import BlockType
from ..postprocess import clean
from ..serialize import serialize


def _load_model():
    try:
        from pix2tex.cli import LatexOCR  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "pix2tex is not installed. Install the optional extra with:\n"
            "    pip install -r requirements-math.txt"
        ) from exc
    return LatexOCR()


def enhance_math_blocks(pdf_path: str, converter, padding: float = 4.0) -> str:
    """Convert `pdf_path`, replacing display-math LaTeX with pix2tex output."""
    import pdfplumber  # transitive (MIT pdfplumber); used here for cropping

    model = _load_model()
    document = converter.convert(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page in document.pages:
            plumber_page = pdf.pages[page.number]
            for block in page.blocks:
                if block.type is not BlockType.MATH_DISPLAY:
                    continue
                bbox = (
                    max(0, block.x0 - padding),
                    max(0, block.top - padding),
                    min(plumber_page.width, block.x1 + padding),
                    min(plumber_page.height, block.bottom + padding),
                )
                try:
                    image = plumber_page.crop(bbox).to_image(resolution=200).original
                    latex = model(image).strip()
                except Exception:  # pragma: no cover - best effort
                    continue
                if latex:
                    block.meta["latex_override"] = latex

    _apply_overrides(document)
    full, _ = serialize(document, converter.config["PAGE_DELIMITER"])
    return clean(full)


def _apply_overrides(document) -> None:
    """Bake pix2tex overrides into single-span math lines for serialization."""
    from ..ir import Line, Span

    for page in document.pages:
        for block in page.blocks:
            latex = block.meta.get("latex_override")
            if latex:
                block.lines = [Line(spans=[Span(text=latex, math=True)])]
