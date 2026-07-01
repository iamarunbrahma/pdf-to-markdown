"""Pipeline orchestration: PDF path in, Markdown out.

Two passes over the document:
  A. Load every page's primitives, group characters into styled lines, and detect
     tables. Collect all lines to measure document-wide statistics and to find
     repeating headers/footers.
  B. Per page, strip headers/footers and footnotes, remove text inside table and
     image regions, compute reading order, classify into blocks, and splice
     tables/images/footnotes back in at their positions.
"""

from __future__ import annotations

import os
import re

import pdfplumber

from . import fonts, headers
from .analyze import compute_stats
from .blocks import group_into_lines
from .classify import classify_segment
from .images import extract_images
from .ir import Block, BlockType, Document, Page
from .loader import load_page
from .postprocess import clean
from .regions import drop_lines_in_boxes, reading_order_segments
from .serialize import serialize
from .tables import detect_tables

DEFAULT_CONFIG = {
    "OUTPUT_DIR": "outputs",
    "PAGE_DELIMITER": "",
    "detect_borderless_tables": True,
    "detect_blockquote": True,
    "detect_footnotes": True,
    "bold_as_heading": True,
    "extract_images": True,
}

_FOOTNOTE_MARKER = re.compile(r"^(\d{1,3})[\s.)\]]")


class PdfToMarkdown:
    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    # ---- public API -------------------------------------------------------
    def convert(self, pdf_path: str) -> Document:
        image_map = {}
        if self.config["extract_images"]:
            try:
                image_map = extract_images(pdf_path, self.config["OUTPUT_DIR"])
            except Exception:
                image_map = {}

        pages_data, all_lines = self._pass_a(pdf_path)
        page_width = pages_data[0]["width"] if pages_data else 612.0
        stats = compute_stats(all_lines, page_width, self.config)

        sigs = headers.collect_signatures(
            [(p["height"], p["lines"]) for p in pages_data]
        )
        repeating = headers.repeating_set(sigs, len(pages_data))

        document = Document(meta={"source": pdf_path})
        for pdata in pages_data:
            page = self._pass_b(
                pdata, stats, repeating, image_map.get(pdata["number"], [])
            )
            document.pages.append(page)
        return document

    def convert_to_markdown(self, pdf_path: str) -> tuple[str, list[str]]:
        document = self.convert(pdf_path)
        full, page_md = serialize(document, self.config["PAGE_DELIMITER"])
        return clean(full), [clean(p) for p in page_md]

    def extract(self, pdf_path: str) -> tuple[str, list[str]]:
        """Convert and write `<OUTPUT_DIR>/<name>.md`; returns (markdown, pages)."""
        full, pages = self.convert_to_markdown(pdf_path)
        out_dir = self.config["OUTPUT_DIR"]
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(full)
        return full, pages

    # ---- internals --------------------------------------------------------
    def _pass_a(self, pdf_path: str):
        pages_data, all_lines = [], []
        with pdfplumber.open(pdf_path) as pdf:
            raws = [load_page(page) for page in pdf.pages]
            # Monospace fonts are detected document-wide (advance-width uniformity)
            # so code is recognized even when the font name lacks "mono"/"courier".
            mono_fonts = fonts.detect_mono_fonts([c for r in raws for c in r.chars])
            for page, raw in zip(pdf.pages, raws):
                lines = group_into_lines(
                    raw.chars, raw.hyperlinks, raw.width, mono_fonts
                )
                # pdfplumber's table geometry is in un-rotated space, which no
                # longer matches our rotation-normalized chars, so on a rotated
                # page we let the (now-upright) text flow instead.
                tables = [] if raw.rotated else detect_tables(page, self.config)
                pages_data.append(
                    {
                        "number": raw.number,
                        "width": raw.width,
                        "height": raw.height,
                        "lines": lines,
                        "tables": tables,
                    }
                )
                all_lines.extend(lines)
        return pages_data, all_lines

    def _pass_b(self, pdata, stats, repeating, page_images) -> Page:
        height = pdata["height"]
        lines = [
            ln
            for ln in pdata["lines"]
            if not headers.is_header_footer(ln, height, repeating)
        ]

        footnote_blocks, lines = self._extract_footnotes(lines, height, stats.body_size)

        table_boxes = [t["bbox"] for t in pdata["tables"]]
        image_boxes = [im["bbox"] for im in page_images]
        lines = drop_lines_in_boxes(lines, table_boxes + image_boxes)

        text_blocks = []
        for seg in reading_order_segments(lines, pdata["width"]):
            text_blocks.extend(classify_segment(seg, stats))

        floating = []
        for t in pdata["tables"]:
            floating.append(
                Block(
                    BlockType.TABLE,
                    meta={"rows": t["rows"]},
                    top=t["bbox"][1],
                    bottom=t["bbox"][3],
                    x0=t["bbox"][0],
                    x1=t["bbox"][2],
                )
            )
        for im in page_images:
            floating.append(
                Block(
                    BlockType.IMAGE,
                    meta={
                        "path": im["path"],
                        "alt": os.path.splitext(os.path.basename(im["path"]))[0],
                    },
                    top=im["bbox"][1],
                    bottom=im["bbox"][3],
                    x0=im["bbox"][0],
                    x1=im["bbox"][2],
                )
            )

        blocks = _splice_by_top(text_blocks, floating)

        if not text_blocks and floating and self.config["extract_images"]:
            blocks[0].meta["note"] = (
                "No selectable text layer on this page; image embedded as-is."
            )

        blocks.extend(footnote_blocks)
        return Page(
            number=pdata["number"], width=pdata["width"], height=height, blocks=blocks
        )

    def _extract_footnotes(self, lines, height, body_size):
        if not self.config["detect_footnotes"]:
            return [], lines
        zone_top = 0.80 * height
        candidates = [
            ln
            for ln in lines
            if ln.top > zone_top and ln.size and ln.size <= body_size - 1
        ]
        if not any(_FOOTNOTE_MARKER.match(ln.text.strip()) for ln in candidates):
            return [], lines

        cand_set = set(id(ln) for ln in candidates)
        footnotes, current = [], None
        for ln in sorted(candidates, key=lambda c: c.top):
            m = _FOOTNOTE_MARKER.match(ln.text.strip())
            if m:
                current = Block(
                    BlockType.FOOTNOTE,
                    [ln],
                    meta={"marker": m.group(1)},
                    top=ln.top,
                    bottom=ln.bottom,
                )
                footnotes.append(current)
            elif current is not None:
                current.lines.append(ln)
        remaining = [ln for ln in lines if id(ln) not in cand_set]
        return footnotes, remaining


def _splice_by_top(text_blocks: list[Block], floating: list[Block]) -> list[Block]:
    floating = sorted(floating, key=lambda b: b.top)
    result, fi = [], 0
    for tb in text_blocks:
        while fi < len(floating) and floating[fi].top < tb.top:
            result.append(floating[fi])
            fi += 1
        result.append(tb)
    result.extend(floating[fi:])
    return result


def convert_pdf(pdf_path: str, config: dict | None = None) -> str:
    """Convenience: return the full Markdown for a PDF."""
    full, _ = PdfToMarkdown(config).convert_to_markdown(pdf_path)
    return full
