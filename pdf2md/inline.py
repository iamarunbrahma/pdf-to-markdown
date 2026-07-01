"""Stage 3: build styled `Line`/`Span` objects from raw character groups.

Given the characters of one visual line, this:
  * detects per-character bold / italic / monospace (font name) and
    superscript / subscript (geometry relative to the line baseline),
  * tags math characters,
  * attaches hyperlink targets,
  * coalesces runs of identically-styled characters into spans, inserting spaces
    where horizontal gaps imply them.
"""

from __future__ import annotations

from statistics import median

from . import fonts
from .ir import Line, Span
from .math_detect import is_math_char
from .textnorm import is_wide, normalize_char


def _char_style(ch: dict, line_size: float, baseline: float, mono_fonts: set) -> dict:
    size = ch["size"] or line_size
    bottom = ch["bottom"]
    small = size <= line_size * 0.85 if line_size else False
    superscript = small and (baseline - bottom) > line_size * 0.12
    subscript = small and (bottom - baseline) > line_size * 0.08
    is_mono = (
        fonts.is_mono(ch["fontname"])
        or fonts.clean_fontname(ch["fontname"]) in mono_fonts
    )
    return {
        "bold": fonts.is_bold(ch["fontname"]),
        "italic": fonts.is_italic(ch["fontname"]),
        "mono": is_mono,
        "superscript": superscript,
        "subscript": subscript,
        "math": is_math_char(ch),
    }


def _link_for(ch: dict, hyperlinks: list[dict]):
    cx = (ch["x0"] + ch["x1"]) / 2
    cy = (ch["top"] + ch["bottom"]) / 2
    for h in hyperlinks:
        if h["x0"] - 1 <= cx <= h["x1"] + 1 and h["top"] - 1 <= cy <= h["bottom"] + 1:
            return h["uri"]
    return None


def build_line(
    chars: list[dict], hyperlinks: list[dict], mono_fonts: set | None = None
) -> Line:
    """Coalesce a list of raw chars (one visual line) into a styled Line."""
    mono_fonts = mono_fonts or set()
    chars = sorted(chars, key=lambda c: c["x0"])
    sizes = [c["size"] for c in chars if c["size"]] or [10.0]
    line_size = median(sizes)
    baseline = median([c["bottom"] for c in chars])

    spans: list[Span] = []
    prev = None
    for ch in chars:
        ctext = normalize_char(ch["text"])
        if ctext == "":  # dropped control char / zero-width glyph
            continue
        style = _char_style(ch, line_size, baseline, mono_fonts)
        link = _link_for(ch, hyperlinks)
        key = (
            style["bold"],
            style["italic"],
            style["mono"],
            style["superscript"],
            style["subscript"],
            style["math"],
            link,
        )
        gap = (ch["x0"] - prev["x1"]) if prev else 0.0
        need_space = (
            prev is not None
            and gap > max(1.0, 0.2 * (ch["size"] or line_size))
            and not ctext.startswith(" ")
            # CJK / full-width scripts are written without inter-char spaces.
            and not is_wide(ctext)
            and not is_wide(prev["text"])
        )

        if spans and spans[-1].style_key == key:
            if need_space and not spans[-1].text.endswith(" "):
                spans[-1].text += " "
            spans[-1].text += ctext
            spans[-1].x1 = ch["x1"]
            spans[-1].top = min(spans[-1].top, ch["top"])
            spans[-1].bottom = max(spans[-1].bottom, ch["bottom"])
        else:
            if spans and need_space and not spans[-1].text.endswith(" "):
                spans[-1].text += " "
            spans.append(
                Span(
                    text=ctext,
                    fontname=ch["fontname"],
                    size=ch["size"],
                    bold=style["bold"],
                    italic=style["italic"],
                    mono=style["mono"],
                    superscript=style["superscript"],
                    subscript=style["subscript"],
                    math=style["math"],
                    link=link,
                    x0=ch["x0"],
                    x1=ch["x1"],
                    top=ch["top"],
                    bottom=ch["bottom"],
                )
            )
        prev = ch

    return Line(
        spans=spans,
        x0=min((c["x0"] for c in chars), default=0.0),
        x1=max((c["x1"] for c in chars), default=0.0),
        top=min((c["top"] for c in chars), default=0.0),
        bottom=max((c["bottom"] for c in chars), default=0.0),
    )
