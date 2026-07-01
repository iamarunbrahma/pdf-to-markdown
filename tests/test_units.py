"""Unit tests for the pure, framework-free pipeline stages."""

from __future__ import annotations

from pdf2md import fonts
from pdf2md.ir import Line, Span
from pdf2md.math_detect import is_math_char, to_latex_token
from pdf2md.patterns import is_hr, is_toc_line, list_marker, strip_marker
from pdf2md.postprocess import clean
from pdf2md.serialize import render_inline
from pdf2md.tables import _drop_empty_columns, _looks_tabular, table_to_markdown
from pdf2md.textnorm import is_wide, normalize_char


# ---- fonts ---------------------------------------------------------------
def test_font_bold_italic_mono():
    assert fonts.is_bold("ABCDEF+Arial-BoldMT")
    assert fonts.is_italic("TimesNewRomanPS-ItalicMT")
    assert fonts.is_mono("XYZ+CourierNewPSMT")
    assert not fonts.is_bold("Helvetica")
    assert fonts.clean_fontname("ABCDEF+Arial") == "Arial"


def test_math_font():
    assert fonts.is_math_font("ABCDEF+CMMI10")
    assert fonts.is_math_font("CMSY7")
    assert not fonts.is_math_font("Helvetica")


def test_detect_mono_fonts_by_advance_uniformity():
    # Monospace: every glyph has the same advance -> detected even without a
    # "mono" in the name. Proportional: varied advances -> not detected.
    mono = [
        {"text": c, "fontname": "ABCDEF+mplus1mn", "adv": 6.0, "size": 10.0}
        for c in "abcdefghij"
    ]
    prop = [
        {"text": c, "fontname": "GHIJKL+Prose", "adv": a, "size": 10.0}
        for c, a in zip("abcdefghij", [3, 8, 5, 7, 2, 9, 4, 6, 3, 8])
    ]
    found = fonts.detect_mono_fonts(mono + prop)
    assert "mplus1mn" in found
    assert "Prose" not in found


# ---- patterns ------------------------------------------------------------
def test_list_marker():
    assert list_marker("- item") == ("ul", "-")
    assert list_marker("• item") == ("ul", "•")
    assert list_marker("1. step")[0] == "ol"
    assert list_marker("a) thing")[0] == "ol"
    assert list_marker("normal text") is None


def test_strip_marker():
    assert strip_marker("- hello") == "hello"
    assert strip_marker("1. hello") == "hello"


def test_hr_and_toc():
    assert is_hr("---")
    assert is_hr("____")
    assert not is_hr("a-b")
    assert is_toc_line("Chapter 1 ........ 12")
    assert not is_toc_line("just a sentence.")


# ---- math ----------------------------------------------------------------
def test_unicode_to_latex():
    assert to_latex_token("α") == r"\alpha"
    assert to_latex_token("x≤y") == r"x\leqy"
    assert is_math_char({"fontname": "CMMI10", "text": "x"})
    assert is_math_char({"fontname": "Helvetica", "text": "≤"})
    assert not is_math_char({"fontname": "Helvetica", "text": "a"})


# ---- tables --------------------------------------------------------------
def test_table_to_markdown():
    md = table_to_markdown([["A", "B"], ["1", "2"]])
    lines = md.splitlines()
    assert lines[0].startswith("| A")
    assert set(lines[1]) <= set("|- ")
    assert "| 1" in lines[2]


def test_looks_tabular_rejects_prose_columns():
    # 8 single-word columns spanning the page == two-column prose, not a table.
    rows = [["Left", "column", "line", "1.", "Right", "column", "line", "1."]] * 5
    assert not _looks_tabular(rows, (0, 0, 600, 700), 612, 792)
    # A small 3-column grid with multi-word cells is a real table.
    good = [["Name", "Job title", "Years"], ["Ada", "Lead engineer", "10"]]
    assert _looks_tabular(good, (50, 50, 300, 120), 612, 792)


def test_drop_empty_columns():
    rows = [["A", "", "B"], ["1", "", "2"]]
    assert _drop_empty_columns(rows) == [["A", "B"], ["1", "2"]]


def test_looks_tabular_rejects_full_page_grid():
    # A "table" that fills most of the page height is a mis-read document, not a table.
    rows = [["a", "b"], ["c", "d"]] * 10
    assert not _looks_tabular(rows, (50, 40, 400, 760), 612, 792)


def test_dollar_amounts_are_escaped():
    # Dollar amounts must be escaped so KaTeX/MathJax don't treat them as math.
    out = render_inline(_line(Span(text="Revenue was $5 billion", size=10)))
    assert r"\$5 billion" in out
    assert "$5 billion" not in out.replace(r"\$", "")  # no unescaped '$'


def test_ligatures_and_control_chars():
    assert normalize_char("ﬁ") == "fi"
    assert normalize_char("ﬀ") == "ff"
    assert normalize_char("\xa0") == " "
    assert normalize_char("\x00") == ""  # control char dropped


def test_cjk_handling():
    assert is_wide("世") and is_wide("あ") and is_wide("한")
    assert not is_wide("A") and not is_wide("é")
    # Full-justification spaces between CJK characters are collapsed.
    assert clean("联 合 国 大 会").strip() == "联合国大会"
    # Latin spacing is untouched.
    assert clean("hello world").strip() == "hello world"


# ---- inline serialization ------------------------------------------------
def _line(*spans):
    return Line(spans=list(spans))


def test_render_inline_styles():
    out = render_inline(
        _line(
            Span(text="Hello ", size=10),
            Span(text="bold", size=10, bold=True),
            Span(text=" and ", size=10),
            Span(text="it", size=10, italic=True),
        )
    )
    assert "**bold**" in out
    assert "*it*" in out


def test_render_inline_link_and_super():
    out = render_inline(
        _line(
            Span(text="ref", size=10, link="https://x.io"),
            Span(text="2", size=7, superscript=True),
        )
    )
    assert "[ref](https://x.io)" in out
    assert "<sup>2</sup>" in out


def test_render_inline_math_run():
    out = render_inline(
        _line(
            Span(text="E", size=10, math=True),
            Span(text="=", size=10, math=True),
            Span(text="mc", size=10, math=True),
            Span(text="2", size=7, math=True, superscript=True),
        )
    )
    assert out == "$E=mc^{2}$"
