"""Per-character text hygiene.

PDF text streams frequently contain ligature glyphs (a single "ﬁ" character),
exotic whitespace, and stray control characters. Normalizing these as we read
each character keeps the extracted text faithful — a technique borrowed from
pdftext's postprocessing (reimplemented here, MIT-clean).
"""

from __future__ import annotations

import unicodedata

LIGATURES = {
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬆ": "st",
    "ﬅ": "st",
}

# Assorted Unicode spaces that should collapse to a plain ASCII space.
_SPACES = {"\xa0", " ", " ", " ", " ", " ", " ", " "}


def is_wide(text: str) -> bool:
    """True if the text contains CJK / full-width characters.

    Such scripts are written without spaces between characters, so gap-based
    space insertion must be suppressed around them.
    """
    return any(
        "⺀" <= ch <= "鿿"  # CJK radicals .. unified ideographs
        or "　" <= ch <= "ヿ"  # CJK punctuation, hiragana, katakana
        or "가" <= ch <= "힣"  # hangul syllables
        or "＀" <= ch <= "｠"  # full-width forms
        for ch in text
    )


def normalize_char(text: str) -> str:
    """Normalize a single pdf character's text; returns '' to drop it."""
    if not text:
        return ""
    if text in LIGATURES:
        return LIGATURES[text]
    if text in _SPACES:
        return " "
    # Drop control characters (Unicode category starting with 'C'), keeping tabs.
    if len(text) == 1 and text != "\t" and unicodedata.category(text).startswith("C"):
        return ""
    return text
