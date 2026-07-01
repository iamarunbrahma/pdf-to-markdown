"""Font-name heuristics.

PDF font names are the single richest style signal pdfplumber gives us, but they
are messy: they usually carry a 6-letter subset prefix ("ABCDEF+ArialMT") and
encode weight/slant inconsistently ("Arial-BoldMT", "TimesNewRomanPS-ItalicMT",
"CMBX10"). These helpers normalize and interpret them.
"""

from __future__ import annotations

import re

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")

_BOLD = re.compile(
    r"(bold|black|heavy|semibold|demibold|extrabold|ultra|[-_]bd\b|bx\d|[-_]b\b)", re.I
)
_ITALIC = re.compile(r"(italic|oblique|slanted|[-_]it\b|[-_]i\b|ti\d)", re.I)
_MONO = re.compile(
    r"(mono|courier|consol|menlo|inconsolata|sourcecode|dejavusansmono|cousine|"
    r"liberationmono|robotomono|firacode|jetbrains|cmtt|cmsltt|typewriter)",
    re.I,
)
# Computer Modern + common Unicode math fonts used (almost) exclusively for math.
_MATH_FONT = re.compile(
    r"(cmmi|cmsy|cmex|cmbsy|msam|msbm|stixmath|stix-math|xitsmath|"
    r"latinmodernmath|lmmath|mathjax|texgyre\w*math|euclidmath|mtmi|mt-extra)",
    re.I,
)


def clean_fontname(fontname: str) -> str:
    """Strip the subset prefix, e.g. 'ABCDEF+Arial-BoldMT' -> 'Arial-BoldMT'."""
    if not fontname:
        return ""
    return _SUBSET_PREFIX.sub("", fontname)


def is_bold(fontname: str) -> bool:
    return bool(_BOLD.search(clean_fontname(fontname)))


def is_italic(fontname: str) -> bool:
    return bool(_ITALIC.search(clean_fontname(fontname)))


def is_mono(fontname: str) -> bool:
    return bool(_MONO.search(clean_fontname(fontname)))


def is_math_font(fontname: str) -> bool:
    return bool(_MATH_FONT.search(clean_fontname(fontname)))


def detect_mono_fonts(chars: list) -> set:
    """Identify monospace fonts by advance-width uniformity.

    Font *names* are an unreliable mono signal (e.g. "mplus1mn-regular" is
    monospace but contains neither "mono" nor "courier"). Monospace fonts instead
    have a (near-)constant advance width per glyph, so we flag any font whose
    single-char advance/size ratios cluster tightly around their median.
    """
    from collections import defaultdict
    from statistics import median

    ratios: dict[str, list] = defaultdict(list)
    ascii_count: dict[str, list] = defaultdict(lambda: [0, 0])  # [ascii, total]
    for c in chars:
        text = c.get("text", "")
        adv = c.get("adv") or 0.0
        size = c.get("size") or 0.0
        if len(text) == 1 and text.strip() and adv > 0 and size > 0:
            name = clean_fontname(c["fontname"])
            ratios[name].append(adv / size)
            ascii_count[name][1] += 1
            if ord(text) < 128:
                ascii_count[name][0] += 1

    mono = set()
    for name, rs in ratios.items():
        if len(rs) < 8:
            continue
        m = median(rs)
        if m <= 0:
            continue
        uniform = sum(1 for r in rs if abs(r - m) <= 0.05 * m) / len(rs)
        # True monospace fonts are ~1.0 uniform; keep the bar high so a
        # proportional font with a coincidentally-uniform small sample (e.g. a
        # 7pt serif) is not mistaken for code.
        if uniform < 0.93:
            continue
        # CJK fonts are uniform-advance too but are prose, not code. Only treat a
        # font as code-monospace if its glyphs are ASCII-dominant.
        a, total = ascii_count[name]
        if total and a / total >= 0.5:
            mono.add(name)
    return mono
