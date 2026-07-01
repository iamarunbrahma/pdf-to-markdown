"""Detect and strip repeating running headers / footers and page numbers.

A line in the top or bottom band of the page whose (zone, normalized-text)
signature recurs on many pages is treated as a running header/footer. Digit runs
are normalized so "Page 1" / "Page 2" collapse to one signature, and bare page
numbers are always dropped.
"""

from __future__ import annotations

import re
from collections import Counter

from .ir import Line

_DIGITS = re.compile(r"\d+")
_BARE_NUMBER = re.compile(r"^\W*\d{1,4}\W*$")
_TOP_FRAC = 0.10
_BOTTOM_FRAC = 0.90


def _zone(line: Line, height: float):
    if line.bottom < _TOP_FRAC * height:
        return "top"
    if line.top > _BOTTOM_FRAC * height:
        return "bottom"
    return None


def _norm(text: str) -> str:
    return _DIGITS.sub("#", text.strip().lower())


def collect_signatures(pages: list[tuple]) -> Counter:
    """pages: list of (height, [Line, ...]). Returns a signature -> count counter."""
    sigs: Counter = Counter()
    for height, lines in pages:
        for ln in lines:
            z = _zone(ln, height)
            if z and ln.text.strip():
                sigs[(z, _norm(ln.text))] += 1
    return sigs


def repeating_set(sigs: Counter, npages: int) -> set:
    threshold = max(3, int(0.5 * npages))
    if npages < 3:
        return set()
    return {sig for sig, count in sigs.items() if count >= threshold}


def is_header_footer(line: Line, height: float, repeating: set) -> bool:
    z = _zone(line, height)
    if not z:
        return False
    if (z, _norm(line.text)) in repeating:
        return True
    return bool(_BARE_NUMBER.match(line.text.strip()))
