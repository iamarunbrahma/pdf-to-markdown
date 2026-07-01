"""Shared line-level regex/text patterns used by analysis and classification."""

from __future__ import annotations

import re

# Bullet glyphs commonly used as unordered-list markers.
BULLETS = "•◦▪▫●○‣⁃∙"

_ORDERED = re.compile(
    r"^\(?(?:\d{1,3}|[ivxlcdm]{1,6}|[A-Za-z])[.)]"  # 1.  1)  (a)  iv.  A)
)
_DOT_LEADER = re.compile(r".+?\.{3,}\s*\d{1,4}\s*$")  # "Chapter 1 ........ 12"
_HR = re.compile(r"^[-_*=]{3,}$")


def list_marker(text: str):
    """Return ('ul'|'ol', marker_str) if the line starts with a list marker."""
    t = text.lstrip()
    if not t:
        return None
    if t[0] in BULLETS:
        return ("ul", t[0])
    if t[0] in "-*" and len(t) > 1 and t[1] == " ":
        return ("ul", t[0])
    m = _ORDERED.match(t)
    if m and (len(t) == m.end() or t[m.end() : m.end() + 1] in (" ", "\t")):
        return ("ol", m.group(0))
    return None


def strip_marker(text: str) -> str:
    """Remove a leading list marker, returning the item content."""
    t = text.lstrip()
    marker = list_marker(t)
    if not marker:
        return text.strip()
    return t[len(marker[1]) :].lstrip()


def is_hr(text: str) -> bool:
    return bool(_HR.match(text.strip()))


def is_toc_line(text: str) -> bool:
    return bool(_DOT_LEADER.match(text.strip()))
