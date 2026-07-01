"""Stage 2: cluster raw characters into visual lines.

Characters are grouped by vertical position (with a tolerance derived from the
median glyph height so that super/subscripts stay attached to their line), then
each cluster is handed to `inline.build_line` to become a styled `Line`.
"""

from __future__ import annotations

from statistics import median

from .inline import build_line
from .ir import Line


def _split_on_gaps(chars: list[dict], page_width: float) -> list[list[dict]]:
    """Split a same-baseline char run at large horizontal gaps (column gutters).

    Without this, two-column text on the same baseline would be merged into one
    full-width line and the column structure would be lost before reading order.
    Normal word spacing (a few points) never trips the threshold; a column
    gutter (tens of points) does.
    """
    chars = sorted(chars, key=lambda c: c["x0"])
    groups = [[chars[0]]]
    for c in chars[1:]:
        prev = groups[-1][-1]
        gap = c["x0"] - prev["x1"]
        threshold = max(18.0, 0.045 * page_width, 3.0 * (c["size"] or 10.0))
        if gap > threshold:
            groups.append([c])
        else:
            groups[-1].append(c)
    return groups


def _median_size(chars: list[dict]) -> float:
    sizes = [c["size"] for c in chars if c["size"]]
    return median(sizes) if sizes else 10.0


def _deduplicate(chars: list[dict]) -> list[dict]:
    """Drop overlapping duplicate glyphs.

    Some PDFs fake bold by drawing each glyph twice at (nearly) the same spot,
    which yields doubled text like "TThhee". Collapse chars that share the same
    text at the same rounded position.
    """
    seen = set()
    out = []
    for c in chars:
        # Require full-box match (x0, x1, top) so genuinely adjacent identical
        # letters are never collapsed - only true overprint duplicates are.
        key = (round(c["x0"]), round(c["x1"]), round(c["top"]), c["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def group_into_lines(
    chars: list[dict],
    hyperlinks: list[dict],
    page_width: float = 612.0,
    mono_fonts: set | None = None,
) -> list[Line]:
    """Cluster chars into visual lines (gutter-aware) and style them.

    Clustering anchors on each line's *fixed* baseline (the bottom of its first
    char) rather than a growing band, which prevents tightly-leaded lines from
    chaining together into one giant block. The tolerance is wide enough to keep
    super/subscripts attached but far smaller than typical line pitch.
    """
    chars = [c for c in chars if c["text"] != ""]
    if not chars:
        return []
    chars = _deduplicate(chars)

    tol = max(2.0, 0.5 * _median_size(chars))
    chars = sorted(chars, key=lambda c: (c["top"], c["x0"]))

    clusters: list[dict] = []
    for c in chars:
        placed = None
        for cl in reversed(clusters):
            if abs(c["bottom"] - cl["base"]) <= tol:
                placed = cl
                break
            if cl["base"] < c["bottom"] - 3 * tol:  # clusters are roughly top-ordered
                break
        if placed is None:
            clusters.append({"base": c["bottom"], "chars": [c]})
        else:
            placed["chars"].append(c)

    lines: list[Line] = []
    for cl in clusters:
        for seg in _split_on_gaps(cl["chars"], page_width):
            line = build_line(seg, hyperlinks, mono_fonts)
            if line.text.strip():
                lines.append(line)
    lines.sort(key=lambda ln: (round(ln.top, 1), ln.x0))
    return lines
