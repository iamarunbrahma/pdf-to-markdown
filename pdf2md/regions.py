"""Stage 4: reading order and region filtering.

Determines the order in which lines should be read on a page using a
band-and-column model (a practical specialization of the recursive XY-cut):
full-width elements split the page into horizontal bands; within each band, text
is grouped into columns ordered left-to-right. This yields correct order for
single-column and multi-column pages with full-width titles/section headers.
"""

from __future__ import annotations

from .ir import Line


def line_in_box(line: Line, box: tuple) -> bool:
    """True if the line's center sits inside (x0, top, x1, bottom)."""
    x0, top, x1, bottom = box
    cx = (line.x0 + line.x1) / 2
    cy = (line.top + line.bottom) / 2
    return x0 - 1 <= cx <= x1 + 1 and top - 1 <= cy <= bottom + 1


def drop_lines_in_boxes(lines: list[Line], boxes: list[tuple]) -> list[Line]:
    if not boxes:
        return lines
    return [ln for ln in lines if not any(line_in_box(ln, b) for b in boxes)]


def _detect_columns(lines, body_left, body_right, min_gutter):
    """Find column x-ranges using only strict column-width lines.

    Wider lines (titles, full-width headers) are excluded so they cannot pollute
    the column boundaries.
    """
    body_w = body_right - body_left
    col_lines = [ln for ln in lines if (ln.x1 - ln.x0) < 0.5 * body_w]
    if len(col_lines) < 6:
        return [(body_left, body_right)]

    intervals = sorted((ln.x0, ln.x1) for ln in col_lines)
    merged = [list(intervals[0])]
    for x0, x1 in intervals[1:]:
        if x0 <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    cols = [list(merged[0])]
    for x0, x1 in merged[1:]:
        if x0 - cols[-1][1] >= min_gutter:
            cols.append([x0, x1])
        else:
            cols[-1][1] = max(cols[-1][1], x1)

    if len(cols) < 2:
        return [(body_left, body_right)]

    # Require every column to carry real content; otherwise treat as single.
    counts = [0] * len(cols)
    for ln in col_lines:
        counts[_assign_column(ln, cols)] += 1
    if any(c < 3 for c in counts):
        return [(body_left, body_right)]
    return [tuple(c) for c in cols]


def _assign_column(line, cols):
    best, best_ov = 0, -1.0
    for i, (a, b) in enumerate(cols):
        ov = min(line.x1, b) - max(line.x0, a)
        if ov > best_ov:
            best_ov, best = ov, i
    return best


def reading_order_segments(lines: list[Line], page_width: float):
    """Return a list of segments (each a list of Lines) in reading order."""
    if not lines:
        return []
    body_left = min(ln.x0 for ln in lines)
    body_right = max(ln.x1 for ln in lines)
    min_gutter = max(18.0, 0.03 * page_width)
    cols = _detect_columns(lines, body_left, body_right, min_gutter)

    if len(cols) <= 1:
        return [sorted(lines, key=lambda ln: (round(ln.top, 1), ln.x0))]

    body_w = body_right - body_left
    gutters = [(cols[i][1] + cols[i + 1][0]) / 2 for i in range(len(cols) - 1)]

    def is_spanning(ln: Line) -> bool:
        if (ln.x1 - ln.x0) >= 0.5 * body_w:
            return True
        return any(ln.x0 < g < ln.x1 for g in gutters)

    segments: list[list[Line]] = []
    buffers: list[list[Line]] = [[] for _ in cols]
    fw_buffer: list[Line] = []

    def flush_cols():
        for buf in buffers:
            if buf:
                segments.append(sorted(buf, key=lambda ln: ln.top))
                buf.clear()

    def flush_fw():
        if fw_buffer:
            segments.append(sorted(fw_buffer, key=lambda ln: ln.top))
            fw_buffer.clear()

    for ln in sorted(lines, key=lambda t: (round(t.top, 1), t.x0)):
        if is_spanning(ln):
            flush_cols()
            fw_buffer.append(ln)
        else:
            flush_fw()
            buffers[_assign_column(ln, cols)].append(ln)
    flush_fw()
    flush_cols()
    return segments
