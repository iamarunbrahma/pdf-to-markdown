"""Stage 5: classify an ordered run of lines into typed blocks.

Operates on one reading-order *segment* (a column/band chunk) at a time so
paragraph merging never crosses a column boundary.
"""

from __future__ import annotations

from .analyze import DocStats, indent_level
from .ir import Block, BlockType, Line
from .patterns import is_hr, is_toc_line, list_marker


def _mk(btype, lines, **kw) -> Block:
    return Block(
        type=btype,
        lines=list(lines),
        x0=min(ln.x0 for ln in lines),
        x1=max(ln.x1 for ln in lines),
        top=min(ln.top for ln in lines),
        bottom=max(ln.bottom for ln in lines),
        **kw,
    )


def _all_bold(line: Line) -> bool:
    spans = [s for s in line.spans if s.text.strip()]
    return bool(spans) and all(s.bold for s in spans)


def heading_level(line: Line, stats: DocStats) -> int:
    # A monospace line is code, never a heading - even if its font happens to be
    # larger than the body text (common: code set slightly larger than prose).
    if is_code_line(line):
        return 0
    size = round(line.size * 2) / 2
    if size in stats.heading_levels:
        return stats.heading_levels[size]
    text = line.text.strip()
    if (
        stats.bold_heading_level
        and _all_bold(line)
        and 0 < len(text) <= 70
        and not text.endswith((".", ":", ";", ","))
        and not list_marker(text)
    ):
        return stats.bold_heading_level
    return 0


def is_code_line(line: Line) -> bool:
    # Require the whole line to be monospace: a line that is only *partly* mono is
    # prose with inline code, not a code block.
    total = sum(len(s.text.strip()) for s in line.spans) or 1
    mono = sum(len(s.text.strip()) for s in line.spans if s.mono)
    return mono / total > 0.9


def is_display_math(line: Line) -> bool:
    total = sum(len(s.text.strip()) for s in line.spans) or 1
    math = sum(len(s.text.strip()) for s in line.spans if s.math)
    return total < 90 and math / total > 0.55


def _para_break(prev: Line, nxt: Line) -> bool:
    gap = nxt.top - prev.bottom
    return gap > 0.9 * (nxt.size or 10.0)


def _is_blockquote(lines: list[Line], stats: DocStats) -> bool:
    if not stats.config.get("detect_blockquote", True):
        return False
    # Left indentation of the whole block is the reliable signal (right-edge
    # inset is unstable when the block is among the widest lines on the page).
    indent = min(ln.x0 for ln in lines) - stats.body_left
    text_len = sum(len(ln.text.strip()) for ln in lines)
    return indent >= 18 and text_len >= 20


def _consume_list(lines, i, stats, out):
    n = len(lines)
    while i < n:
        ln = lines[i]
        marker = list_marker(ln.text.strip())
        if not marker:
            break
        kind, mk = marker
        level = indent_level(ln.x0, stats.indent_levels)
        item = _mk(
            BlockType.LIST_ITEM,
            [ln],
            level=level,
            meta={"ordered": kind == "ol", "marker": mk},
        )
        i += 1
        while i < n:
            nl = lines[i]
            t = nl.text.strip()
            if list_marker(t) or heading_level(nl, stats) or is_hr(t):
                break
            if nl.top - item.lines[-1].bottom > 0.9 * (nl.size or 10.0):
                break
            if nl.x0 < ln.x0 - 2:  # dedented past the marker => not a continuation
                break
            item.lines.append(nl)
            item.bottom = nl.bottom
            i += 1
        out.append(item)
    return i


def classify_segment(lines: list[Line], stats: DocStats) -> list[Block]:
    out: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        text = ln.text.strip()

        if is_hr(text):
            out.append(_mk(BlockType.HR, [ln]))
            i += 1
            continue

        if is_toc_line(text):
            run = [ln]
            j = i + 1
            while j < n and is_toc_line(lines[j].text.strip()):
                run.append(lines[j])
                j += 1
            out.append(_mk(BlockType.TOC, run))
            i = j
            continue

        if heading_level(ln, stats):
            out.append(_mk(BlockType.HEADING, [ln], level=heading_level(ln, stats)))
            i += 1
            continue

        if is_display_math(ln):
            out.append(_mk(BlockType.MATH_DISPLAY, [ln]))
            i += 1
            continue

        if is_code_line(ln):
            run = [ln]
            j = i + 1
            while j < n and is_code_line(lines[j]):
                run.append(lines[j])
                j += 1
            out.append(_mk(BlockType.CODE, run))
            i = j
            continue

        if list_marker(text):
            i = _consume_list(lines, i, stats, out)
            continue

        # Paragraph: accumulate continuation lines.
        para = [ln]
        j = i + 1
        while j < n:
            nxt = lines[j]
            t = nxt.text.strip()
            if (
                is_hr(t)
                or is_toc_line(t)
                or heading_level(nxt, stats)
                or list_marker(t)
                or is_code_line(nxt)
                or is_display_math(nxt)
                or _para_break(para[-1], nxt)
            ):
                break
            para.append(nxt)
            j += 1
        btype = (
            BlockType.BLOCKQUOTE if _is_blockquote(para, stats) else BlockType.PARAGRAPH
        )
        out.append(_mk(btype, para))
        i = j
    return out
