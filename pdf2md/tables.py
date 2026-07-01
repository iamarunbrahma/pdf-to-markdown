"""Table detection (ruled + borderless) and GFM serialization.

Uses pdfplumber's `lines` strategy first; for regions with aligned text but no
ruling lines it falls back to the `text` strategy, filtered conservatively so
ordinary prose is not mistaken for a table.
"""

from __future__ import annotations


def _boxes_overlap(a, b) -> bool:
    ax0, atop, ax1, abot = a
    bx0, btop, bx1, bbot = b
    return not (ax1 <= bx0 or bx1 <= ax0 or abot <= btop or bbot <= atop)


def _drop_empty_columns(rows):
    """Remove columns that are empty in every row.

    pdfplumber's text strategy often inserts spurious separators between real
    columns, producing all-blank columns; dropping them cleans up the table.
    """
    if not rows:
        return rows
    ncols = max(len(r) for r in rows)
    padded = [r + [""] * (ncols - len(r)) for r in rows]
    keep = [c for c in range(ncols) if any(padded[r][c] for r in range(len(padded)))]
    if not keep:
        return rows
    return [[padded[r][c] for c in keep] for r in range(len(padded))]


def _clean_rows(rows):
    cleaned = []
    for row in rows:
        cleaned.append([(c or "").strip().replace("\n", " ") for c in row])
    # Drop fully empty rows, then fully empty columns.
    cleaned = [r for r in cleaned if any(cell for cell in r)]
    return _drop_empty_columns(cleaned)


def _is_real_grid(rows) -> bool:
    """Reject diagram/figure regions that pdfplumber reports as huge sparse grids."""
    if len(rows) < 2:
        return False
    ncols = max(len(r) for r in rows)
    if not 2 <= ncols <= 20:
        return False
    cells = [c for r in rows for c in r]
    nonempty = sum(1 for c in cells if c)
    return bool(cells) and nonempty / len(cells) >= 0.3


def _looks_tabular(rows, bbox, page_w, page_h) -> bool:
    """Conservative borderless-table test: rejects multi-column prose grids.

    pdfplumber's text strategy will split repetitive column text into one
    "column" per word, turning two-column body text into a fake wide table. We
    reject candidates that look like prose: single-word-heavy wide grids, or
    grids that span most of the page.
    """
    if len(rows) < 2:
        return False
    ncols = max(len(r) for r in rows)
    if not 2 <= ncols <= 8:
        return False

    cells = [c for r in rows for c in r]
    nonempty = [c for c in cells if c]
    if not nonempty or len(nonempty) / len(cells) < 0.6:
        return False

    # A long single-token cell means mis-extracted prose (e.g. an abstract), not
    # a real table cell.
    if any(len(c) > 40 and " " not in c for c in nonempty):
        return False

    single_word = sum(1 for c in nonempty if " " not in c) / len(nonempty)
    if single_word > 0.75 and ncols >= 4:
        return False

    x0, top, x1, bottom = bbox
    # A borderless "table" that fills most of the page height is almost always an
    # entire single-column document mis-read as a grid, not a real inline table.
    if (bottom - top) > 0.5 * page_h:
        return False

    # Require at least two genuinely populated columns (non-empty in >=50% of
    # rows); prose split into fake columns has one dominant column plus artifacts.
    nrows = len(rows)
    padded = [r + [""] * (ncols - len(r)) for r in rows]
    filled_cols = sum(
        1
        for c in range(ncols)
        if sum(1 for r in range(nrows) if padded[r][c]) >= 0.5 * nrows
    )
    return filled_cols >= 2


def detect_tables(page, config: dict) -> list[dict]:
    """Return [{'bbox': (x0, top, x1, bottom), 'rows': [[str, ...], ...]}]."""
    found = []
    try:
        ruled = page.find_tables()
    except Exception:
        ruled = []
    for t in ruled:
        rows = _clean_rows(t.extract())
        if _is_real_grid(rows):
            found.append({"bbox": tuple(t.bbox), "rows": rows})

    if config.get("detect_borderless_tables", True):
        try:
            textual = page.find_tables(
                {"vertical_strategy": "text", "horizontal_strategy": "text"}
            )
        except Exception:
            textual = []
        for t in textual:
            bbox = tuple(t.bbox)
            if any(_boxes_overlap(bbox, f["bbox"]) for f in found):
                continue
            rows = _clean_rows(t.extract())
            if _looks_tabular(rows, bbox, float(page.width), float(page.height)):
                found.append({"bbox": bbox, "rows": rows})

    found.sort(key=lambda f: f["bbox"][1])
    return found


def _escape_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    norm = [[_escape_cell(c) for c in r] + [""] * (ncols - len(r)) for r in rows]

    widths = [max(len(norm[r][c]) for r in range(len(norm))) for c in range(ncols)]
    widths = [max(3, w) for w in widths]

    def fmt(row):
        return "| " + " | ".join(row[c].ljust(widths[c]) for c in range(ncols)) + " |"

    out = [
        fmt(norm[0]),
        "| " + " | ".join("-" * widths[c] for c in range(ncols)) + " |",
    ]
    out += [fmt(r) for r in norm[1:]]
    return "\n".join(out)
