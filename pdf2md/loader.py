"""Stage 1: turn a pdfplumber page into normalized, framework-agnostic primitives.

Nothing downstream of this module imports pdfplumber, so the rest of the pipeline
stays a pure function of plain dicts/lists and is trivial to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawPage:
    number: int  # 0-based
    width: float
    height: float
    chars: list[dict] = field(default_factory=list)
    rulings: list[dict] = field(
        default_factory=list
    )  # horizontal/vertical lines + rect edges
    images: list[dict] = field(default_factory=list)
    hyperlinks: list[dict] = field(default_factory=list)
    rotated: bool = False


_CHAR_KEYS = (
    "text",
    "fontname",
    "size",
    "x0",
    "x1",
    "top",
    "bottom",
    "upright",
    "height",
    "width",
    "adv",
)


def _norm_char(c: dict) -> dict:
    out = {k: c.get(k) for k in _CHAR_KEYS}
    out["size"] = float(out.get("size") or 0.0)
    out["fontname"] = out.get("fontname") or ""
    out["text"] = out.get("text") or ""
    for k in ("x0", "x1", "top", "bottom", "height", "width", "adv"):
        out[k] = float(out.get(k) or 0.0)
    return out


def _rulings(page) -> list[dict]:
    """Collect thin horizontal/vertical segments from lines and rect edges."""
    out = []
    for ln in page.lines:
        out.append(
            {
                "x0": float(ln["x0"]),
                "x1": float(ln["x1"]),
                "top": float(ln["top"]),
                "bottom": float(ln["bottom"]),
                "orientation": "h" if abs(ln["bottom"] - ln["top"]) <= 1.5 else "v",
            }
        )
    for r in page.rects:
        w = float(r["x1"]) - float(r["x0"])
        h = float(r["bottom"]) - float(r["top"])
        if h <= 2.0 and w > 2.0:  # thin rect == a horizontal rule
            out.append(
                {
                    "x0": float(r["x0"]),
                    "x1": float(r["x1"]),
                    "top": float(r["top"]),
                    "bottom": float(r["bottom"]),
                    "orientation": "h",
                }
            )
    return out


def _hyperlinks(page) -> list[dict]:
    links = []
    for h in getattr(page, "hyperlinks", []) or []:
        uri = h.get("uri")
        if uri:
            links.append(
                {
                    "uri": uri,
                    "x0": float(h["x0"]),
                    "x1": float(h["x1"]),
                    "top": float(h["top"]),
                    "bottom": float(h["bottom"]),
                }
            )
    # Some PDFs expose URIs only through annotations.
    for a in getattr(page, "annots", []) or []:
        uri = (
            (a.get("uri") or (a.get("data") or {}).get("uri"))
            if isinstance(a, dict)
            else None
        )
        if uri:
            links.append(
                {
                    "uri": uri,
                    "x0": float(a["x0"]),
                    "x1": float(a["x1"]),
                    "top": float(a["top"]),
                    "bottom": float(a["bottom"]),
                }
            )
    return links


def _rotate_bbox(x0, top, x1, bottom, rotation, w, h):
    """Map a bbox from pdfplumber's page space into upright reading space.

    pdfplumber reports a /Rotate-d page's glyphs in the (swapped) display
    dimensions but laid out sideways; this rotates coordinates so downstream
    left-to-right / top-to-bottom order is correct.
    """
    if rotation == 90:
        nx0, nx1, ntop, nbottom = top, bottom, w - x1, w - x0
    elif rotation == 270:
        nx0, nx1, ntop, nbottom = h - bottom, h - top, x0, x1
    elif rotation == 180:
        nx0, nx1, ntop, nbottom = w - x1, w - x0, h - bottom, h - top
    else:
        nx0, nx1, ntop, nbottom = x0, top, x1, bottom  # unreachable
    return min(nx0, nx1), min(ntop, nbottom), max(nx0, nx1), max(ntop, nbottom)


def _rotate_char(c, rotation, w, h):
    c["x0"], c["top"], c["x1"], c["bottom"] = _rotate_bbox(
        c["x0"], c["top"], c["x1"], c["bottom"], rotation, w, h
    )
    c["width"], c["height"] = c["x1"] - c["x0"], c["bottom"] - c["top"]
    # pdfplumber's advance is unreliable for rotated glyphs (reports adv==size for
    # every char); zero it so it can't poison monospace-font detection.
    c["adv"] = 0.0
    return c


def _rotate_box(b, rotation, w, h):
    b["x0"], b["top"], b["x1"], b["bottom"] = _rotate_bbox(
        b["x0"], b["top"], b["x1"], b["bottom"], rotation, w, h
    )
    return b


def _effective_rotation(page) -> int:
    """Page rotation from /Rotate, or derived from char matrices when a page is
    drawn rotated without a /Rotate flag."""
    rot = int(getattr(page, "rotation", 0) or 0) % 360
    if rot in (90, 180, 270):
        return rot
    chars = [c for c in page.chars if (c.get("text") or "").strip()]
    if not chars:
        return 0
    nonup = [c for c in chars if not c.get("upright", True) and c.get("matrix")]
    if len(nonup) < 0.6 * len(chars):  # a minority => watermark, not a rotated page
        return 0
    import statistics

    if statistics.mean(c["matrix"][0] for c in nonup) < -0.5:
        return 180
    return 270 if statistics.mean(c["matrix"][1] for c in nonup) > 0 else 90


def load_page(page) -> RawPage:
    """Build a RawPage from a live pdfplumber Page.

    On an upright page, non-upright characters are dropped: they are almost always
    sidebar watermarks (e.g. the arXiv stamp). On a rotated page (via /Rotate or
    drawn sideways), the whole content is "non-upright", so instead we rotate
    every coordinate into upright reading space and keep the text.
    """
    rotation = _effective_rotation(page)
    w, h = float(page.width), float(page.height)

    if rotation in (90, 180, 270):
        chars = [
            _rotate_char(_norm_char(c), rotation, w, h)
            for c in page.chars
            if (c.get("text") or "") != ""
        ]
        images = [
            _rotate_box(
                {
                    "x0": float(im["x0"]),
                    "x1": float(im["x1"]),
                    "top": float(im["top"]),
                    "bottom": float(im["bottom"]),
                    "name": im.get("name"),
                },
                rotation,
                w,
                h,
            )
            for im in page.images
        ]
        links = [_rotate_box(link, rotation, w, h) for link in _hyperlinks(page)]
        out_w, out_h = (h, w) if rotation in (90, 270) else (w, h)
        return RawPage(
            number=page.page_number - 1,
            width=out_w,
            height=out_h,
            chars=chars,
            rulings=[],
            images=images,
            hyperlinks=links,
            rotated=True,
        )

    chars = [
        _norm_char(c)
        for c in page.chars
        if (c.get("text") or "") != "" and c.get("upright", True)
    ]
    return RawPage(
        number=page.page_number - 1,
        width=w,
        height=h,
        chars=chars,
        rulings=_rulings(page),
        images=[
            {
                "x0": float(im["x0"]),
                "x1": float(im["x1"]),
                "top": float(im["top"]),
                "bottom": float(im["bottom"]),
                "name": im.get("name"),
            }
            for im in page.images
        ],
        hyperlinks=_hyperlinks(page),
    )
