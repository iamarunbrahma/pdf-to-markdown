"""Stage 8: final whitespace cleanup, fence-aware.

Collapses excess blank lines and trims trailing spaces without touching the
interior of fenced code blocks.
"""

from __future__ import annotations

import re

# CJK / full-width characters (written without inter-character spaces).
_CJK = r"[⺀-鿿぀-ヿ가-힣　-〿＀-｠]"


def clean(markdown: str) -> str:
    lines = markdown.split("\n")
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line.rstrip())
            continue
        out.append(line if in_fence else line.rstrip())

    text = "\n".join(out)
    text = re.sub(r"\(cid:\d+\)", "", text)  # drop undecodable CID-font glyphs
    # Collapse full-justification spaces between CJK characters.
    text = re.sub(rf"({_CJK}) +(?={_CJK})", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip() + "\n"
