"""Stage 7: render the typed IR to Markdown.

Inline styling, math-run combining, nested-list numbering, de-hyphenation, and
block spacing all live here. This is the only module that emits Markdown syntax.
"""

from __future__ import annotations

import re

from .ir import Block, BlockType, Document, Line, Span
from .math_detect import to_latex_token
from .patterns import strip_marker
from .tables import table_to_markdown

# Escape literal '$' too, so dollar amounts in financial text are not parsed as
# math delimiters by KaTeX/MathJax-enabled Markdown renderers.
_ESCAPE = re.compile(r"([\\`*<$])")
_MARKER_RE = re.compile(
    r"^\s*(?:[•◦▪▫●○‣⁃∙]|[-*]|\(?(?:\d{1,3}|[ivxlcdm]{1,6}|[A-Za-z])[.)])\s+"
)


def _escape_inline(text: str) -> str:
    return _ESCAPE.sub(r"\\\1", text)


def _wrap(text: str, left: str, right: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    return f"{lead}{left}{stripped}{right}{trail}"


def _render_span(s: Span, suppress_emphasis: bool = False) -> str:
    if s.mono and s.text.strip():
        body = s.text.strip().replace("`", "\\`")
        core = f"`{body}`"
    else:
        core = _escape_inline(s.text)
        if suppress_emphasis:
            pass
        elif s.bold and s.italic:
            core = _wrap(core, "***", "***")
        elif s.bold:
            core = _wrap(core, "**", "**")
        elif s.italic:
            core = _wrap(core, "*", "*")
    if s.superscript:
        core = _wrap(core, "<sup>", "</sup>")
    if s.subscript:
        core = _wrap(core, "<sub>", "</sub>")
    if s.link:
        core = f"[{core.strip()}]({s.link})"
    return core


def _render_math_run(spans: list[Span]) -> str:
    latex = ""
    plain = ""
    has_script = False
    for s in spans:
        raw = s.text.strip()
        if not raw:
            continue
        tok = to_latex_token(raw)
        if s.superscript:
            latex += "^{" + tok + "}"
            has_script = True
        elif s.subscript:
            latex += "_{" + tok + "}"
            has_script = True
        else:
            latex += tok
        plain += raw
    latex = latex.strip()
    if not latex:
        return ""
    # A "math" run that is only digits/punctuation (e.g. a decimal point set in a
    # math font, "8.30") is not real math - emit it literally, not as $...$.
    has_real_math = has_script or bool(re.search(r"[A-Za-z\\]", latex))
    if not has_real_math:
        return _escape_inline(plain.strip())
    return f"${latex}$"


def render_inline(line: Line, suppress_emphasis: bool = False) -> str:
    parts, i, spans = [], 0, line.spans
    while i < len(spans):
        if spans[i].math:
            run = []
            while i < len(spans) and spans[i].math:
                run.append(spans[i])
                i += 1
            parts.append(_render_math_run(run))
        else:
            parts.append(_render_span(spans[i], suppress_emphasis))
            i += 1
    return "".join(parts).strip()


def _join_text_lines(lines: list[Line]) -> str:
    """Join wrapped lines into a paragraph, de-hyphenating soft breaks."""
    out = ""
    for idx, ln in enumerate(lines):
        piece = render_inline(ln)
        if idx == 0:
            out = piece
        elif re.search(r"[A-Za-z]-$", out) and re.match(r"[a-z]", piece):
            out = out[:-1] + piece  # soft hyphen across a line break
        else:
            out = out.rstrip() + " " + piece
    return out


def _guess_language(text: str) -> str:
    if re.search(r"^\s*(def |class |import |from \w+ import|print\()", text, re.M):
        return "python"
    if re.search(r"\b(function|const|let|var)\b|=>|console\.", text):
        return "javascript"
    if re.search(r"^\s*(#include|using namespace|int main)", text, re.M):
        return "cpp"
    if re.search(
        r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)\b", text, re.I | re.M
    ):
        return "sql"
    return ""


class _Ctx:
    def __init__(self):
        self.counters: dict[int, int] = {}

    def list_number(self, level: int) -> int:
        for lvl in list(self.counters):
            if lvl > level:
                del self.counters[lvl]
        self.counters[level] = self.counters.get(level, 0) + 1
        return self.counters[level]

    def reset_lists(self):
        self.counters.clear()


def _render_block(block: Block, ctx: _Ctx) -> str:
    t = block.type
    if t is BlockType.HEADING:
        text = render_inline(block.lines[0], suppress_emphasis=True)
        return f"{'#' * max(1, min(6, block.level))} {text}"
    if t is BlockType.HR:
        return "---"
    if t is BlockType.TABLE:
        return table_to_markdown(block.meta.get("rows", []))
    if t is BlockType.IMAGE:
        alt = block.meta.get("alt", "image")
        path = block.meta.get("path", "")
        note = block.meta.get("note")
        ref = f"![{alt}]({path})"
        return f"{ref}\n\n*{note}*" if note else ref
    if t is BlockType.MATH_DISPLAY:
        spans = [s for ln in block.lines for s in ln.spans]
        for s in spans:
            s.math = True
        latex = _render_math_run(spans)
        inner = latex[1:-1] if latex.startswith("$") else latex
        return f"$$\n{inner}\n$$"
    if t is BlockType.FOOTNOTE:
        return f"[^{block.meta.get('marker', '1')}]: {_join_text_lines(block.lines)}"
    if t is BlockType.TOC:
        items = []
        for ln in block.lines:
            txt = re.sub(r"\.{3,}\s*\d+\s*$", "", ln.text).strip()
            if txt:
                items.append(f"- {txt}")
        return "\n".join(items)
    if t is BlockType.CODE:
        body = "\n".join(ln.text for ln in block.lines)
        return f"```{_guess_language(body)}\n{body}\n```"
    if t is BlockType.BLOCKQUOTE:
        text = _join_text_lines(block.lines)
        return "\n".join(f"> {line}" for line in text.split("\n"))
    if t is BlockType.LIST_ITEM:
        lines_md = [render_inline(ln) for ln in block.lines]
        lines_md[0] = _MARKER_RE.sub("", lines_md[0], count=1) or strip_marker(
            block.lines[0].text
        )
        content = " ".join(p for p in lines_md if p)
        indent = "  " * max(0, block.level)
        if block.meta.get("ordered"):
            marker = f"{ctx.list_number(block.level)}."
        else:
            marker = "-"
        return f"{indent}{marker} {content}"
    # paragraph
    return _join_text_lines(block.lines)


def serialize(document: Document, page_delimiter: str = "") -> tuple[str, list[str]]:
    """Return (full_markdown, [per_page_markdown, ...])."""
    page_strings = []
    for page in document.pages:
        ctx = _Ctx()
        chunks: list[str] = []
        prev_type = None
        prev_ordered = None
        prev_level = None
        for block in page.blocks:
            if block.type is not BlockType.LIST_ITEM:
                ctx.reset_lists()
            rendered = _render_block(block, ctx)
            if rendered.strip() == "":
                continue
            # Consecutive list items stay in one tight list, except a same-level
            # switch between ordered/unordered starts a distinct list.
            same_list = (
                block.type is BlockType.LIST_ITEM
                and prev_type is BlockType.LIST_ITEM
                and not (
                    block.level == prev_level
                    and block.meta.get("ordered") != prev_ordered
                )
            )
            if chunks and same_list:
                chunks.append("\n" + rendered)
            else:
                chunks.append("\n\n" + rendered if chunks else rendered)
            prev_type = block.type
            if block.type is BlockType.LIST_ITEM:
                prev_ordered = block.meta.get("ordered")
                prev_level = block.level
            else:
                prev_ordered = None
                prev_level = None
        page_strings.append("".join(chunks).strip())

    delim = f"\n\n{page_delimiter}\n\n" if page_delimiter else "\n\n"
    full = delim.join(p for p in page_strings)
    return full, page_strings
