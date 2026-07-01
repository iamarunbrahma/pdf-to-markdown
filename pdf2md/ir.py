"""Typed intermediate representation shared across all pipeline stages.

The whole point of the IR is to decouple *extraction* (PDF -> structured blocks)
from *serialization* (structured blocks -> Markdown), so each direction can be
reasoned about and unit-tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Optional


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE = "code"
    BLOCKQUOTE = "blockquote"
    HR = "hr"
    TABLE = "table"
    IMAGE = "image"
    FOOTNOTE = "footnote"
    MATH_DISPLAY = "math_display"
    TOC = "toc"


@dataclass
class Span:
    """A run of characters sharing the same visual style."""

    text: str
    fontname: str = ""
    size: float = 0.0
    bold: bool = False
    italic: bool = False
    mono: bool = False
    superscript: bool = False
    subscript: bool = False
    math: bool = False
    link: Optional[str] = None
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0

    @property
    def style_key(self):
        return (
            self.bold,
            self.italic,
            self.mono,
            self.superscript,
            self.subscript,
            self.math,
            self.link,
        )


@dataclass
class Line:
    """One visual line of text, made of one or more styled spans."""

    spans: list[Span] = field(default_factory=list)
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def size(self) -> float:
        sizes = [s.size for s in self.spans if s.size]
        return median(sizes) if sizes else 0.0

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class Block:
    """A logical block of content (paragraph, heading, list item, ...)."""

    type: BlockType
    lines: list[Line] = field(default_factory=list)
    level: int = 0
    meta: dict = field(default_factory=dict)
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(ln.text for ln in self.lines)


@dataclass
class Page:
    number: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Document:
    pages: list[Page] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
