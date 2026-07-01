"""pdf2md — a thorough, MIT-only PDF -> Markdown converter."""

from .converter import PdfToMarkdown, convert_pdf
from .ir import Block, BlockType, Document, Line, Page, Span

__version__ = "1.0.0"
__all__ = [
    "PdfToMarkdown",
    "convert_pdf",
    "Document",
    "Page",
    "Block",
    "Line",
    "Span",
    "BlockType",
]
