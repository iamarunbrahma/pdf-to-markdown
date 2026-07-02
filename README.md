# PDF to Markdown

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org)
![Dependencies: MIT-only](https://img.shields.io/badge/dependencies-MIT--only-brightgreen.svg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Turn PDFs into clean, structured Markdown**

`pdf2md` extracts a PDF's *structure* - headings, lists, tables, code, math,
links, and images - not just its raw text. It is built on the MIT-licensed
[`pdfplumber`](https://github.com/jsvine/pdfplumber) / `pdfminer.six`, with no
AGPL (`PyMuPDF`) and no heavy ML/OCR stack anywhere in the dependency tree.
Install, import, and run it as **`pdf2md`**.

## Install

```bash
uv sync
```

## Usage

```bash
uv run pdf2md --pdf_path paper.pdf --out out/
# → writes out/paper.md
```

## Features

- **Headings** from document-wide font-size statistics
- **Text**: paragraphs (de-hyphenated), bold / italic / inline code, super/subscript, links
- **Lists** (ordered, unordered, nested), blockquotes, horizontal rules
- **Code blocks** via monospace-font detection
- **Tables** - ruled *and* borderless, as GFM pipe tables
- **Math → LaTeX** - inline `$…$` and display `$$…$$` (heuristic)
- **Layout** - multi-column reading order, rotated/landscape pages, header/footer stripping
- **Footnotes**, **table of contents**, **images**, and **CJK / Unicode** text

Behaviour is tunable in [`config/config.yaml`](config/config.yaml).

## Optional: high-accuracy math

Core math extraction is heuristic. For accurate equations, install the opt-in ML
extra (`pix2tex`, which pulls in BSD `torch` - kept out of the core so the
default install stays lean):

```bash
uv sync --extra math
```

## Limitations

- **No OCR** - scanned / image-only pages yield embedded images only.
- **Math is heuristic** - complex equations are best-effort (use `--extra math`).
- **Dense multi-table pages** and **right-to-left** scripts may need manual review.

## License

MIT - see [LICENSE](LICENSE).
