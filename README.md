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
# → writes out/paper.md; extracted images land in out/ and are linked relative to the .md
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

## Optional: progress bar

Every step of the pipeline reports itself, so a long conversion is never a black
box. Steps that walk the pages get a bar; one-off steps print a `[pdf2md]` line:

```text
[pdf2md] features on: images, borderless tables, blockquotes, footnotes, bold headings
extracting images: 100%|##########| 1477/1477 [01:33<00:00, 15.76page/s]
reading pages:     100%|##########| 1477/1477 [01:39<00:00, 14.81page/s]
analyzing layout:  100%|##########| 1477/1477 [01:17<00:00, 18.98page/s]
[pdf2md] measuring document
[pdf2md] detecting headers/footers
rendering pages:   100%|##########| 1477/1477 [00:01<00:00, 1115.97page/s]
[pdf2md] writing markdown
```

It is opt-in because tqdm is dual-licensed MPL-2.0/MIT, and the core stays
MIT-only; without it the CLI just runs quietly.

```bash
uv sync --extra progress
uv run pdf2md --pdf_path paper.pdf                 # bar when stderr is a terminal
uv run pdf2md --pdf_path paper.pdf --progress      # force on (piped output, CI)
uv run pdf2md --pdf_path paper.pdf --no-progress   # completely silent
```

Auto-detection can also be overridden with `PDF2MD_PROGRESS=1` /
`PDF2MD_NO_PROGRESS=1`.

As a library, pass any `progress(stage, done, total)` callable - `total > 0` is
per-page progress, `total == 0` is a one-off step. The pipeline never imports
tqdm itself, so nothing is pulled in unless you ask for it:

```python
from pdf2md import PdfToMarkdown

PdfToMarkdown(progress=lambda stage, done, total: print(stage, done, total))
```

## Limitations

- **No OCR** - scanned / image-only pages yield embedded images only.
- **Math is heuristic** - complex equations are best-effort (use `--extra math`).
- **Dense multi-table pages** and **right-to-left** scripts may need manual review.

## License

MIT - see [LICENSE](LICENSE).
