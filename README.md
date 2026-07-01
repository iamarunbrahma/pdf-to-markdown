# PDF to Markdown

A thorough, **MIT-only** PDF → Markdown converter built on
[`pdfplumber`](https://github.com/jsvine/pdfplumber) / `pdfminer.six`. It extracts
document *structure* - headings, lists, tables, code, math, links, images - not
just plain text. No AGPL (`PyMuPDF`), no ML/OCR stack, and no copyleft anywhere in
the dependency tree.

## Install

Managed with [uv](https://docs.astral.sh/uv/) via `pyproject.toml`:

```bash
uv sync
```

## Usage

```bash
uv run pdf2md --pdf_path file.pdf [--out DIR] [--config config/config.yaml]
```

```python
from pdf2md import convert_pdf

markdown = convert_pdf("file.pdf")
```

Output is written to `<OUTPUT_DIR>/<name>.md` (default `outputs/`) with any
extracted images alongside. Behaviour is tunable in `config/config.yaml`.

## Features

Headings, paragraphs (de-hyphenated), bold/italic/inline-code, super/subscript,
ordered/unordered/nested lists, links, blockquotes, code blocks, GFM tables
(ruled + borderless), images, multi-column reading order, rotated pages,
math → LaTeX (`$…$` / `$$…$$`), footnotes, TOC, and header/footer stripping.

## Optional: high-accuracy math

Core math is heuristic. For accurate equations, install the opt-in ML extra
(`pix2tex`, which pulls in BSD `torch`, kept out of the core):

```bash
uv sync --extra math
```

## Limitations

No OCR (scanned pages yield images only); heuristic math is best-effort; dense
multi-table and right-to-left layouts may need review.

## Testing

```bash
uv run pytest -q
```

## License

MIT - see [LICENSE](LICENSE).
