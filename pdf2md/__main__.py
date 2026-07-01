# Absolute import so this works whether run as `python -m pdf2md`, as the
# installed console script, or via `uv run pdf2md`.
from pdf2md.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
