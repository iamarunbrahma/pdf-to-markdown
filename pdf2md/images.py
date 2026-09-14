"""Image extraction via pdfminer's MIT-licensed ImageWriter.

Runs one lightweight pdfminer pass to export every embedded raster image and
records each image's page and position (converted to top-based coordinates) so
the serializer can place the reference in the right spot in reading order.
"""

from __future__ import annotations

import os
from collections import defaultdict

from pdfminer.high_level import extract_pages
from pdfminer.image import ImageWriter
from pdfminer.layout import LTFigure, LTImage

from .progress import STAGE_IMAGES, ProgressCallback


def _iter_images(container):
    for elem in container:
        if isinstance(elem, LTImage):
            yield elem
        elif isinstance(elem, LTFigure):
            yield from _iter_images(elem)


def extract_images(
    pdf_path: str,
    out_dir: str,
    progress: ProgressCallback | None = None,
    total: int = 0,
) -> dict:
    """Return {page_index: [{'bbox': (x0, top, x1, bottom), 'path': str}, ...]}.

    `path` is the image filename relative to `out_dir`, which is also where the
    Markdown file is written, so the emitted link resolves from the .md file.

    `progress`/`total` are the caller's callback and its page count: this pass
    runs over the whole document on its own, before the pipeline starts, so
    without them it is the one stretch of a long run a user cannot see.
    """
    results: dict = defaultdict(list)
    os.makedirs(out_dir, exist_ok=True)
    writer = ImageWriter(out_dir)
    try:
        pages = enumerate(extract_pages(pdf_path, laparams=None))
    except Exception:
        return results

    for page_index, layout in pages:
        page_height = float(layout.height)
        for img in _iter_images(layout):
            x0, y0, x1, y1 = img.bbox
            if (x1 - x0) < 10 or (y1 - y0) < 10:
                continue
            try:
                name = writer.export_image(img)
            except Exception:
                continue
            results[page_index].append(
                {
                    "bbox": (
                        float(x0),
                        page_height - float(y1),
                        float(x1),
                        page_height - float(y0),
                    ),
                    "path": name,
                }
            )
        if progress is not None and total > 0:
            progress(STAGE_IMAGES, page_index + 1, total)
    return results
