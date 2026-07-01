"""Document-level statistics used to classify blocks consistently.

Headings, indentation levels, and body margins only make sense *relative to the
document*, so we measure them once across every line before classifying.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .ir import Line
from .patterns import list_marker


@dataclass
class DocStats:
    body_size: float = 10.0
    heading_levels: dict = field(default_factory=dict)  # rounded size -> level (1..6)
    bold_heading_level: int = 0
    indent_levels: list = field(
        default_factory=list
    )  # sorted marker-x0 cluster centers
    body_left: float = 0.0
    body_right: float = 612.0
    config: dict = field(default_factory=dict)


def _cluster(values: list[float], tol: float = 12.0) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _percentile(sorted_vals: list[float], frac: float, default: float) -> float:
    if not sorted_vals:
        return default
    idx = min(len(sorted_vals) - 1, max(0, int(len(sorted_vals) * frac)))
    return sorted_vals[idx]


def compute_stats(all_lines: list[Line], page_width: float, config: dict) -> DocStats:
    size_weight: Counter = Counter()
    for ln in all_lines:
        for s in ln.spans:
            if s.size:
                size_weight[round(s.size * 2) / 2] += max(1, len(s.text.strip()))

    body_size = size_weight.most_common(1)[0][0] if size_weight else 10.0

    # Heading tiers: sizes meaningfully larger than body, with real support.
    bigger = sorted(
        {sz for sz, w in size_weight.items() if sz > body_size + 0.4 and w >= 2},
        reverse=True,
    )[:6]
    heading_levels = {sz: i + 1 for i, sz in enumerate(bigger)}
    bold_heading_level = (
        min(len(bigger) + 1, 6) if config.get("bold_as_heading", True) else 0
    )

    marker_x = [ln.x0 for ln in all_lines if list_marker(ln.text.strip())]
    indent_levels = _cluster(marker_x, tol=12.0)

    xs0 = sorted(ln.x0 for ln in all_lines)
    xs1 = sorted(ln.x1 for ln in all_lines)
    body_left = _percentile(xs0, 0.05, 0.0)
    body_right = _percentile(xs1, 0.95, page_width)

    return DocStats(
        body_size=body_size,
        heading_levels=heading_levels,
        bold_heading_level=bold_heading_level,
        indent_levels=indent_levels,
        body_left=body_left,
        body_right=body_right,
        config=config,
    )


def indent_level(x0: float, indent_levels: list) -> int:
    """Map a marker's x0 to a 0-based nesting depth."""
    if not indent_levels:
        return 0
    nearest = min(range(len(indent_levels)), key=lambda i: abs(indent_levels[i] - x0))
    return nearest
