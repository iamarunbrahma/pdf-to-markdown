"""Heuristic math detection + Unicode -> LaTeX reconstruction (pure-MIT core).

This is best-effort by design: with no ML in core, we detect math from font
names (Computer Modern math families), the presence of math Unicode symbols, and
super/subscript geometry, then reconstruct LaTeX from the positioned glyphs. The
optional `plugins.math_ml` path is the high-accuracy alternative.
"""

from __future__ import annotations

from .fonts import is_math_font

# Common math Unicode -> LaTeX command. Applied ONLY inside detected math runs,
# so ordinary prose keeps its Unicode characters untouched.
UNICODE_TO_LATEX = {
    # Greek lowercase
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "ϕ": r"\varphi",
    # Greek uppercase
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    # Operators / relations
    "×": r"\times",
    "÷": r"\div",
    "±": r"\pm",
    "∓": r"\mp",
    "·": r"\cdot",
    "∗": r"\ast",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "≡": r"\equiv",
    "∝": r"\propto",
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "√": r"\sqrt{}",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∮": r"\oint",
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "⇔": r"\Leftrightarrow",
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊃": r"\supset",
    "⊇": r"\supseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    "∀": r"\forall",
    "∃": r"\exists",
    "¬": r"\neg",
    "∧": r"\land",
    "∨": r"\lor",
    "⊕": r"\oplus",
    "⊗": r"\otimes",
    "≪": r"\ll",
    "≫": r"\gg",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "°": r"^{\circ}",
    "′": r"'",
    "″": r"''",
    "…": r"\dots",
    "⋯": r"\cdots",
}

# Characters that strongly imply math even in an otherwise normal font.
_MATH_SYMBOLS = set(UNICODE_TO_LATEX) | set("=+−<>|/∖")


def is_math_char(ch: dict) -> bool:
    """True if a single pdfplumber char looks mathematical."""
    if is_math_font(ch.get("fontname", "")):
        return True
    text = ch.get("text", "")
    return any(c in _MATH_SYMBOLS for c in text)


def to_latex_token(text: str) -> str:
    """Map a span's text through the Unicode->LaTeX table, char by char."""
    return "".join(UNICODE_TO_LATEX.get(c, c) for c in text)
