"""Shared color palette for credit_risk's plotting modules.

Colors are the validated reference categorical/sequential palette from the
dataviz skill (light-mode slots) - not re-picked per chart. Import from here
rather than redefining hex values in `univariate_analysis.py`,
`bivariate_analysis.py`, or `correlations.py`.
"""

from __future__ import annotations

# Reference categorical palette, light mode, fixed slot order (dataviz skill).
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
OTHER_COLOR = "#c3c2b7"  # muted gray for a folded "Other" wedge/category
SEQUENTIAL_BLUE = "#2a78d6"  # single-hue default for a magnitude comparison

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
