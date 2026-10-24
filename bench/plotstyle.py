"""Shared matplotlib style. Categorical slots in fixed order (validated: dataviz validate_palette.js, light mode,
all checks pass; three slots are below 3:1 contrast, so every figure ships with a table in TABLES.md and
direct value labels where bars are used)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLOR = {
    "static": SLOTS[0],
    "ecmp": SLOTS[1],
    "independent": SLOTS[2],
    "joint": SLOTS[3],
    "oracle": SLOTS[4],
}


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT2,
            "xtick.color": TEXT2,
            "ytick.color": TEXT2,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "font.size": 9,
            "legend.frameon": False,
            "axes.axisbelow": True,
        }
    )
