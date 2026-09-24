"""One figure style for the whole report.

Figures go into an APA paper, so they are light-mode, print-legible, and quiet:
thin marks, recessive grid and axes, text in ink colors rather than series
colors. The three categorical colors validate as colorblind-safe for any pair;
the third (aqua) sits below 3:1 contrast on white, so any chart that uses it
carries direct labels. Every figure's underlying numbers are also written to
results/ as CSV, which is the table view.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # headless: Colab, servers, and CI all render to file
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

# Colab lacks Times New Roman; fall back quietly instead of warning per figure.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.edgecolor": INK_3,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 2,
})


def save(fig, name):
    """Write a figure to figures/<name>.png and close it."""
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def label_bars(ax, bars, fmt="{:,.0f}", inside=False):
    """Direct labels on bars, in ink color, never the series color."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, -12 if inside else 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5,
                    color="white" if inside else INK_2)
