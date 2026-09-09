"""Publication-quality matplotlib style for this project.

Usage
-----
    from scads_drvi.viz.style import apply_style
    apply_style()                          # default (constrained_layout=True)
    apply_style(constrained_layout=False)  # for colorbar + tight_layout layouts

Design choices
--------------
Font sizes follow Nature/Cell single-column conventions (8–9 pt).
Colors use the Wong (2011) 8-color colorblind-safe palette
(doi:10.1038/nmeth.1618).  PDF fonttype 42 keeps text editable in
Illustrator / Inkscape.  No dependence on ~/.config/matplotlib — the
style is fully self-contained.
"""
from __future__ import annotations

# Wong (2011) colorblind-safe palette — 8 colours
WONG = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#CC78BC",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#029E73",  # bluish green
    "#F0E442",  # yellow
    "#949494",  # grey
]

# ------------------------------------------------------------------
# Core rcParams dict (everything except axes.prop_cycle, which must
# be set as a cycler object — handled in apply_style() below).
# ------------------------------------------------------------------
_BASE: dict = {
    # ── Typography ─────────────────────────────────────────────────
    "font.family":              "sans-serif",
    "font.sans-serif":          ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size":                8,          # tick labels, legend body
    "axes.titlesize":           9,
    "axes.titleweight":         "normal",
    "axes.titlepad":            5,
    "axes.labelsize":           9,
    "axes.labelpad":            3,
    "xtick.labelsize":          8,
    "ytick.labelsize":          8,
    "legend.fontsize":          8,
    "legend.title_fontsize":    9,
    # ── Colours ────────────────────────────────────────────────────
    "text.color":               "black",
    "axes.edgecolor":           "black",
    "axes.labelcolor":          "black",
    "xtick.color":              "black",
    "ytick.color":              "black",
    "figure.facecolor":         "white",
    "axes.facecolor":           "white",
    "savefig.facecolor":        "white",
    "savefig.edgecolor":        "white",
    # ── Axes ───────────────────────────────────────────────────────
    "axes.spines.top":          False,
    "axes.spines.right":        False,
    "axes.spines.left":         True,
    "axes.spines.bottom":       True,
    "axes.linewidth":           0.8,
    "axes.grid":                False,
    "axes.axisbelow":           True,
    # ── Ticks ──────────────────────────────────────────────────────
    "xtick.direction":          "out",
    "ytick.direction":          "out",
    "xtick.major.width":        0.8,
    "ytick.major.width":        0.8,
    "xtick.major.size":         3.0,
    "ytick.major.size":         3.0,
    "xtick.minor.size":         2.0,
    "ytick.minor.size":         2.0,
    # ── Lines / markers ────────────────────────────────────────────
    "lines.linewidth":          1.2,
    "lines.markersize":         4,
    # ── Legend ─────────────────────────────────────────────────────
    "legend.frameon":           False,
    "legend.borderpad":         0.3,
    "legend.labelspacing":      0.3,
    "legend.handlelength":      1.5,
    "legend.handletextpad":     0.4,
    # ── Figure ─────────────────────────────────────────────────────
    # 6.5 × 4 in = ~double-column width; scale panels up/down per figure.
    "figure.figsize":           [6.5, 4.0],
    "figure.dpi":               120,        # screen preview
    "figure.constrained_layout.use": True,  # disable per-figure for colorbar+tight_layout
    # ── Save ───────────────────────────────────────────────────────
    "savefig.dpi":              300,        # print / journal submission
    "savefig.format":           "pdf",      # vector; switch to png for raster-heavy figures
    "savefig.bbox":             "tight",
    "savefig.pad_inches":       0.05,
    "savefig.transparent":      False,
    # PDF/PS text editable in Illustrator and Inkscape
    "pdf.fonttype":             42,
    "ps.fonttype":              42,
}


def apply_style(constrained_layout: bool = True) -> None:
    """Apply the project publication style to matplotlib's global rcParams.

    Parameters
    ----------
    constrained_layout:
        Set to False when mixing ``fig.colorbar`` with ``tight_layout`` —
        constrained_layout and tight_layout cannot be active simultaneously.
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(_BASE)
    plt.rcParams["axes.prop_cycle"] = cycler("color", WONG)
    plt.rcParams["figure.constrained_layout.use"] = constrained_layout
