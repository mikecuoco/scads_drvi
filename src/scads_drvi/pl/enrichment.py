"""Enrichment figures.

Every function takes tidy frames -- never a path -- and returns the figure plus its axes.
Tick labels are the results table's own ``factor`` value plus a ``+``/``-`` suffix when the
row carries a ``direction`` (see :func:`scads_drvi.enrich.ldsc.read_results`); pass
``titles=embed.var["title"]`` to show DRVI's own ``"DR 1"`` naming instead of the raw
``factor`` value (``"dim_47"``) -- the same ``"DR 1+"``/``"DR 1-"`` convention
:func:`scads_drvi.pl.umap.latent_umap_grid` uses. Any function that would otherwise
receive a per-cell table takes a :class:`~scads_drvi.pl.frugal.BoxStats` instead, which
makes handing it a million rows impossible rather than merely inadvisable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from scads_drvi.pl.color import SignificanceRamp

__all__ = [
    "heritability_landscape",
    "trait_concordance",
    "risk_manhattan",
    "chrom_sort_key",
]

Z_COLUMN = "Coefficient_z-score"

_SUFFIX = {"pos": "+", "neg": "-", "combined": ""}


def _labelled(results: pd.DataFrame, titles: Mapping[str, str] | None = None) -> list[str]:
    """A reader-facing name per row: ``factor`` (or its display title, from `titles`, when
    given -- e.g. DRVI's own ``"DR 1"``) plus a ``+``/``-`` suffix when the row carries
    a ``direction``, without ever materializing a ``dim_47/pos``-shaped column name.
    """
    from scads_drvi._util.compat import with_factor_column

    results = with_factor_column(results)
    factors = results["factor"].astype(str)
    if titles is not None:
        factors = factors.map(titles).fillna(factors)
    if "direction" not in results.columns:
        return factors.tolist()
    suffix = results["direction"].map(_SUFFIX).fillna("")
    return (factors + suffix).tolist()


def heritability_landscape(
    results: pd.DataFrame,
    *,
    trait: str | None = None,
    top_n: int = 4,
    ramp: SignificanceRamp | None = None,
    q_column: str = "fdr_q",
    z_column: str = Z_COLUMN,
    tau_column: str = "Coefficient",
    tau_se_column: str | None = "Coefficient_std_error",
    alpha: float = 0.05,
    titles: Mapping[str, str] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Two ranked bar panels of the same factors: raw tau with error bars, and z-scores.

    Left, the S-LDSC coefficient `tau` (`tau_column`) with its standard error
    (`tau_se_column`; pass None or leave the column out for no error bars). Right, the
    coefficient z-score. Both panels share the factor order (ranked by z) and one
    significance colour ramp, so a factor reads the same in each.

    The BH boundary is recomputed from this run's q-values rather than drawn at a
    remembered z -- that number depends on the whole p-vector, so one copied from a
    previous run is wrong for this one. It is drawn on the z panel only.

    `alpha` is passed to both the boundary and its label, so the line and the text naming
    it cannot be set at different levels. `titles` (e.g. `embed.var["title"]`) shows
    each top-`top_n` annotation as DRVI's own `"DR 1"` naming plus a `+`/`-` suffix,
    rather than the raw `dim` value.
    """
    import matplotlib.pyplot as plt

    from scads_drvi.pl.color import (
        DEFAULT_RAMP,
        Z_HIGH_CONFIDENCE,
        Z_NOMINAL_ONE_TAILED,
        add_threshold_lines,
        significance_colors,
        significance_handles,
    )
    from scads_drvi.stats import bh_threshold_z

    ramp = ramp or DEFAULT_RAMP
    frame = results if trait is None else results.loc[results["trait"] == trait]
    if frame.empty:
        raise ValueError(f"no results to plot for trait={trait!r}")
    for column in (z_column, q_column):
        if column not in frame.columns:
            raise KeyError(f"{column!r} not in results; run stats.add_fdr first")
    if tau_column not in frame.columns:
        raise KeyError(f"{tau_column!r} not in results; it is the S-LDSC coefficient (tau)")

    frame = frame.sort_values(z_column, ascending=False)
    names = _labelled(frame, titles)
    z = frame[z_column].to_numpy(dtype=float)
    q = frame[q_column].to_numpy(dtype=float)
    tau = frame[tau_column].to_numpy(dtype=float)
    tau_se = (
        frame[tau_se_column].to_numpy(dtype=float)
        if tau_se_column is not None and tau_se_column in frame.columns
        else None
    )
    colors = significance_colors(q, ramp=ramp)
    boundary = bh_threshold_z(q, z, alpha=alpha)

    fig, (tau_ax, z_ax) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    positions = np.arange(len(z))

    tau_ax.bar(
        positions,
        tau,
        yerr=tau_se,
        color=colors,
        width=0.9,
        error_kw={"ecolor": "0.2", "elinewidth": 0.8, "capsize": 1.5},
    )
    tau_ax.axhline(0, color="0.3", linewidth=0.6)
    tau_ax.set_xlabel("factor, ranked by z")
    tau_ax.set_ylabel("coefficient tau")
    tau_ax.set_xticks([])

    z_ax.bar(positions, z, color=colors, width=0.9)
    z_ax.set_xlabel("factor, ranked by z")
    z_ax.set_ylabel("coefficient z")
    z_ax.set_xticks([])
    add_threshold_lines(
        z_ax,
        z=(Z_NOMINAL_ONE_TAILED, Z_HIGH_CONFIDENCE),
        axis="y",
        bh=boundary,
        alpha=alpha,
    )
    for slot in range(min(top_n, len(z))):
        z_ax.annotate(
            names[slot],
            (positions[slot], z[slot]),
            textcoords="offset points",
            xytext=(10, -2 - 10 * slot),  # a cascade to the right: the bars there are shorter
            ha="left",
            va="top",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "0.4"},
        )
    z_ax.legend(handles=significance_handles(ramp), frameon=False, fontsize=7, loc="upper right")

    if trait is not None:
        fig.suptitle(trait)
    return fig, (tau_ax, z_ax)


def trait_concordance(
    results: pd.DataFrame,
    *,
    traits: Sequence[str],
    z_column: str = Z_COLUMN,
    top_n: int = 4,
    titles: Mapping[str, str] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Two traits' z-scores against each other, plus the distribution of the difference.

    A sensitivity analysis reads as concordance plus a shift; separating the two panels
    keeps "the same factors rank highly" distinguishable from "every z moved down".
    `titles` (e.g. `embed.var["title"]`) shows each top-`top_n` annotation as DRVI's own
    `"DR 1"` naming plus a `+`/`-` suffix, rather than the raw `factor` value.
    """
    import matplotlib.pyplot as plt

    from scads_drvi._util.compat import with_factor_column

    results = with_factor_column(results)
    traits = list(traits)
    if len(traits) != 2:
        raise ValueError(f"need exactly two traits, got {traits}")

    # Pivoting on `factor` alone would silently average a directional arm's pos and neg
    # rows together -- pivot on both when `direction` is present so each row of `wide`
    # is still one factor, not one factor with two directions blended into it.
    index = ["factor", "direction"] if "direction" in results.columns else "factor"
    wide = results.pivot_table(index=index, columns="trait", values=z_column)
    missing = [t for t in traits if t not in wide.columns]
    if missing:
        raise KeyError(f"no results for trait(s) {missing}")
    wide = wide[traits].dropna()
    if wide.empty:
        raise ValueError("no factor has a z-score for both traits")

    left, right = wide[traits[0]].to_numpy(), wide[traits[1]].to_numpy()
    titled = (lambda f: titles.get(f, f)) if titles is not None else (lambda f: f)
    if isinstance(index, list):
        names = [f"{titled(f)}{_SUFFIX.get(direction, '')}" for f, direction in wide.index]
    else:
        names = [titled(str(f)) for f in wide.index]

    fig, (scatter, hist) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    scatter.scatter(left, right, s=14, linewidths=0, color="#0173B2")
    span = [min(left.min(), right.min()), max(left.max(), right.max())]
    scatter.plot(span, span, color="#949494", linestyle=":", linewidth=0.8)
    scatter.set_xlabel(f"z ({traits[0]})")
    scatter.set_ylabel(f"z ({traits[1]})")
    for slot in np.argsort(-left)[:top_n]:
        scatter.annotate(
            names[slot],
            (left[slot], right[slot]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=7,
        )

    delta = right - left
    # A constant delta -- two traits ranking identically, or a single factor -- gives the
    # histogram a zero-width range, and numpy 2 raises "Too many bins for data range"
    # there where numpy 1 quietly widened it.
    #
    # The comparison has to be against a TOLERANCE, not against zero: a delta that is
    # constant in intent still varies by ~1e-16 once the two z-scores have been through
    # floating-point arithmetic, so `ptp > 0` is true and the range is still unusable.
    centre = float(np.mean(delta)) if delta.size else 0.0
    spread = float(np.ptp(delta)) if delta.size else 0.0
    scale = max(abs(centre), 1.0)
    if spread > scale * 1e-9:
        bins = min(30, max(5, delta.size // 2))
        hist.hist(delta, bins=bins, color="#949494")
    else:
        pad = max(abs(centre) * 0.05, 0.05)
        hist.hist(delta, bins=5, range=(centre - pad, centre + pad), color="#949494")
    hist.axvline(0.0, color="#D55E00", linestyle="--", linewidth=0.9)
    hist.set_xlabel(f"z({traits[1]}) - z({traits[0]})")
    hist.set_ylabel("factors")
    hist.set_title(f"median shift {np.median(delta):+.2f}", fontsize=8)
    return fig, (scatter, hist)


def chrom_sort_key(chrom: str) -> tuple[int, str]:
    """Numeric chromosomes first, in numeric order; anything else after, alphabetic.

    No hardcoded assembly table -- this package makes no organism/build assumption
    (see `tests/test_portability.py`), so chromosome order is derived from whichever
    labels are actually present, not looked up.
    """
    bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
    return (0, f"{int(bare):09d}") if bare.isdigit() else (1, bare)


# Private alias kept for existing importers.
_chrom_sort_key = chrom_sort_key


def risk_manhattan(
    risk: pd.DataFrame,
    *,
    value_col: str = "risk_total",
    factor_subset: Sequence[str] | None = None,
    titles: Mapping[str, str] | None = None,
    s: float = 8.0,
    dim_subset: Sequence[str] | None = None,
) -> tuple[Figure, Axes]:
    """Genome-wide scatter of peak-level risk (:func:`scads_drvi.enrich.risk.peak_risk`'s
    output) -- position on x, `value_col` on y.

    Default view (`factor_subset=None`): one point per peak, its highest `value_col`
    across every `(factor, direction)` row that mentions it -- the peak-resolution
    analogue of picking a signal's single top-capturing factor -- coloured by *which*
    factor that is. `factor_subset` shows every row for those specific factors instead
    (still one point per `(peak, factor, direction)`), e.g. to look at one factor's own
    risk track alone rather than the genome-wide maximum.

    X position is a running genomic offset built from `risk`'s own `chrom`/`start`/
    `end` columns, chromosomes ordered numerically then alphabetically
    (:func:`chrom_sort_key`) -- not a hardcoded assembly length table, so a
    chromosome's span here is only as wide as the peaks actually plotted make it, not
    its real length.

    `dim_subset` is the deprecated old name of `factor_subset`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from scads_drvi._util.compat import renamed_kwarg, with_factor_column
    from scads_drvi.pl.color import categorical_palette

    factor_subset = renamed_kwarg(factor_subset, dim_subset, new="factor_subset", old="dim_subset")
    risk = with_factor_column(risk)
    for column in ("peak", "chrom", "start", "factor", "direction", value_col):
        if column not in risk.columns:
            raise KeyError(f"{column!r} not in risk; run enrich.risk.peak_risk first")

    frame = risk if factor_subset is None else risk[risk["factor"].isin(factor_subset)]
    if frame.empty:
        raise ValueError(
            "no rows to plot"
            if factor_subset is None
            else f"no rows for factor_subset={list(factor_subset)!r}"
        )

    if factor_subset is None:
        frame = frame.loc[frame.groupby("peak")[value_col].idxmax()]

    end_col = "end" if "end" in frame.columns else "start"
    chroms = sorted(frame["chrom"].astype(str).unique(), key=chrom_sort_key)
    offsets: dict[str, int] = {}
    running = 0
    for chrom in chroms:
        offsets[chrom] = running
        running += int(frame.loc[frame["chrom"] == chrom, end_col].max()) + 1

    x = frame["chrom"].astype(str).map(offsets).to_numpy() + frame["start"].to_numpy()
    y = frame[value_col].to_numpy(dtype=float)

    titled = (lambda f: titles.get(f, f)) if titles is not None else (lambda f: f)
    palette = categorical_palette(frame["factor"].astype(str).tolist())
    colors = frame["factor"].astype(str).map(palette)

    fig, ax = plt.subplots(figsize=(max(9.0, 0.4 * len(chroms)), 3.6))
    ax.scatter(x, y, c=colors, s=s, linewidths=0)
    ax.set_xticks([offsets[c] + frame.loc[frame["chrom"] == c, "start"].median() for c in chroms])
    ax.set_xticklabels(chroms, rotation=90, fontsize=7)
    ax.set_xlabel("genomic position")
    ax.set_ylabel(value_col)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=titled(factor))
        for factor, color in palette.items()
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=6,
        loc="upper left",
        ncol=max(1, len(handles) // 12),
    )
    return fig, ax
