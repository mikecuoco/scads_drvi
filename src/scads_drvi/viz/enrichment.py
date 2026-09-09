"""Enrichment figures.

Every function takes tidy frames -- never a path -- and returns the figure plus its axes.
Tick labels go through the factor labels so a panel cannot be titled with an index name,
and any function that would otherwise receive a per-cell table takes a
:class:`~scads_drvi.viz.frugal.BoxStats` instead, which makes handing it a million rows
impossible rather than merely inadvisable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from scads_drvi.labels import FactorLabels
    from scads_drvi.viz.frugal import BoxStats

__all__ = [
    "heritability_landscape",
    "trait_concordance",
    "covariate_audit",
    "score_by_group",
    "grouped_landscape",
]

Z_COLUMN = "Coefficient_z-score"


def _labelled(results: "pd.DataFrame", labels: "FactorLabels | None") -> list[str]:
    if "display" in results.columns:
        return results["display"].astype(str).tolist()
    if labels is not None:
        return labels.display_labels(results["dim"].astype(str))
    return results["dim"].astype(str).tolist()


def heritability_landscape(
    results: "pd.DataFrame",
    *,
    trait: str | None = None,
    labels: "FactorLabels | None" = None,
    top_n: int = 4,
    ramp=None,
    q_column: str = "fdr_q",
    z_column: str = Z_COLUMN,
) -> tuple["Figure", tuple["Axes", "Axes"]]:
    """Ranked z-scores and a volcano, sharing one significance ramp.

    The BH boundary is recomputed from this run's q-values rather than drawn at a
    remembered z -- that number depends on the whole p-vector, so one copied from a
    previous run is wrong for this one.
    """
    import matplotlib.pyplot as plt

    from scads_drvi.stats import bh_threshold_z
    from scads_drvi.viz.color import (
        DEFAULT_RAMP,
        Z_HIGH_CONFIDENCE,
        Z_NOMINAL_ONE_TAILED,
        add_threshold_lines,
        significance_colors,
        significance_handles,
    )

    ramp = ramp or DEFAULT_RAMP
    frame = results if trait is None else results.loc[results["trait"] == trait]
    if frame.empty:
        raise ValueError(f"no results to plot for trait={trait!r}")
    for column in (z_column, q_column):
        if column not in frame.columns:
            raise KeyError(f"{column!r} not in results; run stats.add_fdr first")

    frame = frame.sort_values(z_column, ascending=False)
    names = _labelled(frame, labels)
    z = frame[z_column].to_numpy(dtype=float)
    q = frame[q_column].to_numpy(dtype=float)
    colors = significance_colors(q, ramp=ramp)
    boundary = bh_threshold_z(q, z)

    fig, (bars, volcano) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    positions = np.arange(len(z))
    bars.bar(positions, z, color=colors, width=0.9)
    bars.set_xlabel("factor, ranked")
    bars.set_ylabel("coefficient z")
    bars.set_xticks([])
    add_threshold_lines(
        bars, z=(Z_NOMINAL_ONE_TAILED, Z_HIGH_CONFIDENCE), axis="y", bh=boundary
    )
    for slot in range(min(top_n, len(z))):
        bars.annotate(
            names[slot], (positions[slot], z[slot]),
            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7,
        )

    with np.errstate(divide="ignore"):
        neglog = -np.log10(np.clip(q, 1e-300, 1.0))
    volcano.scatter(z, neglog, c=colors, s=14, linewidths=0)
    volcano.set_xlabel("coefficient z")
    volcano.set_ylabel(f"-log10({ramp.label})")
    volcano.axhline(
        -np.log10(ramp.thresholds[0]), color="#949494", linestyle=":", linewidth=0.8
    )
    for slot in range(min(top_n, len(z))):
        volcano.annotate(
            names[slot], (z[slot], neglog[slot]),
            textcoords="offset points", xytext=(3, 3), fontsize=7,
        )
    volcano.legend(
        handles=significance_handles(ramp), frameon=False, fontsize=7, loc="upper left"
    )

    if trait is not None:
        fig.suptitle(trait)
    return fig, (bars, volcano)


def trait_concordance(
    results: "pd.DataFrame",
    *,
    traits: Sequence[str],
    labels: "FactorLabels | None" = None,
    z_column: str = Z_COLUMN,
    top_n: int = 4,
) -> tuple["Figure", tuple["Axes", "Axes"]]:
    """Two traits' z-scores against each other, plus the distribution of the difference.

    A sensitivity analysis reads as concordance plus a shift; separating the two panels
    keeps "the same factors rank highly" distinguishable from "every z moved down".
    """
    import matplotlib.pyplot as plt

    traits = list(traits)
    if len(traits) != 2:
        raise ValueError(f"need exactly two traits, got {traits}")

    wide = results.pivot_table(index="dim", columns="trait", values=z_column)
    missing = [t for t in traits if t not in wide.columns]
    if missing:
        raise KeyError(f"no results for trait(s) {missing}")
    wide = wide[traits].dropna()
    if wide.empty:
        raise ValueError("no factor has a z-score for both traits")

    left, right = wide[traits[0]].to_numpy(), wide[traits[1]].to_numpy()
    names = (
        labels.display_labels(wide.index.astype(str))
        if labels is not None
        else wide.index.astype(str).tolist()
    )

    fig, (scatter, hist) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    scatter.scatter(left, right, s=14, linewidths=0, color="#0173B2")
    span = [min(left.min(), right.min()), max(left.max(), right.max())]
    scatter.plot(span, span, color="#949494", linestyle=":", linewidth=0.8)
    scatter.set_xlabel(f"z ({traits[0]})")
    scatter.set_ylabel(f"z ({traits[1]})")
    for slot in np.argsort(-left)[:top_n]:
        scatter.annotate(
            names[slot], (left[slot], right[slot]),
            textcoords="offset points", xytext=(3, 3), fontsize=7,
        )

    delta = right - left
    hist.hist(delta, bins=30, color="#949494")
    hist.axvline(0.0, color="#D55E00", linestyle="--", linewidth=0.9)
    hist.set_xlabel(f"z({traits[1]}) - z({traits[0]})")
    hist.set_ylabel("factors")
    hist.set_title(f"median shift {np.median(delta):+.2f}", fontsize=8)
    return fig, (scatter, hist)


def covariate_audit(
    cells: "pd.DataFrame",
    *,
    value: str,
    covariates: Sequence[str],
    log_x: Sequence[str] = (),
    n_sample: int = 50_000,
    gridsize: int = 40,
) -> tuple["Figure", np.ndarray]:
    """Hexbin of a per-cell value against each technical covariate, with Spearman rho.

    Hexbin rather than scatter because a million-point scatter of a covariate against a
    score is a solid block that hides the relationship it exists to show.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    from scads_drvi.viz.umap import subsample

    covariates = list(covariates)
    if not covariates:
        raise ValueError("no covariates to audit")
    for column in (value, *covariates):
        if column not in cells.columns:
            raise KeyError(f"{column!r} is not a column of the cell table")

    plotted = subsample(cells, n_sample)
    fig, axes = plt.subplots(
        1, len(covariates), figsize=(3.4 * len(covariates), 3.2), squeeze=False
    )
    flat = axes.ravel()

    for ax, covariate in zip(flat, covariates):
        block = plotted[[covariate, value]].replace([np.inf, -np.inf], np.nan).dropna()
        x = block[covariate].to_numpy(dtype=float)
        y = block[value].to_numpy(dtype=float)
        if covariate in log_x:
            keep = x > 0
            x, y = np.log10(x[keep]), y[keep]
        ax.hexbin(x, y, gridsize=gridsize, mincnt=1, cmap="viridis", linewidths=0)
        rho = spearmanr(x, y).statistic if len(x) > 2 else float("nan")
        ax.set_xlabel(f"log10({covariate})" if covariate in log_x else covariate)
        ax.set_ylabel(value)
        ax.set_title(f"rho = {rho:+.3f}", fontsize=8)
    return fig, flat


def score_by_group(
    stats: "BoxStats",
    *,
    group_label: str,
    value_label: str = "score",
    null: float | None = None,
    highlight: Sequence[str] = (),
    top_n: int | None = None,
) -> tuple["Figure", "Axes"]:
    """Score distribution across groups, from PRECOMPUTED quantiles.

    Taking a :class:`BoxStats` is the point: this function cannot be handed the per-cell
    table, so the expensive mistake is unavailable rather than discouraged.

    `null` belongs to the score -- read it off the CellScores object, which carries its
    own, rather than typing 0 or 1.
    """
    import matplotlib.pyplot as plt

    from scads_drvi.viz.color import categorical_palette
    from scads_drvi.viz.frugal import draw_boxes

    ordered = stats.order_by("median")
    if top_n is not None:
        ordered = ordered.select(list(ordered.labels)[:top_n])

    palette = categorical_palette(
        ordered.labels, highlight=list(highlight) or None
    )
    colors = [palette[label] for label in ordered.labels]

    width = max(4.0, 0.32 * len(ordered))
    fig, ax = plt.subplots(figsize=(width, 3.4))
    draw_boxes(ax, ordered, colors=colors, hline=null)
    ax.set_xticks(np.arange(1, len(ordered) + 1))
    ax.set_xticklabels(ordered.labels, rotation=90, fontsize=7)
    ax.set_xlabel(group_label)
    ax.set_ylabel(value_label)
    return fig, ax


def grouped_landscape(
    means: "pd.DataFrame",
    counts: "pd.DataFrame",
    *,
    row_label: str,
    column_label: str,
    value_label: str = "mean score",
    blocks: Sequence[tuple[str, int, int]] = (),
    gamma: float = 1.0,
    cmap: str = "magma",
    masked_color: str = "#C2CCD6",
) -> tuple["Figure", "Axes"]:
    """Heatmap of a value over two groupings, with thin bins visibly masked.

    Masked bins are drawn in a distinct cool grey, and the colourbar label says so: grey
    means "no reliable value", not "measured, and low". `blocks` draws alternating
    background bands for a blocked row axis (see ``scores.aggregate.block_order``).
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    from scads_drvi.viz.color import robust_norm

    if means.shape != counts.shape:
        raise ValueError(
            f"means {means.shape} and counts {counts.shape} must have the same shape"
        )

    palette = colormaps[cmap].with_extremes(bad=masked_color)
    norm = robust_norm(means.to_numpy(dtype=float), percentiles=(0.0, 99.0), gamma=gamma)

    height = max(3.0, 0.22 * means.shape[0])
    fig, ax = plt.subplots(figsize=(1.1 * means.shape[1] + 3.0, height))

    for slot, (_, start, stop) in enumerate(blocks):
        if slot % 2 == 0:
            ax.axhspan(start - 0.5, stop - 0.5, color="#F2F2F2", zorder=0)

    image = ax.imshow(
        np.ma.masked_invalid(means.to_numpy(dtype=float)),
        aspect="auto", cmap=palette, norm=norm, zorder=2,
    )
    ax.set_xticks(np.arange(means.shape[1]))
    ax.set_xticklabels(means.columns.astype(str), rotation=90, fontsize=7)
    ax.set_yticks(np.arange(means.shape[0]))
    ax.set_yticklabels(means.index.astype(str), fontsize=6)
    ax.set_xlabel(column_label)
    ax.set_ylabel(row_label)

    for name, start, stop in blocks:
        ax.text(
            means.shape[1] - 0.4, (start + stop - 1) / 2.0, name,
            va="center", ha="left", fontsize=7,
        )

    bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.03)
    bar.set_label(f"{value_label}\n(grey: below the count threshold)")
    return fig, ax
