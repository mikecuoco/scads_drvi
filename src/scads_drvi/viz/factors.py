"""Factorization diagnostics.

Figures about the factors themselves rather than about a trait: which dimensions carry
signal, which are dead, which track a nuisance covariate, and how groups load onto them.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from scads_drvi.viz.frugal import BoxStats

__all__ = [
    "latent_dimension_stats",
    "factor_distributions",
    "factor_correlation",
    "group_factor_heatmaps",
    "covariate_association",
]


def latent_dimension_stats(
    latent_stats: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    vanished_column: str = "vanished",
    ncols: int = 2,
) -> tuple[Figure, np.ndarray]:
    """Per-dimension summary statistics, with vanished dimensions marked.

    A vanished dimension is not a small one -- it is a dimension the model stopped using.
    Plotting the two the same way invites reading a dead factor as a weak signal, so they
    are drawn in a separate colour and counted in the title.
    """
    import matplotlib.pyplot as plt
    from pandas.api.types import is_numeric_dtype

    frame = latent_stats
    if columns is None:
        # is_numeric_dtype, not np.issubdtype: from pandas 3.0 a text column is a
        # StringDtype extension dtype and np.issubdtype raises on it instead of
        # returning False.
        columns = [
            c
            for c in frame.columns
            if c != vanished_column and is_numeric_dtype(frame[c])
        ][:4]
    columns = list(columns)
    if not columns:
        raise ValueError("no numeric columns to plot")

    vanished = (
        frame[vanished_column].astype(bool).to_numpy()
        if vanished_column in frame.columns
        else np.zeros(len(frame), dtype=bool)
    )

    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows), squeeze=False
    )
    flat = axes.ravel()

    positions = np.arange(len(frame))
    # strict=False is deliberate: the grid is padded to nrows*ncols >= len(columns),
    # so the trailing axes have no column and are left blank.
    for ax, column in zip(flat, columns, strict=False):
        values = frame[column].to_numpy(dtype=float)
        ax.bar(positions[~vanished], values[~vanished], color="#0173B2", width=0.9)
        if vanished.any():
            ax.bar(
                positions[vanished], values[vanished], color="#D55E00", width=0.9,
                label="vanished",
            )
            ax.legend(frameon=False, fontsize=7)
        ax.set_xlabel("latent dimension")
        ax.set_ylabel(column)

    for slot in range(len(columns), len(flat)):
        flat[slot].set_visible(False)

    fig.suptitle(f"{int(vanished.sum())} of {len(frame)} dimensions vanished")
    return fig, flat


def factor_distributions(
    panels: Mapping[str, BoxStats],
    *,
    value_label: str = "score",
    yscale: str | None = "symlog",
    linthresh: float = 1e-3,
) -> Figure:
    """One stacked box panel per score source, over the same factor axis.

    Takes precomputed statistics, so the per-cell matrices behind these panels are never
    handed to a plotting call.
    """
    import matplotlib.pyplot as plt

    from scads_drvi.viz.frugal import draw_boxes

    if not panels:
        raise ValueError("no panels to draw")
    names = list(panels)
    width = max(6.0, 0.14 * max(len(p) for p in panels.values()))
    fig, axes = plt.subplots(
        len(names), 1, figsize=(width, 2.4 * len(names)), squeeze=False, sharex=True
    )
    flat = axes.ravel()

    for ax, name in zip(flat, names, strict=True):
        stats = panels[name]
        draw_boxes(ax, stats, show_maximum=stats.maximum is not None)
        ax.set_ylabel(f"{name}\n{value_label}", fontsize=7)
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=linthresh)
        elif yscale:
            ax.set_yscale(yscale)

    last = flat[-1]
    stats = panels[names[-1]]
    last.set_xticks(np.arange(1, len(stats) + 1))
    last.set_xticklabels(stats.labels, rotation=90, fontsize=5)
    last.set_xlabel("factor")
    return fig


def factor_correlation(
    correlation: pd.DataFrame,
    *,
    vanished: Sequence[str] = (),
    method: str = "average",
    cmap: str = "RdBu_r",
) -> tuple[Figure, Axes]:
    """Clustered factor-factor correlation, with dead factors greyed rather than hidden.

    Removing vanished factors would make the matrix look cleaner and quietly change what
    fraction of the model the figure describes.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError(f"expected a square matrix, got {correlation.shape}")

    matrix = correlation.to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0)
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    distance = 0.5 * (distance + distance.T)

    if len(matrix) > 2:
        order = leaves_list(linkage(squareform(distance, checks=False), method=method))
    else:
        order = np.arange(len(matrix))

    labels = correlation.index.astype(str).to_numpy()[order]
    ordered = matrix[np.ix_(order, order)]

    size = max(4.0, 0.11 * len(labels))
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    image = ax.imshow(ordered, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)

    dead = set(str(v) for v in vanished)
    if dead:
        for position, label in enumerate(labels):
            if label in dead:
                ax.get_xticklabels()[position].set_color("#D55E00")
                ax.get_yticklabels()[position].set_color("#D55E00")

    fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045).set_label("Pearson r")
    return fig, ax


def group_factor_heatmaps(
    panels: Sequence[tuple[str, pd.DataFrame, str]],
    *,
    row_order: Sequence[str] | None = None,
    column_order: Sequence[str] | None = None,
    grey_columns: Container[str] = (),
) -> tuple[Figure, np.ndarray]:
    """Several heatmaps sharing ONE row and column order.

    Letting each panel choose its own ordering is how a reader compares two panels and
    draws a conclusion about factors that are not in the same place in both.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    from scads_drvi.viz.color import robust_norm

    panels = list(panels)
    if not panels:
        raise ValueError("no panels to draw")

    first = panels[0][1]
    rows = list(row_order) if row_order is not None else list(first.index.astype(str))
    cols = (
        list(column_order)
        if column_order is not None
        else list(first.columns.astype(str))
    )

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), max(3.0, 0.2 * len(rows))),
        squeeze=False,
    )
    flat = axes.ravel()

    for ax, (title, frame, cmap) in zip(flat, panels, strict=True):
        aligned = frame.reindex(index=rows, columns=cols)
        raw = aligned.to_numpy(dtype=float)
        values = np.ma.masked_invalid(raw)
        # A panel holding negative values is a signed quantity, so centre it on zero;
        # otherwise the midpoint of a diverging map lands somewhere arbitrary.
        signed = bool(np.isfinite(raw).any() and np.nanmin(raw) < 0)
        norm = robust_norm(raw, symmetric=signed)
        image = ax.imshow(
            values,
            aspect="auto",
            cmap=colormaps[cmap].with_extremes(bad="#C2CCD6"),
            norm=norm,
        )
        ax.set_title(title, fontsize=8)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows if ax is flat[0] else [], fontsize=6)
        for position, name in enumerate(cols):
            if name in grey_columns:
                ax.get_xticklabels()[position].set_color("#D55E00")
        fig.colorbar(image, ax=ax, pad=0.02, fraction=0.03)

    return fig, flat


def covariate_association(
    association: pd.Series,
    *,
    threshold: float = 0.7,
    ylabel: str = "|Spearman rho|",
) -> tuple[Figure, Axes]:
    """Per-factor association with a nuisance covariate, with a concern threshold drawn.

    The threshold is an argument because what counts as too much depends on the covariate
    and the claim being made about the factor.
    """
    import matplotlib.pyplot as plt

    values = association.to_numpy(dtype=float)
    labels = association.index.astype(str)
    order = np.argsort(-np.abs(values))

    fig, ax = plt.subplots(figsize=(max(5.0, 0.13 * len(values)), 3.0))
    colors = [
        "#D55E00" if abs(v) >= threshold else "#0173B2" for v in values[order]
    ]
    ax.bar(np.arange(len(values)), np.abs(values[order]), color=colors, width=0.9)
    ax.axhline(threshold, color="#949494", linestyle=":", linewidth=0.8)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels[order], rotation=90, fontsize=5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("factor, ranked by association")
    flagged = int((np.abs(values) >= threshold).sum())
    ax.set_title(f"{flagged} factor(s) at or above {threshold:g}", fontsize=8)
    return fig, ax
