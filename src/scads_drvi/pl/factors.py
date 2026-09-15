"""Factorization diagnostics.

Figures about the factors themselves rather than about a trait: which track a nuisance
covariate, how factors correlate with one another, and how groups load onto them.

Two figures this module used to draw are now DRVI's own, and are not reimplemented here:
per-dimension summary statistics (``drvi.utils.pl.plot_latent_dimension_stats``, driven
by the ``var`` columns ``model.set_latent_dimension_stats`` writes directly onto the
embed -- see the getting-started guide's training example) and a factor-value-by-category
heatmap (``drvi.utils.pl.plot_latent_dims_in_heatmap``). Call those directly and layer
:func:`scads_drvi.pl.style.apply_style`/:func:`scads_drvi.pl.save.save_figure` on the
figure they return, the same as any other figure in this package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from scads_drvi.pl.frugal import BoxStats

__all__ = [
    "factor_distributions",
    "factor_correlation",
    "covariate_association",
]


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

    from scads_drvi.pl.frugal import draw_boxes

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


def covariate_association(
    association: pd.Series,
    *,
    threshold: float = 0.7,
    ylabel: str = "|Spearman rho|",
) -> tuple[Figure, Axes]:
    """Per-factor association with a nuisance covariate, with a concern threshold drawn.

    The threshold is an argument because what counts as too much depends on the covariate
    and the claim being made about the factor.

    `association` is any per-factor score, computed however the caller likes -- for a
    categorical covariate, ``drvi.utils.metrics.spearman_correlation_score`` (or
    ``nn_alignment_score`` / ``discrete_scaled_mutual_info_score`` /
    ``binary_maximum_mutual_information_score``) run against ``embed.X`` is the DRVI-
    native way to get one; this function only draws it.
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
