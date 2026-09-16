"""Factorization diagnostics.

Figures about the factors themselves rather than about a trait: which track a nuisance
covariate, how factors correlate with one another, how groups load onto them, and (below)
this project's own in-house ports of the two DRVI-specific figures this module used to
defer entirely -- per-dimension summary statistics (:func:`latent_dimension_stats`,
ported from ``drvi.utils.pl.plot_latent_dimension_stats``) and a factor-value-by-category
heatmap (:func:`latent_heatmap`, ported from ``drvi.utils.pl.plot_latent_dims_in_heatmap``)
-- matching each one's own colours, per-dimension scaling and sampling mechanics exactly
(see each function's own docstring for where this project's return-value contract and one
already-fixed layout bug are kept instead), so no `drvi-py` install is needed just to draw
a figure that looks like DRVI's own. :func:`latent_heatmap_with_heritability` pairs that
heatmap with a per-dim heritability bar sharing its column order -- a genuinely
complementary companion panel, unlike
:func:`scads_drvi.pl.enrichment.heritability_landscape`'s bar+volcano, which collapse to
the same ranking when the volcano's x-axis is itself a z-score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

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
    "latent_dimension_stats",
    "latent_heatmap",
    "latent_heatmap_with_heritability",
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


def _cluster_order(matrix: np.ndarray, *, method: str = "average") -> np.ndarray:
    """Leaf order from hierarchical clustering of a correlation matrix.

    Shared by :func:`factor_correlation` and :func:`latent_heatmap` so the two don't
    carry two copies of the same distance-from-correlation-and-linkage call. `matrix` is
    a square, symmetric correlation matrix (NaN treated as 0 -- no evidence of
    correlation, not evidence of none); a matrix too small to cluster (``n <= 2``)
    returns its rows unreordered rather than asking scipy to cluster nothing.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    matrix = np.nan_to_num(matrix, nan=0.0)
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    distance = 0.5 * (distance + distance.T)

    if len(matrix) > 2:
        return leaves_list(linkage(squareform(distance, checks=False), method=method))
    return np.arange(len(matrix))


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

    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError(f"expected a square matrix, got {correlation.shape}")

    matrix = np.nan_to_num(correlation.to_numpy(dtype=float), nan=0.0)
    order = _cluster_order(matrix, method=method)
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


def latent_dimension_stats(
    dim_stats: pd.DataFrame,
    *,
    columns: Sequence[str] = ("reconstruction_effect", "max_value", "mean", "std"),
    order_col: str = "order",
    titles: Mapping[str, str] | None = None,
    remove_vanished: bool = False,
    ncols: int = 5,
    log_scale: bool | Literal["try"] = "try",
) -> tuple[Figure, np.ndarray]:
    """One rank-vs-value panel per named column of `dim_stats` -- e.g. ``embed.var``.

    Ported to match ``drvi.utils.pl.plot_latent_dimension_stats``'s own mechanics
    exactly: points split vanished (black) / kept (blue) -- DRVI's own hardcoded
    colours -- connected by a plain grey line ranked by `order_col`, under the x-axis
    label DRVI always uses ("Rank based on Explanation Share") regardless of
    `order_col`'s own name. `log_scale="try"` (the default) switches a panel to log-y
    only when that column's *finite* values are all positive -- DRVI's own condition,
    checked only on finite values here so one non-finite entry does not silently leave
    an otherwise-positive column on a linear scale. A shared figure-level legend is
    drawn when `remove_vanished=False`, positioned to coexist with `constrained_layout`
    rather than DRVI's raw `bbox_to_anchor` (which this project already hit and fixed a
    real overlap bug from).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if order_col not in dim_stats.columns:
        raise KeyError(f"{order_col!r} is not a column of dim_stats")
    if "vanished" not in dim_stats.columns:
        raise KeyError('"vanished" is not a column of dim_stats')
    missing = [c for c in columns if c not in dim_stats.columns]
    if missing:
        raise KeyError(f"columns not in dim_stats: {missing}")
    if not columns:
        raise ValueError("no columns to plot")

    frame = dim_stats
    if remove_vanished:
        frame = frame.loc[~frame["vanished"].astype(bool)]
    frame = frame.sort_values(order_col)
    vanished = frame["vanished"].astype(bool).to_numpy()
    rank = frame[order_col].to_numpy(dtype=float)
    palette = {"vanished": "black", "kept": "blue"}

    ncols = min(ncols, len(columns))
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), squeeze=False
    )
    flat = axes.ravel()

    for ax, column in zip(flat, columns, strict=False):
        values = frame[column].to_numpy(dtype=float)
        ax.plot(rank, values, "-", color="grey", linewidth=0.8, zorder=1)
        ax.scatter(
            rank[~vanished], values[~vanished], color=palette["kept"], s=10,
            zorder=2, label="kept",
        )
        ax.scatter(
            rank[vanished], values[vanished], color=palette["vanished"], s=10,
            zorder=2, label="vanished",
        )
        ax.set_xlabel("Rank based on Explanation Share")
        ax.set_ylabel((titles or {}).get(column, column))
        finite = values[np.isfinite(values)]
        use_log = finite.size > 0 and finite.min() > 0 if log_scale == "try" else bool(log_scale)
        if use_log:
            ax.set_yscale("log")
            # The default log-scale formatter switches to "N x 10^k" scientific
            # notation whenever a panel's range does not span a full decade (every
            # column here regularly doesn't) -- plain numbers read better and are what
            # every other panel in this package already uses.
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    for ax in flat[len(columns):]:
        fig.delaxes(ax)

    if not remove_vanished:
        handles, labels = flat[0].get_legend_handles_labels()
        # "outside ..." is constrained-layout-aware -- it reserves real figure space for
        # the legend, unlike a raw bbox_to_anchor outside the axes grid, which
        # constrained_layout knows nothing about and can end up squeezing into the last
        # panel, overlapping its own y-axis label.
        fig.legend(handles, labels, loc="outside center right", frameon=False)

    return fig, flat[: len(columns)]


def _ordered_latent_matrix(
    values: pd.DataFrame,
    categories: pd.Series,
    dim_stats: pd.DataFrame,
    *,
    title_col: str,
    order_col: str,
    order: Literal["rank", "cluster"],
    method: str,
    remove_vanished: bool,
    balance: int | None,
    seed: int,
) -> tuple[np.ndarray, list[str], list[str], list[tuple[str, int, int]]]:
    """Column order, row grouping and the cells x dims matrix behind a latent heatmap.

    Used by :func:`latent_heatmap_with_heritability`, whose heritability bar panel needs
    exactly this computation exposed separately so the bar is guaranteed to describe the
    same dims in the same order as the heatmap under it. :func:`latent_heatmap` itself
    does not use this -- see its own docstring for why it is a separate, `embed`-based
    implementation.
    """
    import pandas as pd

    from scads_drvi.pl.umap import subsample

    for column in (order_col, title_col):
        if column not in dim_stats.columns:
            raise KeyError(f"{column!r} is not a column of dim_stats")
    if remove_vanished and "vanished" not in dim_stats.columns:
        raise KeyError('"vanished" is not a column of dim_stats')
    if order not in ("rank", "cluster"):
        raise ValueError(f"order must be 'rank' or 'cluster', got {order!r}")

    shared = values.index.intersection(categories.index)
    if shared.empty:
        raise ValueError("values and categories share no index")
    values = values.loc[shared]
    categories = categories.loc[shared].astype(str)

    dims = dim_stats
    if remove_vanished:
        dims = dims.loc[~dims["vanished"].astype(bool)]
    missing = [d for d in dims.index if d not in values.columns]
    if missing:
        raise KeyError(f"dim_stats names dims not present in values: {missing}")
    values = values[list(dims.index)]

    if order == "rank":
        dims_ordered = list(dims.sort_values(order_col).index)
    else:
        correlation = values.corr().to_numpy(dtype=float)
        positions = _cluster_order(correlation, method=method)
        dims_ordered = [values.columns[i] for i in positions]

    cell_frame = pd.DataFrame({"category": categories})
    if balance is not None:
        n_groups = cell_frame["category"].nunique()
        cell_frame = subsample(cell_frame, n=balance * n_groups, seed=seed, stratify="category")
    cell_frame = cell_frame.sort_values("category", kind="stable")
    row_order = cell_frame.index

    blocks: list[tuple[str, int, int]] = []
    start = 0
    for name, group in cell_frame.groupby("category", sort=False, observed=True):
        stop = start + len(group)
        blocks.append((str(name), start, stop))
        start = stop

    image = values.loc[row_order, dims_ordered].to_numpy(dtype=float)
    titles = dim_stats.loc[dims_ordered, title_col].astype(str).tolist()
    return image, dims_ordered, titles, blocks


def _draw_latent_heatmap(ax: Axes, image: np.ndarray, titles: list[str], blocks, *, cmap: str):
    """The imshow + shading + block labels shared by both heatmap-drawing entry points."""
    from scads_drvi.pl.color import robust_norm

    for slot, (_, span_start, span_stop) in enumerate(blocks):
        if slot % 2 == 0:
            ax.axhspan(span_start - 0.5, span_stop - 0.5, color="#F2F2F2", zorder=0)

    norm = robust_norm(image, symmetric=True)
    im = ax.imshow(image, cmap=cmap, norm=norm, aspect="auto", zorder=2)
    ax.set_xticks(np.arange(len(titles)))
    ax.set_xticklabels(titles, rotation=90, fontsize=6)
    ax.set_yticks([])
    ax.set_ylabel("cells (grouped by category)")

    for name, span_start, span_stop in blocks:
        ax.text(
            len(titles) - 0.4, (span_start + span_stop - 1) / 2.0, name,
            va="center", ha="left", fontsize=7,
        )
    return im


def _balanced_rows(categories: pd.Series, *, min_count: int = 10, seed: int = 0) -> np.ndarray:
    """DRVI's own ``make_balanced_subsample``, ported exactly: every category
    contributes the same count -- the smallest category's size, floored at
    `min_count` -- via ``groupby(...).sample(random_state=seed, replace=...)`` (DRVI's
    own mechanism, hardcoded to ``random_state=0``; kept configurable here), not a
    separately-seeded draw, so the rows actually selected match DRVI's own output.
    """
    n_sample_per_cond = int(categories.value_counts().min())
    n = max(min_count, n_sample_per_cond)
    return (
        categories.groupby(categories, observed=True)
        .sample(n=n, random_state=seed, replace=n_sample_per_cond < n)
        .index.to_numpy()
    )


def latent_heatmap(
    embed,
    categorical_column: str,
    *,
    title_col: str | None = "title",
    order_col: str | None = "order",
    sort_by_categorical: bool = False,
    order: Literal["rank", "cluster"] = "rank",
    method: str = "average",
    make_balanced: bool = True,
    remove_vanished: bool = True,
    cmap=None,
    figsize: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[Figure, Axes]:
    """Heatmap of latent dimensions (columns) x cells (rows), grouped by `categorical_column`.

    Ported to call ``scanpy.pl.heatmap`` directly, the same way DRVI's own
    ``drvi.utils.pl.plot_latent_dims_in_heatmap`` does -- same `embed` input, parameter
    names and defaults (`title_col`, `order_col`, `sort_by_categorical`,
    `make_balanced`, `remove_vanished`), DRVI's own ``SaturatedRdBu`` colour scale
    (`cmap`, default :data:`scads_drvi.pl.color.SATURATED_RED_BLUE_CMAP`) centred at 0
    and no dendrogram, so it is a comfortable swap-in -- not a hand-rolled
    ``seaborn.heatmap`` draw with its own category shading/colourbar. `seed` defaults
    to 0, matching DRVI's own hardcoded ``make_balanced_subsample`` seed (see
    :func:`_balanced_rows`), so the balanced subsample actually drawn is the same one
    DRVI's own call draws.

    `sort_by_categorical=True` reproduces DRVI's own heuristic exactly (each dimension
    ordered by which cell holds its largest-magnitude value). `order="cluster"`
    (ignored when `sort_by_categorical=True`) is this project's own addition instead of
    DRVI's default rank ordering: hierarchical clustering of the dimensions by
    correlation, reusing :func:`_cluster_order` (the same clustering
    :func:`factor_correlation` uses) -- unlike `sort_by_categorical`, it does not depend
    on the incidental fact that a balanced subsample happens to come out
    category-blocked.

    Returns `scanpy`'s own ``"heatmap_ax"`` as `ax`; its own category color bar
    (`"groupby_ax"`) lives on the same `fig` alongside it.
    """
    import scanpy as sc

    from scads_drvi.pl.color import SATURATED_RED_BLUE_CMAP

    cmap = SATURATED_RED_BLUE_CMAP if cmap is None else cmap

    if order_col is not None and order_col not in embed.var.columns:
        raise KeyError(f"{order_col!r} is not a column of embed.var")
    if categorical_column not in embed.obs.columns:
        raise KeyError(f"{categorical_column!r} is not a column of embed.obs")
    if remove_vanished:
        if "vanished" not in embed.var.columns:
            raise KeyError('"vanished" is not a column of embed.var')
        embed = embed[:, ~embed.var["vanished"].to_numpy(dtype=bool)]

    if make_balanced:
        rows = _balanced_rows(embed.obs[categorical_column].astype(str), seed=seed)
        embed = embed[rows]

    x = np.asarray(embed.X.toarray() if hasattr(embed.X, "toarray") else embed.X, dtype=float)

    if sort_by_categorical:
        dim_order = np.argsort(np.abs(x).argmax(axis=0))
    elif order == "cluster":
        dim_order = _cluster_order(np.corrcoef(x, rowvar=False), method=method)
    elif order_col is not None:
        dim_order = np.argsort(embed.var[order_col].to_numpy())
    else:
        dim_order = np.arange(x.shape[1])

    ordered_var = embed.var.iloc[dim_order]
    vars_to_show = list(
        ordered_var.index if title_col is None else ordered_var[title_col].astype(str)
    )

    if figsize is None:
        figsize = (10, len(embed.obs[categorical_column].unique()) / 6)

    axes = sc.pl.heatmap(
        embed, vars_to_show, categorical_column, gene_symbols=title_col,
        figsize=figsize, show_gene_labels=True, show=False,
        vcenter=0, cmap=cmap, dendrogram=False,
    )
    return axes["heatmap_ax"].figure, axes["heatmap_ax"]


def latent_heatmap_with_heritability(
    values: pd.DataFrame,
    categories: pd.Series,
    dim_stats: pd.DataFrame,
    heritability: pd.Series,
    *,
    heritability_q: pd.Series | None = None,
    heritability_label: str = "coefficient z",
    alpha: float = 0.05,
    title_col: str = "title",
    order_col: str = "order",
    order: Literal["rank", "cluster"] = "rank",
    method: str = "average",
    remove_vanished: bool = True,
    balance: int | None = None,
    cmap=None,
    seed: int = 42,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """:func:`latent_heatmap`, with a per-dim heritability bar above it, same column order.

    Pairs the interpretability heatmap with an actually complementary panel: which cells
    drive each factor (below) against how heritability-enriched that same factor is
    (above) -- rather than :func:`scads_drvi.pl.enrichment.heritability_landscape`'s bar
    + volcano, whose two panels collapse to the same ranking whenever the volcano's x is
    already a z-score (its p comes from that same z one-tailed, so -log10(q) is just a
    monotone function of it; the panels differ only when the x-axis carries information
    the significance test doesn't, e.g. an unstandardised effect size).

    `heritability` is a per-dim Series indexed like `dim_stats` -- one value per factor,
    already reduced to whichever direction or summary the caller wants (e.g.
    ``results.query("direction == 'pos'").set_index("dim")["Coefficient_z-score"]``);
    this function never derives it and never resolves a pos/neg split itself. It is
    reindexed to the heatmap's own column order, so the two panels always describe the
    same factors in the same order regardless of what order `heritability`'s index came
    in. Pass `heritability_q` (same index) to colour bars by the BH-significance ramp
    :func:`scads_drvi.pl.enrichment.heritability_landscape` uses and draw its boundary
    line; omit it for a single plain-coloured bar with no significance lines, when no
    q-value is available for this `heritability`.
    """
    import matplotlib.pyplot as plt

    from scads_drvi.pl.color import (
        DEFAULT_RAMP,
        SATURATED_RED_BLUE_CMAP,
        Z_HIGH_CONFIDENCE,
        Z_NOMINAL_ONE_TAILED,
        add_threshold_lines,
        significance_colors,
        significance_handles,
    )
    from scads_drvi.stats import bh_threshold_z

    cmap = SATURATED_RED_BLUE_CMAP if cmap is None else cmap

    image, dims_ordered, titles, blocks = _ordered_latent_matrix(
        values, categories, dim_stats,
        title_col=title_col, order_col=order_col, order=order, method=method,
        remove_vanished=remove_vanished, balance=balance, seed=seed,
    )

    missing = [d for d in dims_ordered if d not in heritability.index]
    if missing:
        raise KeyError(
            f"heritability has no value for {len(missing)} of the heatmap's factor(s), "
            f"e.g. {missing[:5]}"
        )
    z = heritability.loc[dims_ordered].to_numpy(dtype=float)

    q = None
    if heritability_q is not None:
        missing_q = [d for d in dims_ordered if d not in heritability_q.index]
        if missing_q:
            raise KeyError(
                f"heritability_q has no value for {len(missing_q)} of the heatmap's "
                f"factor(s), e.g. {missing_q[:5]}"
            )
        q = heritability_q.loc[dims_ordered].to_numpy(dtype=float)

    heatmap_height = max(3.0, min(0.02 * image.shape[0], 14.0))
    bar_height = 1.8
    width = max(4.0, 0.15 * len(dims_ordered)) + 1.5
    fig, (bar, heat) = plt.subplots(
        2, 1, figsize=(width, bar_height + heatmap_height), sharex=True,
        gridspec_kw={"height_ratios": [bar_height, heatmap_height]},
    )

    positions = np.arange(len(dims_ordered))
    colors = significance_colors(q, ramp=DEFAULT_RAMP) if q is not None else "#0173B2"
    bar.bar(positions, z, color=colors, width=0.9, zorder=2)
    bar.set_ylabel(heritability_label)
    bar.set_xlim(-0.5, len(dims_ordered) - 0.5)
    plt.setp(bar.get_xticklabels(), visible=False)
    if q is not None:
        boundary = bh_threshold_z(q, z, alpha=alpha)
        add_threshold_lines(
            bar, z=(Z_NOMINAL_ONE_TAILED, Z_HIGH_CONFIDENCE), axis="y",
            bh=boundary, alpha=alpha,
        )
        bar.legend(
            handles=significance_handles(DEFAULT_RAMP), frameon=False, fontsize=6,
            loc="upper right",
        )

    im = _draw_latent_heatmap(heat, image, titles, blocks, cmap=cmap)
    fig.colorbar(im, ax=(bar, heat), pad=0.02, fraction=0.03 / 2).set_label("value")
    fig.subplots_adjust(hspace=0.05)
    return fig, (bar, heat)
