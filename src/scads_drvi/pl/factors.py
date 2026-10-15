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
a figure that looks like DRVI's own. :func:`latent_heatmap` also takes any number of
marginal bar plots (:class:`BarSeries`) on its top, right and left edges -- e.g. a per-factor
heritability bar sharing its column order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from scads_drvi.pl.frugal import BoxStats

#: The linkage methods `scipy.cluster.hierarchy.linkage` accepts -- shared by every
#: `method` parameter in this module that eventually reaches :func:`_cluster_order`, so
#: an invalid method is caught at the call site rather than inside scipy.
LinkageMethod = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]

__all__ = [
    "factor_distributions",
    "factor_correlation",
    "latent_dimension_stats",
    "latent_heatmap",
    "BarSeries",
    "LatentHeatmapAxes",
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


def _cluster_order(matrix: np.ndarray, *, method: LinkageMethod = "average") -> np.ndarray:
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


def _correlation_panel(
    ax: Axes, correlation: pd.DataFrame, *, dead: set[str], method: LinkageMethod, cmap: str
) -> Any:
    """Draw one clustered correlation matrix on `ax`; a trailing +/- is ignored when matching `dead`."""
    if correlation.shape[0] != correlation.shape[1]:
        raise ValueError(f"expected a square matrix, got {correlation.shape}")

    matrix = np.nan_to_num(correlation.to_numpy(dtype=float), nan=0.0)
    order = _cluster_order(matrix, method=method)
    labels = correlation.index.astype(str).to_numpy()[order]
    image = ax.imshow(matrix[np.ix_(order, order)], cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)
    for position, label in enumerate(labels):
        if label in dead or label.rstrip("+-") in dead:
            ax.get_xticklabels()[position].set_color("#D55E00")
            ax.get_yticklabels()[position].set_color("#D55E00")
    return image


def factor_correlation(
    correlation: pd.DataFrame,
    *,
    split: pd.DataFrame | None = None,
    vanished: Sequence[str] = (),
    method: LinkageMethod = "average",
    cmap: str = "RdBu_r",
) -> tuple[Figure, Axes | tuple[Axes, Axes]]:
    """Clustered factor-factor correlation, with dead factors greyed rather than hidden.

    Removing vanished factors would make the matrix look cleaner and quietly change what
    fraction of the model the figure describes. Pass the correlation of the directional
    arms (``"DR 5+"``/``"DR 5-"``) as `split` for a second panel: `correlation` (signed
    factors) on the left, `split` on the right, each clustered on its own. The return is
    then ``(fig, (ax_signed, ax_split))``; without `split` it is ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    dead = {str(v) for v in vanished}
    panels = [correlation] if split is None else [correlation, split]
    size = max(4.0, 0.11 * max(len(p) for p in panels))
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(len(panels) * size + 1.2, size),
        squeeze=False,
        layout="constrained",
    )
    for ax, panel in zip(axes[0], panels, strict=True):
        image = _correlation_panel(ax, panel, dead=dead, method=method, cmap=cmap)
    fig.colorbar(image, ax=list(axes[0]), pad=0.02, fraction=0.045 / len(panels)).set_label(
        "Pearson r"
    )
    if split is None:
        return fig, axes[0, 0]
    axes[0, 0].set_title("signed factors", fontsize=8)
    axes[0, 1].set_title("split (+/-) arms", fontsize=8)
    return fig, (axes[0, 0], axes[0, 1])


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, column in zip(flat, columns, strict=False):
        values = frame[column].to_numpy(dtype=float)
        ax.plot(rank, values, "-", color="grey", linewidth=0.8, zorder=1)
        ax.scatter(
            rank[~vanished],
            values[~vanished],
            color=palette["kept"],
            s=10,
            zorder=2,
            label="kept",
        )
        ax.scatter(
            rank[vanished],
            values[vanished],
            color=palette["vanished"],
            s=10,
            zorder=2,
            label="vanished",
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

    for ax in flat[len(columns) :]:
        fig.delaxes(ax)

    if not remove_vanished:
        handles, labels = flat[0].get_legend_handles_labels()
        # "outside ..." is constrained-layout-aware -- it reserves real figure space for
        # the legend, unlike a raw bbox_to_anchor outside the axes grid, which
        # constrained_layout knows nothing about and can end up squeezing into the last
        # panel, overlapping its own y-axis label.
        fig.legend(handles, labels, loc="outside right center", frameon=False)

    return fig, flat[: len(columns)]


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


def _directional_split(embed: AnnData, *, title_col: str | None, order_col: str | None) -> AnnData:
    """Every kept dimension becomes two columns, ``"{dim}+"``/``"{dim}-"`` -- a ReLU'd
    positive part and a ReLU'd negative part (as a positive magnitude), zero elsewhere.

    The same directional split DRVI's own ``get_effect_of_splits_within_distribution``
    produces and this package's own `enrich`/`scores` modules already build by hand
    (``np.clip(embed.X, 0, None)``/``np.clip(-embed.X, 0, None)``) -- not
    :func:`scads_drvi.pl.umap.latent_umap_grid`'s directional split, which negates the
    *whole* column for its own colour-scale reasons rather than zeroing out either half;
    a heatmap's diverging colour already encodes sign, so the useful new information
    here is which cells actually carry each direction's loading, not a recoloured copy
    of the same values.
    """
    import anndata as ad
    import pandas as pd
    from scipy.sparse import _base as sparse_base

    # `.obs`/`.var` are typed `DataFrame | Dataset2D` (scvi-tools' lazy/backed frame) in
    # current anndata stubs; this function (like the rest of this module) only ever
    # handles an ordinary in-memory AnnData.
    var, obs = embed.var, embed.obs
    if not isinstance(var, pd.DataFrame) or not isinstance(obs, pd.DataFrame):
        raise TypeError(
            "latent_heatmap's directional split needs an in-memory AnnData "
            "(pandas .obs/.var); a lazy/backed AnnData (e.g. scvi-tools' Dataset2D) is "
            "not supported here."
        )

    x_raw = embed.X
    if x_raw is None:
        raise ValueError("embed.X must not be None")
    x = np.asarray(
        x_raw.toarray() if isinstance(x_raw, sparse_base._spbase) else x_raw,
        dtype=float,
    )
    pos_x, neg_x = np.clip(x, 0, None), np.clip(-x, 0, None)

    pos_var, neg_var = var.copy(), var.copy()
    pos_var.index = pd.Index([f"{i}+" for i in embed.var_names])
    neg_var.index = pd.Index([f"{i}-" for i in embed.var_names])
    if title_col is not None and title_col in var.columns:
        pos_var[title_col] = pos_var[title_col].astype(str) + "+"
        neg_var[title_col] = neg_var[title_col].astype(str) + "-"
    if order_col is not None and order_col in var.columns:
        pos_var[order_col] = pos_var[order_col] + 1e-8
        neg_var[order_col] = neg_var[order_col] - 1e-8

    return ad.AnnData(
        X=np.concatenate([pos_x, neg_x], axis=1),
        obs=obs,
        var=pd.concat([pos_var, neg_var]),
    )


@dataclass(frozen=True)
class BarSeries:
    """One marginal bar plot of a :func:`latent_heatmap`.

    `values` is a Series. In ``column_series`` it holds one value per heatmap column, keyed
    by the column's var name or its displayed title (``"DR 5+"``). In ``row_series`` and
    ``row_series_left`` it holds one value per category of the heatmap's
    ``categorical_column`` -- or, with `agg`, one value per cell (indexed like
    ``embed.obs_names``), reduced per category by that statistic.

    `q` (same index as `values`) colours the bars by BH significance and draws the
    boundary line at level `alpha`, as :func:`scads_drvi.pl.enrichment.heritability_landscape`
    does. `se` (same index) draws error bars; with `agg` the standard error is computed from
    the cells instead -- of the mean or median, or of the sum (``sqrt(n) * std``, the
    un-normalised total rather than a per-cell average). `color` is a fixed colour, a mapping or
    Series of colours keyed like `values` (a column series: var name or title; a row series:
    category), or a matplotlib colormap that colours each bar by its value; for a row series
    the default is the category palette (``"group"``). A column series with `q` is coloured by
    significance, so it cannot also take `color`. `size_in` is the bar panel's thickness in
    inches. `name` only labels error messages.
    """

    values: pd.Series
    label: str = ""
    se: pd.Series | None = None
    q: pd.Series | None = None
    color: str | Mapping[str, str] | pd.Series | Colormap | None = None
    agg: Literal["mean", "median", "sum"] | None = None
    size_in: float = 1.0
    alpha: float = 0.05
    name: str = ""


class LatentHeatmapAxes(NamedTuple):
    """The axes of a :func:`latent_heatmap` that was given bar series, in the order given."""

    heat: Axes
    columns: list[Axes]
    rows_left: list[Axes]
    rows_right: list[Axes]


_BAR_GAP_IN = 0.08
_FONT_PT = 7  # one text size for every label, tick and legend of a latent_heatmap figure
_AGG_FUNCTIONS: dict[str, Callable[[np.ndarray], Any]] = {
    "mean": np.mean,
    "median": np.median,
    "sum": np.sum,
}


def _align(series: pd.Series, keys: Sequence[Any], *, what: str, unit: str) -> np.ndarray:
    """`series` reindexed to `keys` as floats, or a ``KeyError`` naming what is missing."""
    missing = [k for k in keys if k not in series.index]
    if missing:
        raise KeyError(
            f"{what} has no value for {len(missing)} of the heatmap's {unit}(s), e.g. {missing[:5]}"
        )
    return series.loc[list(keys)].to_numpy(dtype=float)


def _align_columns(
    series: pd.Series, ordered_var: pd.DataFrame, title_col: str | None, *, what: str
) -> np.ndarray:
    """`series` in the heatmap's column order, keyed by var name or by displayed title."""
    keys = list(ordered_var.index)
    if title_col is not None and not all(k in series.index for k in keys):
        titles = ordered_var[title_col].astype(str).tolist()
        if all(t in series.index for t in titles):
            return series.loc[titles].to_numpy(dtype=float)
    return _align(series, keys, what=what, unit="factor")


def _bar_colors(
    color: Any,
    keys: Sequence[Any],
    values: np.ndarray,
    *,
    default: Any,
    alt_keys: Sequence[Any] | None = None,
    name: str = "",
) -> Any:
    """Per-bar colours from a BarSeries `color`: a colour, a key -> colour mapping, or a colormap."""
    from matplotlib.colors import Colormap, Normalize

    if color is None or (isinstance(color, str) and color == "group"):
        return default
    if isinstance(color, str):
        return color
    if isinstance(color, Colormap):
        finite = values[np.isfinite(values)]
        norm = Normalize(finite.min(), finite.max()) if finite.size else Normalize(0, 1)
        return [color(norm(v)) if np.isfinite(v) else "#BBBBBB" for v in values]
    lookup = dict(color)
    out, missing = [], []
    for i, key in enumerate(keys):
        for candidate in (key, None if alt_keys is None else alt_keys[i]):
            if candidate in lookup:
                out.append(lookup[candidate])
                break
        else:
            missing.append(key)
    if missing:
        raise KeyError(
            f"{name or 'color'} has no colour for {len(missing)} bar(s), e.g. {missing[:5]}"
        )
    return out


def _unify_fonts(fig: Figure) -> None:
    """Give every tick label, axis label and legend entry of `fig` the same size (scanpy's own
    heatmap axes use their own defaults, which otherwise differ from the marginal bars')."""
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=_FONT_PT)
        ax.xaxis.label.set_fontsize(_FONT_PT)
        ax.yaxis.label.set_fontsize(_FONT_PT)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(_FONT_PT)


def _series_name(series: BarSeries, kind: str, i: int) -> str:
    return series.name or series.label or f"{kind}[{i}]"


def _per_category(
    series: BarSeries,
    obs: pd.DataFrame,
    categorical_column: str,
    cat_order: Sequence[str],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """A row series' value (and error) per category, in `cat_order`."""
    if series.agg is None:
        values = _align(series.values, cat_order, what=name, unit="category")
        se = None
        if series.se is not None:
            se = _align(series.se, cat_order, what=f"{name}_se", unit="category")
        return values, se

    if series.agg not in _AGG_FUNCTIONS:
        raise ValueError(
            f"agg (cell_score_agg) must be 'mean', 'median' or 'sum', got {series.agg!r}"
        )
    cells = _align(series.values, list(obs.index), what=name, unit="cell")
    cats = obs[categorical_column].astype(str).to_numpy()
    groups = [cells[cats == cat] for cat in cat_order]
    agg = _AGG_FUNCTIONS[series.agg]
    values = np.array([agg(g) if len(g) else np.nan for g in groups])

    # Standard error of the mean for "mean"/"median" (an average estimate); of the *sum* --
    # sqrt(n) * std, not std / sqrt(n) -- for "sum", the un-normalised total a group's cells
    # contribute rather than a per-cell average.
    def _factor(n: int) -> float:
        return float(np.sqrt(n)) if series.agg == "sum" else 1.0 / float(np.sqrt(n))

    se = np.array([g.std(ddof=1) * _factor(len(g)) if len(g) > 1 else 0.0 for g in groups])
    return values, se


def _specificity_order(
    var: pd.DataFrame, specificity: pd.Series | None, title_col: str | None
) -> np.ndarray:
    """Column positions sorted by descending `specificity` (NaN last); ties keep var order."""
    if specificity is None:
        raise ValueError(
            'order="specificity" needs `specificity`, a Series of one score per column'
        )
    score = _align_columns(specificity, var, title_col, what="specificity")
    return np.argsort(-np.where(np.isnan(score), -np.inf, score), kind="stable")


def _draw_column_bar(
    fig: Figure,
    heat: Axes,
    rect: tuple[float, float, float, float],
    series: BarSeries,
    values: np.ndarray,
    *,
    se: np.ndarray | None,
    q: np.ndarray | None,
    colors: Any = None,
) -> Axes:
    from matplotlib.ticker import MaxNLocator

    from scads_drvi.pl.color import (
        DEFAULT_RAMP,
        Z_HIGH_CONFIDENCE,
        Z_NOMINAL_ONE_TAILED,
        add_threshold_lines,
        significance_colors,
        significance_handles,
    )
    from scads_drvi.stats import bh_threshold_z

    bar = fig.add_axes(rect, sharex=heat)
    if q is not None:
        colors = significance_colors(q, ramp=DEFAULT_RAMP)
    elif colors is None:
        colors = "#0173B2"
    bar.bar(
        np.arange(len(values)),
        values,
        yerr=se,
        color=colors,
        width=0.9,
        zorder=2,
        error_kw={"ecolor": "0.2", "elinewidth": 0.8, "capsize": 2},
    )
    bar.axhline(0, color="0.3", lw=0.6, zorder=1)
    for spine in ("top", "right", "bottom"):
        bar.spines[spine].set_visible(False)
    bar.tick_params(axis="x", bottom=False, labelbottom=False)
    bar.tick_params(axis="y", labelsize=_FONT_PT, length=2)
    bar.yaxis.set_major_locator(MaxNLocator(nbins=3))
    bar.set_ylabel(series.label, fontsize=_FONT_PT)
    bar.margins(x=0)
    if q is not None:
        add_threshold_lines(
            bar,
            z=(Z_NOMINAL_ONE_TAILED, Z_HIGH_CONFIDENCE),
            axis="y",
            bh=bh_threshold_z(q, values, alpha=series.alpha),
            alpha=series.alpha,
        )
        # Outside the axes, right of the last column: inside it covers the tallest bars.
        bar.legend(
            handles=significance_handles(DEFAULT_RAMP),
            frameon=False,
            fontsize=_FONT_PT,
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
            borderaxespad=0,
        )
    return bar


def _draw_row_bar(
    fig: Figure,
    heat: Axes,
    rect: tuple[float, float, float, float],
    series: BarSeries,
    values: np.ndarray,
    *,
    se: np.ndarray | None,
    mids: np.ndarray,
    heights: np.ndarray,
    colors: Any,
    left: bool,
) -> Axes:
    from matplotlib.ticker import MaxNLocator

    bar = fig.add_axes(rect, sharey=heat)
    bar.barh(
        mids,
        values,
        xerr=se,
        height=heights,
        color=colors,
        zorder=2,
        error_kw={"ecolor": "0.2", "elinewidth": 0.8, "capsize": 2},
    )
    bar.axvline(0, color="0.3", lw=0.6, zorder=1)
    for spine in ("top", "left", "right"):
        bar.spines[spine].set_visible(False)
    bar.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    bar.tick_params(axis="x", labelsize=_FONT_PT, length=2)
    bar.xaxis.set_major_locator(MaxNLocator(nbins=3))
    bar.set_xlabel(series.label, fontsize=_FONT_PT)
    bar.margins(y=0)
    if left:
        bar.invert_xaxis()  # positive values grow away from the heatmap
    return bar


def latent_heatmap(
    embed: AnnData,
    categorical_column: str,
    *,
    title_col: str | None = "title",
    order_col: str | None = "order",
    sort_by_categorical: bool = False,
    order: Literal["rank", "cluster", "specificity"] = "rank",
    specificity: pd.Series | None = None,
    method: LinkageMethod = "average",
    make_balanced: bool = True,
    remove_vanished: bool = True,
    directional: bool = True,
    aggregate: Literal["cells", "mean"] = "cells",
    cmap: Colormap | str | None = None,
    directional_cmap: Colormap | str | None = None,
    figsize: tuple[float, float] | None = None,
    seed: int = 0,
    column_series: Sequence[BarSeries] = (),
    row_series: Sequence[BarSeries] = (),
    row_series_left: Sequence[BarSeries] = (),
    heritability: pd.Series | None = None,
    heritability_se: pd.Series | None = None,
    heritability_q: pd.Series | None = None,
    heritability_label: str = "coefficient z",
    alpha: float = 0.05,
    cell_scores: pd.Series | None = None,
    cell_score_agg: Literal["mean", "median", "sum"] = "mean",
    cell_score_label: str = "score",
) -> tuple[Figure, Axes | LatentHeatmapAxes | tuple[Axes, ...]]:
    """Heatmap of latent dimensions (columns) x cells (rows), grouped by `categorical_column`.

    Calls ``scanpy.pl.heatmap`` directly, the same way DRVI's own
    ``drvi.utils.pl.plot_latent_dims_in_heatmap`` does -- same `embed` input, parameter
    names and defaults (`title_col`, `order_col`, `sort_by_categorical`, `make_balanced`,
    `remove_vanished`), DRVI's own ``SaturatedRdBu`` colour scale (`cmap`) centred at 0 and no
    dendrogram. `seed` defaults to 0, matching DRVI's own hardcoded ``make_balanced_subsample``
    seed (see :func:`_balanced_rows`).

    Column order. `order="rank"` is DRVI's explained-share ranking. `order="cluster"` clusters
    the columns by correlation (:func:`_cluster_order`, as :func:`factor_correlation` does).
    `order="specificity"` sorts them by descending `specificity`, a Series of one score per
    column keyed by var name or displayed title (e.g. the ``specificity`` column of
    :func:`scads_drvi.specificity.graph_specificity`, indexed by
    ``split_directions(..., naming="signed")``'s names). `sort_by_categorical=True` instead
    reproduces DRVI's heuristic (each column ordered by which cell holds its largest value) and
    overrides `order`.

    `directional=True` (the default) splits every kept dimension into two columns,
    ``"{title}+"``/``"{title}-"``: a ReLU'd positive part (``np.clip(X, 0, None)``) and a ReLU'd
    negative part (``np.clip(-X, 0, None)``), the split DRVI's own
    ``get_effect_of_splits_within_distribution(..., directional=True)`` uses. These values are
    never negative, so the heatmap uses `directional_cmap` (a one-sided ramp from 0) instead of
    the diverging `cmap`. Pass `directional=False` for one column per factor.

    `aggregate="mean"` draws one row per category -- each cell of the heatmap is that category's
    mean loading on that column, over all its cells (`make_balanced` is ignored) -- instead of one
    row per cell.

    Bar series. Each :class:`BarSeries` is a marginal bar plot sharing an axis with the heatmap.
    `column_series` are stacked upward from the top edge (one value per column);
    `row_series` are stacked to the right and `row_series_left` to the left (one value per
    category, or one per cell with ``agg=``). Any number of each; the first in each list is
    nearest the heatmap. Per-cell series are reduced over the cells actually drawn (the balanced
    subsample, when `make_balanced`), or over all cells with ``aggregate="mean"``.

    Returns `ax` (``scanpy``'s ``"heatmap_ax"``) alone when no series is given, or a
    :class:`LatentHeatmapAxes` when any of the three lists is. The old arguments `heritability`
    (+ `_se`, `_q`, `_label`, `alpha`) and `cell_scores` (+ `_agg`, `_label`) still work for one
    release: they build a column and a row :class:`BarSeries` and warn. Used alone they keep the
    old return shape: ``(bar, ax)``, ``(ax, score)`` or ``(bar, ax, score)``.
    """
    import warnings

    import pandas as pd
    import scanpy as sc
    from scipy.sparse import _base as sparse_base

    from scads_drvi.pl.color import SATURATED_JUST_SKY_CMAP, SATURATED_RED_BLUE_CMAP

    if order not in ("rank", "cluster", "specificity"):
        raise ValueError(f"order must be 'rank', 'cluster' or 'specificity', got {order!r}")
    if aggregate not in ("cells", "mean"):
        raise ValueError(f"aggregate must be 'cells' or 'mean', got {aggregate!r}")
    cmap = SATURATED_RED_BLUE_CMAP if cmap is None else cmap
    directional_cmap = SATURATED_JUST_SKY_CMAP if directional_cmap is None else directional_cmap

    new_style = bool(column_series or row_series or row_series_left)
    column_series, row_series = list(column_series), list(row_series)
    row_series_left = list(row_series_left)
    if heritability is not None:
        warnings.warn(
            "`heritability*` is deprecated: pass `column_series=[BarSeries(...)]`",
            DeprecationWarning,
            stacklevel=2,
        )
        column_series.insert(
            0,
            BarSeries(
                heritability,
                label=heritability_label,
                se=heritability_se,
                q=heritability_q,
                alpha=alpha,
                name="heritability",
            ),
        )
    if cell_scores is not None:
        warnings.warn(
            "`cell_scores*` is deprecated: pass `row_series=[BarSeries(..., agg=...)]`",
            DeprecationWarning,
            stacklevel=2,
        )
        row_series.insert(
            0,
            BarSeries(
                cell_scores,
                label=cell_score_label,
                agg=cell_score_agg,
                name="cell_scores",
            ),
        )
    has_series = bool(column_series or row_series or row_series_left)

    # `.obs`/`.var` are typed `DataFrame | Dataset2D` (scvi-tools' lazy/backed frame) in
    # current anndata stubs; this function only ever handles an ordinary in-memory
    # AnnData, so a non-DataFrame here is a genuine usage error.
    def _as_frame(obj: object, what: str) -> pd.DataFrame:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"{what} must be a pandas DataFrame; a lazy/backed AnnData (e.g. "
                "scvi-tools' Dataset2D) is not supported here."
            )
        return obj

    var = _as_frame(embed.var, "embed.var")
    obs = _as_frame(embed.obs, "embed.obs")
    if order_col is not None and order_col not in var.columns:
        raise KeyError(f"{order_col!r} is not a column of embed.var")
    if categorical_column not in obs.columns:
        raise KeyError(f"{categorical_column!r} is not a column of embed.obs")
    if remove_vanished:
        if "vanished" not in var.columns:
            raise KeyError('"vanished" is not a column of embed.var')
        embed = embed[:, ~var["vanished"].to_numpy(dtype=bool)]

    if directional:
        embed = _directional_split(embed, title_col=title_col, order_col=order_col)

    if make_balanced and aggregate == "cells":
        obs = _as_frame(embed.obs, "embed.obs")
        rows = _balanced_rows(obs[categorical_column].astype(str), seed=seed)
        embed = embed[rows]

    x_raw = embed.X
    if x_raw is None:
        raise ValueError("embed.X must not be None")
    x = np.asarray(
        x_raw.toarray() if isinstance(x_raw, sparse_base._spbase) else x_raw,
        dtype=float,
    )

    var = _as_frame(embed.var, "embed.var")
    obs = _as_frame(embed.obs, "embed.obs")

    col = obs[categorical_column]
    if isinstance(col.dtype, pd.CategoricalDtype):
        cat_order = [str(c) for c in col.cat.categories]
    else:
        cat_order = sorted(col.astype(str).unique())

    # Per-category values of every row series, from the cells as drawn -- BEFORE a mean
    # collapse replaces them by one row per category.
    row_values: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    for tag, group in (("row_series", row_series), ("row_series_left", row_series_left)):
        for i, s in enumerate(group):
            row_values[id(s)] = _per_category(
                s, obs, categorical_column, cat_order, name=_series_name(s, tag, i)
            )

    if sort_by_categorical:
        dim_order = np.argsort(np.abs(x).argmax(axis=0))
    elif order == "cluster":
        dim_order = _cluster_order(np.corrcoef(x, rowvar=False), method=method)
    elif order == "specificity":
        dim_order = _specificity_order(var, specificity, title_col)
    elif order_col is not None:
        dim_order = np.argsort(var[order_col].to_numpy())
    else:
        dim_order = np.arange(x.shape[1])

    ordered_var = var.iloc[dim_order]
    vars_to_show = list(
        ordered_var.index if title_col is None else ordered_var[title_col].astype(str)
    )

    column_values: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Any]] = []
    for i, s in enumerate(column_series):
        name = _series_name(s, "column_series", i)
        if isinstance(s.color, str) and s.color == "group":
            raise ValueError("color='group' is for row series; a column series has no categories")
        if s.q is not None and s.color is not None:
            raise ValueError(
                f"{name}: q colours the bars by significance, so color cannot be set too"
            )
        z = _align_columns(s.values, ordered_var, title_col, what=name)
        se = (
            _align_columns(s.se, ordered_var, title_col, what=f"{name}_se")
            if s.se is not None
            else None
        )
        q = (
            _align_columns(s.q, ordered_var, title_col, what=f"{name}_q")
            if s.q is not None
            else None
        )
        colors = _bar_colors(
            s.color, vars_to_show, z, default=None, alt_keys=list(ordered_var.index), name=name
        )
        column_values.append((z, se, q, colors))

    if aggregate == "mean":
        import anndata as ad

        means = (
            pd.DataFrame(x)
            .groupby(obs[categorical_column].astype(str).to_numpy(), sort=False)
            .mean()
        )
        means = means.loc[cat_order]
        group_colors_in = embed.uns.get(f"{categorical_column}_colors")
        embed = ad.AnnData(
            X=means.to_numpy(dtype=np.float32),
            obs=pd.DataFrame(
                {categorical_column: pd.Categorical(cat_order, categories=cat_order)},
                index=pd.Index(cat_order, name=None),
            ),
            var=var,
        )
        if group_colors_in is not None and len(group_colors_in) == len(cat_order):
            embed.uns[f"{categorical_column}_colors"] = list(group_colors_in)
        obs = _as_frame(embed.obs, "embed.obs")

    if figsize is None:
        n_cat = len(cat_order)
        figsize = (10, max(2.0, 0.4 * n_cat)) if aggregate == "mean" else (10, n_cat / 6)

    # Marginal panels need real inches of their own, not a fraction carved out of
    # `figsize` -- that figsize is sized only for the heatmap itself (as little as a
    # fraction of an inch tall for a handful of categories), so reserving a *fraction*
    # of it would squash both the heatmap and the new panels into nothing.
    def _extent(group: Sequence[BarSeries]) -> float:
        return sum(s.size_in + _BAR_GAP_IN for s in group)

    top_in, right_in, left_in = (
        _extent(column_series),
        _extent(row_series),
        _extent(row_series_left),
    )
    call_figsize = figsize
    if has_series:
        call_w, call_h = figsize
        call_figsize = (call_w + right_in + left_in, max(call_h, 2.0) + top_in)

    heatmap_kwargs = (
        {"cmap": directional_cmap, "vmin": 0} if directional else {"cmap": cmap, "vcenter": 0}
    )
    # Whichever slice above (remove_vanished/make_balanced) ran last may still leave
    # embed a view -- sc.pl.heatmap below mutates it in place (sanitizing `.obs`,
    # then assigning a `.uns` colors key), which warns unless it's an actual AnnData.
    if embed.is_view:
        embed = embed.copy()
    axes = sc.pl.heatmap(
        embed,
        vars_to_show,
        categorical_column,
        gene_symbols=title_col,
        figsize=call_figsize,
        show_gene_labels=True,
        show=False,
        dendrogram=False,
        **heatmap_kwargs,
    )
    heat = axes["heatmap_ax"]
    fig = heat.figure
    if aggregate == "mean":
        # scanpy centres each category label at start + size/2 in edge coordinates, but imshow
        # pixels are centred on integers: with one row per category the labels sit half a row low.
        for a in fig.axes:
            ticks = a.get_yticks()
            if (
                a is not heat
                and len(a.get_yticklabels()) == len(cat_order)
                and len(ticks) == len(cat_order)
            ):
                a.set_yticks(ticks - 0.5, [t.get_text() for t in a.get_yticklabels()])

    if not has_series:
        _unify_fonts(fig)
        return fig, heat

    import matplotlib.pyplot as plt

    adjust: dict[str, float] = {}
    if column_series:
        adjust["top"] = 1 - (top_in + 0.05) / call_figsize[1]
    if row_series:
        adjust["right"] = 1 - right_in / call_figsize[0]
    if row_series_left:
        adjust["left"] = fig.subplotpars.left + left_in / call_figsize[0]
    fig.subplots_adjust(**adjust)
    pos = heat.get_position()
    fig_w, fig_h = call_figsize
    # `scanpy.pl.heatmap` draws its own category colour bar and colourbar in thin axes it never
    # exposes in the returned dict -- start the side panels beyond every axes scanpy made.
    scanpy_axes = list(fig.axes)
    right_edge = max(a.get_position().x1 for a in scanpy_axes)
    left_edge = min(a.get_position().x0 for a in scanpy_axes)
    if row_series_left:
        # The category names are tick labels of scanpy's colour bar: keep the left panels clear
        # of them, measuring the rendered extent rather than the axes box.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        left_edge = min(to_fig.transform_bbox(a.get_tightbbox(renderer)).x0 for a in scanpy_axes)

    column_axes: list[Axes] = []
    y = pos.y1 + _BAR_GAP_IN / fig_h
    for s, (z, se, q, bar_colors) in zip(column_series, column_values, strict=True):
        height = s.size_in / fig_h
        column_axes.append(
            _draw_column_bar(
                fig, heat, (pos.x0, y, pos.width, height), s, z, se=se, q=q, colors=bar_colors
            )
        )
        y += height + _BAR_GAP_IN / fig_h

    group_colors = embed.uns.get(f"{categorical_column}_colors")
    if group_colors is None or len(group_colors) != len(cat_order):
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#0173B2"])
        group_colors = [cycle[i % len(cycle)] for i in range(len(cat_order))]
    cats = obs[categorical_column].astype(str)
    counts = cats.value_counts().reindex(cat_order).fillna(0).to_numpy(dtype=int)
    edges = np.concatenate([[0], np.cumsum(counts)])
    mids = (edges[:-1] + edges[1:] - 1) / 2.0
    heights = (edges[1:] - edges[:-1]) * 0.9

    def _row_axes(group: Sequence[BarSeries], *, left: bool) -> list[Axes]:
        out: list[Axes] = []
        x0 = left_edge - _BAR_GAP_IN / fig_w if left else right_edge + _BAR_GAP_IN / fig_w
        for s in group:
            values, se = row_values[id(s)]
            width = s.size_in / fig_w
            x_start = x0 - width if left else x0
            out.append(
                _draw_row_bar(
                    fig,
                    heat,
                    (x_start, pos.y0, width, pos.height),
                    s,
                    values,
                    se=se,
                    mids=mids,
                    heights=heights,
                    colors=_bar_colors(
                        s.color,
                        cat_order,
                        values,
                        default=group_colors,
                        name=_series_name(s, "row_series", 0),
                    ),
                    left=left,
                )
            )
            x0 = x_start - _BAR_GAP_IN / fig_w if left else x0 + width + _BAR_GAP_IN / fig_w
        return out

    rows_right = _row_axes(row_series, left=False)
    rows_left = _row_axes(row_series_left, left=True)

    # scanpy's own colourbar sits in a thin axes it never exposes in `axes`, by default
    # sandwiched between `heat` and the first right panel -- move it to the figure's top right
    # corner instead, so it reads as a legend rather than a squeezed-in extra column.
    if row_series:
        known = set(axes.values()) | {heat} | set(column_axes) | set(rows_right) | set(rows_left)
        colorbar_ax = next((a for a in fig.axes if a not in known), None)
        if colorbar_ax is not None:
            cb_w, cb_h = 0.05, 0.12
            colorbar_ax.set_position([0.94 - cb_w, 0.99 - cb_h, cb_w, cb_h])

    _unify_fonts(fig)
    if new_style:
        return fig, LatentHeatmapAxes(heat, column_axes, rows_left, rows_right)
    # Legacy arguments alone: the old return shapes.
    if column_axes and rows_right:
        return fig, (column_axes[0], heat, rows_right[0])
    if column_axes:
        return fig, (column_axes[0], heat)
    return fig, (heat, rows_right[0])
