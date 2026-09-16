"""Embedding scatters.

A plain categorical or continuous scatter -- `sc.pl.embedding(embed, basis="umap",
color="cell_type")`, or `color="score"` with `cmap=`/`vmin=`/`vmax=` (scanpy accepts
percentile strings directly, e.g. ``vmin="p1"``) -- already does everything such a plot
needs: point sizing by cell count, a stable per-category palette or a colorbar, a
legend. Call scanpy directly; this module does not wrap it a second time. What earns a
wrapper here is a grid of one panel per latent dimension (:func:`latent_umap_grid`),
ported to match ``drvi.utils.pl.plot_latent_dims_in_umap``'s own mechanics exactly --
DRVI's own colours, per-dimension colour limits and directional +/- split -- see its
docstring.

:func:`latent_umap_grid` takes the ``embed`` ``AnnData`` directly (``obs`` = cells, a
named ``obsm`` basis, ``var`` = one row per latent dimension), the same convention
``scvi``/``drvi-py`` already use for this object, rather than a plain x/y frame: unlike
the rest of :mod:`scads_drvi.pl`, it needs ``anndata``'s aligned fancy-indexing to
subsample cells and keep ``obs``/``obsm`` in sync, so there is no benefit to decomposing
to a frame first only to reassemble it for scanpy. Apply
:func:`scads_drvi.pl.style.apply_style`/:func:`scads_drvi.pl.save.save_figure` to the
figure it returns, same as any other figure in this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from anndata import AnnData
    from matplotlib.figure import Figure

__all__ = [
    "SUBSAMPLE_DEFAULT",
    "SUBSAMPLE_SEED",
    "subsample",
    "latent_umap_grid",
]

SUBSAMPLE_DEFAULT = 60_000
SUBSAMPLE_SEED = 42


def subsample(
    frame: pd.DataFrame,
    n: int = SUBSAMPLE_DEFAULT,
    *,
    seed: int = SUBSAMPLE_SEED,
    stratify: str | None = None,
) -> pd.DataFrame:
    """A reproducible row subsample. Returns `frame` unchanged when it is small enough.

    `stratify` caps the draw per group instead of overall, so a rare category is not
    sampled out of existence by a uniform draw.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if len(frame) <= n:
        return frame

    rng = np.random.default_rng(seed)
    if stratify is None:
        take = rng.choice(len(frame), size=n, replace=False)
        return frame.iloc[np.sort(take)]

    if stratify not in frame.columns:
        raise KeyError(f"{stratify!r} is not a column of the frame")
    groups = frame.groupby(stratify, observed=True, sort=False)
    per_group = max(1, n // max(groups.ngroups, 1))
    parts = []
    for _, block in groups:
        if len(block) <= per_group:
            parts.append(block)
        else:
            take = rng.choice(len(block), size=per_group, replace=False)
            parts.append(block.iloc[np.sort(take)])
    import pandas as pd

    return pd.concat(parts)


def _subsample_embed(embed: AnnData, n: int | None, *, seed: int = SUBSAMPLE_SEED) -> AnnData:
    """`subsample`'s reproducible row selection, applied to an ``AnnData``'s ``obs``.

    Returns a **copy**, not a view: every caller here goes on to write a temporary
    ``obs`` column (a directional ReLU split, a masked continuous value) onto the
    result, and mutating a view would silently write through to the caller's `embed`.

    A plain categorical/continuous scatter needs none of this -- ``sc.pl.embedding``
    (or ``sc.pl.umap``) already handles point sizing, a stable per-category palette and
    a colorbar on its own; call it directly rather than through a wrapper here.
    """
    if n is None:
        return embed.copy()
    positions = subsample(embed.obs.reset_index(drop=True), n, seed=seed).index.to_numpy()
    return embed[positions].copy()


def latent_umap_grid(
    embed: AnnData,
    *,
    dim_subset: Sequence[str] | None = None,
    directional: bool = False,
    remove_vanished: bool = True,
    order_col: str = "order",
    title_col: str = "title",
    ncols: int = 5,
    cmap=None,
    directional_cmap=None,
    min_max_thresholds: tuple[float, float] | None = (-1.0, 1.0),
    color_bar_rescale_ratio: float = 1.0,
    rearrange_titles: bool = True,
    n: int | None = SUBSAMPLE_DEFAULT,
    seed: int = SUBSAMPLE_SEED,
    **kwargs,
) -> Figure:
    """One embedding panel per latent dimension, ported to match
    ``drvi.utils.pl.plot_latent_dims_in_umap``'s own mechanics exactly: DRVI's
    ``SaturatedRdBu``/``SaturatedSky`` colours (`cmap`/`directional_cmap`, default
    :data:`scads_drvi.pl.color.SATURATED_RED_BLUE_CMAP`/:data:`~.SATURATED_SKY_CMAP`),
    per-dimension colour limits from `embed.var["min"]`/`["max"]` (widened to
    `min_max_thresholds` rather than this project's own robust percentile limits), a
    directional split built the same way DRVI builds it -- concatenating a negated copy
    of `embed` (`ad.concat`), not a temporary ``obs`` column -- and DRVI's own trick of
    moving each panel's title into the plot itself (`rearrange_titles`) with the
    negative panel's y-axis (colorbar) flipped and relabelled.

    `dim_subset` still names raw `embed.var_names` (this project's own convention
    throughout), not `title_col` values as DRVI's own `dim_subset` does.
    """
    import anndata as ad
    import scanpy as sc

    from scads_drvi.pl.color import SATURATED_RED_BLUE_CMAP, SATURATED_SKY_CMAP

    cmap = SATURATED_RED_BLUE_CMAP if cmap is None else cmap
    directional_cmap = SATURATED_SKY_CMAP if directional_cmap is None else directional_cmap

    for column in (order_col, title_col):
        if column not in embed.var.columns:
            raise KeyError(f"{column!r} is not a column of embed.var")
    if remove_vanished and "vanished" not in embed.var.columns:
        raise KeyError('"vanished" is not a column of embed.var')

    dim_stats = embed.var
    if remove_vanished:
        dim_stats = dim_stats.loc[~dim_stats["vanished"].astype(bool)]
    dim_stats = dim_stats.sort_values(order_col)

    if dim_subset is not None:
        missing = [d for d in dim_subset if d not in dim_stats.index]
        if missing:
            raise KeyError(f"dim_subset names dims not present (after filtering): {missing}")
        dim_stats = dim_stats.loc[list(dim_subset)]
    dims = list(dim_stats.index)
    if not dims:
        raise ValueError("no latent dimensions to plot after filtering")

    plotted = _subsample_embed(embed, n, seed=seed)[:, dims].copy()
    # A plain 0..len(dims)-1 rank, in the same order `dims` was already sorted into
    # above -- DRVI's own `order_col` arithmetic (+/- a small epsilon per direction)
    # only needs *an* ordering to offset, not the original fit-wide rank.
    plotted.var[order_col] = np.arange(len(dims), dtype=float)

    if directional:
        pos, neg = plotted.copy(), plotted.copy()
        neg.X = -neg.X
        neg.var["min"], neg.var["max"] = -neg.var["max"], -neg.var["min"]
        pos.var["_direction"], neg.var["_direction"] = "+", "-"
        pos.var[title_col] = pos.var[title_col].astype(str) + "+"
        neg.var[title_col] = neg.var[title_col].astype(str) + "-"
        pos.var[order_col] = pos.var[order_col] + 1e-8
        neg.var[order_col] = neg.var[order_col] - 1e-8
        plotted = ad.concat([pos, neg], axis=1, join="inner", merge="first")
        # dim_0..dim_{K-1} now appears twice (once negated) -- var_names must be
        # unique for scanpy's gene_symbols= lookup below to resolve one row per title.
        plotted.var_names = [str(i) for i in range(plotted.n_vars)]

    # `.var`'s own row order need not match -- color=cols_to_show below is an
    # explicit, already-sorted panel list; gene_symbols= resolves each title to its
    # var row regardless of where that row physically sits.
    tmp_df = plotted.var.sort_values(order_col)
    cols_to_show = list(tmp_df[title_col])

    if min_max_thresholds is not None:
        vmin = list(np.minimum(tmp_df["min"].to_numpy(dtype=float), min_max_thresholds[0]))
        vmax = list(np.maximum(tmp_df["max"].to_numpy(dtype=float), min_max_thresholds[1]))
    else:
        vmin = list(tmp_df["min"].to_numpy(dtype=float))
        vmax = list(tmp_df["max"].to_numpy(dtype=float))

    fig = sc.pl.umap(
        plotted, gene_symbols=title_col, color=cols_to_show, return_fig=True,
        frameon=False, cmap=directional_cmap if directional else cmap,
        vmin=vmin, vcenter=0, vmax=vmax, ncols=ncols, show=False, **kwargs,
    )
    for i, ax in enumerate(fig.axes[1 : 2 * len(tmp_df) : 2]):
        pos_ax = ax.get_position()
        ax.set_position(
            [pos_ax.x0, pos_ax.y0, pos_ax.width, pos_ax.height * color_bar_rescale_ratio]
        )
        if directional and tmp_df["_direction"].iloc[i] == "-":
            ax.invert_yaxis()
            labels = -ax.get_yticks()
            if all(x == int(x) for x in labels):
                labels = [int(x) for x in labels]
            ax.set_yticklabels(labels)
    if rearrange_titles:
        for ax in fig.axes:
            ax.text(
                0.935, 0.05, ax.get_title(), size=15, ha="left", color="black",
                rotation=90, transform=ax.transAxes,
            )
            ax.set_title("")

    return fig
