"""Embedding scatters.

A plain categorical or continuous scatter -- `sc.pl.embedding(embed, basis="umap",
color="cell_type")`, or `color="score"` with `cmap=`/`vmin=`/`vmax=` (scanpy accepts
percentile strings directly, e.g. ``vmin="p1"``) -- already does everything such a plot
needs: point sizing by cell count, a stable per-category palette or a colorbar, a
legend. Call scanpy directly; this module does not wrap it a second time. What earns a
wrapper here is a grid of one panel per latent dimension (:func:`latent_umap_grid`),
which scanpy's own `color=` API does not give you for free: per-panel titles/limits and
a directional +/- split (``relu(x)``/``relu(-x)``) assembled from one
``sc.pl.embedding`` call, under this project's own robust percentile limits
(:func:`scads_drvi.pl.color.percentile_bounds`).

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
    cmap: str = "RdBu_r",
    directional_cmap: str = "viridis",
    percentiles: tuple[float, float] | None = None,
    n: int | None = SUBSAMPLE_DEFAULT,
    seed: int = SUBSAMPLE_SEED,
    **kwargs,
) -> Figure:
    """One embedding panel per latent dimension, in this project's own style.

    Replaces ``drvi.utils.pl.plot_latent_dims_in_umap``. `directional=True` doubles
    every kept dimension into a ``+``/``-`` panel (``relu(x)``/``relu(-x)``, computed
    inline as temporary ``obs`` columns -- a one-line clip on a column already in hand,
    not a reason for a dedicated helper). Those panels use `directional_cmap`
    (one-sided, since a ReLU output is non-negative) rather than the diverging `cmap`
    the signed panels use.

    One ``sc.pl.embedding`` call draws the whole grid (`color=` takes the full list of
    panel names, with a matching `vmin`/`vmax` per panel), so panel layout, spacing and
    per-panel colorbars all come from scanpy rather than being rebuilt here.
    """
    import scanpy as sc

    from scads_drvi.pl.color import ROBUST_LIMITS, percentile_bounds

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

    plotted = _subsample_embed(embed, n, seed=seed)
    values = plotted[:, dims].X
    values = np.asarray(values.toarray() if hasattr(values, "toarray") else values, dtype=float)

    vmin_default, vmax_default = percentile_bounds(percentiles or ROBUST_LIMITS)
    panels: list[str] = []
    cmaps: list[str] = []
    vmins: list = []
    vmaxs: list = []
    titles: list[str] = []

    for j, dim in enumerate(dims):
        title = str(dim_stats.loc[dim, title_col])
        if directional:
            pos, neg = f"{dim}__pos", f"{dim}__neg"
            plotted.obs[pos] = np.clip(values[:, j], 0, None)
            plotted.obs[neg] = np.clip(-values[:, j], 0, None)
            panels += [pos, neg]
            cmaps += [directional_cmap, directional_cmap]
            vmins += ["p0", "p0"]
            vmaxs += [vmax_default, vmax_default]
            titles += [f"{title}+", f"{title}-"]
        else:
            plotted.obs[dim] = values[:, j]
            panels.append(dim)
            cmaps.append(cmap)
            vmins.append(vmin_default)
            vmaxs.append(vmax_default)
            titles.append(title)

    # sc.pl.embedding takes one `cmap=`, not one per panel -- every panel here shares
    # `cmap` (non-directional) or `directional_cmap` (directional), never a mix, so this
    # is safe; a future per-panel-cmap need would have to draw panels individually.
    fig = sc.pl.embedding(
        plotted, basis="umap", color=panels, cmap=cmaps[0], vmin=vmins, vmax=vmaxs,
        vcenter=0 if not directional else None, title=titles, ncols=ncols,
        show=False, return_fig=True, **kwargs,
    )
    return fig
