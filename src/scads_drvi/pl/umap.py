"""Embedding scatters.

Every function takes a tidy frame and returns ``(fig, ax)``. None of them touches a path
or a global, so a caller can compose them and a test can assert on them.

Point size is derived from the point count rather than typed. The code this replaces used
``s=5`` in one place and ``s=0.10`` in another -- a 50x difference that is a consequence
of drawing 60,000 points versus 1.26 million, not a stylistic choice, and therefore
something to compute.

A per-factor grid over one embedding -- what this module used to call
``umap_factor_grid`` -- is now ``drvi.utils.pl.plot_latent_dims_in_umap(embed,
directional=..., dim_subset=..., order_col="order")``, driven by the ``obsm["X_umap"]``
and ``var`` columns :func:`scads_drvi.factorize.result.build_embed` already sets. Apply
:func:`scads_drvi.pl.style.apply_style`/:func:`scads_drvi.pl.save.save_figure` to the
figure it returns, same as any other figure here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "SUBSAMPLE_DEFAULT",
    "SUBSAMPLE_SEED",
    "point_style",
    "subsample",
    "bare",
    "umap_categorical",
    "umap_continuous",
]

SUBSAMPLE_DEFAULT = 60_000
SUBSAMPLE_SEED = 42


def point_style(n: int) -> dict:
    """Marker properties for a scatter of `n` points.

    Size falls as the count rises, so a 60k panel and a 1.2M panel both read. Always
    rasterised: a vector scatter of a million points produces a PDF no viewer will open.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n <= 2_000:
        size = 12.0
    elif n <= 20_000:
        size = 6.0
    elif n <= 100_000:
        size = 3.0
    elif n <= 500_000:
        size = 1.0
    else:
        size = 0.3
    alpha = 0.9 if n <= 20_000 else (0.6 if n <= 200_000 else 0.35)
    return {
        "s": size,
        "alpha": alpha,
        "linewidths": 0.0,
        "edgecolors": "none",
        "rasterized": True,
    }


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


def bare(ax: Axes) -> None:
    """Strip ticks and spines, for a panel whose axes carry no units."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _coords(frame: pd.DataFrame, x: str | None, y: str | None) -> tuple[str, str]:
    if x is not None and y is not None:
        for name in (x, y):
            if name not in frame.columns:
                raise KeyError(f"{name!r} is not a column of the frame")
        return x, y
    from pandas.api.types import is_numeric_dtype

    # is_numeric_dtype, not np.issubdtype: from pandas 3.0 a text column is a
    # StringDtype extension dtype, and np.issubdtype raises TypeError on it rather than
    # returning False -- so the numpy spelling turns "this column is not a coordinate"
    # into a crash.
    numeric = [c for c in frame.columns if is_numeric_dtype(frame[c])]
    if len(numeric) < 2:
        raise ValueError(
            "cannot find two numeric coordinate columns; name them with x= and y="
        )
    return numeric[0], numeric[1]


def umap_categorical(
    frame: pd.DataFrame,
    hue: str,
    *,
    ax: Axes | None = None,
    x: str | None = None,
    y: str | None = None,
    palette: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    legend: str = "right",
    n: int | None = SUBSAMPLE_DEFAULT,
) -> tuple[Figure, Axes]:
    """Scatter an embedding coloured by a categorical column.

    The palette is built from the **full** category list before subsampling, so the same
    category is the same colour in every panel of a figure.
    """
    import matplotlib.pyplot as plt

    from scads_drvi.pl.color import categorical_palette

    if hue not in frame.columns:
        raise KeyError(f"{hue!r} is not a column of the frame")
    xcol, ycol = _coords(frame, x, y)

    categories = order or sorted(frame[hue].astype(str).unique())
    palette = palette or categorical_palette(categories)

    plotted = subsample(frame, n, stratify=hue) if n else frame
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    style = point_style(len(plotted))

    for category in categories:
        block = plotted[plotted[hue].astype(str) == str(category)]
        if block.empty:
            continue
        ax.scatter(
            block[xcol], block[ycol],
            color=palette.get(str(category), "#949494"),
            label=str(category), **style,
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    if legend == "right":
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
            markerscale=4.0, handletextpad=0.3,
        )
    elif legend == "inside":
        ax.legend(frameon=False, markerscale=4.0)
    return fig, ax


def umap_continuous(
    frame: pd.DataFrame,
    value: str,
    *,
    ax: Axes | None = None,
    x: str | None = None,
    y: str | None = None,
    cmap: str = "viridis",
    norm=None,
    percentiles: tuple[float, float] | None = None,
    zero_as_background: bool = False,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    n: int | None = SUBSAMPLE_DEFAULT,
) -> tuple[Figure, Axes]:
    """Scatter an embedding coloured by a continuous column, under robust limits.

    `zero_as_background` draws exact zeros in grey underneath and sets the colour scale
    on the non-zero values. Loadings are ReLU output, so a large fraction are exactly
    zero; including them in the scale wastes most of the ramp on a spike at zero.
    """
    import matplotlib.pyplot as plt

    from scads_drvi.pl.color import ROBUST_LIMITS, robust_norm

    if value not in frame.columns:
        raise KeyError(f"{value!r} is not a column of the frame")
    xcol, ycol = _coords(frame, x, y)

    plotted = subsample(frame, n) if n else frame
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    values = plotted[value].to_numpy(dtype=float)
    if norm is None:
        norm = robust_norm(
            values,
            percentiles=percentiles or ROBUST_LIMITS,
            nonzero_only=zero_as_background,
        )

    style = point_style(len(plotted))
    if zero_as_background:
        zero = values == 0
        if zero.any():
            ax.scatter(
                plotted[xcol][zero], plotted[ycol][zero],
                color="#E6E6E6", zorder=1, **style,
            )
        plotted, values = plotted[~zero], values[~zero]

    mappable = ax.scatter(
        plotted[xcol], plotted[ycol], c=values, cmap=cmap, norm=norm, zorder=2, **style
    )
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    if colorbar:
        bar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.045)
        bar.set_label(colorbar_label or value)
    return fig, ax
