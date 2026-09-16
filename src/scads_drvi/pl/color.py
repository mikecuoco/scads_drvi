"""Colour limits, palettes and significance ramps -- with one default each.

The code this replaces used four different percentile pairs for the same "robust colour
limits" intent (1/99.5, 1/99, 0.2/99.8, 2/98) plus one hardcoded ``clip(0.5, 35.0)`` that
is silently wrong for any run but the one it was typed for. There is one default here and
it is a keyword argument, so a deviation shows up in the diff instead of being buried in
a scatter call.

The significance ramp keeps its thresholds and its colours in the same frozen object, so
they cannot drift apart -- the failure mode being a legend that says ``q<0.05`` next to
points coloured by a threshold somebody else changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.colors import Normalize

__all__ = [
    "ROBUST_LIMITS",
    "Z_NOMINAL_ONE_TAILED",
    "Z_HIGH_CONFIDENCE",
    "robust_limits",
    "robust_norm",
    "percentile_bounds",
    "SignificanceRamp",
    "DEFAULT_RAMP",
    "significance_class",
    "significance_colors",
    "significance_handles",
    "categorical_palette",
    "add_threshold_lines",
    "SATURATED_RED_BLUE_CMAP",
    "SATURATED_JUST_SKY_CMAP",
    "SATURATED_SKY_CMAP",
]

#: The one default percentile pair for robust colour limits.
ROBUST_LIMITS: tuple[float, float] = (1.0, 99.5)

# Copied verbatim from DRVI's own `drvi.utils.plotting._cmap`
# (https://github.com/theislab/DRVI/blob/main/src/drvi/utils/plotting/_cmap.py), so
# `latent_umap_grid`/`latent_heatmap` render with the exact colours DRVI's own
# `plot_latent_dims_in_umap`/`plot_latent_dims_in_heatmap` do.
_cmap_data = {
    "red": ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 1.0, 1.0), (0.75, 1.0, 1.0), (1.0, 0.5, 0.0)),
    "green": ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 1.0, 1.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "blue": ((0.0, 0.0, 0.5), (0.25, 1.0, 1.0), (0.5, 1.0, 1.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0)),
}
SATURATED_RED_BLUE_CMAP = LinearSegmentedColormap("SaturatedRdBu", _cmap_data)

_cmap_data = {
    "red": ((0.0, 0.0, 240 / 256), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 240 / 256), (0.75, 200 / 255, 200 / 255), (1.0, 63 / 255, 0.0)),
    "blue": ((0.0, 0.0, 240 / 256), (0.75, 255 / 255, 255 / 255), (1.0, 80 / 255, 0.0)),
}
SATURATED_JUST_SKY_CMAP = LinearSegmentedColormap("SaturatedJSky", _cmap_data)

_cmap_data = {
    "red": ((0.0, 0.0, 136 / 255), (0.5, 250 / 256, 250 / 256), (0.65, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 136 / 255), (0.5, 250 / 256, 250 / 256), (0.65, 200 / 255, 200 / 255), (1.0, 63 / 255, 0.0)),
    "blue": ((0.0, 0.0, 136 / 255), (0.5, 250 / 256, 250 / 256), (0.65, 255 / 255, 255 / 255), (1.0, 80 / 255, 0.0)),
}
SATURATED_SKY_CMAP = LinearSegmentedColormap("SaturatedSky", _cmap_data)
del _cmap_data

#: One-tailed nominal significance in z. Derived, not the retyped 1.645.
Z_NOMINAL_ONE_TAILED: float = 1.6448536269514722

#: The conventional "high confidence" z used in this project's figures.
Z_HIGH_CONFIDENCE: float = 3.0


def robust_limits(
    values,
    *,
    percentiles: tuple[float, float] = ROBUST_LIMITS,
    symmetric: bool = False,
    nonzero_only: bool = False,
    floor: float | None = None,
) -> tuple[float, float]:
    """Percentile colour limits that never come back degenerate.

    `nonzero_only` sets the scale on the non-zero entries, which matters when a matrix is
    mostly exact zeros -- including them puts the whole useful range in the bottom of the
    ramp. `symmetric` centres the limits on zero for a diverging quantity.

    A degenerate range (all values equal, or fewer than two finite values) is widened
    rather than returned, because a zero-width normalisation renders as a single flat
    colour and reads as "no data" rather than "no variation".
    """
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if nonzero_only:
        array = array[array != 0]

    if array.size == 0:
        return (0.0, 1.0)

    low_pct, high_pct = percentiles
    if not 0 <= low_pct < high_pct <= 100:
        raise ValueError(
            f"percentiles must satisfy 0 <= low < high <= 100, got {percentiles}"
        )

    low, high = (float(x) for x in np.percentile(array, [low_pct, high_pct]))
    if floor is not None:
        low = max(low, float(floor))
    if symmetric:
        extent = max(abs(low), abs(high))
        low, high = -extent, extent
    if high <= low:
        # widen around the value rather than returning a zero-width range
        centre = 0.5 * (low + high)
        spread = max(abs(centre) * 1e-6, 1e-9)
        low, high = centre - spread, centre + spread
    return low, high


def robust_norm(
    values,
    *,
    percentiles: tuple[float, float] = ROBUST_LIMITS,
    gamma: float = 1.0,
    symmetric: bool = False,
    nonzero_only: bool = False,
    floor: float | None = None,
) -> Normalize:
    """A matplotlib norm from :func:`robust_limits`; ``gamma != 1`` gives a PowerNorm."""
    from matplotlib.colors import Normalize, PowerNorm

    low, high = robust_limits(
        values,
        percentiles=percentiles,
        symmetric=symmetric,
        nonzero_only=nonzero_only,
        floor=floor,
    )
    if gamma == 1.0:
        return Normalize(vmin=low, vmax=high)
    if low < 0:
        raise ValueError(
            f"PowerNorm needs vmin >= 0, got {low}. Use gamma=1 for a diverging scale."
        )
    return PowerNorm(gamma=gamma, vmin=low, vmax=high)


def percentile_bounds(percentiles: tuple[float, float] = ROBUST_LIMITS) -> tuple[str, str]:
    """This project's percentile pair, spelled the way ``scanpy.pl.embedding`` wants it.

    scanpy accepts ``vmin``/``vmax`` as a string ``"pN"`` meaning "the Nth percentile of
    the data actually being plotted," computed by scanpy itself at draw time -- so this
    is not a second implementation of :func:`robust_limits`, only the one place this
    project's default percentile pair is spelled in scanpy's own syntax, so it is never
    retyped (and never drifts from :data:`ROBUST_LIMITS`) at a call site.
    """
    low, high = percentiles
    return f"p{low:g}", f"p{high:g}"


@dataclass(frozen=True)
class SignificanceRamp:
    """Thresholds and colours together, so they cannot be changed independently.

    `thresholds` is ascending; there is one more colour than threshold, the last being
    the "not significant" colour.
    """

    thresholds: tuple[float, ...] = (0.05, 0.10)
    colors: tuple[str, ...] = ("#D55E00", "#DE8F05", "#0173B2")
    label: str = "BH q"

    def __post_init__(self) -> None:
        if len(self.colors) != len(self.thresholds) + 1:
            raise ValueError(
                f"{len(self.thresholds)} thresholds need "
                f"{len(self.thresholds) + 1} colours, got {len(self.colors)}"
            )
        if list(self.thresholds) != sorted(self.thresholds):
            raise ValueError(f"thresholds must be ascending, got {self.thresholds}")

    @property
    def legend_labels(self) -> list[str]:
        parts = [f"{self.label} < {t:g}" for t in self.thresholds]
        parts.append(f"{self.label} >= {self.thresholds[-1]:g}")
        return parts


DEFAULT_RAMP = SignificanceRamp()


def significance_class(
    values, *, thresholds: Sequence[float] = DEFAULT_RAMP.thresholds
) -> np.ndarray:
    """Ordinal class per test: 0 is the most significant, len(thresholds) the least.

    NaN sorts into the least-significant class -- a test that could not be evaluated is
    not evidence, and colouring it as significant would be a claim.
    """
    array = np.asarray(values, dtype=float)
    out = np.full(array.shape, len(thresholds), dtype=int)
    for level in range(len(thresholds) - 1, -1, -1):
        out[np.isfinite(array) & (array < thresholds[level])] = level
    return out


def significance_colors(values, *, ramp: SignificanceRamp = DEFAULT_RAMP) -> list[str]:
    """Per-point colours from q-values."""
    classes = significance_class(values, thresholds=ramp.thresholds)
    return [ramp.colors[c] for c in classes.ravel()]


def significance_handles(ramp: SignificanceRamp = DEFAULT_RAMP) -> list:
    """Legend proxies, so a ramped panel can actually be read."""
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], marker="o", linestyle="none", color=color, label=label)
        for color, label in zip(ramp.colors, ramp.legend_labels, strict=True)
    ]


def categorical_palette(
    categories: Sequence[str],
    *,
    highlight: str | Sequence[str] | None = None,
    highlight_color: str = "#D55E00",
    base: str | None = None,
) -> dict[str, str]:
    """A stable category -> colour map.

    Keyed on the **sorted full** list of categories, not on whichever survived a
    subsample. Building the map from ``unique()`` of a sampled frame is why two panels of
    the same figure could give one category two different colours.

    `highlight` is a caller's claim about which categories are interesting, so it is an
    argument rather than a value baked in here.

    Up to 8 categories use the project's colourblind-safe cycle; beyond that a
    perceptually spaced continuous map is sampled, since no 8-colour palette extends
    honestly to 29 levels.
    """
    from matplotlib import colormaps

    from scads_drvi.pl.style import WONG

    unique = sorted(dict.fromkeys(str(c) for c in categories))
    n = len(unique)
    if n == 0:
        return {}

    if base is None and n <= len(WONG):
        colours = list(WONG[:n])
    else:
        cmap = colormaps[base or "turbo"]
        colours = [
            "#{:02x}{:02x}{:02x}".format(*(int(255 * v) for v in cmap(x)[:3]))
            for x in np.linspace(0.02, 0.98, n)
        ]

    palette = dict(zip(unique, colours, strict=True))

    if highlight is not None:
        wanted = [highlight] if isinstance(highlight, str) else list(highlight)
        unknown = [h for h in wanted if str(h) not in palette]
        if unknown:
            raise KeyError(
                f"cannot highlight {unknown}: not among the categories given"
            )
        for name in wanted:
            palette[str(name)] = highlight_color
    return palette


def add_threshold_lines(
    ax,
    *,
    z: Sequence[float] = (Z_NOMINAL_ONE_TAILED, Z_HIGH_CONFIDENCE),
    axis: str = "y",
    label: bool = True,
    bh: float | None = None,
    alpha: float = 0.05,
    color: str = "#949494",
) -> None:
    """Draw significance reference lines consistently on one axis.

    `bh` is a data-derived boundary (see ``stats.bh_threshold_z``) and is drawn
    differently from the fixed conventions, because it is a property of this run rather
    than a constant.

    `alpha` is the level `bh` was computed at, and exists only to label the line. It was
    previously the literal ``0.05`` in the label string, so a caller correcting at any
    other level got a line annotated with a threshold it was not drawn at -- the same
    class of drift the frozen :class:`SignificanceRamp` above exists to prevent.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    draw = ax.axhline if axis == "y" else ax.axvline

    for value in z:
        draw(
            value,
            color=color,
            linestyle=":",
            linewidth=0.8,
            label=f"z = {value:.2f}" if label else None,
        )
    if bh is not None:
        draw(
            bh,
            color="#D55E00",
            linestyle="--",
            linewidth=0.9,
            label=f"BH q < {alpha:g}" if label else None,
        )
