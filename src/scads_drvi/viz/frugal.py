"""Box statistics computed once, then drawn.

Handing a seaborn boxplot 1.2 million rows makes it re-sort the frame per category to
recompute five numbers. Those five numbers are what a box *is*, so they are computed in
one pass here and fed to matplotlib's ``bxp``, which draws from statistics directly.

The practical consequence is that a plotting function can take a :class:`BoxStats` and
thereby be *unable* to receive the full table -- the guard is in the type, not in a
comment asking the caller to be careful.

One inherited convention is made explicit: the whiskers are drawn **at the quartiles**,
not at 1.5 IQR. That is what the code being replaced did, it is unusual, and leaving it
implicit would let a reader mistake a quartile whisker for a range.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["BoxStats", "box_stats", "box_stats_by_column", "draw_boxes"]

Whis = Literal["quartile", "1.5iqr", "minmax"]


@dataclass(frozen=True)
class BoxStats:
    """Per-group box statistics: exactly what ``Axes.bxp`` needs, and nothing else."""

    labels: tuple[str, ...]
    median: np.ndarray
    q1: np.ndarray
    q3: np.ndarray
    whislo: np.ndarray
    whishi: np.ndarray
    n: np.ndarray
    whis: Whis = "quartile"
    maximum: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def to_bxp(self) -> list[dict]:
        """The list of dicts ``Axes.bxp`` consumes."""
        return [
            {
                "label": label,
                "med": float(self.median[i]),
                "q1": float(self.q1[i]),
                "q3": float(self.q3[i]),
                "whislo": float(self.whislo[i]),
                "whishi": float(self.whishi[i]),
                "fliers": [],
            }
            for i, label in enumerate(self.labels)
        ]

    def order_by(
        self, key: Literal["median", "q3", "label", "n"] = "median", *, ascending=False
    ) -> BoxStats:
        """Same boxes, reordered for an axis."""
        if key == "label":
            order = np.argsort(np.asarray(self.labels))
        else:
            order = np.argsort(getattr(self, "median" if key == "median" else key))
        if not ascending:
            order = order[::-1]
        return replace(
            self,
            labels=tuple(np.asarray(self.labels)[order]),
            median=self.median[order],
            q1=self.q1[order],
            q3=self.q3[order],
            whislo=self.whislo[order],
            whishi=self.whishi[order],
            n=self.n[order],
            maximum=None if self.maximum is None else self.maximum[order],
        )

    def select(self, labels: Sequence[str]) -> BoxStats:
        """The subset named, in the order named."""
        index = {label: i for i, label in enumerate(self.labels)}
        missing = [label for label in labels if label not in index]
        if missing:
            raise KeyError(f"no box for {missing}")
        order = np.array([index[label] for label in labels])
        return replace(
            self,
            labels=tuple(labels),
            median=self.median[order],
            q1=self.q1[order],
            q3=self.q3[order],
            whislo=self.whislo[order],
            whishi=self.whishi[order],
            n=self.n[order],
            maximum=None if self.maximum is None else self.maximum[order],
        )


def _whiskers(values: np.ndarray, q1: float, q3: float, whis: Whis) -> tuple[float, float]:
    if whis == "quartile":
        return q1, q3
    if whis == "minmax":
        return float(values.min()), float(values.max())
    if whis == "1.5iqr":
        iqr = q3 - q1
        low = values[values >= q1 - 1.5 * iqr]
        high = values[values <= q3 + 1.5 * iqr]
        return (
            float(low.min()) if low.size else q1,
            float(high.max()) if high.size else q3,
        )
    raise ValueError(f"unknown whis {whis!r}")


def box_stats(
    values,
    groups=None,
    *,
    whis: Whis = "quartile",
    min_n: int = 1,
    with_maximum: bool = False,
) -> BoxStats:
    """Per-group quantiles in one pass.

    With `groups` omitted the whole vector is a single box. Groups with fewer than
    `min_n` finite observations are omitted -- five summary statistics of three points
    are not a distribution.
    """
    import pandas as pd

    series = pd.Series(np.asarray(values, dtype=float))
    if groups is None:
        keys = pd.Series(["all"] * len(series))
    else:
        keys = pd.Series(np.asarray(groups)).astype(str)
        if len(keys) != len(series):
            raise ValueError(
                f"values and groups must be the same length, got {len(series)} "
                f"and {len(keys)}"
            )

    labels: list[str] = []
    med, q1s, q3s, lo, hi, counts, maxima = [], [], [], [], [], [], []
    for label, index in keys.groupby(keys, sort=True).groups.items():
        block = series.iloc[np.asarray(index)].to_numpy()
        block = block[np.isfinite(block)]
        if block.size < int(min_n):
            continue
        first, middle, third = np.percentile(block, [25, 50, 75])
        whislo, whishi = _whiskers(block, float(first), float(third), whis)
        labels.append(str(label))
        q1s.append(first)
        med.append(middle)
        q3s.append(third)
        lo.append(whislo)
        hi.append(whishi)
        counts.append(block.size)
        maxima.append(block.max())

    if not labels:
        raise ValueError(f"no group reached min_n={min_n}")

    return BoxStats(
        labels=tuple(labels),
        median=np.asarray(med, dtype=float),
        q1=np.asarray(q1s, dtype=float),
        q3=np.asarray(q3s, dtype=float),
        whislo=np.asarray(lo, dtype=float),
        whishi=np.asarray(hi, dtype=float),
        n=np.asarray(counts, dtype=int),
        whis=whis,
        maximum=np.asarray(maxima, dtype=float) if with_maximum else None,
    )


def box_stats_by_column(
    matrix,
    labels: Sequence[str] | None = None,
    *,
    whis: Whis = "quartile",
    with_maximum: bool = False,
) -> BoxStats:
    """One box per **column** of a (rows x k) matrix.

    The per-factor case: a column is a factor and the rows are cells, so no group vector
    exists. Computed with a single ``percentile`` call over the whole matrix rather than
    a Python loop over columns.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {array.shape}")

    if labels is None:
        columns = getattr(matrix, "columns", None)
        labels = (
            [str(c) for c in columns]
            if columns is not None
            else [str(i) for i in range(array.shape[1])]
        )
    labels = [str(label) for label in labels]
    if len(labels) != array.shape[1]:
        raise ValueError(
            f"{len(labels)} labels for {array.shape[1]} columns"
        )

    q1, med, q3 = np.nanpercentile(array, [25, 50, 75], axis=0)
    if whis == "quartile":
        whislo, whishi = q1, q3
    elif whis == "minmax":
        whislo, whishi = np.nanmin(array, axis=0), np.nanmax(array, axis=0)
    else:
        iqr = q3 - q1
        whislo, whishi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    return BoxStats(
        labels=tuple(labels),
        median=np.asarray(med, dtype=float),
        q1=np.asarray(q1, dtype=float),
        q3=np.asarray(q3, dtype=float),
        whislo=np.asarray(whislo, dtype=float),
        whishi=np.asarray(whishi, dtype=float),
        n=np.sum(np.isfinite(array), axis=0).astype(int),
        whis=whis,
        maximum=np.nanmax(array, axis=0) if with_maximum else None,
    )


def _orientation(vertical: bool) -> dict:
    """The box-orientation keyword this matplotlib accepts.

    ``vert=bool`` is deprecated from 3.11 and removed in 3.13; ``orientation=`` does not
    exist before 3.11. The declared floor is 3.8, so both spellings have to be reachable
    -- and passing the wrong one is either a warning now or a TypeError later.
    """
    import matplotlib

    parts = matplotlib.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):  # a dev or unusual version string
        major, minor = 3, 11
    if (major, minor) >= (3, 11):
        return {"orientation": "vertical" if vertical else "horizontal"}
    return {"vert": vertical}


def draw_boxes(
    ax,
    stats: BoxStats,
    *,
    colors: Sequence[str] | None = None,
    vertical: bool = True,
    show_maximum: bool = False,
    hline: float | None = None,
    width: float = 0.65,
) -> None:
    """Draw precomputed boxes with the project's styling.

    `hline` is where a null belongs -- read it off a
    :class:`~scads_drvi.scores.cell.CellScores`, which carries its own, rather than
    typing 0 or 1 here.
    """
    boxes = ax.bxp(
        stats.to_bxp(),
        showfliers=False,
        patch_artist=True,
        widths=width,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 0.6},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
        **_orientation(vertical),
    )
    if colors is not None:
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")

    if show_maximum and stats.maximum is not None:
        positions = np.arange(1, len(stats) + 1)
        if vertical:
            ax.plot(positions, stats.maximum, "o", ms=2.0, color="#333333", zorder=5)
        else:
            ax.plot(stats.maximum, positions, "o", ms=2.0, color="#333333", zorder=5)

    if hline is not None:
        (ax.axhline if vertical else ax.axvline)(
            hline, color="#949494", linestyle="--", linewidth=0.8, zorder=1
        )
