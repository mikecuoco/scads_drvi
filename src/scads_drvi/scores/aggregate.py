"""Summarising per-cell values over groups.

Every grouping is named by the caller. The module does not know what a group is, only
that thin groups produce unreliable means and must be visible as such rather than
quietly averaged in.

Two conventions worth stating, because they are choices and not defaults:

- A group below ``min_cells`` is **dropped and counted**, not silently kept. The count
  is on the returned object, so a summary can say how much it discarded.
- A masked cell in a matrix comes back as **NaN, not 0**. Grey means "no reliable value"
  and zero means "measured, and low"; conflating them turns missing data into a claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "GroupStats",
    "summarize_by",
    "group_matrix",
    "block_order",
    "profile_by_group",
    "eta_squared",
    "streaming_correlation",
]


@dataclass(frozen=True)
class GroupStats:
    """Per-group summary, plus what it took to produce it."""

    frame: pd.DataFrame
    keys: tuple[str, ...]
    value: str
    min_cells: int
    n_dropped: int = 0
    dropped: tuple[str, ...] = field(default=(), repr=False)

    def __len__(self) -> int:
        return len(self.frame)

    def order_by(self, column: str = "mean", *, ascending: bool = False) -> GroupStats:
        """Same statistics, reordered -- for a figure axis."""
        from dataclasses import replace

        if column not in self.frame.columns:
            raise KeyError(
                f"{column!r} not in the summary; columns are {list(self.frame.columns)}"
            )
        return replace(
            self, frame=self.frame.sort_values(column, ascending=ascending)
        )


def summarize_by(
    values: pd.Series,
    cells: pd.DataFrame,
    *,
    by: str | Sequence[str],
    min_cells: int = 20,
    ci: float = 0.95,
    quantiles: Sequence[float] = (),
) -> GroupStats:
    """Mean, SEM and confidence interval of `values` per group.

    `values` is indexed by cell; `cells` carries the grouping columns. Only the cells
    present in both are used, and that count is reported as ``n`` so a shrunken join is
    visible rather than inferred.
    """
    from scipy.stats import norm

    keys = [by] if isinstance(by, str) else list(by)
    missing = [k for k in keys if k not in cells.columns]
    if missing:
        raise KeyError(f"cannot group by {missing}; columns are {list(cells.columns)}")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    name = values.name or "value"
    joined = cells[keys].join(values.rename(name), how="inner")
    if joined.empty:
        raise ValueError(
            "the values and the cell table share no cells -- check that both are "
            "indexed the same way"
        )

    grouped = joined.groupby(keys, dropna=False, observed=True)[name]
    summary = grouped.agg(
        n="size", n_scored="count", mean="mean", median="median", std="std"
    )
    for q in quantiles:
        summary[f"q{q:g}"] = grouped.quantile(q / 100.0 if q > 1 else q)

    with np.errstate(invalid="ignore", divide="ignore"):
        sem = summary["std"].to_numpy(dtype=float) / np.sqrt(
            summary["n_scored"].to_numpy(dtype=float)
        )
    summary["sem"] = sem
    half = norm.ppf(0.5 + ci / 2.0) * sem
    summary["ci_low"] = summary["mean"] - half
    summary["ci_high"] = summary["mean"] + half

    thin = summary["n_scored"] < int(min_cells)
    dropped = tuple(str(i) for i in summary.index[thin])
    summary = summary.loc[~thin].reset_index()

    return GroupStats(
        frame=summary,
        keys=tuple(keys),
        value=str(name),
        min_cells=int(min_cells),
        n_dropped=int(thin.sum()),
        dropped=dropped,
    )


def group_matrix(
    values: pd.Series,
    cells: pd.DataFrame,
    *,
    index: str,
    columns: str,
    min_cells: int = 20,
    index_order: Sequence[str] | None = None,
    column_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(mean matrix, count matrix)`` over two grouping columns.

    Cells in a bin with fewer than `min_cells` observations are set to NaN in the mean
    matrix, and the count matrix says why. Returning both is the point: a reader of the
    mean alone cannot tell an empty bin from a low one.
    """

    for column in (index, columns):
        if column not in cells.columns:
            raise KeyError(f"{column!r} not in the cell table")

    name = values.name or "value"
    joined = cells[[index, columns]].join(values.rename(name), how="inner")
    if joined.empty:
        raise ValueError("the values and the cell table share no cells")

    means = joined.pivot_table(
        index=index, columns=columns, values=name, aggfunc="mean", observed=True
    )
    counts = joined.pivot_table(
        index=index, columns=columns, values=name, aggfunc="count", observed=True
    ).reindex(index=means.index, columns=means.columns).fillna(0).astype(int)

    means = means.mask(counts < int(min_cells))

    if index_order is not None:
        means = means.reindex(index=list(index_order))
        counts = counts.reindex(index=list(index_order), fill_value=0)
    if column_order is not None:
        means = means.reindex(columns=list(column_order))
        counts = counts.reindex(columns=list(column_order), fill_value=0)
    return means, counts


def block_order(
    stats: GroupStats | pd.DataFrame,
    *,
    group: str,
    block: str,
    value: str = "mean",
    ascending: bool = False,
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Order groups within parent blocks, for a blocked categorical axis.

    Returns the group order and, per block, its ``(name, start, stop)`` span in that
    order -- which is what a figure needs to draw the alternating background bands and
    the block labels without recomputing the grouping.

    Blocks follow the **column's own order**: a pandas categorical is laid out in its
    declared category order, anything else lexicographically. That is deliberate -- a
    categorical carries a meaningful ordering (developmental stage, anatomical
    position), and re-sorting it alphabetically would discard the one piece of
    information the dtype exists to record. Pass an already-ordered categorical to
    control the axis.
    """
    frame = stats.frame if isinstance(stats, GroupStats) else stats
    for column in (group, block, value):
        if column not in frame.columns:
            raise KeyError(f"{column!r} not in the summary; columns are "
                           f"{list(frame.columns)}")

    ordered = frame.sort_values([block, value], ascending=[True, ascending])
    names = ordered[group].astype(str).tolist()

    # `dropna=False` is load-bearing. `sort_values` KEEPS a row whose block is NaN (at the
    # end), so without it the group appeared in `names` while belonging to no span -- and a
    # caller drawing bands and block labels from those spans, as
    # `viz.enrichment.grouped_landscape` does, mislabelled the axis from that row onwards.
    spans: list[tuple[str, int, int]] = []
    start = 0
    for block_name, chunk in ordered.groupby(
        block, sort=True, observed=True, dropna=False
    ):
        stop = start + len(chunk)
        spans.append((str(block_name), start, stop))
        start = stop

    covered = sum(stop - start for _, start, stop in spans)
    if covered != len(names):
        raise ValueError(
            f"{covered} of {len(names)} groups fall inside a block span. The spans index "
            f"`names` positionally, so a gap silently shifts every block label after it."
        )
    return names, spans


def profile_by_group(
    matrix: pd.DataFrame,
    labels: Iterable,
    *,
    min_cells: int = 1,
    zscore: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Mean of each column per group, and its z within each column.

    Returns ``(means, z, counts)``. The z is taken **within a column**, i.e. across
    groups for one factor, because the question is which groups stand out on that
    factor -- not how a group's factors compare to each other, which is a different
    question with a different normalisation.

    Groups below `min_cells` become NaN rows rather than being dropped: a masked row
    and an absent row are different claims about the data.
    """
    import pandas as pd

    labels = pd.Series(list(labels), index=matrix.index, name="_group")
    grouped = matrix.groupby(labels, observed=True)
    means = grouped.mean()
    counts = grouped.size()

    thin = counts < int(min_cells)
    if thin.any():
        means.loc[thin[thin].index, :] = np.nan

    if zscore:
        centre = means.mean(axis=0)
        spread = means.std(axis=0, ddof=0).replace(0.0, np.nan)
        z = (means - centre) / spread
    else:
        z = means.copy()

    return means, z, counts.to_dict()


def eta_squared(values, groups) -> float:
    """One-way ANOVA eta-squared: the fraction of variance explained by a grouping.

    The categorical analogue of r-squared, for asking how much of a score is accounted
    for by a nuisance factor. Returns NaN when there is no variance to explain.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if values.shape != groups.shape:
        raise ValueError(
            f"values and groups must have the same shape, got {values.shape} "
            f"and {groups.shape}"
        )
    finite = np.isfinite(values)
    values, groups = values[finite], groups[finite]
    if values.size == 0:
        return float("nan")

    grand = values.mean()
    total = float(((values - grand) ** 2).sum())
    if total == 0.0:
        return float("nan")

    between = 0.0
    for group in np.unique(groups):
        member = values[groups == group]
        between += member.size * (member.mean() - grand) ** 2
    return float(between / total)


def streaming_correlation(chunks: Iterable[np.ndarray], n_columns: int) -> np.ndarray:
    """Pearson correlation between columns, from row chunks.

    Accumulates first and second moments instead of holding the matrix, so the peak
    memory is the chunk plus a k x k gram matrix rather than a float64 copy of the whole
    thing.
    """
    k = int(n_columns)
    total = np.zeros(k, dtype=np.float64)
    gram = np.zeros((k, k), dtype=np.float64)
    n = 0

    for chunk in chunks:
        block = np.asarray(chunk, dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != k:
            raise ValueError(
                f"each chunk must be (rows, {k}); got {getattr(block, 'shape', None)}"
            )
        total += block.sum(axis=0)
        gram += block.T @ block
        n += block.shape[0]

    if n < 2:
        raise ValueError(f"need at least 2 rows to correlate, got {n}")

    mean = total / n
    covariance = gram / n - np.outer(mean, mean)
    spread = np.sqrt(np.diag(covariance))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.outer(spread, spread)
    return np.clip(correlation, -1.0, 1.0)
