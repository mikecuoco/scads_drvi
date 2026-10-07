"""Per-cell disease scores -- both of them, named apart.

Two different quantities have been called ``cs`` in this pipeline, and nothing on disk
recorded which one produced a given column:

``ScoreKind.Z_WEIGHTED``
    ``cs_i = sum_k L_ik * max(0, z_k)``. An unnormalised, non-negative weighted sum of
    loadings. **Null is 0.** This is what the interpretation notebooks computed.

``ScoreKind.SCADS_RATIO``
    ``cs_i = (sum_k L_ik a_k e_k) / (sum_k L_ik a_k)``. A weighted mean enrichment, with
    empirical-Bayes shrunk enrichments. **Null is 1.** :func:`cs_from_enrichment` computes
    it; the workflow's scoring step writes it to disk.

They are not on the same scale and do not share a null, so a plot that draws a reference
line at 0 for one and 1 for the other is right both times and wrong if the columns are
swapped. :class:`CellScores` carries its own null so a figure reads it from the data
rather than from a constant typed next to the axis.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "ScoreKind",
    "NULL_VALUE",
    "CellScores",
    "factor_weights",
    "cs_from_z",
    "cs_from_enrichment",
    "read_cell_scores",
    "compare_scores",
]


class ScoreKind(StrEnum):
    """Which formula produced a score column. Determines its null and its label."""

    Z_WEIGHTED = "z_weighted"
    SCADS_RATIO = "scads_ratio"


#: The value a score takes under no association, per kind.
NULL_VALUE: Mapping[ScoreKind, float] = {
    ScoreKind.Z_WEIGHTED: 0.0,
    ScoreKind.SCADS_RATIO: 1.0,
}

_LABEL: Mapping[ScoreKind, str] = {
    ScoreKind.Z_WEIGHTED: "$CS_i$ (z-weighted loading sum)",
    ScoreKind.SCADS_RATIO: "$CS_i$ (mean enrichment)",
}


@dataclass(frozen=True)
class CellScores:
    """Per-cell scores plus the provenance needed to plot and interpret them."""

    values: pd.Series
    kind: ScoreKind
    model: str
    trait: str
    factors: tuple[str, ...] = ()
    weights: np.ndarray | None = field(default=None, repr=False)
    n_unscored: int = 0

    @property
    def dims(self) -> tuple[str, ...]:
        """Deprecated name of :attr:`factors`."""
        warnings.warn(
            "`CellScores.dims` is now `CellScores.factors`", DeprecationWarning, stacklevel=2
        )
        return self.factors

    @property
    def null(self) -> float:
        """The no-association value. Read this; never hardcode 0 or 1 beside an axis."""
        return NULL_VALUE[self.kind]

    @property
    def label(self) -> str:
        """Axis label naming which score this is."""
        return _LABEL[self.kind]

    def __len__(self) -> int:
        return len(self.values)

    def describe(self) -> pd.Series:
        """Summary statistics, with the null included for reference."""
        import pandas as pd

        stats = self.values.describe()
        stats["null"] = self.null
        return pd.Series(stats, name=f"{self.model}:{self.trait}")


def factor_weights(
    results: pd.DataFrame,
    *,
    factors: Iterable[str] | None = None,
    column: str = "Coefficient_z-score",
    clip_negative: bool = True,
    factor_column: str | None = None,
    dims: Iterable[str] | None = None,
    dim_column: str | None = None,
) -> np.ndarray:
    """Weights aligned to `factors`, in that order.

    `results` has one row per factor, keyed by `factor_column` (default ``"factor"``).
    `clip_negative` applies ``max(0, z)``: a factor whose annotation carries *less*
    heritability than baseline is evidence against involvement, and letting it subtract
    from a cell's score would let depletion in one factor mask enrichment in another.

    `dims` and `dim_column` are the deprecated old names of `factors` and `factor_column`.
    """
    from scads_drvi._util.compat import renamed_kwarg, with_factor_column

    factors = renamed_kwarg(factors, dims, new="factors", old="dims")
    if factors is None:
        raise TypeError("factor_weights() missing required keyword argument: 'factors'")
    factor_column = renamed_kwarg(
        factor_column, dim_column, new="factor_column", old="dim_column"
    ) or "factor"
    factors = list(factors)
    if column not in results.columns:
        raise KeyError(f"{column!r} not in results; columns are {list(results.columns)}")
    if factor_column == "factor":
        results = with_factor_column(results)
    if factor_column not in results.columns:
        raise KeyError(f"{factor_column!r} not in results")

    indexed = results.set_index(factor_column)
    missing = [d for d in factors if d not in indexed.index]
    if missing:
        raise KeyError(f"no result row for {len(missing)} requested factor(s), e.g. {missing[:5]}")
    if indexed.index.has_duplicates:
        raise ValueError(
            f"{factor_column!r} is not unique in results -- did you pass more than one "
            f"trait? Filter to one trait before computing weights."
        )

    weights = indexed.loc[factors, column].to_numpy(dtype=float)
    if clip_negative:
        weights = np.maximum(weights, 0.0)
    return weights


def cs_from_z(
    loadings: pd.DataFrame,
    results: pd.DataFrame,
    *,
    model: str,
    trait: str,
    column: str = "Coefficient_z-score",
    clip_negative: bool = True,
    chunk_rows: int = 200_000,
) -> CellScores:
    """``cs_i = sum_k L_ik * max(0, z_k)`` -- the interpretation notebooks' score.

    `loadings` is cells x factors -- for a directional score, pass a ReLU'd view of
    `embed.X` (``np.clip(embed.X, 0, None)`` for positive, ``np.clip(-embed.X, 0, None)``
    for negative). Only the factors present in `results` are used, in the order
    `results` gives them; when `results` carries a
    ``direction`` column, filter it to one direction before calling this (e.g.
    ``results.query("direction == 'pos'")``) so the weights line up with a `loadings`
    that was ReLU'd the same way.

    Computed in row chunks: the loadings table is hundreds of megabytes and the
    notebooks materialised the whole product at once.
    """
    import pandas as pd

    from scads_drvi._util.compat import with_factor_column

    if not isinstance(loadings, pd.DataFrame):
        raise TypeError("loadings must be a DataFrame of cells x factors")

    results = with_factor_column(results)
    factors = [d for d in results["factor"].tolist() if d in loadings.columns]
    if not factors:
        raise ValueError(
            "no factor in the results table is a column of the loadings matrix. "
            f"Results name e.g. {results['factor'].tolist()[:3]}; loadings columns are "
            f"e.g. {list(loadings.columns)[:3]}."
        )

    weights = factor_weights(results, factors=factors, column=column, clip_negative=clip_negative)

    n = len(loadings)
    out = np.empty(n, dtype=np.float64)
    step = max(int(chunk_rows), 1)
    for start in range(0, n, step):
        block = loadings.iloc[start : start + step][factors].to_numpy(dtype=np.float64)
        out[start : start + step] = block @ weights

    values = pd.Series(out, index=loadings.index, name="cs")
    return CellScores(
        values=values,
        kind=ScoreKind.Z_WEIGHTED,
        model=model,
        trait=trait,
        factors=tuple(factors),
        weights=weights,
        n_unscored=int(len(results) - len(factors)),
    )


def cs_from_enrichment(
    loadings: pd.DataFrame,
    enrichment: pd.Series,
    annot_size: pd.Series,
    *,
    model: str,
    trait: str,
    exclude: Iterable[str] = (),
    chunk_rows: int = 200_000,
) -> CellScores:
    """``cs_i = sum_k L_ik a_k e_k / sum_k L_ik a_k`` -- the SCADS enrichment ratio (null 1).

    The score is a mean of the factors' enrichments `e_k`, weighted by how much of cell
    i's loading sits on each factor and by how large the factor's annotation is (`a_k`;
    SCADS's variant count, here the annotation's genome-wide ``.l2.M``). A cell with no
    loading on any scored factor has no defined score: it is ``NaN`` and counted in
    ``n_unscored`` (never 0, which would read as "measured, and low").

    `loadings` is cells x keys, where each column is one annotation: for a directional
    factor pass a ReLU'd view per direction (``np.clip(L, 0, None)`` for ``pos``,
    ``np.clip(-L, 0, None)`` for ``neg``) as separate columns, named like the keys of
    `enrichment` and `annot_size` (both indexed by key). Scaling a cell's loadings, or
    every `annot_size`, by a constant leaves the score unchanged.

    `exclude` names keys that get no weight (SCADS drops annotations covering under 0.5%
    of the genome, whose S-LDSC error estimates are unreliable). Every other key must
    have a finite `enrichment` and a positive `annot_size`, or this raises naming them:
    guessing a weight for a bad estimate would move every cell's score.

    `CellScores.weights` holds the numerator weights ``a_k * e_k`` of the scored keys.
    Computed in row chunks, as :func:`cs_from_z` is.
    """
    import pandas as pd

    if not isinstance(loadings, pd.DataFrame):
        raise TypeError("loadings must be a DataFrame of cells x keys")

    excluded = set(exclude)
    keys = [k for k in loadings.columns if k not in excluded]
    if not keys:
        raise ValueError("no key is left to score: every loadings column is excluded")
    for name, series in (("enrichment", enrichment), ("annot_size", annot_size)):
        missing = [k for k in keys if k not in series.index]
        if missing:
            raise KeyError(f"{name} has no value for {len(missing)} key(s), e.g. {missing[:5]}")

    e = enrichment.loc[keys].to_numpy(dtype=np.float64)
    a = annot_size.loc[keys].to_numpy(dtype=np.float64)
    bad_e = [k for k, v in zip(keys, e, strict=True) if not np.isfinite(v)]
    bad_a = [k for k, v in zip(keys, a, strict=True) if not (np.isfinite(v) and v > 0)]
    if bad_e or bad_a:
        raise ValueError(
            f"{len(bad_e)} key(s) have a non-finite enrichment (e.g. {bad_e[:3]}) and "
            f"{len(bad_a)} have a non-positive annot_size (e.g. {bad_a[:3]}); pass them in "
            "`exclude` to leave them out of the score"
        )

    numerator_weights = a * e
    n = len(loadings)
    out = np.full(n, np.nan, dtype=np.float64)
    step = max(int(chunk_rows), 1)
    for start in range(0, n, step):
        block = loadings.iloc[start : start + step][keys].to_numpy(dtype=np.float64)
        denominator = block @ a
        scored = denominator > 0
        out[start : start + step][scored] = (block @ numerator_weights)[scored] / denominator[scored]

    values = pd.Series(out, index=loadings.index, name="cs")
    return CellScores(
        values=values,
        kind=ScoreKind.SCADS_RATIO,
        model=model,
        trait=trait,
        factors=tuple(keys),
        weights=numerator_weights,
        n_unscored=int(np.isnan(out).sum()),
    )


def read_cell_scores(
    path: str | Path,
    *,
    model: str,
    trait: str,
    column: str = "cs",
    index_column: str = "barcode",
) -> CellScores:
    """Read the scoring stage's ``cell_scores.tsv`` as a SCADS-ratio score.

    Kept distinct from :func:`cs_from_z` on purpose -- this column has a null of 1, not
    0, and the two are not comparable without :func:`compare_scores`.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no cell scores at {path}. They are written by the scoring stage; the "
            f"interpretation notebooks recomputed a different score instead of reading "
            f"this file, which is how the two came to share a name."
        )
    frame = pd.read_csv(path, sep="\t")
    if column not in frame.columns:
        raise KeyError(f"{column!r} not in {path}; columns are {list(frame.columns)}")
    if index_column in frame.columns:
        frame = frame.set_index(index_column)
    values = frame[column].astype(float)
    values.name = "cs"
    return CellScores(
        values=values,
        kind=ScoreKind.SCADS_RATIO,
        model=model,
        trait=trait,
        n_unscored=int(values.isna().sum()),
    )


def compare_scores(left: CellScores, right: CellScores) -> pd.DataFrame:
    """Correlate two scores over the cells they share.

    Both Pearson and Spearman: the two formulas are on different scales with different
    nulls, so a high rank correlation with a low linear one would mean they order cells
    the same way while disagreeing on magnitude -- which is the interesting case, and
    the one a single coefficient would hide.
    """
    import pandas as pd

    if left.kind is right.kind:
        raise ValueError(
            f"both scores are {left.kind.value}; compare_scores is for telling the two "
            f"different formulas apart"
        )

    shared = left.values.index.intersection(right.values.index)
    if len(shared) == 0:
        raise ValueError("the two score sets share no cells")

    a = left.values.loc[shared]
    b = right.values.loc[shared]
    both = pd.DataFrame({left.kind.value: a, right.kind.value: b}).dropna()
    return pd.DataFrame(
        {
            "n_shared": [len(shared)],
            "n_compared": [len(both)],
            "pearson": [both.iloc[:, 0].corr(both.iloc[:, 1], method="pearson")],
            "spearman": [both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")],
            "null_left": [left.null],
            "null_right": [right.null],
        }
    )
