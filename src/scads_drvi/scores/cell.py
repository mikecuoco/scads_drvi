"""Per-cell disease scores -- both of them, named apart.

Two different quantities have been called ``cs`` in this pipeline, and nothing on disk
recorded which one produced a given column:

``ScoreKind.Z_WEIGHTED``
    ``cs_i = sum_k L_ik * max(0, z_k)``. An unnormalised, non-negative weighted sum of
    loadings. **Null is 0.** This is what the interpretation notebooks computed.

``ScoreKind.SCADS_RATIO``
    ``cs_i = (sum_k L_ik a_k e_k) / (sum_k L_ik a_k)``. A weighted mean enrichment, with
    empirical-Bayes shrunk enrichments. **Null is 1.** This is what the scoring stage
    writes to disk.

They are not on the same scale and do not share a null, so a plot that draws a reference
line at 0 for one and 1 for the other is right both times and wrong if the columns are
swapped. :class:`CellScores` carries its own null so a figure reads it from the data
rather than from a constant typed next to the axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from scads_drvi.labels import FactorLabels

__all__ = [
    "ScoreKind",
    "NULL_VALUE",
    "CellScores",
    "factor_weights",
    "cs_from_z",
    "read_cell_scores",
    "compare_scores",
]


class ScoreKind(str, Enum):
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

    values: "pd.Series"
    kind: ScoreKind
    model: str
    trait: str
    dims: tuple[str, ...] = ()
    weights: np.ndarray | None = field(default=None, repr=False)
    n_unscored: int = 0

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

    def describe(self) -> "pd.Series":
        """Summary statistics, with the null included for reference."""
        import pandas as pd

        stats = self.values.describe()
        stats["null"] = self.null
        return pd.Series(stats, name=f"{self.model}:{self.trait}")


def factor_weights(
    results: "pd.DataFrame",
    *,
    dims: Iterable[str],
    column: str = "Coefficient_z-score",
    clip_negative: bool = True,
    dim_column: str = "dim",
) -> np.ndarray:
    """Weights aligned to `dims`, in that order.

    `clip_negative` applies ``max(0, z)``: a factor whose annotation carries *less*
    heritability than baseline is evidence against involvement, and letting it subtract
    from a cell's score would let depletion in one factor mask enrichment in another.
    """
    dims = list(dims)
    if column not in results.columns:
        raise KeyError(f"{column!r} not in results; columns are {list(results.columns)}")
    if dim_column not in results.columns:
        raise KeyError(f"{dim_column!r} not in results")

    indexed = results.set_index(dim_column)
    missing = [d for d in dims if d not in indexed.index]
    if missing:
        raise KeyError(
            f"no result row for {len(missing)} requested factor(s), e.g. {missing[:5]}"
        )
    if indexed.index.has_duplicates:
        raise ValueError(
            f"{dim_column!r} is not unique in results -- did you pass more than one "
            f"trait? Filter to one trait before computing weights."
        )

    weights = indexed.loc[dims, column].to_numpy(dtype=float)
    if clip_negative:
        weights = np.maximum(weights, 0.0)
    return weights


def cs_from_z(
    loadings: "pd.DataFrame",
    results: "pd.DataFrame",
    *,
    model: str,
    trait: str,
    labels: "FactorLabels | None" = None,
    column: str = "Coefficient_z-score",
    clip_negative: bool = True,
    chunk_rows: int = 200_000,
) -> CellScores:
    """``cs_i = sum_k L_ik * max(0, z_k)`` -- the interpretation notebooks' score.

    `loadings` is cells x factors. Only the factors present in `results` are used, in
    the order `results` gives them.

    Computed in row chunks: the loadings table is hundreds of megabytes and the
    notebooks materialised the whole product at once.
    """
    import pandas as pd

    if not isinstance(loadings, pd.DataFrame):
        raise TypeError("loadings must be a DataFrame of cells x factors")

    dims = [d for d in results["dim"].tolist() if d in loadings.columns]
    if not dims:
        raise ValueError(
            "no factor in the results table is a column of the loadings matrix. "
            f"Results name e.g. {results['dim'].tolist()[:3]}; loadings columns are "
            f"e.g. {list(loadings.columns)[:3]}."
        )
    if labels is not None:
        labels.assert_index_dims(dims)

    weights = factor_weights(
        results, dims=dims, column=column, clip_negative=clip_negative
    )

    n = len(loadings)
    out = np.empty(n, dtype=np.float64)
    step = max(int(chunk_rows), 1)
    for start in range(0, n, step):
        block = loadings.iloc[start : start + step][dims].to_numpy(dtype=np.float64)
        out[start : start + step] = block @ weights

    values = pd.Series(out, index=loadings.index, name="cs")
    return CellScores(
        values=values,
        kind=ScoreKind.Z_WEIGHTED,
        model=model,
        trait=trait,
        dims=tuple(dims),
        weights=weights,
        n_unscored=int(len(results) - len(dims)),
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


def compare_scores(left: CellScores, right: CellScores) -> "pd.DataFrame":
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
