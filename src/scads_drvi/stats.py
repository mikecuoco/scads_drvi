"""One-tailed p-values and Benjamini-Hochberg, with one implementation.

Three different BH step-ups existed across this pipeline, and the one the interpretation
notebooks used did not work::

    ranks = pd.Series(p).rank(method="first")
    fdr = p * len(p) / ranks
    fdr = np.minimum.accumulate(fdr.sort_values(ascending=False))[fdr.index]

``np.minimum.accumulate`` over a sequence that is *already sorted descending* returns it
unchanged, so the step-up monotonicity enforcement never fired and the reported q-values
were raw ``p*m/rank``, which is not monotone in p. :func:`compare_bh` reproduces that
arithmetic beside the correct one so the difference can be measured on real results
before anything downstream is re-run.

:func:`bh_qvalues` matches R's ``p.adjust(method="BH")``, including its explicit ``n``
argument -- the SCADS binarisation deliberately inflates the denominator to a
genome-wide bin count rather than the number of tests performed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "p_one_tailed",
    "bh_qvalues",
    "bh_threshold_z",
    "legacy_bh_qvalues",
    "compare_bh",
    "add_fdr",
    "significant",
]

Tail = Literal["upper", "lower"]


def p_one_tailed(z, *, tail: Tail = "upper") -> np.ndarray:
    """One-tailed p-value for a z-score.

    ``upper`` is the default because an S-LDSC coefficient z-score is tested against the
    one-sided alternative that an annotation carries *more* heritability than baseline;
    a two-tailed p there would be answering a question nobody asked.
    """
    from scipy.stats import norm

    z = np.asarray(z, dtype=float)
    if tail == "upper":
        return np.asarray(norm.sf(z), dtype=float)
    if tail == "lower":
        return np.asarray(norm.cdf(z), dtype=float)
    raise ValueError(f"tail must be 'upper' or 'lower', got {tail!r}")


def bh_qvalues(p, n: int | None = None) -> np.ndarray:
    """Benjamini-Hochberg q-values. Equivalent to R's ``p.adjust(p, "BH", n=n)``.

    `n` is the multiplicity denominator, defaulting to the number of finite p-values.
    Pass it explicitly when the tests performed are a subset of the tests conceptually
    being corrected over.

    NaN p-values map to q=1 and do not contribute to `n`: a test that could not be run
    is not evidence, but dropping it silently would shrink the denominator and make
    every other q optimistic.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p must be 1-D, got shape {p.shape}")

    finite = np.isfinite(p)
    m = int(finite.sum())
    q = np.ones_like(p, dtype=float)
    if m == 0:
        return q

    denominator = m if n is None else int(n)
    if denominator < m:
        raise ValueError(
            f"n={denominator} is smaller than the {m} finite p-values being corrected"
        )

    values = p[finite]
    order = np.argsort(values, kind="stable")
    ranks = np.arange(1, m + 1, dtype=float)

    scaled = values[order] * denominator / ranks
    # Enforce monotonicity from the largest p downwards. This is the step that the
    # notebooks' version silently skipped.
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.clip(scaled, 0.0, 1.0)
    q[finite] = adjusted
    return q


def legacy_bh_qvalues(p) -> np.ndarray:
    """The interpretation notebooks' BH arithmetic, reproduced exactly.

    Kept only so :func:`compare_bh` can quantify what changes. Do not use it: the
    monotonicity step is a no-op (see the module docstring), so the result is
    ``p*m/rank`` clipped at 1.
    """
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    # rank(method="first") over the raw p-values, ties broken by position.
    order = np.argsort(p, kind="stable")
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m + 1, dtype=float)
    return np.clip(p * m / ranks, 0.0, 1.0)


def compare_bh(p, n: int | None = None) -> pd.DataFrame:
    """Correct vs legacy q-values side by side, with the significance flips called out.

    Columns: ``p``, ``q_correct``, ``q_legacy``, ``delta``, and ``flips_at_05``. The last
    is what actually matters -- a q-value that moves but stays on the same side of the
    threshold changes no conclusion.
    """
    import pandas as pd

    p = np.asarray(p, dtype=float)
    correct = bh_qvalues(p, n=n)
    legacy = legacy_bh_qvalues(p)
    return pd.DataFrame(
        {
            "p": p,
            "q_correct": correct,
            "q_legacy": legacy,
            "delta": correct - legacy,
            "flips_at_05": (correct < 0.05) != (legacy < 0.05),
        }
    )


def bh_threshold_z(
    q, z, *, alpha: float = 0.05, tail: Tail = "upper"
) -> float | None:
    """The least extreme z still passing BH at `alpha`, or None if nothing passes.

    For drawing a significance line on an axis whose units are z. Computing it beats
    hardcoding one: the boundary depends on the whole p-vector, so a number copied from
    a previous run is wrong for this one.

    `tail` must match the tail the p-values were computed under. Under ``upper`` the
    boundary is the smallest passing z; under ``lower`` it is the largest. Taking the
    minimum either way puts the line on the wrong side of the distribution and, because
    it is still a real z from the data, it looks entirely plausible there.
    """
    if tail not in ("upper", "lower"):
        raise ValueError(f"tail must be 'upper' or 'lower', got {tail!r}")

    q = np.asarray(q, dtype=float)
    z = np.asarray(z, dtype=float)
    if q.shape != z.shape:
        raise ValueError(f"q and z must have the same shape, got {q.shape} and {z.shape}")
    passing = np.isfinite(q) & np.isfinite(z) & (q < alpha)
    if not passing.any():
        return None
    return float(z[passing].min() if tail == "upper" else z[passing].max())


def add_fdr(
    frame: pd.DataFrame,
    *,
    by: str | Iterable[str] | None = None,
    z_col: str = "Coefficient_z-score",
    p_col: str = "p_1tailed",
    q_col: str = "fdr_q",
    n: int | None = None,
    tail: Tail = "upper",
) -> pd.DataFrame:
    """Return a copy of `frame` with one-tailed p and BH q columns added.

    `by` names the columns that delimit a correction family -- typically whatever
    distinguishes independent analyses in the same table, such as the trait. Making it an
    argument keeps the scope of the correction visible at the call site instead of
    implicit in a loop, which is how a table ends up corrected across families that were
    never meant to be pooled.

    With ``by=None`` the whole frame is one family.
    """
    import pandas as pd

    if z_col not in frame.columns:
        raise KeyError(
            f"{z_col!r} not in frame; columns are {list(frame.columns)}"
        )

    out = frame.copy()
    out[p_col] = p_one_tailed(out[z_col].to_numpy(), tail=tail)

    if by is None:
        out[q_col] = bh_qvalues(out[p_col].to_numpy(), n=n)
        return out

    keys = [by] if isinstance(by, str) else list(by)
    missing = [k for k in keys if k not in out.columns]
    if missing:
        raise KeyError(f"cannot group by {missing}; columns are {list(out.columns)}")

    q = pd.Series(np.ones(len(out), dtype=float), index=out.index)
    for _, block in out.groupby(keys, dropna=False, observed=True, sort=False):
        q.loc[block.index] = bh_qvalues(block[p_col].to_numpy(), n=n)
    out[q_col] = q
    return out


def significant(
    frame: pd.DataFrame, *, alpha: float = 0.05, q_col: str = "fdr_q"
) -> pd.Series:
    """Boolean mask at a stated alpha, so no caller re-types ``q < 0.05``."""
    if q_col not in frame.columns:
        raise KeyError(f"{q_col!r} not in frame; run add_fdr first")
    return frame[q_col] < alpha
