#!/usr/bin/env python3
"""Factor selection for the enrichment stage.

The one algorithm with real content here is factor selection, and it is a
STAGE rather than a loop filter. K_eff sets the annotation column count, the
number of --h2 runs, a_k, the annotation correlation matrix, and the pool the
empirical-Bayes shrinkage is fitted over -- a vanished dimension left in that
pool moves every other factor's shrunk estimate. So it is resolved once, here,
and everything downstream keys on the factor_map :func:`select_factors` builds.

**This module imports numpy and pandas at module scope**, and is the only one
in the package that does. That is deliberate rather than an oversight: it is
reachable only from an environment that already carries both, so the lazy
convention the rest of the package follows would buy nothing here.
tests/test_import_surface.py records the exemption.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData

# `log` is re-exported: enrichment scripts import it here.
from scads_drvi._util.progress import log_out as log  # noqa: F401  (re-exported)

# ---------------------------------------------------------------------------
# Factor selection (stage 1a)
# ---------------------------------------------------------------------------

def _as_bool(col: pd.Series) -> pd.Series:
    """Coerce a possibly-string boolean column to real bools.

    latent_stats.tsv is written by pandas from numpy bools, so it round-trips
    as literal "True"/"False". pandas usually re-infers that as bool dtype, but
    not if the column picked up any other value -- in which case guessing
    silently is worse than failing.
    """
    if col.dtype == bool:
        return col
    mapped = col.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        bad = sorted(set(col.astype(str)[mapped.isna()]))
        raise ValueError(f"non-boolean values in `vanished`: {bad[:5]}")
    return mapped.astype(bool)


def select_factors(
    latent_stats: pd.DataFrame,
    dim_names: Sequence[str],
    exclude_vanished: bool = True,
    exclude_dims: Iterable[str] = (),
) -> pd.DataFrame:
    """Build the factor map: which dimensions get enriched, and under which k index.

    `latent_stats` is indexed by dimension name and needs only a `vanished` column
    (plus, optionally, `vanished_positive_direction`) -- typically `embed.var` itself,
    since DRVI's own ``model.set_latent_dimension_stats`` already wrote those columns
    onto it (see the getting-started guide's training example). No reshaping is
    needed to call this.

    ``dim_names`` is authoritative for ordering and membership -- it comes from
    topic_loadings.tsv, i.e. the contract, not from the inspect artifact.

    Returns a frame with one row per dimension and columns
    ``dim, vanished, kept, drop_reason, annot_index``. ``annot_index`` is the
    1-based k used in S-LDSC filenames (k1..kK_eff) and is assigned over KEPT
    dimensions only; dropped rows get <NA>.

    Upstream SCADS names annotations positionally, so without this mapping the
    route from `k7.results` back to `dim_*` after a drop is guesswork. Nothing
    downstream may renumber on its own.
    """
    dim_names = list(dim_names)
    stats = latent_stats.set_axis(latent_stats.index.astype(str), axis=0)
    stats_dims = list(stats.index)
    if set(stats_dims) != set(dim_names):
        only_stats = sorted(set(stats_dims) - set(dim_names))
        only_contract = sorted(set(dim_names) - set(stats_dims))
        raise ValueError(
            "latent_stats does not describe the same dimensions as the "
            f"loadings. Only in latent_stats: {only_stats}; only in loadings: "
            f"{only_contract}. These must come from the same fit."
        )

    exclude_dims = set(exclude_dims or ())
    unknown = exclude_dims - set(dim_names)
    if unknown:
        raise ValueError(f"factors.exclude_dims names unknown dimensions: {sorted(unknown)}")

    vanished_col = _as_bool(stats["vanished"])

    # DRVI derives both of these from the signed latent: `vanished` is
    # `|X|.max(over cells) < threshold` and `vanished_positive_direction` is
    # `X.max(...) < threshold`. So `vanished` is exactly
    # (`vanished_positive_direction` and `vanished_negative_direction`), which
    # means these two columns can only ever disagree one way -- a dimension live
    # in the negative direction alone.
    #
    # Which flag is the right one is a property of the *caller*, not of the fit,
    # so report the condition rather than prescribing a fix. A positive-only
    # consumer (loadings ReLU'd to the positive side, factors read as the positive
    # OOD log-fold-change) should treat such a dimension as vanished. A
    # direction-aware consumer -- one that scores `_pos` and `_neg` separately,
    # the way the S-LDSC sweep does -- should keep it and use only its negative
    # direction. `select_factors` filters on `vanished`, so it keeps it either
    # way; a positive-only caller that wants it dropped should pass it via
    # `exclude_dims`.
    pos_col = "vanished_positive_direction"
    if pos_col in stats.columns:
        pos_vanished_col = _as_bool(stats[pos_col])
        disagree = stats.index[vanished_col != pos_vanished_col].tolist()
        if disagree:
            log(
                f"NOTE: {disagree} are live in the negative direction only "
                f"(`vanished` and `{pos_col}` disagree). Kept, since `vanished` "
                "governs selection. Positive-only consumers should drop them; "
                "direction-aware consumers should use their negative direction."
            )

    rows = []
    for dim in dim_names:
        vanished = bool(vanished_col.loc[dim])
        reason = ""
        if exclude_vanished and vanished:
            reason = "vanished"
        elif dim in exclude_dims:
            reason = "exclude_dims"
        rows.append({"dim": dim, "vanished": vanished, "kept": reason == "",
                     "drop_reason": reason})

    fmap = pd.DataFrame(rows)
    fmap["annot_index"] = pd.array(
        [None] * len(fmap), dtype="Int64"
    )
    fmap.loc[fmap["kept"], "annot_index"] = np.arange(1, int(fmap["kept"].sum()) + 1)

    if not fmap["kept"].any():
        raise ValueError(
            "every dimension was excluded -- nothing left to enrich. Check "
            "factors.exclude_vanished / factors.exclude_dims in config.yaml."
        )
    return fmap


def kept_dims(fmap: pd.DataFrame) -> list[str]:
    """Kept dimension names, in annot_index order."""
    keep = fmap[fmap["kept"]].sort_values("annot_index")
    return keep["dim"].tolist()


# ---------------------------------------------------------------------------
# The result h5ad -- obs = cells, var = one row per latent dimension, built and
# written with plain anndata calls at the call site (see the getting-started guide)
# ---------------------------------------------------------------------------

def kept_loadings(embed: AnnData, fmap: pd.DataFrame) -> pd.DataFrame:
    """cells x kept-dims, straight from a result h5ad's signed ``X`` -- no ReLU.

    A caller that needs one direction's non-negative loadings (for S-LDSC's top-frac
    annotation ranking) derives it explicitly first (``np.clip(embed.X, 0, None)`` for
    the positive direction, ``np.clip(-embed.X, 0, None)`` for negative); this function
    only subsets to the kept dimensions, in `fmap`'s order, from whatever `embed` it is
    given.
    """
    import pandas as pd

    keep = kept_dims(fmap)
    # embed.X is typed as a broad ndarray/sparse/backed union; this module's own
    # docstring is explicit that it's reachable only from an in-memory, dense fit.
    X = cast(np.ndarray, embed[:, keep].X)
    frame = pd.DataFrame(X, index=embed.obs_names, columns=keep)
    return frame


def kept_feature_loadings(feature_loadings: AnnData, fmap: pd.DataFrame) -> pd.DataFrame:
    """peaks x kept-dims, from the companion loadings h5ad (however it was saved and
    loaded, e.g. plain ``anndata.read_h5ad``)."""
    import pandas as pd

    keep = kept_dims(fmap)
    X = cast(np.ndarray, feature_loadings[:, keep].X)
    frame = pd.DataFrame(X, index=feature_loadings.obs_names, columns=keep)
    return frame


def parse_peaks(index: Iterable[str]) -> pd.DataFrame:
    """Split canonical chr:start-end names into a chrom/start/end frame.

    Coordinates are returned as the peak name states them, which for this pipeline's
    output is 0-based half-open (BED), matching what the peak merge wrote.
    """
    chrom, start, end = [], [], []
    for name in index:
        seq, _, rest = str(name).partition(":")
        lo, _, hi = rest.partition("-")
        chrom.append(seq)
        start.append(int(lo))
        end.append(int(hi))
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end})
