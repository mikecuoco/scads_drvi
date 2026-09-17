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

import numpy as np
import pandas as pd

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
    stats_dims = list(latent_stats["dim"].astype(str))
    if set(stats_dims) != set(dim_names):
        only_stats = sorted(set(stats_dims) - set(dim_names))
        only_contract = sorted(set(dim_names) - set(stats_dims))
        raise ValueError(
            "latent_stats.tsv does not describe the same dimensions as the "
            f"loadings. Only in latent_stats: {only_stats}; only in loadings: "
            f"{only_contract}. These must come from the same fit."
        )

    stats = latent_stats.set_index(latent_stats["dim"].astype(str))
    exclude_dims = set(exclude_dims or ())
    unknown = exclude_dims - set(dim_names)
    if unknown:
        raise ValueError(f"factors.exclude_dims names unknown dimensions: {sorted(unknown)}")

    # The loadings are a ReLU of the latent (positive side only) and the factors
    # are the positive OOD log-fold-change, so `vanished_positive_direction` is
    # strictly the flag that matches the contract. On both current fits it
    # agrees with `vanished` on every dimension; say so loudly if that ever
    # stops being true rather than quietly picking one.
    pos_col = "vanished_positive_direction"
    if pos_col in stats.columns:
        disagree = stats.index[stats["vanished"] != stats[pos_col]].tolist()
        if disagree:
            log(
                f"WARNING: `vanished` and `{pos_col}` disagree on {disagree}. "
                "The contract's loadings are a ReLU, so the positive-direction "
                "flag is the relevant one -- review before trusting this run."
            )

    rows = []
    for dim in dim_names:
        vanished = bool(stats.loc[dim, "vanished"])
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

def latent_stats_from_embed(embed) -> pd.DataFrame:
    """``embed.var`` reshaped to the ``dim, vanished, ...`` frame :func:`select_factors`
    expects.

    DRVI's own ``model.set_latent_dimension_stats`` already wrote these columns onto
    ``embed.var`` when the object was built (see the getting-started guide's training
    example), so there is no file to read -- this just gives the in-memory table the
    shape the rest of this module was written against.
    """
    frame = embed.var.reset_index(names="dim")
    frame["dim"] = frame["dim"].astype(str)
    for column in ("vanished", "vanished_positive_direction", "vanished_negative_direction"):
        if column in frame.columns:
            frame[column] = _as_bool(frame[column])
    return frame


def kept_loadings(embed, fmap: pd.DataFrame):
    """cells x kept-dims, straight from a result h5ad's signed ``X`` -- no ReLU.

    A caller that needs one direction's non-negative loadings (for S-LDSC's top-frac
    annotation ranking) derives it explicitly first (``np.clip(embed.X, 0, None)`` for
    the positive direction, ``np.clip(-embed.X, 0, None)`` for negative); this function
    only subsets to the kept dimensions, in `fmap`'s order, from whatever `embed` it is
    given.
    """
    import pandas as pd

    keep = kept_dims(fmap)
    frame = pd.DataFrame(embed[:, keep].X, index=embed.obs_names, columns=keep)
    return frame


def kept_feature_loadings(feature_loadings, fmap: pd.DataFrame):
    """peaks x kept-dims, from the companion loadings h5ad (however it was saved and
    loaded, e.g. plain ``anndata.read_h5ad``)."""
    import pandas as pd

    keep = kept_dims(fmap)
    frame = pd.DataFrame(
        feature_loadings[:, keep].X, index=feature_loadings.obs_names, columns=keep
    )
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
