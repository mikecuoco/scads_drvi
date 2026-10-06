"""Reading a PLINK ``.bim`` and aligning its SNPs to genomic features.

`ldsc.py --l2 --thin-annot` accepts a thin annotation (bare annotation columns, no
``CHR BP SNP CM``) directly -- so all this package needs from
a `.bim` is its row order, since a per-SNP annotation must return one value per `.bim`
row, in `.bim` order.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    import pandas as pd

__all__ = ["BIM_COLUMNS", "read_bim", "assign_peaks", "restrict_to_reference_snps"]

#: Columns of a PLINK ``.bim``, in file order.
BIM_COLUMNS = ("CHR", "SNP", "CM", "BP", "A1", "A2")


def read_bim(path: str | Path) -> pd.DataFrame:
    """Read a PLINK ``.bim``, preserving file order.

    No index is set and no sorting is applied: row order *is* the contract, because
    every consumer of the result aligns positionally.
    """
    import pandas as pd

    frame = pd.read_csv(
        path, sep=r"\s+", header=None, names=list(BIM_COLUMNS), dtype={"SNP": str}
    )
    if frame.empty:
        raise ValueError(f"{path} has no variants")
    return frame


def assign_peaks(bim: pd.DataFrame, peaks_index: Iterable[str]) -> pd.Series:
    """The one peak (from `peaks_index`) each `.bim` row falls in, or `pd.NA`.

    Peak coordinates come from :func:`scads_drvi.enrich.config.parse_peaks` applied to
    `peaks_index`, so a SNP overlaps a peak under that function's half-open
    ``[start, end)`` convention. `bim` may be a single chromosome (`read_bim`'s own
    return) or the whole genome concatenated -- either way, a SNP is only ever compared
    against peaks that share its own `CHR`. A leading ``chr``/``Chr``/``CHR`` is
    stripped from both `bim`'s `CHR` and each peak's own chromosome before comparing,
    since PLINK ``.bim`` files use bare chromosome names (``"1"``) while this package's
    own peak-naming convention carries the prefix (``"chr1:..."``) -- exactly the
    ``f"chr{chrom}"`` glue every caller of this comparison has had to write by hand.

    Vectorized per chromosome (peaks sorted by start, `numpy.searchsorted`), since `bim`
    here is genome-wide in scale -- millions of rows, not the handful-of-loci case a
    caller might otherwise expect. A SNP overlapping more than one peak raises: peaks
    from :func:`scads_drvi.enrich.config.kept_feature_loadings` are assumed
    non-overlapping, and silently picking one would hide a caller data problem instead
    of naming it.
    """
    import numpy as np
    import pandas as pd

    from scads_drvi.enrich.config import parse_peaks

    for column in ("CHR", "BP"):
        if column not in bim.columns:
            raise KeyError(f"bim is missing the {column!r} column")

    peaks_index = list(peaks_index)
    if not peaks_index:
        raise ValueError("no peaks to assign against")

    peaks = parse_peaks(peaks_index)
    peaks.index = peaks_index
    peaks["chrom"] = peaks["chrom"].astype(str).str.replace(r"(?i)^chr", "", regex=True)
    bim_chrom = bim["CHR"].astype(str).str.replace(r"(?i)^chr", "", regex=True)
    positions = bim["BP"].to_numpy()

    matched: list[object] = [pd.NA] * len(bim)
    for chrom, group in peaks.groupby("chrom", sort=False):
        rows = np.flatnonzero((bim_chrom == chrom).to_numpy())
        if rows.size == 0:
            continue

        order = np.argsort(group["start"].to_numpy())
        starts = group["start"].to_numpy()[order]
        ends = group["end"].to_numpy()[order]
        names = group.index.to_numpy()[order]

        pos = positions[rows]
        idx = np.searchsorted(starts, pos, side="right") - 1
        valid = idx >= 0
        hit = np.zeros(pos.shape, dtype=bool)
        hit[valid] = pos[valid] < ends[idx[valid]]

        # Peaks are assumed non-overlapping, so at most one candidate should ever
        # contain `pos` -- check the immediately preceding peak (by start) too, the
        # only other one that could, to catch a caller's peak set that isn't.
        left = idx - 1
        left_valid = left >= 0
        also_left = np.zeros(pos.shape, dtype=bool)
        also_left[left_valid] = pos[left_valid] < ends[left[left_valid]]
        ambiguous = np.flatnonzero(hit & also_left)
        if ambiguous.size:
            bad_row = rows[ambiguous[0]]
            raise ValueError(
                f"bim row {bad_row} (chrom {chrom!r}, BP {int(positions[bad_row])}) "
                "overlaps more than one peak -- peaks are assumed non-overlapping."
            )

        for row, j in zip(rows[hit], idx[hit], strict=True):
            matched[row] = names[j]

    return pd.Series(matched, index=bim.index, name="peak", dtype="object")


def restrict_to_reference_snps(
    ldscore: pd.DataFrame, reference_snps: Iterable[str], *, snp_col: str = "SNP"
) -> pd.DataFrame:
    """Rows of `ldscore` (a ``.l2.ldscore.gz``-shaped frame, one row per SNP)
    restricted to `reference_snps`, reordered to match `reference_snps`'s own order.

    `ldsc.py --h2 --overlap-annot` requires every `--ref-ld-chr` stem to cover
    identical SNP sets per chromosome. Our own factor's LD scores are computed over the
    full reference panel, while a real baseline's own precomputed scores (e.g.
    baselineLD_v2.2) are restricted to its regression SNP set (HapMap3 minus MHC) --
    this filters our side down to match. `.l2.M`/`.l2.M_5_50` are deliberately *not*
    touched by this function: those are genome-wide counts of tagged SNPs over the full
    panel, not the regression SNP set, and baseline's own M files are computed the same
    way -- only the per-SNP `.l2.ldscore.gz` rows need aligning.

    Every id in `reference_snps` must already be present in `ldscore[snp_col]` -- a
    reference SNP genuinely missing from our own panel means a mismatched build or
    plink prefix, a caller data problem to raise on, not silently drop.
    """
    reference_snps = list(reference_snps)
    if snp_col not in ldscore.columns:
        raise KeyError(f"{snp_col!r} not in ldscore columns")

    have = set(ldscore[snp_col])
    missing = [snp for snp in reference_snps if snp not in have]
    if missing:
        raise ValueError(
            f"{len(missing)} reference SNP(s) not found in ldscore[{snp_col!r}] "
            f"(e.g. {missing[:5]!r}) -- ldscore and reference_snps must come from the "
            "same build/panel"
        )

    order = {snp: i for i, snp in enumerate(reference_snps)}
    restricted = ldscore[ldscore[snp_col].isin(order)].copy()
    restricted = restricted.sort_values(snp_col, key=lambda s: s.map(order))
    return restricted.reset_index(drop=True)
