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


def _bare_chrom(chrom: pd.Series) -> pd.Series:
    """`chrom` as strings with any leading ``chr``/``Chr``/``CHR`` removed."""
    return chrom.astype(str).str.replace(r"(?i)^chr", "", regex=True)


def _check_disjoint(peaks: pd.DataFrame) -> None:
    """Raise if any two peaks (cols `chrom`, `start`, `end`) on one chromosome overlap.

    Adjacent peaks (``end == start``) are fine. Vectorised: after sorting by start, a peak
    overlaps an earlier one exactly when its start is below the running maximum end.
    """
    import numpy as np

    ordered = peaks.sort_values(["chrom", "start"], kind="stable")
    running_end = ordered.groupby("chrom", sort=False)["end"].cummax()
    previous_end = running_end.groupby(ordered["chrom"], sort=False).shift()
    bad = np.flatnonzero((ordered["start"] < previous_end).to_numpy())
    if bad.size:
        i = int(bad[0])
        later = ordered.iloc[i]
        before = ordered.iloc[:i]
        before = before[(before["chrom"] == later["chrom"]) & (before["end"] > later["start"])]
        raise ValueError(f"peaks must not overlap: {before.index[0]} and {ordered.index[i]}")


def assign_peaks(
    bim: pd.DataFrame,
    peaks_index: Iterable[str],
    coords: pd.DataFrame | None = None,
) -> pd.Series:
    """The one peak (from `peaks_index`) each `.bim` row falls in, or `pd.NA`.

    Peaks are 0-based half-open ``[start, end)`` intervals (the BED convention), taken from
    :func:`scads_drvi.enrich.config.parse_peaks` applied to `peaks_index`, or from `coords`.
    `.bim` positions are 1-based, so a SNP at `BP` is the interval ``(BP - 1, BP)`` and lies
    in a peak when ``start < BP <= end``. (Comparing `BP` with ``[start, end)`` directly
    is off by one at both edges.)

    `coords` (columns `chrom`, `start`, `end`; indexed by peak name) replaces the
    coordinates parsed from the names -- for peaks lifted to another genome build. A peak
    whose `start`/`end` are missing is never assigned.

    `bim` may be one chromosome (`read_bim`'s own return) or the whole genome -- a SNP is
    only compared with peaks on its own `CHR`. A leading ``chr``/``Chr``/``CHR`` is
    stripped from both sides, since PLINK uses bare names (``"1"``) and peak names carry
    the prefix (``"chr1:..."``). Peaks on chromosomes absent from `bim` get no SNPs.

    Peaks must not overlap one another: that is checked for the whole peak set (every
    chromosome, whatever `bim` holds) before the search and raises
    ``ValueError("peaks must not overlap: a and b")``. The check is global because looking
    only at neighbours misses nested intervals. A zero-length peak contains no SNP and is
    left out of the check.
    """
    import bioframe as bf
    import numpy as np
    import pandas as pd

    from scads_drvi.enrich.config import parse_peaks

    for column in ("CHR", "BP"):
        if column not in bim.columns:
            raise KeyError(f"bim is missing the {column!r} column")

    peaks_index = list(peaks_index)
    if not peaks_index:
        raise ValueError("no peaks to assign against")

    if coords is None:
        peaks = parse_peaks(peaks_index)
        peaks.index = peaks_index
    else:
        peaks = coords.loc[peaks_index, ["chrom", "start", "end"]].dropna(subset=["start", "end"])
    peaks = peaks.astype({"start": "int64", "end": "int64"})
    peaks["chrom"] = _bare_chrom(peaks["chrom"])
    snp_chrom = _bare_chrom(bim["CHR"])
    peaks = peaks[peaks["end"] > peaks["start"]]
    _check_disjoint(peaks)  # the whole set, so validity does not depend on which `bim` is passed
    peaks = peaks[peaks["chrom"].isin(set(snp_chrom))]

    matched = np.full(len(bim), pd.NA, dtype=object)
    if not peaks.empty:
        bp = bim["BP"].to_numpy(dtype="int64")
        snps = pd.DataFrame(
            {"chrom": snp_chrom.to_numpy(), "start": bp - 1, "end": bp, "row": np.arange(len(bim))}
        )
        peak_frame = peaks.rename_axis("peak").reset_index()[["chrom", "start", "end", "peak"]]
        hits = bf.overlap(snps, peak_frame, how="inner", suffixes=("", "_p"))
        matched[hits["row"].to_numpy()] = hits["peak_p"].to_numpy()

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
