"""Weighted (non-binarized) factor-motif enrichment.

WHY NOT BINARIZE. Reducing each factor to a top-fraction region indicator and
rank-testing it (e.g. pycisTarget's DEM) has two costs: the cut point is arbitrary -- a
region just inside the cut counts exactly as much as the most extreme one, and just
outside counts zero -- and a uniform random background controls nothing.

THE METHOD. Score a (factor, motif) pair by the weighted expectation of the motif's
score under the factor's region weights,

    T(k, m) = sum_r W[k, r] * S[r, m]

and calibrate it against permuting region identity WITHIN matched strata (e.g. GC x
width bins, see :func:`gc_width_strata`):

    NES(k, m) = (T - E[T]) / sd[T]

THE PERMUTATION NULL IS EXACT -- IT IS NEVER SAMPLED. For a permutation pi of R regions,
T = sum_r a_r b_pi(r) has, over the uniform permutation distribution,

    E[T]   = (sum a)(sum b) / R
    Var[T] = SS_a * SS_b / (R - 1)              SS = centred sum of squares

and permuting within strata g makes the strata independent, so

    E[T]   = sum_g (sum_g a)(sum_g b) / n_g
    Var[T] = sum_g SS_a(g) * SS_b(g) / (n_g - 1)

Every term is a per-stratum sum or sum-of-squares, so ONE streaming pass over the score
database yields the exact null -- deterministic, no seed, and orders of magnitude
cheaper than sampling it by explicit matmul (see ``tests/test_motif.py`` for the proof
against brute-force and sampled permutations).

With a single stratum this NES equals ``r * sqrt(R - 1)`` exactly -- the permutation
framing and a plain Pearson correlation are the same estimand. Stratification is
precisely what makes it differ.

SCALE INVARIANCE. NES is exactly invariant to rescaling ``W[k, :]`` by any positive
constant, because ``T``, ``E[T]`` and ``sd[T]`` all scale linearly in it. Only
within-factor structure is used -- pass ``check_invariance=True`` to
:func:`weighted_motif_enrichment` to assert this rather than trust the argument.

REGIONS, NOT THE CALLER'S FEATURES, ARE THE UNIT. The score database lives on its own
region set (e.g. a SCREEN-style catalogue); :func:`build_region_map` maps a caller's
features onto it and :func:`regionise` averages weights per matched region.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scads_drvi._util.progress import log

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "DEFAULT_SLAB",
    "DEFAULT_SOFT_THRESHOLD_PCT",
    "DEFAULT_N_GC_BINS",
    "DEFAULT_N_LEN_BINS",
    "MIN_STRATUM_SIZE",
    "soft_threshold",
    "covariate_strata",
    "gc_width_strata",
    "read_score_db_regions",
    "read_score_db_motif_ids",
    "build_region_map",
    "regionise",
    "stream_accumulate",
    "stratified_nes",
    "unstratified_nes",
    "weighted_motif_enrichment",
]

#: Contiguous score-database columns read per slab; see `stream_accumulate`.
DEFAULT_SLAB = 100_000
DEFAULT_SOFT_THRESHOLD_PCT = 50.0
DEFAULT_N_GC_BINS = 10
DEFAULT_N_LEN_BINS = 10
#: A stratum needs at least this many regions or its permutation variance term is
#: undefined (a single region has zero centred sum of squares).
MIN_STRATUM_SIZE = 2


# --------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------
def soft_threshold(W: np.ndarray, pct: float) -> np.ndarray:
    """Zero each factor's (row's) weights below `pct` percentile OF ITS POSITIVE VALUES.

    Leaves marginal, near-zero-but-nonzero weights from diluting the specific signal a
    factor actually carries. `pct <= 0` is a no-op.
    """
    if pct <= 0:
        return W
    out = np.array(W, copy=True)
    for k in range(out.shape[0]):
        row = out[k]
        pos = row[row > 0]
        if pos.size == 0:
            continue
        row[row < np.percentile(pos, pct)] = 0.0
    log(f"soft-threshold at p{pct:g} of positives -> {100 * (out == 0).mean():.1f}% zeros")
    return out


# --------------------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------------------
def _quantile_bin(x: np.ndarray, ok: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def covariate_strata(
    covariates: Sequence[np.ndarray],
    n_bins: int | Sequence[int],
    min_size: int = MIN_STRATUM_SIZE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Quantile-bin one or more per-region covariates and cross them into strata.

    Generalises "GC x width" (see :func:`gc_width_strata`) to any set of continuous
    covariates: each is independently quantile-binned into `n_bins` bins (one int
    applied to all, or one per covariate), and the strata are the distinct
    combinations of bins.

    Returns `(keep, strata_ix, n_strata)`. `keep` is a boolean mask into the original
    region order (regions with a non-finite covariate, or landing in a stratum smaller
    than `min_size`, are dropped); `strata_ix`/`n_strata` describe the KEPT regions only.
    """
    covariates = [np.asarray(c, dtype=np.float64) for c in covariates]
    if not covariates:
        raise ValueError("at least one covariate is required")
    n = len(covariates[0])
    if any(len(c) != n for c in covariates):
        raise ValueError("all covariates must have the same length")
    bins = [n_bins] * len(covariates) if isinstance(n_bins, int) else list(n_bins)
    if len(bins) != len(covariates):
        raise ValueError("n_bins must be a single int or one per covariate")

    ok = np.ones(n, dtype=bool)
    for c in covariates:
        ok &= np.isfinite(c)
    if not ok.all():
        log(f"dropping {int((~ok).sum()):,} region(s) with non-finite covariate value(s)")

    combined = np.zeros(n, dtype=np.int64)
    multiplier = 1
    for c, nb in zip(covariates, bins, strict=True):
        combined += _quantile_bin(c, ok, nb) * multiplier
        multiplier *= nb
    combined = combined.astype(np.int32)
    combined[~ok] = -1

    keep = ok.copy()
    counts = np.bincount(combined[ok], minlength=int(combined.max()) + 1)
    for g in np.where((counts > 0) & (counts < min_size))[0]:
        keep &= combined != g
    if keep.sum() != ok.sum():
        log(
            f"dropping {int(ok.sum() - keep.sum()):,} region(s) in strata with < {min_size} members"
        )

    uniq, strata_ix = np.unique(combined[keep], return_inverse=True)
    log(
        f"{len(uniq)} strata over {int(keep.sum()):,} regions "
        f"(median {int(np.median(np.bincount(strata_ix))):,} regions/stratum)"
    )
    return keep, strata_ix.astype(np.int32), len(uniq)


def gc_width_strata(
    region_names: Sequence[str],
    genome_fasta: str | Path,
    n_gc_bins: int = DEFAULT_N_GC_BINS,
    n_len_bins: int = DEFAULT_N_LEN_BINS,
    min_size: int = MIN_STRATUM_SIZE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """GC x width quantile strata for named regions -- the default confounder pair.

    Width is included alongside GC because it is often the STRONGER confounder for a
    Cluster-Buster-style motif score: score scales with region length, and GC alone
    would leave that uncontrolled.
    """
    from scads_drvi.annotate.gc import peak_gc
    from scads_drvi.enrich.config import parse_peaks

    coords = parse_peaks(region_names)
    width = (coords["end"] - coords["start"]).to_numpy(dtype=np.float64)
    log(f"computing GC for {len(region_names):,} regions from {Path(genome_fasta).name}...")
    gc = peak_gc(genome_fasta, coords)
    return covariate_strata([gc, width], [n_gc_bins, n_len_bins], min_size=min_size)


# --------------------------------------------------------------------------------------
# features -> database regions
# --------------------------------------------------------------------------------------
def _format_peak(chrom: str, start: int, end: int) -> str:
    """The canonical ``chrom:start-end`` spelling :func:`~scads_drvi.enrich.config.parse_peaks`
    reads back."""
    return f"{chrom}:{int(start)}-{int(end)}"


def build_region_map(
    peaks: Sequence[str],
    db_region_names: Sequence[str],
    fraction_overlap: float = 0.4,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Map `peaks` onto `db_region_names` by genomic overlap.

    A pair counts as a match if EITHER side's overlap fraction exceeds
    `fraction_overlap` (the rule pycisTarget's DEM route uses; reproduced here directly
    on `bioframe` so this module depends on it rather than on `pycistarget` for two
    helper functions).

    Returns `(used_cols, peak_rows, region_names)`: `used_cols` are DATABASE COLUMN
    INDICES (sorted, into `db_region_names`), and `peak_rows` is, per used region, the
    array of `peaks` indices overlapping it.
    """
    import bioframe

    from scads_drvi.enrich.config import parse_peaks

    log("building region overlap map...")
    peaks = list(peaks)
    db_region_names = list(db_region_names)

    joined = bioframe.overlap(
        parse_peaks(peaks),
        parse_peaks(db_region_names),
        how="inner",
        return_overlap=True,
        suffixes=("_1", "_2"),
    )
    if len(joined) == 0:
        raise ValueError(
            "no overlap between the two region sets -- check they share a genome "
            "build and chromosome naming."
        )
    overlap_len = joined["overlap_end"] - joined["overlap_start"]
    overlap_target = overlap_len / (joined["end_1"] - joined["start_1"])
    overlap_query = overlap_len / (joined["end_2"] - joined["start_2"])
    joined = joined[(overlap_target > fraction_overlap) | (overlap_query > fraction_overlap)]

    target_names = [
        _format_peak(c, s, e)
        for c, s, e in zip(joined["chrom_1"], joined["start_1"], joined["end_1"], strict=True)
    ]
    query_names = [
        _format_peak(c, s, e)
        for c, s, e in zip(joined["chrom_2"], joined["start_2"], joined["end_2"], strict=True)
    ]

    col_of = {name: i for i, name in enumerate(db_region_names)}
    row_of = {name: i for i, name in enumerate(peaks)}
    by_region: dict[int, list[int]] = {}
    for tgt, qry in zip(target_names, query_names, strict=True):
        by_region.setdefault(col_of[qry], []).append(row_of[tgt])

    used_cols = np.array(sorted(by_region), dtype=np.int64)
    peak_rows = [np.asarray(by_region[c], dtype=np.int64) for c in used_cols]
    region_names = [db_region_names[c] for c in used_cols]
    n_peaks_hit = len({r for rows in peak_rows for r in rows})
    log(
        f"matched {len(used_cols):,} / {len(db_region_names):,} db regions "
        f"({100 * len(used_cols) / len(db_region_names):.1f}%), covering "
        f"{n_peaks_hit:,} / {len(peaks):,} peaks ({100 * n_peaks_hit / len(peaks):.1f}%)"
    )
    return used_cols, peak_rows, region_names


def regionise(W: np.ndarray, peak_rows: Sequence[np.ndarray]) -> np.ndarray:
    """(K, n_peaks) -> (K, n_regions), averaging over each region's overlapping peaks.

    Mean rather than max: a max's positive bias grows with the number of overlapping
    peaks, which tends to correlate with region width, which tends to correlate with
    score -- exactly the confound stratification exists to remove.
    """
    out = np.empty((W.shape[0], len(peak_rows)), dtype=np.float32)
    for j, rows in enumerate(peak_rows):
        out[:, j] = W[:, rows].mean(axis=1) if rows.size > 1 else W[:, rows[0]]
    return out


# --------------------------------------------------------------------------------------
# the score database
# --------------------------------------------------------------------------------------
def read_score_db_regions(db_path: str | Path, motif_id_column: str = "motifs") -> list[str]:
    """Region column names of a region x motif score database.

    The database's LAST column is the motif-id column (`motif_id_column`); every other
    column is a region, stored lexicographically sorted (see `stream_accumulate`).
    """
    import pyarrow.dataset as ds

    names = ds.dataset(db_path, format="feather").schema.names
    if names[-1] != motif_id_column:
        raise ValueError(
            f"expected the last column of {db_path} to be {motif_id_column!r}, got {names[-1]!r}"
        )
    return names[:-1]


def read_score_db_motif_ids(db_path: str | Path, motif_id_column: str = "motifs") -> list[str]:
    """The motif-id column of a region x motif score database, in row order."""
    import pyarrow.dataset as ds

    values = (
        ds.dataset(db_path, format="feather")
        .to_table(columns=[motif_id_column])
        .column(0)
        .to_pylist()
    )
    return [str(v) for v in values]


def stream_accumulate(
    db_path: str | Path,
    used_cols: np.ndarray,
    Wr: np.ndarray,
    strata_ix: np.ndarray,
    n_strata: int,
    slab: int = DEFAULT_SLAB,
    expect_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One contiguous pass over the score database.

    Contiguity is the whole trick: scattered column selections above a few thousand
    columns are pathological for most columnar-on-disk formats, but contiguous index
    slabs are linear. The database's region columns are assumed lexicographically
    sorted, so we walk them in order and pick out the used ones inside each slab.

    Accumulates, in float64:
        T       (K, M)          sum_r W[k,r] S[r,m]
        sumS    (n_strata, M)   sum_{r in g} S[r,m]
        sumS2   (n_strata, M)   sum_{r in g} S[r,m]^2

    `expect_names`, when given, is checked against the database's own column names at
    each slab: a slab/row misalignment would otherwise be silent -- the scores would
    simply belong to the wrong regions.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(db_path, format="feather")
    schema_names = dataset.schema.names

    K = Wr.shape[0]
    B = np.zeros((len(strata_ix), n_strata), dtype=np.float32)
    B[np.arange(len(strata_ix)), strata_ix] = 1.0

    T: np.ndarray | None = None
    sumS: np.ndarray | None = None
    sumS2: np.ndarray | None = None
    n_done = 0
    t0 = time.time()
    for lo in range(0, int(used_cols[-1]) + 1, slab):
        hi = min(lo + slab, int(used_cols[-1]) + 1)
        i0, i1 = np.searchsorted(used_cols, [lo, hi])
        if i1 <= i0:
            continue
        sel = used_cols[i0:i1]
        tbl = dataset.to_table(columns=schema_names[lo:hi], use_threads=True)
        names = tbl.column_names
        X = np.empty((tbl.num_rows, len(sel)), dtype=np.float32)  # (M, n_sel)
        for j, c in enumerate(sel):
            X[:, j] = tbl.column(int(c) - lo).chunk(0).to_numpy(zero_copy_only=False)
        if expect_names is not None:
            got = [names[int(c) - lo] for c in sel[:: max(1, len(sel) // 8)]]
            want = [expect_names[i] for i in range(i0, i1)[:: max(1, len(sel) // 8)]]
            assert got == want, f"slab/column misalignment at [{lo},{hi}): {got[:2]} != {want[:2]}"
        del tbl

        if T is None or sumS is None or sumS2 is None:
            M = X.shape[0]
            T = np.zeros((K, M), dtype=np.float64)
            sumS = np.zeros((n_strata, M), dtype=np.float64)
            sumS2 = np.zeros((n_strata, M), dtype=np.float64)

        T += (X @ Wr[:, i0:i1].T).T.astype(np.float64)
        sumS += (X @ B[i0:i1]).T.astype(np.float64)
        sumS2 += ((X * X) @ B[i0:i1]).T.astype(np.float64)
        n_done += len(sel)
        del X
        log(
            f"  slab [{lo:,},{hi:,}) -> {len(sel):,} used ({n_done:,} total, "
            f"{time.time() - t0:.0f}s)"
        )

    assert n_done == len(used_cols), f"{n_done} != {len(used_cols)}"
    assert T is not None and sumS is not None and sumS2 is not None, (
        "no slabs were read -- used_cols is empty"
    )
    return T, sumS, sumS2


# --------------------------------------------------------------------------------------
# the exact permutation null
# --------------------------------------------------------------------------------------
def stratified_nes(
    T: np.ndarray,
    sumS: np.ndarray,
    sumS2: np.ndarray,
    Wr: np.ndarray,
    strata_ix: np.ndarray,
    n_strata: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact permutation mean/sd from per-stratum moments; see the module docstring."""
    n_g = np.bincount(strata_ix, minlength=n_strata).astype(np.float64)  # (G,)
    Wd = Wr.astype(np.float64)
    sumW = np.zeros((Wd.shape[0], n_strata))
    sumW2 = np.zeros((Wd.shape[0], n_strata))
    np.add.at(sumW.T, strata_ix, Wd.T)
    np.add.at(sumW2.T, strata_ix, (Wd * Wd).T)

    SS_W = sumW2 - sumW**2 / np.maximum(n_g, 1)  # (K, G)
    SS_S = sumS2 - sumS**2 / np.maximum(n_g, 1)[:, None]  # (G, M)

    E = (sumW / np.maximum(n_g, 1)) @ sumS  # (K, M)
    var = (SS_W / np.maximum(n_g - 1, 1)) @ SS_S  # (K, M)
    sd = np.sqrt(np.maximum(var, 0.0))
    nes = np.divide(T - E, sd, out=np.zeros_like(T), where=sd > 0)

    assert np.isfinite(nes).all(), "non-finite NES"
    return nes, E, sd


def unstratified_nes(
    T: np.ndarray, sumS: np.ndarray, sumS2: np.ndarray, Wr: np.ndarray
) -> np.ndarray:
    """Same statistic with a single stratum -- the un-corrected comparison column."""
    n = Wr.shape[1]
    Wd = Wr.astype(np.float64)
    sW, sW2 = Wd.sum(1), (Wd * Wd).sum(1)
    sS, sS2 = sumS.sum(0), sumS2.sum(0)
    SS_W = sW2 - sW**2 / n
    SS_S = sS2 - sS**2 / n
    E = np.outer(sW, sS) / n
    sd = np.sqrt(np.maximum(np.outer(SS_W, SS_S) / (n - 1), 0.0))
    # `out=`/`where=` push np.divide's overload resolution to `Any`; the annotated
    # local (rather than returning the call directly) keeps the declared -> np.ndarray.
    result: np.ndarray = np.divide(T - E, sd, out=np.zeros_like(T), where=sd > 0)
    return result


# --------------------------------------------------------------------------------------
def weighted_motif_enrichment(
    Wr: np.ndarray,
    labels: Sequence[str],
    used_cols: np.ndarray,
    region_names: Sequence[str],
    strata_ix: np.ndarray,
    n_strata: int,
    db_path: str | Path,
    *,
    tf_of: Mapping[str, str] | None = None,
    slab: int = DEFAULT_SLAB,
    motif_id_column: str = "motifs",
    check_invariance: bool = False,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Score every (factor, motif) pair and assemble the result table.

    `Wr` is (K, n_regions) non-negative factor weights already mapped onto the score
    database's regions and restricted to `region_names`/`used_cols` (see
    `build_region_map` + `regionise`, and `gc_width_strata`/`covariate_strata` for
    `strata_ix`/`n_strata`). A factor whose weights sum to zero after that filtering is
    dropped; every surviving factor is renormalised to sum to 1 (`T`'s scale is
    otherwise arbitrary -- NES itself is exactly invariant to it).

    `tf_of`, when given, maps a motif id to a transcription-factor annotation string
    for the `tf_annotation` output column.

    Returns `(rows, meta)`. `rows` has one row per (factor, motif) with columns
    `factor`, `motif_id`, `weighted_score`, `nes`, `nes_unstratified`, `tf_annotation`
    (if `tf_of` given), `n_regions`; sorted by `factor` then descending `nes`.
    """
    import pandas as pd

    t0 = time.time()
    labels = list(labels)
    region_names = list(region_names)

    if not np.isfinite(Wr).all():
        raise ValueError("non-finite weights")
    if Wr.min() < 0.0:
        raise ValueError(f"negative weights ({Wr.min():.3g})")

    live = Wr.sum(axis=1) > 0
    if not live.all():
        log(f"dropping {int((~live).sum())} factor(s) with all-zero weights")
    Wr = Wr[live]
    labels = [name for name, keep in zip(labels, live, strict=True) if keep]
    Wr = Wr / np.maximum(Wr.sum(axis=1, keepdims=True), 1e-30)

    T, sumS, sumS2 = stream_accumulate(
        db_path, used_cols, Wr, strata_ix, n_strata, slab, expect_names=region_names
    )
    nes, _, _ = stratified_nes(T, sumS, sumS2, Wr, strata_ix, n_strata)
    nes_un = unstratified_nes(T, sumS, sumS2, Wr)

    if check_invariance:
        Wr2 = Wr.copy()
        Wr2[0] *= 1e6
        T2 = T.copy()
        T2[0] *= 1e6
        nes2, _, _ = stratified_nes(T2, sumS, sumS2, Wr2, strata_ix, n_strata)
        d = np.abs(nes2[0] - nes[0]).max()
        if d >= 1e-6:
            raise AssertionError(f"NES not scale-invariant: max |delta| = {d:.3g}")
        log(f"scale-invariance check passed (max |delta| = {d:.3g})")

    motif_ids = read_score_db_motif_ids(db_path, motif_id_column)
    if len(motif_ids) != T.shape[1]:
        raise ValueError(f"{len(motif_ids)} motif ids != {T.shape[1]} score columns")

    K, M = nes.shape
    rows = pd.DataFrame(
        {
            "factor": np.repeat(labels, M),
            "motif_id": np.tile(motif_ids, K),
            "weighted_score": T.ravel(),
            "nes": nes.ravel(),
            "nes_unstratified": nes_un.ravel(),
        }
    )
    if tf_of is not None:
        rows["tf_annotation"] = rows["motif_id"].map(tf_of)
    rows["n_regions"] = int(Wr.shape[1])
    rows = rows.sort_values(["factor", "nes"], ascending=[True, False]).reset_index(drop=True)

    meta = {
        "n_regions": int(Wr.shape[1]),
        "n_factors": int(K),
        "n_motifs": int(M),
        "n_strata": int(n_strata),
        "elapsed_s": round(time.time() - t0, 1),
    }
    log(f"scored {len(rows):,} (factor, motif) pairs in {time.time() - t0:.0f}s")
    return rows, meta
