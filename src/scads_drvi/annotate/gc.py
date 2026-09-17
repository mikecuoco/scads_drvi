"""G+C content per region, read directly from an indexed FASTA.

One pass per contig, not one read per region. Seek+read per region was measured at 68
regions/s over an NFS mount -- 155 minutes for 635,485 regions -- because NFS charges
full request latency per call. Slurping a contig and gathering only the requested bases
costs ~2 s per contig and answers every region on it in one shot.

A whole-contig cumulative sum was tried and rejected: two int64 prefix sums over
chromosome 1 is ~2 GB on top of the byte and mask arrays, and OOM-killed. Regions
typically cover a small fraction of the genome, so gathering only their bases is both
smaller and less work.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["read_fai", "peak_gc"]

#: Regions are chunked so gathered bases stay bounded regardless of contig size.
_CHUNK = 20_000


def read_fai(path: str | Path) -> dict[str, tuple[int, int, int, int]]:
    """name -> (length, offset, line_bases, line_width) from a samtools ``.fai``."""
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 5:
            out[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
    return out


def _contig_counts(
    fh, meta: tuple[int, int, int, int], starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-region ``(n_GC, n_ACGT)`` for one contig, in one sequential read."""
    length, offset, line_bases, line_width = meta
    n_lines = (length + line_bases - 1) // line_bases
    nbytes = length + n_lines * (line_width - line_bases)
    fh.seek(offset)
    raw = np.frombuffer(fh.read(nbytes), dtype=np.uint8)
    raw = raw[(raw != 10) & (raw != 13)][:length]  # strip \n and \r
    up = (raw & 0xDF).copy()  # ASCII upper-case, cheaply
    del raw

    s = np.clip(starts, 0, length).astype(np.int64)
    e = np.clip(ends, 0, length).astype(np.int64)
    width = e - s
    n_gc = np.zeros(s.size, dtype=np.int64)
    n_acgt = np.zeros(s.size, dtype=np.int64)
    for lo in range(0, s.size, _CHUNK):
        hi = min(lo + _CHUNK, s.size)
        ss, ww = s[lo:hi], width[lo:hi]
        keep = ww > 0
        if not keep.any():
            continue
        flat = np.repeat(ss[keep], ww[keep]) + (
            np.arange(ww[keep].sum(), dtype=np.int64)
            - np.repeat(np.concatenate([[0], np.cumsum(ww[keep])[:-1]]), ww[keep]))
        b = up[flat]
        gcb = (b == 71) | (b == 67)
        acgtb = gcb | (b == 65) | (b == 84)
        bounds = np.concatenate([[0], np.cumsum(ww[keep])[:-1]])
        idx = np.flatnonzero(keep) + lo
        n_gc[idx] = np.add.reduceat(gcb, bounds)
        n_acgt[idx] = np.add.reduceat(acgtb, bounds)
        del flat, b, gcb, acgtb
    del up
    return n_gc, n_acgt


def peak_gc(fasta: str | Path, coords: pd.DataFrame, ndigits: int = 3) -> np.ndarray:
    """G+C fraction per region, over the region's own width.

    `coords` is a frame with ``chrom``/``start``/``end`` columns (0-based half-open),
    e.g. from :func:`scads_drvi.enrich.config.parse_peaks`. Regions on a contig absent
    from the index, or with no ACGT bases, come back NaN rather than 0 -- a zero would be
    a legitimate GC value and would silently join a downstream fit.
    """
    fasta = Path(fasta)
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if not fai.exists():
        raise FileNotFoundError(f"{fai} not found; index with `samtools faidx {fasta}`")
    idx = read_fai(fai)
    gc = np.full(len(coords), np.nan)
    chrom_arr = coords["chrom"].to_numpy()
    start = coords["start"].to_numpy(dtype=np.int64)
    end = coords["end"].to_numpy(dtype=np.int64)
    with fasta.open("rb") as fh:
        for chrom in dict.fromkeys(chrom_arr):
            key = chrom if chrom in idx else str(chrom).replace("chr", "")
            if key not in idx:
                continue
            sel = np.flatnonzero(chrom_arr == chrom)
            n_gc, n_acgt = _contig_counts(fh, idx[key], start[sel], end[sel])
            with np.errstate(invalid="ignore", divide="ignore"):
                gc[sel] = np.where(n_acgt > 0, n_gc / n_acgt, np.nan)
    return np.round(gc, ndigits)
