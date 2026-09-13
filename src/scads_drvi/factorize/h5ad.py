#!/usr/bin/env python3
"""Backed-h5ad readers: scattered CSR row gathers, shape, and var names.

No torch at module scope, deliberately: a CPU env must be able to import this. Obs
access goes through ``anndata`` itself (:func:`keep_rows` reads its depth column via a
backed ``anndata.read_h5ad``, not a hand-rolled decoder) -- the CSR-gathering code below
is the genuinely custom part, a deliberate performance optimization for scattered-row
reads on a huge sparse matrix that anndata's own backed-mode indexing does not do
efficiently.
"""
from __future__ import annotations

import h5py
import numpy as np
import scipy.sparse as sp

from scads_drvi._util.progress import log


def to_numpy(x) -> np.ndarray:
    """Decode an h5py value to a numpy array of str or numbers."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _iter_index_blocks(path: str, srows: np.ndarray, indptr: np.ndarray,
                       block_nnz: int, max_gap: int):
    """Yield (indices, n_rows) blocks under `block_nnz`, coalescing runs closer than `max_gap`."""
    nnz = (indptr[srows + 1] - indptr[srows]).astype(np.int64)
    edges = np.searchsorted(np.cumsum(nnz), np.arange(1, int(nnz.sum()) + block_nnz,
                                                      block_nnz), side="left") + 1
    edges = np.unique(np.clip(np.r_[0, edges], 0, len(srows)))
    with h5py.File(path, "r") as f:
        X = f["X"]["indices"]
        for lo_r, hi_r in zip(edges[:-1], edges[1:], strict=True):
            grp = srows[lo_r:hi_r]
            starts, ends = indptr[grp], indptr[grp + 1]
            breaks = np.nonzero(starts[1:] - ends[:-1] > max_gap)[0] + 1
            parts = []
            for r in np.split(np.arange(len(grp)), breaks):
                a0, b0 = starts[r[0]], ends[r[-1]]
                if b0 == a0:
                    continue
                blk = X[a0:b0]
                for t in r:
                    parts.append(blk[starts[t] - a0:ends[t] - a0])
                del blk
            yield (np.concatenate(parts) if parts
                   else np.empty(0, dtype=np.int32)), int(hi_r - lo_r)


def read_rows_csr(path: str, rows: np.ndarray, n_vars: int,
                  max_gap: int | None = None) -> sp.csr_matrix:
    """Read scattered rows out of a backed CSR without loading all of X.

        `max_gap` coalesces runs; its default is DERIVED from the mean nonzeros per row, because a
        hardcoded value read 220 GB to obtain 11.4 GB (19.3x) once the input got denser. The
        amplification ratio is logged so a recurrence is visible.
    """
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    srows = rows[order]

    with h5py.File(path, "r") as f:
        X = f["X"]
        indptr = X["indptr"][:]
        if max_gap is None:
            n_rows_total = max(len(indptr) - 1, 1)
            max_gap = max(int(indptr[-1] // n_rows_total), 1024)
        starts, ends = indptr[srows], indptr[srows + 1]

        # Group sorted rows into runs; break a run when the skipped nonzeros exceed max_gap.
        breaks = np.nonzero(starts[1:] - ends[:-1] > max_gap)[0] + 1
        groups = np.split(np.arange(len(srows)), breaks)

        data_parts, idx_parts, lens = [], [], np.empty(len(srows), dtype=np.int64)
        n_read = 0
        for g in groups:
            lo, hi = starts[g[0]], ends[g[-1]]
            if hi == lo:
                for j in g:
                    lens[j] = 0
                continue
            n_read += int(hi - lo)
            blk_data = X["data"][lo:hi]
            blk_idx = X["indices"][lo:hi]
            for j in g:
                a, b = starts[j] - lo, ends[j] - lo
                data_parts.append(blk_data[a:b])
                idx_parts.append(blk_idx[a:b])
                lens[j] = b - a

    n_want = int(lens.sum())
    log(f"read_rows_csr: {len(rows):,} rows, {len(groups):,} runs, max_gap {max_gap:,} -- "
        f"read {n_read * 8 / 1e9:.1f} GB for {n_want * 8 / 1e9:.1f} GB wanted "
        f"(amplification {n_read / max(n_want, 1):.2f}x)")

    new_indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    data = (np.concatenate(data_parts) if data_parts
            else np.empty(0, dtype=np.float32))
    indices = (np.concatenate(idx_parts) if idx_parts
               else np.empty(0, dtype=np.int32))
    sorted_mat = sp.csr_matrix((data, indices, new_indptr),
                               shape=(len(srows), n_vars))
    # Undo the sort so the caller's row order is preserved.
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sorted_mat[inverse]


def h5ad_shape(path: str):
    with h5py.File(path, "r") as f:
        return tuple(int(x) for x in f["X"].attrs["shape"])


def h5ad_var_names(path: str):
    with h5py.File(path, "r") as f:
        key = f["var"].attrs.get("_index", "_index")
        v = f["var"][key][:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])


def keep_rows(adata_path: str, min_fragment: int, depth_col: str = "n_fragment"):
    """Absolute row indices a gated run may touch."""
    n_obs, _ = h5ad_shape(adata_path)
    if not min_fragment:
        return np.arange(n_obs), None

    import anndata as ad

    try:
        d = ad.read_h5ad(adata_path, backed="r").obs[depth_col].to_numpy(dtype=np.int64)
    except KeyError as exc:
        raise SystemExit(
            f"min_fragment needs obs['{depth_col}'], which is absent"
        ) from exc
    rows = np.flatnonzero(d >= min_fragment)
    info = {"min_fragment": int(min_fragment), "depth_col": depth_col,
            "n_cells_available": int(n_obs), "n_cells_kept": int(rows.size),
            "n_cells_excluded": int(n_obs - rows.size)}
    log(f"gate {depth_col} >= {min_fragment}: {rows.size:,} of {n_obs:,} cells "
        f"({100 * rows.size / n_obs:.2f}%)")
    return rows, info
