"""Clustering-free specificity of factors to cell communities on an independent graph.

The communities are never clustered: they are the kNN neighbourhoods of a latent space
the factors did not shape (for a DRVI fit, e.g. a PoissonVI or scVI latent over the same
cells). Building that graph is the caller's job -- ``scanpy.pp.neighbors`` on the
independent latent, or any exact/approximate kNN -- and :func:`graph_specificity` takes
either its ``(n, k)`` neighbour indices or its ``(n, n)`` sparse matrix (e.g.
``adata.obsp["distances"]``). Edges are unweighted and self edges are dropped: W is the
row-normalised adjacency, so ``(W @ x)[i]`` is the mean of ``x`` over i's neighbours.

For each non-negative factor column ``x`` (for a signed DRVI latent, its directional arms
from :func:`split_directions`):

``moran_i``, ``moran_p``
    ``x'Wx / x'x`` with ``x`` centred, and a one-sided permutation p-value. Does ``x``
    follow the independent structure at all? Low I is noise, depth or something else
    technical, which cannot be community-specific. Not bounded by 1: a row-normalised
    non-symmetric W has spectral norm above 1.
``pr``, ``pr_null``, ``specificity``
    ``e = max(Wx - mean(x), 0)`` is the neighbourhood excess over the mean, and
    ``pr = (sum e)^2 / (N sum e^2)`` in (0, 1] the effective fraction of cells in the
    factor's footprint. ``pr`` is NOT ~1 for a structureless factor: sampling noise in
    ``Wx`` is itself half-positive, so a permuted factor lands near 1/pi, lower for skewed
    marginals. ``pr_null`` is therefore measured per factor, on the same column with its
    cells permuted, and ``specificity = 1 - pr / pr_null``: 0 = no more concentrated than
    noise, -> 1 = a vanishing fraction of the graph, negative = a broad factor whose
    footprint is wider than noise's.
``moran_i_within_group``
    Moran's I with edges restricted to same-group pairs and ``x`` centred *within* group:
    a pure identity factor (constant inside a group) scores ~0, a state factor coherent
    inside groups scores high. Labels only mask edges; they never define communities.
``moran_i_resid``, ``covariate_spearman``
    Moran's I after regressing ``x`` on a per-cell covariate (e.g. log sequencing depth,
    which the independent graph may itself carry), and the rank correlation with it.
``top_<key>``, ``top_<key>_share``
    ``e``-weighted footprint composition by a caller-named labelling -- annotation, not
    part of the score.

Two optional reference rows ride along: ``_perm_control`` (a caller-chosen column with its
cells permuted -- same marginal, structure destroyed; expect I ~ 0, specificity ~ 0) and
``_covariate`` (how much of the covariate the graph itself carries).

Ported from the SCADS capsule's ``code/04_factorize/factor_graph_specificity.py``
(``stage_score``) and ``code/common/embed_metrics.py:moran_i``; graph construction,
approximate-kNN recall and file I/O stay with the caller.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["split_directions", "graph_specificity"]

PERM_CONTROL = "_perm_control"
COVARIATE = "_covariate"


def split_directions(
    X: npt.ArrayLike, names: Sequence[Hashable] | None = None
) -> tuple[np.ndarray, list[str]]:
    """A signed latent as its non-negative directional arms, ``[relu(X), relu(-X)]``.

    Returns the ``(n, 2m)`` arm matrix and its names, ``"{name}_pos"`` for the first m
    columns and ``"{name}_neg"`` for the last m -- the same convention as
    :func:`scads_drvi.pl.umap.latent_umap_grid`'s directional split.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (cells x factors), got shape {X.shape}")
    labels = [str(i) for i in range(X.shape[1])] if names is None else [str(n) for n in names]
    if len(labels) != X.shape[1]:
        raise ValueError(f"{len(labels)} names for {X.shape[1]} columns")
    arms = np.concatenate([np.maximum(X, 0), np.maximum(-X, 0)], axis=1)
    return arms, [f"{n}_pos" for n in labels] + [f"{n}_neg" for n in labels]


def _edge_pattern(graph: Any, n: int, k: int | None) -> sp.csr_matrix:
    """(n, n) 0/1 adjacency, self edges dropped, from neighbour indices or a sparse graph."""
    if sp.issparse(graph):
        if k is not None:
            raise ValueError("k selects neighbour-index columns; it cannot apply to a sparse graph")
        P: sp.csr_matrix = sp.csr_matrix(cast(sp.csr_matrix, graph), dtype=np.float32, copy=True)
        if P.shape != (n, n):
            raise ValueError(f"sparse graph must be ({n}, {n}), got {P.shape}")
        P.setdiag(0)
        P.eliminate_zeros()
        P.data[:] = 1.0
        return P

    nb = np.asarray(graph)
    if nb.ndim != 2 or nb.shape[0] != n or nb.dtype.kind not in "iu":
        raise ValueError(f"neighbour indices must be an integer ({n}, k) array, got "
                         f"{nb.dtype} {nb.shape}")
    if k is not None:
        if not 1 <= k <= nb.shape[1]:
            raise ValueError(f"k={k} outside 1..{nb.shape[1]} neighbour columns")
        nb = nb[:, :k]
    if nb.size and (nb.min() < 0 or nb.max() >= n):
        raise ValueError(f"neighbour indices must lie in [0, {n})")
    rows = np.repeat(np.arange(n, dtype=np.int64), nb.shape[1])
    cols = nb.ravel().astype(np.int64)
    keep = rows != cols
    P = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], cols[keep])), shape=(n, n))
    P.sum_duplicates()
    return P


def _row_normalised(P: sp.csr_matrix) -> sp.csr_matrix:
    """Row-stochastic W from an adjacency; rows with no edges stay all-zero."""
    deg = np.asarray(P.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.csr_matrix(sp.diags(inv.astype(np.float32)) @ P)


def _moran(W: sp.csr_matrix, X: np.ndarray, n_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Moran's I per column and its one-sided permutation p (the same permutations for all)."""
    n, m = X.shape
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(n_perm)]
    stat = np.full(m, np.nan)
    p = np.full(m, np.nan)
    for c in range(m):
        xc = X[:, c].astype(np.float64)
        xc = (xc - xc.mean()).astype(np.float32)
        den = float(xc.astype(np.float64) @ xc)
        if den <= 0:
            continue
        stat[c] = float(xc.astype(np.float64) @ (W @ xc)) / den
        if n_perm:
            Q = np.stack([xc[q] for q in perms], axis=1)
            nulls = (Q * (W @ Q)).sum(0, dtype=np.float64) / den
            p[c] = (1 + int((nulls >= stat[c]).sum())) / (n_perm + 1)
    return stat, p


def _moran_plain(W: sp.csr_matrix, X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Moran's I per column, no null. NaN where the centred column's energy has vanished
    relative to `ref` (the energy it had before residualising)."""
    Xc = X - X.mean(0)
    num = (Xc * np.asarray(W @ Xc)).sum(0)
    den = (Xc**2).sum(0)
    ok = den > 1e-12 * np.maximum(ref, 1e-300)
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def _within_group_moran(P: sp.csr_matrix, codes: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Moran's I over same-group edges only, each column centred within its group."""
    C = P.tocoo()
    same = codes[C.row] == codes[C.col]
    Pg = sp.csr_matrix((C.data[same], (C.row[same], C.col[same])), shape=P.shape)
    has = np.diff(Pg.indptr) > 0
    W = _row_normalised(Pg)
    n_groups = int(codes.max()) + 1
    cnt = np.maximum(np.bincount(codes, minlength=n_groups), 1).astype(np.float64)
    out = np.full(X.shape[1], np.nan)
    for c in range(X.shape[1]):
        x = X[:, c].astype(np.float64)
        xw = x - (np.bincount(codes, weights=x, minlength=n_groups) / cnt)[codes]
        xw[~has] = 0.0
        den = float(xw @ xw)
        if den > 1e-12 * max(float(x @ x), 1e-300):
            out[c] = float(xw @ (W @ xw)) / den
    return out


def _participation(W: sp.csr_matrix, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pr, E): footprint participation ratio per column and the excess matrix it came from."""
    E = np.maximum(np.asarray(W @ X) - X.mean(0, dtype=np.float64).astype(X.dtype), 0.0)
    e1 = E.sum(0, dtype=np.float64)
    e2 = (E.astype(np.float64) ** 2).sum(0)
    ok = e2 > 0
    return np.where(ok, e1**2 / (X.shape[0] * np.where(ok, e2, 1.0)), np.nan), E


def _group_shares(codes: np.ndarray, n_groups: int, E: np.ndarray) -> np.ndarray:
    """(n_groups, n_cols) share of each column's footprint mass in each group (code -1 = none)."""
    lab = codes >= 0
    G = sp.csr_matrix((np.ones(int(lab.sum()), dtype=np.float32),
                       (codes[lab], np.flatnonzero(lab))), shape=(n_groups, codes.size))
    tot = E.sum(0, dtype=np.float64)
    return np.asarray(G @ E, dtype=np.float64) / np.where(tot > 0, tot, 1.0)


def _spearman_to(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ry = rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    out = np.full(X.shape[1], np.nan)
    for c in range(X.shape[1]):
        r = rankdata(X[:, c])
        sd = r.std()
        if sd > 0:
            out[c] = float(((r - r.mean()) / sd * ry).mean())
    return out


def _codes(labels: npt.ArrayLike, n: int, what: str) -> tuple[np.ndarray, list[Any]]:
    import pandas as pd

    cat = pd.Categorical(np.asarray(labels))
    if cat.shape[0] != n:
        raise ValueError(f"{what} has {cat.shape[0]} labels for {n} cells")
    return np.asarray(cat.codes, dtype=np.int64), list(cat.categories)


def graph_specificity(
    graph: Any,
    X: npt.ArrayLike | pd.DataFrame,
    *,
    names: Sequence[Hashable] | None = None,
    k: int | Sequence[int] | None = None,
    groups: npt.ArrayLike | None = None,
    annotate: Mapping[str, npt.ArrayLike] | None = None,
    covariate: npt.ArrayLike | None = None,
    control: Hashable | None = None,
    n_perm: int = 49,
    seed: int = 0,
) -> pd.DataFrame:
    """Score each factor column's specificity to communities of an independent cell graph.

    See the module docstring for what each output column measures.

    Parameters
    ----------
    graph
        Either ``(n, K)`` integer neighbour indices (row i = i's neighbours, nearest first;
        a self entry is dropped wherever it sits) or an ``(n, n)`` scipy sparse graph whose
        non-zeros are edges (weights ignored, diagonal dropped).
    X
        ``(n, m)`` non-negative factor activity, one column per factor. A signed latent
        goes through :func:`split_directions` first. A DataFrame supplies its own names.
    names
        Column labels, defaulting to ``DataFrame`` columns or ``0..m-1``.
    k
        Neighbour counts to score at, each a prefix of the index array's columns (so one
        cached ``K``-NN graph serves every ``k <= K``). Defaults to all ``K`` columns. Not
        valid with a sparse graph.
    groups
        Per-cell labels for ``moran_i_within_group``.
    annotate
        ``{key: per-cell labels}`` for ``top_<key>`` / ``top_<key>_share`` footprint
        composition. May include `groups` again. Missing labels count toward no group.
    covariate
        Per-cell nuisance value (e.g. log10 depth) for ``moran_i_resid`` and
        ``covariate_spearman``; also scored itself as the ``_covariate`` reference row.
    control
        Name of the column to score again with its cells permuted, as ``_perm_control``.
        The most label-structured column is the informative choice -- e.g. the one with
        the highest :func:`scads_drvi.scores.aggregate.eta_squared` against a labelling.
    n_perm, seed
        Permutations for ``moran_p`` (0 skips it: ``moran_p`` is NaN), and the base seed:
        `seed` drives the Moran permutations, ``seed + 1`` the control's permutation and
        ``seed + 2`` the participation null's.

    Returns
    -------
    One row per (factor, k): ``factor``, ``k``, ``moran_i``, ``moran_p``, ``pr``,
    ``pr_null``, ``specificity``, then ``moran_i_within_group`` if `groups`,
    ``moran_i_resid`` and ``covariate_spearman`` if `covariate`, and ``top_<key>``,
    ``top_<key>_share`` per `annotate` key. ``k`` is ``<NA>`` for a sparse graph.
    """
    import pandas as pd

    if names is None and isinstance(X, pd.DataFrame):
        names = list(X.columns)
    Xm = np.asarray(X, dtype=np.float32)
    if Xm.ndim != 2:
        raise ValueError(f"X must be 2-D (cells x factors), got shape {Xm.shape}")
    n, m = Xm.shape
    labels = [str(i) for i in range(m)] if names is None else [str(c) for c in names]
    if len(labels) != m or len(set(labels)) != m:
        raise ValueError(f"need {m} unique names, got {len(labels)} ({len(set(labels))} unique)")
    if not np.isfinite(Xm).all():
        raise ValueError("X must be finite")
    if (Xm < 0).any():
        raise ValueError("X must be non-negative; split a signed latent with split_directions()")

    cols, refs = [Xm], []
    if control is not None:
        if str(control) not in labels:
            raise KeyError(f"control {control!r} is not one of the factor names")
        perm_c = np.random.default_rng(seed + 1).permutation(n)
        cols.append(Xm[perm_c, labels.index(str(control))][:, None])
        refs.append(PERM_CONTROL)
    cov = None
    if covariate is not None:
        cov = np.asarray(covariate, dtype=np.float64).ravel()
        if cov.shape != (n,) or not np.isfinite(cov).all():
            raise ValueError(f"covariate must be {n} finite values, got shape {cov.shape}")
        cols.append(cov.astype(np.float32)[:, None])
        refs.append(COVARIATE)
    if set(refs) & set(labels):
        raise ValueError(f"factor names collide with reference rows {sorted(set(refs) & set(labels))}")
    Xa = np.concatenate(cols, axis=1)
    all_labels = labels + refs

    grp = None if groups is None else _codes(groups, n, "groups")[0]
    if grp is not None and (grp < 0).any():
        raise ValueError("groups has missing labels")
    ann = {key: _codes(v, n, f"annotate[{key!r}]") for key, v in (annotate or {}).items()}

    extra: dict[str, np.ndarray] = {}
    if cov is not None:
        extra["covariate_spearman"] = _spearman_to(Xa, cov)
        D = np.column_stack([np.ones(n), cov])
        Xa64 = Xa.astype(np.float64)
        Xr = Xa64 - D @ np.linalg.lstsq(D, Xa64, rcond=None)[0]
        energy = ((Xa64 - Xa64.mean(0)) ** 2).sum(0)

    ks: list[int | None]
    if k is None:
        ks = [None] if sp.issparse(graph) else [int(np.shape(graph)[-1])]
    else:
        ks = [int(k)] if isinstance(k, (int, np.integer)) else [int(v) for v in k]

    perm = np.random.default_rng(seed + 2).permutation(n)
    records: list[dict[str, Any]] = []
    for kk in ks:
        P = _edge_pattern(graph, n, kk)
        W = _row_normalised(P)
        mi, mp = _moran(W, Xa, n_perm, seed)
        pr_null, _ = _participation(W, Xa[perm])
        pr, E = _participation(W, Xa)
        per_k = dict(extra)
        if grp is not None:
            per_k["moran_i_within_group"] = _within_group_moran(P, grp, Xa)
        if cov is not None:
            per_k["moran_i_resid"] = _moran_plain(W, Xr, energy)
        shares = {key: (_group_shares(codes, len(cats), E), cats)
                  for key, (codes, cats) in ann.items()}
        del E

        for c, lab in enumerate(all_labels):
            rec: dict[str, Any] = {
                "factor": lab, "k": kk, "moran_i": mi[c], "moran_p": mp[c],
                "pr": pr[c], "pr_null": pr_null[c], "specificity": 1.0 - pr[c] / pr_null[c],
            }
            for name in ("moran_i_within_group", "moran_i_resid", "covariate_spearman"):
                if name in per_k:
                    rec[name] = per_k[name][c]
            for key, (sh, cats) in shares.items():
                col = sh[:, c]
                top = int(col.argmax()) if col.size and col.max() > 0 else -1
                rec[f"top_{key}"] = cats[top] if top >= 0 else None
                rec[f"top_{key}_share"] = col[top] if top >= 0 else np.nan
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    df["k"] = df["k"].astype("Int64")
    return df
