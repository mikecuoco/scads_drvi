"""specificity: Moran's I, footprint participation and specificity on a cell graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scads_drvi.specificity import (
    _edge_pattern,
    _moran,
    _row_normalised,
    graph_specificity,
    split_directions,
)

N_BLOB, N_GROUPS, K = 250, 8, 15


def knn(Z: np.ndarray, k: int) -> np.ndarray:
    """Exact kNN by brute force, self excluded."""
    d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(N_GROUPS), N_BLOB)
    centres = rng.normal(scale=20.0, size=(N_GROUPS, 5))
    Z = centres[group] + rng.normal(size=(group.size, 5))
    nb = knn(Z, K)
    n = group.size
    cov = rng.normal(size=n)
    X = pd.DataFrame({
        # identity: on in one blob, constant inside it up to iid noise
        "blob0": (group == 0) + 0.05 * rng.random(n),
        # state: follows a latent direction within every blob
        "state": np.maximum(Z[:, 0] - centres[group, 0], 0),
        # a state inside one blob only: the narrowest footprint
        "narrow": (group == 0) * np.maximum(Z[:, 0] - centres[group, 0], 0),
        "noise": rng.exponential(size=n),
        "lin_cov": cov - cov.min(),
    })
    return {"nb": nb, "X": X, "group": group, "cov": cov, "n": n}


@pytest.fixture(scope="module")
def result(world):
    labels = np.array([f"g{g}" for g in world["group"]])
    return graph_specificity(
        world["nb"], world["X"], k=[10, K], groups=labels, annotate={"blob": labels},
        covariate=world["cov"], control="blob0", n_perm=19,
    ).set_index(["factor", "k"])


def test_columns_and_rows(result):
    assert set(result.index.get_level_values("factor")) == {
        "blob0", "state", "narrow", "noise", "lin_cov", "_perm_control", "_covariate"}
    assert set(result.index.get_level_values("k")) == {10, K}
    for col in ("moran_i", "moran_p", "pr", "pr_null", "specificity", "moran_i_within_group",
                "moran_i_resid", "covariate_spearman", "top_blob", "top_blob_share"):
        assert col in result.columns


def test_community_specific_factor(result):
    r = result.loc[("blob0", K)]
    assert r.moran_i > 0.9
    assert r.moran_p == pytest.approx(1 / 20)
    # one blob of eight: pr is its 1/8 footprint, well under the permuted null's
    assert r.pr == pytest.approx(1 / N_GROUPS, rel=0.05)
    assert r.specificity > 0.4
    assert r.top_blob == "g0" and r.top_blob_share > 0.99
    # identity: nothing left once centred within its blob
    assert abs(r.moran_i_within_group) < 0.1


def test_narrower_footprint_is_more_specific(result):
    spec = result.xs(K, level="k")["specificity"]
    assert spec["narrow"] > spec["blob0"] > spec["state"]
    assert spec["blob0"] > spec["noise"] + 0.3


def test_state_factor_is_coherent_within_groups(result):
    assert result.loc[("state", K), "moran_i_within_group"] > 0.5


@pytest.mark.parametrize("factor", ["noise", "_perm_control"])
def test_structureless_columns_score_zero(result, factor):
    r = result.loc[(factor, K)]
    assert abs(r.moran_i) < 0.05
    assert abs(r.specificity) < 0.1
    assert r.moran_p > 0.05


def test_covariate_residualisation(result):
    r = result.loc[("lin_cov", K)]
    assert r.covariate_spearman == pytest.approx(1.0)
    assert np.isnan(r.moran_i_resid)
    assert np.isnan(result.loc[("_covariate", K), "moran_i_resid"])
    assert abs(result.loc[("blob0", K), "covariate_spearman"]) < 0.1


def test_sparse_graph_matches_index_array(world):
    nb, n = world["nb"], world["n"]
    G = sp.csr_matrix((np.random.default_rng(1).random(nb.size) + 0.1,
                       (np.repeat(np.arange(n), K), nb.ravel())), shape=(n, n))
    G.setdiag(1.0)  # self edges must be dropped
    a = graph_specificity(nb, world["X"], n_perm=9)
    b = graph_specificity(G, world["X"], n_perm=9)
    num = ["moran_i", "moran_p", "pr", "pr_null", "specificity"]
    np.testing.assert_allclose(a[num].to_numpy(), b[num].to_numpy(), rtol=1e-5)
    assert a["k"].tolist() == [K] * 5 and b["k"].isna().all()


def test_k_prefix_equals_separate_call(world, result):
    alone = graph_specificity(world["nb"][:, :10], world["X"], n_perm=19).set_index("factor")
    for f in ("blob0", "noise"):
        assert alone.loc[f, "moran_i"] == pytest.approx(result.loc[(f, 10), "moran_i"], rel=1e-5)
        assert alone.loc[f, "pr"] == pytest.approx(result.loc[(f, 10), "pr"], rel=1e-5)


def test_self_entry_in_index_array_is_dropped():
    nb = np.array([[0, 1], [0, 2], [1, 2]])  # row 0 lists itself; row 2 too
    W = _row_normalised(_edge_pattern(nb, 3, None)).toarray()
    np.testing.assert_allclose(W, [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])


def test_moran_matches_dense_formula():
    rng = np.random.default_rng(3)
    n = 30
    A = (rng.random((n, n)) < 0.2).astype(float)
    np.fill_diagonal(A, 0)
    A[A.sum(1) == 0, 0] = 1
    A[0, 0] = 0
    Wd = A / A.sum(1, keepdims=True)
    x = rng.random((n, 2))
    stat, _ = _moran(sp.csr_matrix(Wd.astype(np.float32)), x.astype(np.float32), 0, 0)
    xc = x - x.mean(0)
    ref = np.einsum("ic,ij,jc->c", xc, Wd, xc) / (xc**2).sum(0)
    np.testing.assert_allclose(stat, ref, rtol=1e-5)


def test_split_directions():
    X = np.array([[1.0, -2.0], [-0.5, 0.0]])
    arms, names = split_directions(X, ["a", "b"])
    assert names == ["a_pos", "b_pos", "a_neg", "b_neg"]
    assert (arms >= 0).all()
    np.testing.assert_array_equal(arms[:, :2] - arms[:, 2:], X)


def test_rejects_bad_input(world):
    nb, X = world["nb"], world["X"]
    with pytest.raises(ValueError, match="non-negative"):
        graph_specificity(nb, -X)
    with pytest.raises(ValueError, match="neighbour indices"):
        graph_specificity(nb[:-1], X)
    with pytest.raises(ValueError, match="sparse graph"):
        graph_specificity(sp.eye(world["n"], format="csr"), X, k=5)
    with pytest.raises(KeyError):
        graph_specificity(nb, X, control="missing")
    with pytest.raises(ValueError, match="k=99"):
        graph_specificity(nb, X, k=99)
