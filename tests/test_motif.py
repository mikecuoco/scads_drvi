"""annotate.motif: the closed-form permutation null, checked against real permutations.

`stratified_nes` never samples its null: it computes the exact permutation mean and
variance of `T = sum_r W[k,r] S[r,m]` from per-stratum moments. That is the single
load-bearing claim in the method -- if it is wrong, every NES is wrong and nothing
downstream would notice. These tests sample the null the slow way and check the closed
form lands on it.
"""

from __future__ import annotations

import itertools

import numpy as np

from scads_drvi.annotate.motif import (
    covariate_strata,
    soft_threshold,
    stratified_nes,
    unstratified_nes,
)


def _toy(seed=0, n=800, K=3, M=40, G=5):
    """Small, deliberately nasty: zero-inflated and heavy-tailed on both sides, uneven strata."""
    rng = np.random.default_rng(seed)
    W = rng.lognormal(0, 2, size=(K, n)).astype(np.float32)
    W[rng.random((K, n)) < 0.6] = 0.0                      # zero inflation, like the real thing
    S = rng.lognormal(0, 1.5, size=(n, M)).astype(np.float32)
    S[rng.random((n, M)) < 0.65] = 0.0
    strata_ix = rng.integers(0, G, size=n).astype(np.int32)
    for g in range(G):                                      # guarantee every stratum >= 2
        strata_ix[rng.choice(n, 2, replace=False)] = g
    return W, S, strata_ix, G


def _moments(W, S, strata_ix, G):
    """The three accumulators `stream_accumulate` produces, computed densely here."""
    T = W.astype(np.float64) @ S.astype(np.float64)
    M = S.shape[1]
    sumS = np.zeros((G, M))
    sumS2 = np.zeros((G, M))
    np.add.at(sumS, strata_ix, S.astype(np.float64))
    np.add.at(sumS2, strata_ix, S.astype(np.float64) ** 2)
    return T, sumS, sumS2


def test_closed_form_is_exact_by_enumeration():
    """THE definitive check: enumerate every permutation, no Monte-Carlo error at all.

    6 regions in 2 strata of 3 gives 3! * 3! = 36 permutations, so the null's true mean and
    variance can be computed by brute force and compared to the closed form exactly. Measured:
    mean agrees to 0.0, sd to ~7e-16. Note the comparison is to the POPULATION sd (ddof=0) --
    the permutation distribution is the whole population, not a sample from one.
    """
    rng = np.random.default_rng(0)
    strata_ix = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    G = 2
    W = rng.lognormal(0, 1, size=(1, 6))
    S = rng.lognormal(0, 1, size=(6, 1))

    T, sumS, sumS2 = _moments(W, S, strata_ix, G)
    _, E, sd = stratified_nes(T, sumS, sumS2, W, strata_ix, G)

    vals = []
    for p0 in itertools.permutations(range(3)):
        for p1 in itertools.permutations(range(3, 6)):
            perm = np.array(p0 + p1)
            vals.append((W[:, perm] @ S).item())
    vals = np.array(vals)
    assert len(vals) == 36

    assert abs(E[0, 0] - vals.mean()) < 1e-12, f"{E[0, 0]} != {vals.mean()}"
    assert abs(sd[0, 0] - vals.std(ddof=0)) < 1e-12, f"{sd[0, 0]} != {vals.std(ddof=0)}"


def test_stratified_null_matches_sampling():
    """The same claim at realistic scale, against real within-stratum permutations.

    The sd tolerance is derived from the draws themselves (spread across batches) rather than
    from normal theory: with zero-inflated, heavy-tailed inputs the permutation distribution of
    T is far from Gaussian, so `sd/sqrt(2(n-1))` understates the sd estimator's own variance by
    ~2.5x here and would make this test either flaky or meaningless.
    """
    W, S, strata_ix, G = _toy()
    T, sumS, sumS2 = _moments(W, S, strata_ix, G)
    _, E, sd = stratified_nes(T, sumS, sumS2, W, strata_ix, G)

    n_perm = 4000
    rng = np.random.default_rng(123)
    by_stratum = [np.where(strata_ix == g)[0] for g in range(G)]
    draws = np.empty((n_perm, W.shape[0], S.shape[1]))
    for i in range(n_perm):
        perm = np.arange(len(strata_ix))
        for idx in by_stratum:                       # permute WITHIN each stratum only
            perm[idx] = rng.permutation(idx)
        draws[i] = W[:, perm].astype(np.float64) @ S.astype(np.float64)

    # The mean's standard error is sd/sqrt(n_perm) with NO estimation noise, because the
    # enumeration test above established that `sd` is the exact null sd. Standardising by an
    # estimated SE instead makes the max over K*M elements unusable: an 8-batch SE carries
    # ~27% relative error, enough to push one of 120 ratios past any fixed threshold.
    z_mean = np.abs(draws.mean(0) - E) / np.maximum(sd / np.sqrt(n_perm), 1e-12)
    assert z_mean.max() < 5.0, f"E[T] off by {z_mean.max():.2f} sigma"

    # For the sd, calibrate against the sd estimator's own variance including the heavy tails:
    # Var(s) ~= sigma^2 (2 + kurtosis) / (4 n). Normal theory (kurtosis 0) understates it ~2.5x
    # on these zero-inflated lognormals and would make this assertion arbitrary.
    emp_sd = draws.std(0, ddof=1)
    z = (draws - draws.mean(0)) / np.maximum(emp_sd, 1e-12)
    kurt = np.maximum((z**4).mean(0) - 3.0, 0.0)
    se_sd = sd * np.sqrt((2.0 + kurt) / (4.0 * n_perm))
    z_sd = np.abs(emp_sd - sd) / np.maximum(se_sd, 1e-12)
    assert z_sd.max() < 5.0, f"sd[T] off by {z_sd.max():.2f} sigma (kurtosis-corrected)"


def test_unstratified_nes_equals_correlation():
    """With one stratum the whole construction collapses to r * sqrt(n-1), exactly."""
    W, S, strata_ix, G = _toy(seed=7)
    T, sumS, sumS2 = _moments(W, S, strata_ix, G)
    nes = unstratified_nes(T, sumS, sumS2, W)

    # float64 on the reference side is required, not cosmetic: the toy weights are
    # lognormal(0,2) so a float32 matmul over 800 terms accumulates ~2e-4 relative error --
    # 10^10 times the real discrepancy, which would look like a failed identity.
    n = W.shape[1]
    Wd, Sd = W.astype(np.float64), S.astype(np.float64)
    Wc = Wd - Wd.mean(1, keepdims=True)
    Sc = Sd - Sd.mean(0, keepdims=True)
    r = (Wc @ Sc) / np.outer(np.linalg.norm(Wc, axis=1), np.linalg.norm(Sc, axis=0))

    assert np.abs(nes - r * np.sqrt(n - 1)).max() < 1e-9


def test_nes_is_invariant_to_per_factor_scale():
    """The property `weighted_motif_enrichment(check_invariance=True)` asserts on real runs."""
    W, S, strata_ix, G = _toy(seed=11)
    T, sumS, sumS2 = _moments(W, S, strata_ix, G)
    nes_a, _, _ = stratified_nes(T, sumS, sumS2, W, strata_ix, G)

    W2 = W.copy()
    W2[0] *= 1e6                       # one factor blown up by six orders of magnitude
    T2, _, _ = _moments(W2, S, strata_ix, G)
    nes_b, _, _ = stratified_nes(T2, sumS, sumS2, W2, strata_ix, G)

    assert np.abs(nes_b - nes_a).max() < 1e-6


def test_stratification_removes_a_planted_confounder():
    """A motif that is high only because its regions sit in one stratum must lose its NES."""
    rng = np.random.default_rng(3)
    n, G = 2000, 4
    strata_ix = rng.integers(0, G, size=n).astype(np.int32)
    hot = strata_ix == 0                       # e.g. the high-GC, wide-region stratum

    W = rng.lognormal(0, 1, size=(1, n)).astype(np.float32)
    W[0][hot] *= 8.0                           # this factor prefers that stratum
    S = rng.lognormal(0, 1, size=(n, 1)).astype(np.float32)
    S[hot, 0] *= 8.0                           # ...and so does this motif. No real link.

    T, sumS, sumS2 = _moments(W, S, strata_ix, G)
    nes_strat, _, _ = stratified_nes(T, sumS, sumS2, W, strata_ix, G)
    nes_raw = unstratified_nes(T, sumS, sumS2, W)

    assert abs(nes_raw[0, 0]) > 5.0, "planted confound should look significant uncorrected"
    assert abs(nes_strat[0, 0]) < abs(nes_raw[0, 0]) / 3.0, (
        f"stratification failed to suppress it: {nes_raw[0, 0]:.2f} -> {nes_strat[0, 0]:.2f}")


def test_soft_threshold_zeros_only_low_positives():
    """Zeros go below the given percentile of each row's OWN positive values."""
    W = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 0.0, 0.0]])
    out = soft_threshold(W, 50.0)
    assert (out[0] == 0).sum() > (W[0] == 0).sum()   # some positives were zeroed in row 0
    assert np.array_equal(out[1], W[1])              # a constant row has no "below" to zero


def test_soft_threshold_noop_at_zero_pct():
    W = np.array([[0.0, 1.0, 2.0]])
    assert np.array_equal(soft_threshold(W, 0.0), W)


def test_covariate_strata_matches_manual_gc_width_crossing():
    """One int `n_bins` applies to every covariate; the strata are the bin cross-product."""
    rng = np.random.default_rng(5)
    gc = rng.uniform(0.3, 0.7, size=500)
    width = rng.uniform(150, 350, size=500)

    keep, strata_ix, n_strata = covariate_strata([gc, width], n_bins=4)
    assert keep.all()
    assert n_strata <= 16
    # Every kept stratum is non-empty and every region belongs to exactly one.
    assert np.bincount(strata_ix, minlength=n_strata).min() >= 1


def test_covariate_strata_drops_small_strata_and_nonfinite():
    x = np.concatenate([np.full(3, np.nan), np.linspace(0, 1, 197)])
    keep, strata_ix, n_strata = covariate_strata([x], n_bins=50, min_size=2)
    assert keep.sum() < 200            # the 3 NaNs are gone, and possibly a thin bin
    assert not keep[:3].any()
    assert np.bincount(strata_ix, minlength=n_strata).min() >= 2
