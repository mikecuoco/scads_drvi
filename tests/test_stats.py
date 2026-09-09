"""stats: one-tailed p, Benjamini-Hochberg, and the legacy comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scads_drvi.stats import (
    add_fdr,
    bh_qvalues,
    bh_threshold_z,
    compare_bh,
    legacy_bh_qvalues,
    p_one_tailed,
    significant,
)


def r_p_adjust_bh(p, n=None):
    """R's p.adjust(p, "BH") transcribed directly, as an independent reference.

        i  <- lp:1L
        o  <- order(p, decreasing = TRUE)
        ro <- order(o)
        pmin(1, cummin(n / i * p[o]))[ro]

    Deliberately a different code path from the implementation under test: descending
    order plus a forward cumulative minimum, rather than ascending plus a reversed one.
    """
    p = np.asarray(p, dtype=float)
    m = p.size
    n = m if n is None else n
    order = np.argsort(-p, kind="stable")
    i = np.arange(m, 0, -1, dtype=float)
    scaled = n / i * p[order]
    adjusted = np.minimum.accumulate(scaled)
    out = np.empty(m, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


class TestOneTailedP:
    def test_upper_tail_is_the_survival_function(self):
        assert p_one_tailed(0.0) == pytest.approx(0.5)
        assert p_one_tailed(1.6448536) == pytest.approx(0.05, abs=1e-6)
        assert p_one_tailed(-1.6448536) == pytest.approx(0.95, abs=1e-6)

    def test_lower_tail_mirrors_the_upper(self):
        z = np.array([-2.0, 0.0, 2.0])
        assert np.allclose(p_one_tailed(z, tail="lower"), 1 - p_one_tailed(z))

    def test_rejects_an_unknown_tail(self):
        with pytest.raises(ValueError, match="upper.*lower"):
            p_one_tailed(0.0, tail="both")


class TestBHAgreesWithR:
    @pytest.mark.parametrize("seed", range(6))
    def test_random_vectors(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.uniform(size=rng.integers(2, 200))
        assert np.allclose(bh_qvalues(p), r_p_adjust_bh(p), atol=1e-12)

    def test_with_heavy_ties(self):
        p = np.array([0.01] * 5 + [0.5] * 5 + [1.0] * 3)
        assert np.allclose(bh_qvalues(p), r_p_adjust_bh(p), atol=1e-12)

    def test_with_zeros_and_ones(self):
        p = np.array([0.0, 0.0, 0.3, 1.0, 1.0])
        assert np.allclose(bh_qvalues(p), r_p_adjust_bh(p), atol=1e-12)

    @pytest.mark.parametrize("n", [50, 200, 6_000_000])
    def test_explicit_denominator(self, n):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=40)
        assert np.allclose(bh_qvalues(p, n=n), r_p_adjust_bh(p, n=n), atol=1e-12)


class TestBHProperties:
    def test_q_is_monotone_in_p(self):
        """The property the notebooks' version failed to enforce."""
        rng = np.random.default_rng(7)
        p = rng.uniform(size=500)
        q = bh_qvalues(p)
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-12)

    def test_q_never_below_p(self):
        rng = np.random.default_rng(8)
        p = rng.uniform(size=200)
        assert np.all(bh_qvalues(p) >= p - 1e-12)

    def test_q_is_bounded(self):
        rng = np.random.default_rng(9)
        q = bh_qvalues(rng.uniform(size=200))
        assert q.min() >= 0.0 and q.max() <= 1.0

    def test_nan_maps_to_one_without_shrinking_the_denominator(self):
        p = np.array([0.001, np.nan, 0.5])
        q = bh_qvalues(p)
        assert q[1] == 1.0
        # the two finite values are corrected over m=2, not m=3
        assert np.allclose(q[[0, 2]], r_p_adjust_bh(np.array([0.001, 0.5])), atol=1e-12)

    def test_empty_and_all_nan(self):
        assert bh_qvalues(np.array([])).size == 0
        assert np.all(bh_qvalues(np.array([np.nan, np.nan])) == 1.0)

    def test_single_value_is_unchanged(self):
        assert bh_qvalues(np.array([0.37]))[0] == pytest.approx(0.37)

    def test_denominator_smaller_than_the_tests_is_refused(self):
        with pytest.raises(ValueError, match="smaller than"):
            bh_qvalues(np.array([0.1, 0.2, 0.3]), n=2)

    def test_two_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match="1-D"):
            bh_qvalues(np.zeros((2, 2)))


class TestLegacyComparison:
    def test_legacy_is_not_monotone(self):
        """The defect, pinned: a q that decreases as p increases."""
        p = np.array([0.04, 0.05, 0.0001])
        legacy = legacy_bh_qvalues(p)
        order = np.argsort(p)
        assert np.any(np.diff(legacy[order]) < -1e-12)
        assert np.all(np.diff(bh_qvalues(p)[order]) >= -1e-12)

    def test_compare_bh_reports_flips(self):
        p = np.array([0.001, 0.02, 0.049, 0.4, 0.9])
        table = compare_bh(p)
        assert list(table.columns) == [
            "p", "q_correct", "q_legacy", "delta", "flips_at_05",
        ]
        assert len(table) == len(p)
        assert table["flips_at_05"].dtype == bool

    def test_the_legacy_values_were_conservative_not_anti_conservative(self):
        """Which direction the defect ran, pinned -- it decides what re-running means.

        The step-up is a cumulative MINIMUM taken from the largest p downwards, so a
        correct q can only be <= the raw p*m/rank the legacy code reported. The
        notebooks' q-values were therefore too large: the defect could hide a real
        signal, never manufacture one. Fixing it can only add significant factors.
        """
        rng = np.random.default_rng(3)
        p = rng.uniform(size=300)
        table = compare_bh(p)
        assert np.all(table["q_correct"] <= table["q_legacy"] + 1e-12)
        assert (table["q_correct"] < table["q_legacy"]).any()


class TestThresholdZ:
    def test_returns_the_smallest_passing_z(self):
        q = np.array([0.01, 0.04, 0.20])
        z = np.array([4.0, 3.1, 1.0])
        assert bh_threshold_z(q, z) == pytest.approx(3.1)

    def test_returns_none_when_nothing_passes(self):
        assert bh_threshold_z(np.array([0.2, 0.9]), np.array([1.0, 0.1])) is None

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            bh_threshold_z(np.array([0.1]), np.array([1.0, 2.0]))


class TestAddFdr:
    def _frame(self):
        return pd.DataFrame(
            {
                "group": ["a", "a", "a", "b", "b", "b"],
                "Coefficient_z-score": [4.0, 2.0, 0.5, 3.0, 1.0, -1.0],
            }
        )

    def test_adds_p_and_q(self):
        out = add_fdr(self._frame())
        assert {"p_1tailed", "fdr_q"} <= set(out.columns)
        assert len(out) == 6

    def test_does_not_mutate_the_input(self):
        frame = self._frame()
        add_fdr(frame)
        assert "fdr_q" not in frame.columns

    def test_grouping_corrects_within_family_only(self):
        frame = self._frame()
        pooled = add_fdr(frame)
        grouped = add_fdr(frame, by="group")
        # Correcting over 3 rather than 6 tests cannot give a larger q.
        assert np.all(grouped["fdr_q"].to_numpy() <= pooled["fdr_q"].to_numpy() + 1e-12)
        for _, block in grouped.groupby("group"):
            expected = r_p_adjust_bh(block["p_1tailed"].to_numpy())
            assert np.allclose(block["fdr_q"].to_numpy(), expected, atol=1e-12)

    def test_index_is_preserved_when_grouping(self):
        frame = self._frame().set_index(pd.Index(list("uvwxyz")))
        out = add_fdr(frame, by="group")
        assert list(out.index) == list("uvwxyz")

    def test_missing_columns_are_named(self):
        with pytest.raises(KeyError, match="Coefficient_z-score"):
            add_fdr(pd.DataFrame({"x": [1.0]}))
        with pytest.raises(KeyError, match="nope"):
            add_fdr(self._frame(), by="nope")

    def test_significant_reads_the_stated_alpha(self):
        out = add_fdr(self._frame(), by="group")
        assert significant(out).sum() <= significant(out, alpha=0.5).sum()
        with pytest.raises(KeyError, match="run add_fdr"):
            significant(pd.DataFrame({"x": [1]}))
