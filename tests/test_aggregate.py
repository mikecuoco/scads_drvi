"""scores.aggregate: grouping is the caller's, thin groups are visible."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scads_drvi.scores.aggregate import (
    GroupStats,
    block_order,
    eta_squared,
    group_matrix,
    profile_by_group,
    streaming_correlation,
    summarize_by,
)


@pytest.fixture
def cells():
    return pd.DataFrame(
        {
            "grouping": ["a"] * 30 + ["b"] * 30 + ["c"] * 3,
            "block": ["B1"] * 30 + ["B1"] * 30 + ["B2"] * 3,
            "axis": (["x", "y"] * 15) + (["x", "y"] * 15) + ["x", "y", "x"],
        },
        index=[f"c{i}" for i in range(63)],
    )


@pytest.fixture
def values(cells):
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(size=len(cells)), index=cells.index, name="cs")


class TestSummarizeBy:
    def test_thin_groups_are_dropped_and_counted(self, values, cells):
        stats = summarize_by(values, cells, by="grouping", min_cells=20)
        assert isinstance(stats, GroupStats)
        assert set(stats.frame["grouping"]) == {"a", "b"}
        assert stats.n_dropped == 1
        assert stats.dropped == ("c",)

    def test_lowering_min_cells_keeps_them(self, values, cells):
        stats = summarize_by(values, cells, by="grouping", min_cells=1)
        assert set(stats.frame["grouping"]) == {"a", "b", "c"}
        assert stats.n_dropped == 0

    def test_reports_n_and_n_scored(self, values, cells):
        stats = summarize_by(values, cells, by="grouping", min_cells=1)
        row = stats.frame.set_index("grouping").loc["a"]
        assert row["n"] == 30 and row["n_scored"] == 30

    def test_nan_values_count_in_n_but_not_n_scored(self, cells):
        vals = pd.Series(np.ones(len(cells)), index=cells.index, name="cs")
        vals.iloc[:5] = np.nan
        stats = summarize_by(vals, cells, by="grouping", min_cells=1)
        row = stats.frame.set_index("grouping").loc["a"]
        assert row["n"] == 30 and row["n_scored"] == 25

    def test_confidence_interval_brackets_the_mean(self, values, cells):
        stats = summarize_by(values, cells, by="grouping", min_cells=1)
        assert (stats.frame["ci_low"] <= stats.frame["mean"]).all()
        assert (stats.frame["mean"] <= stats.frame["ci_high"]).all()

    def test_wider_ci_is_wider(self, values, cells):
        narrow = summarize_by(values, cells, by="grouping", ci=0.5, min_cells=1).frame
        wide = summarize_by(values, cells, by="grouping", ci=0.99, min_cells=1).frame
        assert ((wide["ci_high"] - wide["ci_low"]) >= (narrow["ci_high"] - narrow["ci_low"])).all()

    def test_quantiles_are_added_on_request(self, values, cells):
        stats = summarize_by(
            values, cells, by="grouping", min_cells=1, quantiles=(25, 75)
        )
        assert {"q25", "q75"} <= set(stats.frame.columns)

    def test_multiple_grouping_columns(self, values, cells):
        stats = summarize_by(values, cells, by=["grouping", "axis"], min_cells=1)
        assert stats.keys == ("grouping", "axis")
        assert len(stats.frame) == 6

    def test_unknown_grouping_is_named(self, values, cells):
        with pytest.raises(KeyError, match="cannot group by"):
            summarize_by(values, cells, by="nope")

    def test_disjoint_index_is_refused(self, cells):
        other = pd.Series([1.0], index=["zzz"], name="cs")
        with pytest.raises(ValueError, match="share no cells"):
            summarize_by(other, cells, by="grouping")

    def test_bad_ci_is_refused(self, values, cells):
        with pytest.raises(ValueError, match="ci must be"):
            summarize_by(values, cells, by="grouping", ci=1.5)

    def test_order_by(self, values, cells):
        stats = summarize_by(values, cells, by="grouping", min_cells=1)
        ordered = stats.order_by("mean")
        assert ordered.frame["mean"].is_monotonic_decreasing
        with pytest.raises(KeyError, match="not in the summary"):
            stats.order_by("nope")


class TestGroupMatrix:
    def test_returns_means_and_counts(self, values, cells):
        means, counts = group_matrix(
            values, cells, index="grouping", columns="axis", min_cells=1
        )
        assert means.shape == counts.shape == (3, 2)
        assert counts.to_numpy().sum() == len(cells)

    def test_thin_bins_are_nan_in_the_mean_and_visible_in_the_count(self, values, cells):
        """Grey must mean 'no reliable value', never 'measured and low'."""
        means, counts = group_matrix(
            values, cells, index="grouping", columns="axis", min_cells=10
        )
        assert means.loc["c"].isna().all()
        assert counts.loc["c"].sum() == 3

    def test_orders_can_be_imposed(self, values, cells):
        means, counts = group_matrix(
            values,
            cells,
            index="grouping",
            columns="axis",
            min_cells=1,
            index_order=["c", "b", "a"],
            column_order=["y", "x"],
        )
        assert list(means.index) == ["c", "b", "a"]
        assert list(means.columns) == ["y", "x"]
        assert list(counts.index) == ["c", "b", "a"]

    def test_absent_ordered_level_becomes_nan_not_zero(self, values, cells):
        means, counts = group_matrix(
            values,
            cells,
            index="grouping",
            columns="axis",
            min_cells=1,
            index_order=["a", "absent"],
        )
        assert means.loc["absent"].isna().all()
        assert (counts.loc["absent"] == 0).all()

    def test_unknown_column_is_named(self, values, cells):
        with pytest.raises(KeyError, match="not in the cell table"):
            group_matrix(values, cells, index="nope", columns="axis")


class TestBlockOrder:
    def test_groups_are_ordered_within_blocks(self, values, cells):
        stats = summarize_by(values, cells, by=["block", "grouping"], min_cells=1)
        names, spans = block_order(stats, group="grouping", block="block")
        assert set(names) == {"a", "b", "c"}
        assert [b for b, _, _ in spans] == ["B1", "B2"]

    def test_spans_tile_the_axis_without_gaps(self, values, cells):
        stats = summarize_by(values, cells, by=["block", "grouping"], min_cells=1)
        names, spans = block_order(stats, group="grouping", block="block")
        assert spans[0][1] == 0
        assert spans[-1][2] == len(names)
        for (_, _, stop), (_, start, _) in zip(spans, spans[1:]):
            assert stop == start

    def test_descending_by_default(self, values, cells):
        stats = summarize_by(values, cells, by=["block", "grouping"], min_cells=1)
        names, spans = block_order(stats, group="grouping", block="block")
        frame = stats.frame.set_index("grouping")
        first_block = names[spans[0][1] : spans[0][2]]
        means = [frame.loc[n, "mean"] for n in first_block]
        assert means == sorted(means, reverse=True)

    def test_accepts_a_plain_frame(self, values, cells):
        stats = summarize_by(values, cells, by=["block", "grouping"], min_cells=1)
        names, _ = block_order(stats.frame, group="grouping", block="block")
        assert len(names) == 3

    def test_missing_column_is_named(self, values, cells):
        stats = summarize_by(values, cells, by="grouping", min_cells=1)
        with pytest.raises(KeyError, match="not in the summary"):
            block_order(stats, group="grouping", block="block")


class TestProfileByGroup:
    @pytest.fixture
    def matrix(self, cells):
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            rng.random((len(cells), 4)),
            index=cells.index,
            columns=[f"dim_{i}" for i in range(4)],
        )

    def test_means_and_counts(self, matrix, cells):
        means, z, counts = profile_by_group(matrix, cells["grouping"], min_cells=1)
        assert means.shape == (3, 4)
        assert counts == {"a": 30, "b": 30, "c": 3}

    def test_z_is_taken_within_a_column(self, matrix, cells):
        """Across groups for one factor -- 'which groups stand out on this factor',
        not 'which factor stands out in this group'."""
        _, z, _ = profile_by_group(matrix, cells["grouping"], min_cells=1)
        assert np.allclose(z.mean(axis=0).to_numpy(), 0.0, atol=1e-12)
        assert np.allclose(z.std(axis=0, ddof=0).to_numpy(), 1.0, atol=1e-12)

    def test_thin_groups_become_nan_rows_not_missing_rows(self, matrix, cells):
        means, _, counts = profile_by_group(matrix, cells["grouping"], min_cells=10)
        assert "c" in means.index
        assert means.loc["c"].isna().all()
        assert counts["c"] == 3

    def test_zscore_can_be_disabled(self, matrix, cells):
        means, z, _ = profile_by_group(
            matrix, cells["grouping"], min_cells=1, zscore=False
        )
        pd.testing.assert_frame_equal(means, z)

    def test_a_constant_column_does_not_divide_by_zero(self, cells):
        flat = pd.DataFrame({"dim_0": np.ones(len(cells))}, index=cells.index)
        _, z, _ = profile_by_group(flat, cells["grouping"], min_cells=1)
        assert z["dim_0"].isna().all()


class TestEtaSquared:
    def test_perfect_separation_is_one(self):
        assert eta_squared([1, 1, 2, 2], ["a", "a", "b", "b"]) == pytest.approx(1.0)

    def test_no_group_effect_is_zero(self):
        """Both groups have the same mean, so none of the spread is between them."""
        assert eta_squared([1, 2, 1, 2], ["a", "a", "b", "b"]) == pytest.approx(0.0)

    def test_interleaved_labels_can_still_separate_perfectly(self):
        """Group membership, not position: a/b/a/b over 1,2,1,2 is total separation."""
        assert eta_squared([1, 2, 1, 2], ["a", "b", "a", "b"]) == pytest.approx(1.0)

    def test_bounded_in_zero_one(self):
        rng = np.random.default_rng(0)
        value = eta_squared(rng.normal(size=100), rng.integers(0, 4, size=100))
        assert 0.0 <= value <= 1.0

    def test_no_variance_is_nan(self):
        assert np.isnan(eta_squared([5, 5, 5], ["a", "b", "c"]))

    def test_nan_values_are_ignored(self):
        assert eta_squared([1, 1, 2, 2, np.nan], ["a", "a", "b", "b", "b"]) == pytest.approx(1.0)

    def test_empty_is_nan(self):
        assert np.isnan(eta_squared([], []))

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            eta_squared([1, 2], ["a"])


class TestStreamingCorrelation:
    def test_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(500, 5))
        chunks = [matrix[i : i + 77] for i in range(0, len(matrix), 77)]
        got = streaming_correlation(chunks, 5)
        assert np.allclose(got, np.corrcoef(matrix, rowvar=False), atol=1e-8)

    def test_chunking_does_not_matter(self):
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(200, 3))
        a = streaming_correlation([matrix], 3)
        b = streaming_correlation([matrix[i : i + 1] for i in range(200)], 3)
        assert np.allclose(a, b, atol=1e-8)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(5)
        got = streaming_correlation([rng.normal(size=(50, 4))], 4)
        assert np.allclose(np.diag(got), 1.0, atol=1e-10)

    def test_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="each chunk must be"):
            streaming_correlation([np.zeros((5, 2))], 3)

    def test_too_few_rows_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            streaming_correlation([np.zeros((1, 2))], 2)
