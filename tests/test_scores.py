"""scores.cell: two formulas, two nulls, no way to confuse them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scads_drvi.scores.cell import (
    NULL_VALUE,
    CellScores,
    ScoreKind,
    compare_scores,
    cs_from_enrichment,
    cs_from_z,
    factor_weights,
    read_cell_scores,
)


@pytest.fixture
def loadings():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.random((50, 4)).astype(np.float32),
        columns=["dim_0", "dim_1", "dim_2", "dim_3"],
        index=[f"cell_{i}" for i in range(50)],
    )


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "factor": ["dim_0", "dim_1", "dim_2", "dim_3"],
            "Coefficient_z-score": [3.0, -2.0, 0.0, 1.5],
        }
    )


class TestNulls:
    def test_each_kind_has_its_own_null(self):
        assert NULL_VALUE[ScoreKind.Z_WEIGHTED] == 0.0
        assert NULL_VALUE[ScoreKind.SCADS_RATIO] == 1.0

    def test_cell_scores_reports_its_own_null(self):
        series = pd.Series([1.0, 2.0])
        assert CellScores(series, ScoreKind.Z_WEIGHTED, "m", "t").null == 0.0
        assert CellScores(series, ScoreKind.SCADS_RATIO, "m", "t").null == 1.0

    def test_labels_name_the_formula(self):
        series = pd.Series([1.0])
        a = CellScores(series, ScoreKind.Z_WEIGHTED, "m", "t").label
        b = CellScores(series, ScoreKind.SCADS_RATIO, "m", "t").label
        assert a != b


class TestFactorWeights:
    def test_negative_z_is_clipped_by_default(self, results):
        w = factor_weights(results, factors=["dim_0", "dim_1"])
        assert list(w) == [3.0, 0.0]

    def test_clipping_can_be_disabled(self, results):
        w = factor_weights(results, factors=["dim_1"], clip_negative=False)
        assert w[0] == -2.0

    def test_order_follows_the_requested_dims(self, results):
        w = factor_weights(results, factors=["dim_3", "dim_0"])
        assert list(w) == [1.5, 3.0]

    def test_missing_factor_is_named(self, results):
        with pytest.raises(KeyError, match="no result row"):
            factor_weights(results, factors=["dim_9"])

    def test_duplicate_dims_are_refused(self, results):
        doubled = pd.concat([results, results])
        with pytest.raises(ValueError, match="more than one trait"):
            factor_weights(doubled, factors=["dim_0"])

    def test_missing_column_is_named(self, results):
        with pytest.raises(KeyError, match="nope"):
            factor_weights(results, factors=["dim_0"], column="nope")


class TestCsFromZ:
    def test_matches_the_notebook_expression(self, loadings, results):
        """Pin the collapse: this must equal L[kept] @ max(0, z) exactly."""
        dims = results["factor"].tolist()
        expected = loadings[dims].to_numpy(dtype=np.float64) @ np.maximum(
            results["Coefficient_z-score"].to_numpy(dtype=float), 0.0
        )
        got = cs_from_z(loadings, results, model="m", trait="t")
        assert np.allclose(got.values.to_numpy(), expected, atol=1e-12)

    @pytest.mark.parametrize("chunk", [1, 7, 50, 10_000])
    def test_chunked_equals_unchunked(self, loadings, results, chunk):
        ref = cs_from_z(loadings, results, model="m", trait="t", chunk_rows=10_000)
        got = cs_from_z(loadings, results, model="m", trait="t", chunk_rows=chunk)
        assert np.allclose(got.values.to_numpy(), ref.values.to_numpy(), atol=1e-12)

    def test_result_is_non_negative(self, loadings, results):
        """Loadings are non-negative and weights are clipped, so the score is too."""
        got = cs_from_z(loadings, results, model="m", trait="t")
        assert (got.values >= 0).all()

    def test_index_is_preserved(self, loadings, results):
        got = cs_from_z(loadings, results, model="m", trait="t")
        assert list(got.values.index) == list(loadings.index)

    def test_kind_and_provenance_are_recorded(self, loadings, results):
        got = cs_from_z(loadings, results, model="m", trait="t")
        assert got.kind is ScoreKind.Z_WEIGHTED
        assert got.null == 0.0
        assert got.factors == ("dim_0", "dim_1", "dim_2", "dim_3")
        assert got.model == "m" and got.trait == "t"

    def test_factors_absent_from_the_loadings_are_counted(self, loadings, results):
        extra = pd.concat(
            [results, pd.DataFrame({"factor": ["dim_9"], "Coefficient_z-score": [5.0]})]
        )
        got = cs_from_z(loadings, extra, model="m", trait="t")
        assert got.n_unscored == 1
        assert "dim_9" not in got.factors

    def test_no_overlap_at_all_is_an_error(self, loadings):
        results = pd.DataFrame({"factor": ["other"], "Coefficient_z-score": [1.0]})
        with pytest.raises(ValueError, match="no factor in the results table"):
            cs_from_z(loadings, results, model="m", trait="t")

    def test_non_dataframe_loadings_are_refused(self, results):
        with pytest.raises(TypeError, match="cells x factors"):
            cs_from_z(np.zeros((3, 4)), results, model="m", trait="t")


class TestReadCellScores:
    def test_reads_the_ratio_score(self, tmp_path):
        path = tmp_path / "cell_scores.tsv"
        pd.DataFrame({"barcode": ["a", "b"], "cs": [1.2, 0.8]}).to_csv(
            path, sep="\t", index=False
        )
        scores = read_cell_scores(path, model="m", trait="t")
        assert scores.kind is ScoreKind.SCADS_RATIO
        assert scores.null == 1.0
        assert list(scores.values.index) == ["a", "b"]

    def test_missing_file_explains_the_naming_collision(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="came to share a name"):
            read_cell_scores(tmp_path / "absent.tsv", model="m", trait="t")

    def test_missing_column_is_named(self, tmp_path):
        path = tmp_path / "cs.tsv"
        pd.DataFrame({"barcode": ["a"]}).to_csv(path, sep="\t", index=False)
        with pytest.raises(KeyError, match="'cs'"):
            read_cell_scores(path, model="m", trait="t")


class TestCompareScores:
    def _pair(self):
        index = [f"c{i}" for i in range(20)]
        rng = np.random.default_rng(1)
        base = rng.random(20)
        left = CellScores(pd.Series(base, index=index), ScoreKind.Z_WEIGHTED, "m", "t")
        right = CellScores(
            pd.Series(1.0 + base * 0.5, index=index), ScoreKind.SCADS_RATIO, "m", "t"
        )
        return left, right

    def test_reports_both_correlations_and_both_nulls(self):
        left, right = self._pair()
        out = compare_scores(left, right)
        assert out.loc[0, "pearson"] == pytest.approx(1.0)
        assert out.loc[0, "spearman"] == pytest.approx(1.0)
        assert out.loc[0, "null_left"] == 0.0
        assert out.loc[0, "null_right"] == 1.0

    def test_comparing_two_of_the_same_kind_is_refused(self):
        left, _ = self._pair()
        with pytest.raises(ValueError, match="telling the two"):
            compare_scores(left, left)

    def test_disjoint_cells_are_refused(self):
        left, right = self._pair()
        other = CellScores(
            pd.Series([1.0], index=["zzz"]), ScoreKind.SCADS_RATIO, "m", "t"
        )
        with pytest.raises(ValueError, match="share no cells"):
            compare_scores(left, other)

    def test_only_shared_cells_are_compared(self):
        left, right = self._pair()
        trimmed = CellScores(
            right.values.iloc[:5], ScoreKind.SCADS_RATIO, "m", "t"
        )
        out = compare_scores(left, trimmed)
        assert out.loc[0, "n_shared"] == 5


class TestRenamedNames:
    """`dim` became `factor` (column, `factors=`, `factor_column=`, `CellScores.factors`)."""

    def test_old_column_name_still_works_and_warns(self, loadings, results):
        legacy = results.rename(columns={"factor": "dim"})
        with pytest.warns(DeprecationWarning, match="'dim' column is now called 'factor'"):
            got = cs_from_z(loadings, legacy, model="m", trait="t")
        ref = cs_from_z(loadings, results, model="m", trait="t")
        assert np.allclose(got.values.to_numpy(), ref.values.to_numpy())

    def test_old_dims_keyword_still_works_and_warns(self, results):
        with pytest.warns(DeprecationWarning, match="`dims` is now `factors`"):
            w = factor_weights(results, dims=["dim_0", "dim_1"])
        assert list(w) == [3.0, 0.0]

    def test_old_dim_column_keyword_still_works_and_warns(self, results):
        legacy = results.rename(columns={"factor": "dim"})
        with pytest.warns(DeprecationWarning, match="`dim_column` is now `factor_column`"):
            w = factor_weights(legacy, factors=["dim_0"], dim_column="dim")
        assert list(w) == [3.0]

    def test_both_old_and_new_keyword_is_an_error(self, results):
        with pytest.raises(TypeError, match="pass `factors` only"):
            factor_weights(results, factors=["dim_0"], dims=["dim_0"])

    def test_factors_are_required(self, results):
        with pytest.raises(TypeError, match="'factors'"):
            factor_weights(results)

    def test_cell_scores_dims_property_is_a_deprecated_alias(self, loadings, results):
        got = cs_from_z(loadings, results, model="m", trait="t")
        with pytest.warns(DeprecationWarning, match="`CellScores.dims`"):
            assert got.dims == got.factors


class TestCsFromEnrichment:
    """cs_i = sum_k L_ik a_k e_k / sum_k L_ik a_k, checked against hand arithmetic."""

    @pytest.fixture
    def toy(self):
        # keys k1 (a=2, e=3) and k2 (a=6, e=1); four cells.
        loadings = pd.DataFrame(
            {"k1": [1.0, 0.0, 1.0, 0.0], "k2": [0.0, 1.0, 1.0, 0.0]},
            index=["c0", "c1", "c2", "c3"],
        )
        enrichment = pd.Series({"k1": 3.0, "k2": 1.0})
        annot_size = pd.Series({"k1": 2.0, "k2": 6.0})
        return loadings, enrichment, annot_size

    def test_matches_hand_computed_values(self, toy):
        got = cs_from_enrichment(*toy, model="m", trait="t")
        # c0: (1*2*3)/(1*2) = 3 ; c1: (1*6*1)/(1*6) = 1 ; c2: (2*3 + 6*1)/(2 + 6) = 1.5
        assert got.values.iloc[:3].tolist() == pytest.approx([3.0, 1.0, 1.5])

    def test_cell_with_no_loading_is_nan_and_counted_not_zero(self, toy):
        got = cs_from_enrichment(*toy, model="m", trait="t")
        assert np.isnan(got.values.loc["c3"])
        assert got.n_unscored == 1

    def test_kind_null_and_provenance(self, toy):
        got = cs_from_enrichment(*toy, model="m", trait="t")
        assert got.kind is ScoreKind.SCADS_RATIO
        assert got.null == 1.0
        assert got.factors == ("k1", "k2")
        assert got.weights.tolist() == [6.0, 6.0]  # a_k * e_k
        assert list(got.values.index) == ["c0", "c1", "c2", "c3"]

    def test_excluded_key_gets_no_weight(self, toy):
        got = cs_from_enrichment(*toy, model="m", trait="t", exclude=["k2"])
        # only k1 scores: c0 -> 3, c2 -> 3; c1 loads only on the excluded key -> NaN
        assert got.values.loc["c0"] == pytest.approx(3.0)
        assert got.values.loc["c2"] == pytest.approx(3.0)
        assert np.isnan(got.values.loc["c1"])
        assert got.factors == ("k1",)

    def test_all_enrichment_one_gives_the_null_everywhere_it_is_defined(self, toy):
        loadings, _, annot_size = toy
        got = cs_from_enrichment(
            loadings, pd.Series({"k1": 1.0, "k2": 1.0}), annot_size, model="m", trait="t"
        )
        assert got.values.dropna().tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_scale_invariance_in_loadings_and_annotation_size(self, toy):
        loadings, enrichment, annot_size = toy
        ref = cs_from_enrichment(loadings, enrichment, annot_size, model="m", trait="t")
        scaled = cs_from_enrichment(
            loadings * 7.5, enrichment, annot_size * 1e4, model="m", trait="t"
        )
        assert np.allclose(scaled.values.to_numpy(), ref.values.to_numpy(), equal_nan=True)

    @pytest.mark.parametrize("chunk", [1, 3, 10_000])
    def test_chunked_equals_unchunked(self, toy, chunk):
        ref = cs_from_enrichment(*toy, model="m", trait="t", chunk_rows=10_000)
        got = cs_from_enrichment(*toy, model="m", trait="t", chunk_rows=chunk)
        assert np.allclose(got.values.to_numpy(), ref.values.to_numpy(), equal_nan=True)

    def test_non_finite_enrichment_is_refused_unless_excluded(self, toy):
        loadings, enrichment, annot_size = toy
        enrichment = enrichment.copy()
        enrichment["k2"] = np.nan
        with pytest.raises(ValueError, match="non-finite enrichment"):
            cs_from_enrichment(loadings, enrichment, annot_size, model="m", trait="t")
        got = cs_from_enrichment(
            loadings, enrichment, annot_size, model="m", trait="t", exclude=["k2"]
        )
        assert got.values.loc["c0"] == pytest.approx(3.0)

    def test_non_positive_annotation_size_is_refused(self, toy):
        loadings, enrichment, annot_size = toy
        annot_size = annot_size.copy()
        annot_size["k1"] = 0.0
        with pytest.raises(ValueError, match="non-positive annot_size"):
            cs_from_enrichment(loadings, enrichment, annot_size, model="m", trait="t")

    def test_missing_key_is_named(self, toy):
        loadings, enrichment, annot_size = toy
        with pytest.raises(KeyError, match="enrichment has no value"):
            cs_from_enrichment(loadings, enrichment.drop("k2"), annot_size, model="m", trait="t")

    def test_everything_excluded_is_an_error(self, toy):
        with pytest.raises(ValueError, match="no key is left"):
            cs_from_enrichment(*toy, model="m", trait="t", exclude=["k1", "k2"])

    def test_non_dataframe_loadings_are_refused(self, toy):
        _, enrichment, annot_size = toy
        with pytest.raises(TypeError, match="cells x keys"):
            cs_from_enrichment(np.zeros((2, 2)), enrichment, annot_size, model="m", trait="t")

    def test_does_not_share_a_formula_with_the_z_weighted_score(self, toy, results):
        loadings, enrichment, annot_size = toy
        ratio = cs_from_enrichment(loadings, enrichment, annot_size, model="m", trait="t")
        assert ratio.kind is not ScoreKind.Z_WEIGHTED and ratio.null != 0.0
