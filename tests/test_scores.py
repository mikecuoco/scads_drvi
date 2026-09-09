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
            "dim": ["dim_0", "dim_1", "dim_2", "dim_3"],
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
        w = factor_weights(results, dims=["dim_0", "dim_1"])
        assert list(w) == [3.0, 0.0]

    def test_clipping_can_be_disabled(self, results):
        w = factor_weights(results, dims=["dim_1"], clip_negative=False)
        assert w[0] == -2.0

    def test_order_follows_the_requested_dims(self, results):
        w = factor_weights(results, dims=["dim_3", "dim_0"])
        assert list(w) == [1.5, 3.0]

    def test_missing_factor_is_named(self, results):
        with pytest.raises(KeyError, match="no result row"):
            factor_weights(results, dims=["dim_9"])

    def test_duplicate_dims_are_refused(self, results):
        doubled = pd.concat([results, results])
        with pytest.raises(ValueError, match="more than one trait"):
            factor_weights(doubled, dims=["dim_0"])

    def test_missing_column_is_named(self, results):
        with pytest.raises(KeyError, match="nope"):
            factor_weights(results, dims=["dim_0"], column="nope")


class TestCsFromZ:
    def test_matches_the_notebook_expression(self, loadings, results):
        """Pin the collapse: this must equal L[kept] @ max(0, z) exactly."""
        dims = results["dim"].tolist()
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
        assert got.dims == ("dim_0", "dim_1", "dim_2", "dim_3")
        assert got.model == "m" and got.trait == "t"

    def test_factors_absent_from_the_loadings_are_counted(self, loadings, results):
        extra = pd.concat(
            [results, pd.DataFrame({"dim": ["dim_9"], "Coefficient_z-score": [5.0]})]
        )
        got = cs_from_z(loadings, extra, model="m", trait="t")
        assert got.n_unscored == 1
        assert "dim_9" not in got.dims

    def test_no_overlap_at_all_is_an_error(self, loadings):
        results = pd.DataFrame({"dim": ["other"], "Coefficient_z-score": [1.0]})
        with pytest.raises(ValueError, match="no factor in the results table"):
            cs_from_z(loadings, results, model="m", trait="t")

    def test_non_dataframe_loadings_are_refused(self, results):
        with pytest.raises(TypeError, match="cells x factors"):
            cs_from_z(np.zeros((3, 4)), results, model="m", trait="t")

    def test_labels_veto_a_display_name(self, loadings, tmp_path):
        """If a caller hands in display labels, the score must not be computed on
        whatever columns happen to match."""
        from scads_drvi.labels import load_labels

        fm = tmp_path / "fm.tsv"
        pd.DataFrame(
            {
                "dim": ["dim_0", "dim_1"],
                "vanished": [False, False],
                "kept": [True, True],
                "drop_reason": ["", ""],
                "annot_index": [1, 2],
            }
        ).to_csv(fm, sep="\t", index=False)
        hm = tmp_path / "hm.tsv"
        pd.DataFrame(
            {
                "annot_dim": ["dim_0", "dim_1"],
                "source_dim": ["dim_47", "dim_47"],
                "half": ["pos", "neg"],
            }
        ).to_csv(hm, sep="\t", index=False)
        labels = load_labels(fm, model="m", half_map=hm)

        renamed = loadings.rename(columns={"dim_0": "dim_47/pos"})
        results = pd.DataFrame(
            {"dim": ["dim_47/pos"], "Coefficient_z-score": [2.0]}
        )
        with pytest.raises(KeyError, match="display label"):
            cs_from_z(renamed, results, model="m", trait="t", labels=labels)


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
