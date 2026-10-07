"""enrich.risk: pushing a fitted S-LDSC tau down to individual peaks."""

from __future__ import annotations

import pandas as pd
import pytest

from scads_drvi.enrich.risk import peak_risk, peak_snp_counts, top_risk_peaks


@pytest.fixture
def bim():
    return pd.DataFrame(
        {
            "CHR": ["1", "1", "1", "1", "1"],
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "CM": [0.0] * 5,
            "BP": [110, 120, 310, 320, 330],
            "A1": ["A"] * 5,
            "A2": ["G"] * 5,
        }
    )


@pytest.fixture
def feature_loadings():
    # peak "1:100-200" has 2 overlapping SNPs (rs1, rs2), "1:300-400" has 3 (rs3-5),
    # "1:1000-1100" has none.
    return {
        "pos": pd.DataFrame(
            {"dim_0": [0.5, 0.0], "dim_1": [0.2, 0.1]},
            index=["1:100-200", "1:300-400"],
        ),
        "neg": pd.DataFrame(
            {"dim_0": [0.0, 0.9], "dim_1": [0.3, 0.0]},
            index=["1:100-200", "1:300-400"],
        ),
    }


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "dim": ["dim_0", "dim_0", "dim_1", "dim_1"],
            "direction": ["pos", "neg", "pos", "neg"],
            "Coefficient": [2.0, 1.0, -0.5, 3.0],
        }
    )


class TestPeakSnpCounts:
    def test_counts_overlapping_snps_per_peak(self, bim):
        counts = peak_snp_counts(["1:100-200", "1:300-400", "1:1000-1100"], bim)
        assert counts.loc["1:100-200"] == 2
        assert counts.loc["1:300-400"] == 3
        assert counts.loc["1:1000-1100"] == 0

    def test_every_requested_peak_gets_a_row(self, bim):
        counts = peak_snp_counts(["1:1000-1100"], bim)
        assert list(counts.index) == ["1:1000-1100"]
        assert counts.iloc[0] == 0

    def test_edge_snps_follow_one_based_positions(self):
        # "1:100-200" is 0-based half-open: it covers 1-based positions 101..200, so the SNPs
        # at 101 and 200 count and the ones at 100 and 201 do not.
        edge_bim = pd.DataFrame(
            {
                "CHR": ["1"] * 4,
                "SNP": ["a", "b", "c", "d"],
                "CM": [0.0] * 4,
                "BP": [100, 101, 200, 201],
                "A1": ["A"] * 4,
                "A2": ["G"] * 4,
            }
        )
        assert peak_snp_counts(["1:100-200"], edge_bim).loc["1:100-200"] == 2


class TestPeakRisk:
    def test_risk_rate_is_loading_times_tau(self, feature_loadings, results):
        risk = peak_risk(feature_loadings, results)
        row = risk[
            (risk["peak"] == "1:100-200") & (risk["dim"] == "dim_0") & (risk["direction"] == "pos")
        ]
        assert row["loading"].iloc[0] == pytest.approx(0.5)
        assert row["tau"].iloc[0] == pytest.approx(2.0)
        assert row["risk_rate"].iloc[0] == pytest.approx(1.0)

    def test_risk_total_uses_snp_counts(self, feature_loadings, results, bim):
        counts = peak_snp_counts(["1:100-200", "1:300-400"], bim)
        risk = peak_risk(feature_loadings, results, snp_counts=counts)
        row = risk[
            (risk["peak"] == "1:300-400") & (risk["dim"] == "dim_0") & (risk["direction"] == "neg")
        ]
        # loading=0.9, tau=1.0 -> risk_rate=0.9; 3 overlapping SNPs -> risk_total=2.7
        assert row["risk_rate"].iloc[0] == pytest.approx(0.9)
        assert row["risk_total"].iloc[0] == pytest.approx(2.7)

    def test_no_snp_counts_means_no_risk_total_column(self, feature_loadings, results):
        risk = peak_risk(feature_loadings, results)
        assert "risk_total" not in risk.columns

    def test_zero_loadings_excluded_by_default(self, feature_loadings, results):
        risk = peak_risk(feature_loadings, results)
        # dim_0/pos has loading 0.0 at "1:300-400" -- must not appear
        hit = risk[
            (risk["peak"] == "1:300-400") & (risk["dim"] == "dim_0") & (risk["direction"] == "pos")
        ]
        assert hit.empty

    def test_include_zero_keeps_zero_loadings(self, feature_loadings, results):
        risk = peak_risk(feature_loadings, results, include_zero=True)
        hit = risk[
            (risk["peak"] == "1:300-400") & (risk["dim"] == "dim_0") & (risk["direction"] == "pos")
        ]
        assert len(hit) == 1
        assert hit["loading"].iloc[0] == 0.0
        assert hit["risk_rate"].iloc[0] == 0.0

    def test_missing_fitted_result_raises(self, feature_loadings, results):
        incomplete = results[~((results["dim"] == "dim_1") & (results["direction"] == "neg"))]
        with pytest.raises(KeyError, match="dim_1"):
            peak_risk(feature_loadings, incomplete)

    def test_all_zero_is_refused(self, results):
        all_zero = {
            "pos": pd.DataFrame({"dim_0": [0.0, 0.0], "dim_1": [0.0, 0.0]}, index=["1:100-200", "1:300-400"]),
            "neg": pd.DataFrame({"dim_0": [0.0, 0.0], "dim_1": [0.0, 0.0]}, index=["1:100-200", "1:300-400"]),
        }
        with pytest.raises(ValueError, match="no nonzero loadings"):
            peak_risk(all_zero, results)

    def test_missing_results_columns_is_named(self, feature_loadings):
        bare = pd.DataFrame({"dim": ["dim_0"], "direction": ["pos"]})
        with pytest.raises(KeyError, match="Coefficient"):
            peak_risk(feature_loadings, bare)

    def test_coordinates_are_parsed_from_peak_name(self, feature_loadings, results):
        risk = peak_risk(feature_loadings, results)
        row = risk[risk["peak"] == "1:100-200"].iloc[0]
        assert row["chrom"] == "1"
        assert row["start"] == 100
        assert row["end"] == 200


class TestTopRiskPeaks:
    def test_picks_the_true_top_n(self, feature_loadings, results, bim):
        counts = peak_snp_counts(["1:100-200", "1:300-400"], bim)
        risk = peak_risk(feature_loadings, results, snp_counts=counts)
        top1 = top_risk_peaks(risk, n=1)
        assert len(top1) == 1
        assert top1["risk_total"].iloc[0] == risk["risk_total"].max()

    def test_missing_by_column_is_named(self, feature_loadings, results):
        risk = peak_risk(feature_loadings, results)  # no snp_counts -> no risk_total
        with pytest.raises(KeyError, match="risk_total"):
            top_risk_peaks(risk)
