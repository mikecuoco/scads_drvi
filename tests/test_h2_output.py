"""enrich.h2_output: parsing what the Rust LDSC actually prints.

The fixture is a trimmed copy of real stdout from ldsc 0.5.0 h2 on arm
k96_ind_exp_split / factor k1 / trait bellenguez, including a category name long enough
to run into its first numeric column -- which is the case a naive whitespace split gets
wrong.
"""

from __future__ import annotations

import math

import pytest

from scads_drvi.enrich.h2_output import h2_table, parse_h2_stdout

REAL_STDOUT = """\
  Detected K=98 annotation columns (partitioned h2)
  Read per-annotation M (K=98) from .l2.M_5_50 files; total M = 93711522355825
Total observed-scale h2: 0.0260
Intercept: 1.0598

Category                               Prop_h2  Prop_SNPs   Enrichment    Coefficient
-----------------------------------------------------------------------------------
k1L2_0                                 -0.0855     1.0000      -0.0855    -2.3707e-17
baseL2_1                               -1.3261     0.0000 -20865134.1746     -5.7857e-9
Coding_UCSCL2_1                        -0.0696     0.0000 -76830783.2532     -2.1305e-8
MAF_Adj_Predicted_Allele_AgeL2_1        0.0000    -0.0000          NaN    -3.8743e-10
Human_Promoter_Villar_ExAC.flanking.500L2_1     0.0304     0.0000 828703911.7853      2.2979e-7
"""

# What the Python run recorded for the same factor.
PYTHON_COEFFICIENT = -2.3707e-17
PYTHON_TOTAL_H2 = 0.026
PYTHON_N_CATEGORIES = 98


class TestParseRealOutput:
    def test_summary_scalars(self):
        got = parse_h2_stdout(REAL_STDOUT)
        assert got.total_h2 == pytest.approx(PYTHON_TOTAL_H2, abs=5e-4)
        assert got.intercept == pytest.approx(1.0598)
        assert got.n_categories == PYTHON_N_CATEGORIES

    def test_focal_row_is_the_annotation_under_test(self):
        """Row 0 is ours because the run puts it first in --ref-ld-chr."""
        got = parse_h2_stdout(REAL_STDOUT)
        assert got.categories[0] == "k1L2_0"
        assert got.focal["coefficient"] == pytest.approx(PYTHON_COEFFICIENT, rel=1e-12)

    def test_coefficient_matches_the_python_run_exactly(self):
        got = parse_h2_stdout(REAL_STDOUT)
        assert got.focal["coefficient"] == PYTHON_COEFFICIENT

    def test_a_long_category_name_is_not_merged_into_its_numbers(self):
        """The table looks fixed-width but is not; a whitespace split would fuse the
        name and the first value for any name that overruns the column."""
        got = parse_h2_stdout(REAL_STDOUT)
        assert "Human_Promoter_Villar_ExAC.flanking.500L2_1" in got.categories
        row = got.category("Human_Promoter_Villar_ExAC.flanking.500L2_1")
        assert row["prop_h2"] == pytest.approx(0.0304)
        assert row["coefficient"] == pytest.approx(2.2979e-7)

    def test_nan_enrichment_is_preserved_not_dropped(self):
        got = parse_h2_stdout(REAL_STDOUT)
        row = got.category("MAF_Adj_Predicted_Allele_AgeL2_1")
        assert math.isnan(row["enrichment"])

    def test_every_row_is_parsed(self):
        got = parse_h2_stdout(REAL_STDOUT)
        assert len(got.categories) == 5
        assert len(got.coefficient) == 5

    def test_negative_zero_prop_snps(self):
        got = parse_h2_stdout(REAL_STDOUT)
        assert got.category("MAF_Adj_Predicted_Allele_AgeL2_1")["prop_snps"] == 0.0


class TestStandardErrorsAreAbsent:
    def test_it_says_so_rather_than_guessing(self):
        """A z assembled from an invented SE is indistinguishable downstream from a
        real one, so the parser reports absence instead."""
        got = parse_h2_stdout(REAL_STDOUT)
        assert got.carries_standard_errors is False
        assert got.coefficient_se is None
        assert got.coefficient_z is None


class TestDegenerateInput:
    def test_empty_text(self):
        got = parse_h2_stdout("")
        assert got.categories == ()
        assert got.total_h2 is None
        with pytest.raises(ValueError, match="no categories"):
            _ = got.focal

    def test_summary_without_a_table(self):
        got = parse_h2_stdout("Total observed-scale h2: 0.05\nIntercept: 1.0\n")
        assert got.total_h2 == pytest.approx(0.05)
        assert got.categories == ()

    def test_unknown_category_is_named(self):
        got = parse_h2_stdout(REAL_STDOUT)
        with pytest.raises(KeyError, match="no category"):
            got.category("absent")

    def test_trailing_noise_after_the_table_stops_parsing(self):
        text = REAL_STDOUT + "\nsome trailing prose that is not a row\n"
        got = parse_h2_stdout(text)
        assert len(got.categories) == 5


class TestTable:
    def test_frame_has_the_python_column_names(self):
        frame = h2_table(parse_h2_stdout(REAL_STDOUT))
        assert list(frame.columns) == [
            "Category", "Prop_h2", "Prop_SNPs", "Enrichment", "Coefficient"
        ]
        assert frame.iloc[0]["Category"] == "k1L2_0"
        assert len(frame) == 5
