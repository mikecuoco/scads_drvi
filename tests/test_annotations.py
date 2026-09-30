"""Reading a bim's row order, and aligning its SNPs to peaks."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from scads_drvi.enrich.annotations import (  # noqa: E402
    assign_peaks,
    read_bim,
    restrict_to_reference_snps,
)


def make_bim(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": np.full(n, 22),
            "SNP": [f"rs{1000 + i}" for i in range(n)],
            "CM": np.linspace(0.0, 1.5, n),
            "BP": np.arange(10_000, 10_000 + n),
            "A1": ["A"] * n,
            "A2": ["G"] * n,
        }
    )


class TestReadBim:
    def test_preserves_file_order(self, tmp_path):
        bim = make_bim(20)
        # Shuffle so a reader that sorted would be caught.
        shuffled = bim.sample(frac=1.0, random_state=1).reset_index(drop=True)
        path = tmp_path / "x.bim"
        shuffled.to_csv(path, sep="\t", header=False, index=False)

        got = read_bim(path)
        assert list(got["SNP"]) == list(shuffled["SNP"])

    def test_empty_bim_is_an_error(self, tmp_path):
        # pandas raises EmptyDataError before read_bim's own check can fire; either way
        # the caller must not receive an empty frame and carry on.
        path = tmp_path / "empty.bim"
        path.write_text("")
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            read_bim(path)


def make_snp_bim(chrom, positions):
    return pd.DataFrame(
        {
            "CHR": [chrom] * len(positions),
            "SNP": [f"rs{i}" for i in range(len(positions))],
            "CM": np.zeros(len(positions)),
            "BP": positions,
            "A1": ["A"] * len(positions),
            "A2": ["G"] * len(positions),
        }
    )


class TestAssignPeaks:
    def test_assigns_the_containing_peak(self):
        bim = make_snp_bim("1", [150, 350, 999999])
        peaks = ["chr1:100-200", "chr1:300-400"]
        got = assign_peaks(bim, peaks)
        assert got.iloc[0] == "chr1:100-200"
        assert got.iloc[1] == "chr1:300-400"

    def test_no_overlap_is_na_not_dropped(self):
        bim = make_snp_bim("1", [150, 250])
        peaks = ["chr1:100-200"]
        got = assign_peaks(bim, peaks)
        assert len(got) == 2
        assert pd.isna(got.iloc[1])

    def test_chr_prefix_is_normalized_both_ways(self):
        # bim's bare "1" against a "chr1:..." peak name -- the real-world mismatch
        # every caller of this comparison otherwise has to glue by hand.
        bim = make_snp_bim("1", [150])
        assert assign_peaks(bim, ["chr1:100-200"]).iloc[0] == "chr1:100-200"
        # and the reverse: a "chr1" bim against a bare "1:..." peak name.
        bim_prefixed = make_snp_bim("chr1", [150])
        assert assign_peaks(bim_prefixed, ["1:100-200"]).iloc[0] == "1:100-200"

    def test_wrong_chromosome_is_na(self):
        bim = make_snp_bim("2", [150])
        peaks = ["chr1:100-200"]
        assert pd.isna(assign_peaks(bim, peaks).iloc[0])

    def test_overlapping_peaks_raise(self):
        bim = make_snp_bim("1", [250])
        peaks = ["chr1:100-300", "chr1:200-400"]  # both contain 250
        with pytest.raises(ValueError, match="overlaps more than one peak"):
            assign_peaks(bim, peaks)

    def test_missing_column_is_named(self):
        bim = make_snp_bim("1", [150]).drop(columns="BP")
        with pytest.raises(KeyError, match="BP"):
            assign_peaks(bim, ["chr1:100-200"])

    def test_no_peaks_is_refused(self):
        bim = make_snp_bim("1", [150])
        with pytest.raises(ValueError, match="no peaks"):
            assign_peaks(bim, [])

    def test_genome_wide_bim_matches_only_same_chromosome_peaks(self):
        bim = pd.concat(
            [make_snp_bim("1", [150]), make_snp_bim("2", [150])], ignore_index=True
        )
        peaks = ["chr1:100-200", "chr2:100-200"]
        got = assign_peaks(bim, peaks)
        assert got.iloc[0] == "chr1:100-200"
        assert got.iloc[1] == "chr2:100-200"


def make_ldscore(snps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": [1] * len(snps),
            "SNP": snps,
            "BP": range(len(snps)),
            "fooL2": np.arange(len(snps), dtype=float),
        }
    )


class TestRestrictToReferenceSnps:
    def test_keeps_only_reference_snps(self):
        ldscore = make_ldscore(["rs1", "rs2", "rs3"])
        got = restrict_to_reference_snps(ldscore, ["rs1", "rs3"])
        assert list(got["SNP"]) == ["rs1", "rs3"]

    def test_reorders_to_reference_order_not_original_order(self):
        ldscore = make_ldscore(["rs1", "rs2", "rs3"])
        got = restrict_to_reference_snps(ldscore, ["rs3", "rs1"])
        assert list(got["SNP"]) == ["rs3", "rs1"]

    def test_preserves_other_columns_and_values(self):
        ldscore = make_ldscore(["rs1", "rs2", "rs3"])
        got = restrict_to_reference_snps(ldscore, ["rs2"])
        assert list(got.columns) == list(ldscore.columns)
        assert got.loc[0, "fooL2"] == 1.0

    def test_missing_reference_snp_raises(self):
        ldscore = make_ldscore(["rs1", "rs2"])
        with pytest.raises(ValueError, match="rs99"):
            restrict_to_reference_snps(ldscore, ["rs1", "rs99"])

    def test_missing_snp_column_is_named(self):
        ldscore = make_ldscore(["rs1"]).rename(columns={"SNP": "id"})
        with pytest.raises(KeyError, match="SNP"):
            restrict_to_reference_snps(ldscore, ["rs1"])

    def test_custom_snp_col(self):
        ldscore = make_ldscore(["rs1", "rs2"]).rename(columns={"SNP": "id"})
        got = restrict_to_reference_snps(ldscore, ["rs2"], snp_col="id")
        assert list(got["id"]) == ["rs2"]
