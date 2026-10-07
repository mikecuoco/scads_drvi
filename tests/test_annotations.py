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
        with pytest.raises(ValueError, match="peaks must not overlap"):
            assign_peaks(bim, peaks)

    def test_peak_edges_follow_one_based_positions_and_zero_based_peaks(self):
        # "chr1:100-200" is 0-based half-open, so it covers 1-based positions 101..200.
        bim = make_snp_bim("1", [99, 100, 101, 150, 200, 201])
        got = assign_peaks(bim, ["chr1:100-200"])
        assert got.notna().tolist() == [False, False, True, True, True, False]

    def test_adjacent_peaks_split_at_the_shared_edge(self):
        bim = make_snp_bim("1", [200, 201])
        got = assign_peaks(bim, ["chr1:100-200", "chr1:200-300"])
        assert got.tolist() == ["chr1:100-200", "chr1:200-300"]

    def test_nested_peaks_raise_even_when_a_third_peak_sits_between(self):
        # A contains B and C. Checking only the nearest neighbour misses A-vs-C.
        bim = make_snp_bim("1", [450, 700])
        peaks = ["chr1:100-1000", "chr1:200-300", "chr1:400-500"]
        with pytest.raises(ValueError, match="peaks must not overlap: chr1:100-1000 and"):
            assign_peaks(bim, peaks)

    def test_overlap_is_checked_for_the_whole_peak_set_not_only_where_snps_fall(self):
        bim = make_snp_bim("1", [150])
        with pytest.raises(ValueError, match="peaks must not overlap"):
            assign_peaks(bim, ["chr1:100-200", "chr5:100-300", "chr5:200-400"])

    def test_peaks_on_chromosomes_absent_from_bim_are_ignored(self):
        bim = make_snp_bim("1", [150])
        got = assign_peaks(bim, ["chr1:100-200", "chrUn_x:1-50", "chrUn_x:60-90"])
        assert got.iloc[0] == "chr1:100-200"

    def test_zero_length_peak_inside_another_is_not_an_overlap(self):
        bim = make_snp_bim("1", [150])
        got = assign_peaks(bim, ["chr1:100-200", "chr1:150-150"])
        assert got.iloc[0] == "chr1:100-200"

    def test_zero_length_peak_contains_no_snp(self):
        bim = make_snp_bim("1", [100, 150])
        got = assign_peaks(bim, ["chr1:100-100", "chr1:120-200"])
        assert pd.isna(got.iloc[0])
        assert got.iloc[1] == "chr1:120-200"

    def test_result_is_a_named_object_series_aligned_to_bim(self):
        bim = make_snp_bim("1", [150, 999999]).set_index(pd.Index([7, 3]))
        got = assign_peaks(bim, ["chr1:100-200"])
        assert got.name == "peak"
        assert got.dtype == object
        assert got.index.tolist() == [7, 3]

    def test_no_peak_on_the_bim_chromosomes_gives_all_na(self):
        bim = make_snp_bim("2", [150, 250])
        assert assign_peaks(bim, ["chr1:100-200"]).isna().all()

    def test_coords_replace_the_coordinates_in_the_names(self):
        # Names say 100-200 (the source build); coords say 1000-1100 (the panel's build).
        bim = make_snp_bim("1", [150, 1050])
        coords = pd.DataFrame(
            {"chrom": ["chr1"], "start": [1000], "end": [1100]}, index=["chr1:100-200"]
        )
        got = assign_peaks(bim, ["chr1:100-200"], coords)
        assert pd.isna(got.iloc[0])
        assert got.iloc[1] == "chr1:100-200"

    def test_peak_without_coords_is_never_assigned(self):
        bim = make_snp_bim("1", [150, 350])
        coords = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": pd.array([100, pd.NA], dtype="Int64"),
                "end": pd.array([200, pd.NA], dtype="Int64"),
            },
            index=["chr1:100-200", "chr1:300-400"],
        )
        got = assign_peaks(bim, ["chr1:100-200", "chr1:300-400"], coords)
        assert got.iloc[0] == "chr1:100-200"
        assert pd.isna(got.iloc[1])

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
