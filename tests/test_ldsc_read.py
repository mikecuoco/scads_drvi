"""enrich.ldsc: reading .results back in without losing factors silently."""

from __future__ import annotations

import pandas as pd
import pytest

from scads_drvi.enrich.ldsc import Z_COLUMN, read_results, results_files


def write_results(path, z, *, extra_rows=0):
    """One .results file. Row 0 is the factor's own annotation, as LDSC writes it."""
    rows = [{"Category": f"{path.stem}L2_0", Z_COLUMN: z, "Coefficient": 1.0}]
    for i in range(extra_rows):
        rows.append({"Category": f"baseline_{i}", Z_COLUMN: 0.0, "Coefficient": 0.0})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture
def arm(tmp_path):
    """(annot2dim, results_root) for two kept factors -- the mapping
    :func:`scads_drvi.enrich.config.select_factors`/``kept_dims`` would hand this
    function, not a dropped third dimension (which is that module's own concern, not
    read_results')."""
    annot2dim = {"k1": "dim_0", "k2": "dim_1"}
    root = tmp_path / "results"
    for trait, zs in {"t1": (4.0, 1.0), "t2": (0.5, 3.0)}.items():
        (root / trait).mkdir(parents=True)
        for annot, z in zip(("k1", "k2"), zs, strict=True):
            write_results(root / trait / f"{annot}.results", z)
    return annot2dim, root


class TestResultsFiles:
    def test_lists_by_annotation_name(self, arm):
        _, root = arm
        assert sorted(results_files(root / "t1")) == ["k1", "k2"]

    def test_missing_directory_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no results directory"):
            results_files(tmp_path / "absent")


class TestReadResults:
    def test_one_row_per_factor_per_trait(self, arm):
        annot2dim, root = arm
        frame = read_results(root, traits=["t1", "t2"], annot2dim=annot2dim)
        assert len(frame) == 4
        assert set(frame["trait"]) == {"t1", "t2"}
        assert set(frame["dim"]) == {"dim_0", "dim_1"}

    def test_row_zero_is_the_factors_own_annotation(self, arm):
        """Baseline categories follow it; taking the wrong row silently reports the
        heritability of an unrelated annotation."""
        annot2dim, root = arm
        for annot in ("k1", "k2"):
            write_results(root / "t1" / f"{annot}.results", 4.0, extra_rows=3)
        frame = read_results(root, traits=["t1"], annot2dim=annot2dim)
        assert list(frame["Category"]) == ["k1L2_0", "k2L2_0"]

    def test_row_can_be_overridden(self, arm):
        annot2dim, root = arm
        for annot in ("k1", "k2"):
            write_results(root / "t1" / f"{annot}.results", 4.0, extra_rows=2)
        frame = read_results(
            root, traits=["t1"], annot2dim=annot2dim, row=1, fdr=False
        )
        assert set(frame["Category"]) == {"baseline_0"}

    def test_row_out_of_range_is_explained(self, arm):
        annot2dim, root = arm
        with pytest.raises(ValueError, match="rows; row 9 was requested"):
            read_results(root, traits=["t1"], annot2dim=annot2dim, row=9)

    def test_a_missing_result_file_raises_by_default(self, arm):
        annot2dim, root = arm
        (root / "t1" / "k2.results").unlink()
        with pytest.raises(FileNotFoundError, match="optimistic"):
            read_results(root, traits=["t1"], annot2dim=annot2dim)

    def test_strict_false_allows_a_partial_table(self, arm):
        annot2dim, root = arm
        (root / "t1" / "k2.results").unlink()
        frame = read_results(root, traits=["t1"], annot2dim=annot2dim, strict=False)
        assert len(frame) == 1

    def test_fdr_columns_are_added_within_trait(self, arm):
        annot2dim, root = arm
        frame = read_results(root, traits=["t1", "t2"], annot2dim=annot2dim)
        assert {"p_1tailed", "fdr_q"} <= set(frame.columns)
        # z=4.0 in t1 and z=3.0 in t2 are each the best in their own trait
        best = frame.loc[frame.groupby("trait")["fdr_q"].idxmin()]
        assert set(zip(best["trait"], best["dim"], strict=True)) == {
            ("t1", "dim_0"),
            ("t2", "dim_1"),
        }

    def test_fdr_can_be_skipped(self, arm):
        annot2dim, root = arm
        frame = read_results(root, traits=["t1"], annot2dim=annot2dim, fdr=False)
        assert "fdr_q" not in frame.columns

    def test_direction_defaults_to_combined(self, arm):
        annot2dim, root = arm
        frame = read_results(root, traits=["t1"], annot2dim=annot2dim)
        assert set(frame["direction"]) == {"combined"}

    def test_direction_can_be_a_mapping_per_annotation(self, arm):
        """This is how a caller records which sign of the latent dimension a result
        came from -- see :func:`scads_drvi.factorize.result.directional_loadings`, which is
        where that split is actually derived and computed."""
        annot2dim, root = arm
        frame = read_results(
            root, traits=["t1"], annot2dim=annot2dim, direction={"k1": "pos", "k2": "neg"}
        )
        assert dict(zip(frame["annot"], frame["direction"], strict=True)) == {
            "k1": "pos",
            "k2": "neg",
        }

    def test_no_traits_is_refused(self, arm):
        annot2dim, root = arm
        with pytest.raises(ValueError, match="no traits"):
            read_results(root, traits=[], annot2dim=annot2dim)
