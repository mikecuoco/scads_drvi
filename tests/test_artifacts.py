"""io.artifacts: the h5ad decode, the loadings reader, and load_interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from scads_drvi.io.artifacts import (  # noqa: E402
    Interpretation,
    cell_metadata,
    load_interpretation,
    obs_columns,
    read_loadings,
    read_obs,
    read_umap,
)


def write_h5ad_obs(path, *, n=6, index_name="cell_id"):
    """A minimal h5ad carrying only obs, encoded the way anndata encodes it."""
    ids = np.array([f"lib_{i % 2}:bc_{i}".encode() for i in range(n)])
    groups = np.array([b"g0", b"g1", b"g2"])
    codes = np.array([i % 3 for i in range(n)], dtype=np.int32)
    with h5py.File(path, "w") as fh:
        obs = fh.create_group("obs")
        obs.attrs["_index"] = index_name
        obs.attrs["encoding-type"] = "dataframe"
        obs.create_dataset(index_name, data=ids)
        cat = obs.create_group("grouping")
        cat.create_dataset("categories", data=groups)
        cat.create_dataset("codes", data=codes)
        obs.create_dataset("numerator", data=np.arange(n, dtype=np.int64))
        obs.create_dataset(
            "denominator", data=np.array([0] + [10] * (n - 1), dtype=np.int64)
        )
    return path


@pytest.fixture
def obs_file(tmp_path):
    return write_h5ad_obs(tmp_path / "matrix.h5ad")


class TestReadObs:
    def test_lists_columns_without_reading(self, obs_file):
        assert obs_columns(obs_file) == [
            "cell_id", "denominator", "grouping", "numerator"
        ]

    def test_categoricals_are_decoded(self, obs_file):
        frame = read_obs(obs_file)
        assert str(frame["grouping"].dtype) == "category"
        assert set(frame["grouping"].astype(str)) == {"g0", "g1", "g2"}

    def test_index_comes_from_the_file(self, obs_file):
        """anndata records the index name; guessing it is how a join matches nothing."""
        frame = read_obs(obs_file)
        assert frame.index.name == "cell_id"
        assert frame.index[0] == "lib_0:bc_0"

    def test_bytes_are_decoded_to_str(self, obs_file):
        frame = read_obs(obs_file)
        assert all(isinstance(i, str) for i in frame.index)

    def test_column_subset_still_gets_the_index(self, obs_file):
        frame = read_obs(obs_file, ["grouping"])
        assert list(frame.columns) == ["grouping"]
        assert frame.index[0] == "lib_0:bc_0"

    def test_unknown_column_lists_what_exists(self, obs_file):
        with pytest.raises(KeyError, match="Available"):
            read_obs(obs_file, ["nope"])

    def test_a_file_without_obs_is_refused(self, tmp_path):
        path = tmp_path / "plain.h5"
        with h5py.File(path, "w") as fh:
            fh.create_dataset("x", data=[1])
        with pytest.raises(ValueError, match="no obs group"):
            read_obs(path)


class TestCellMetadata:
    def test_derived_ratio_is_nan_not_zero_when_undefined(self, obs_file):
        """A cell with no observations has an undefined fraction. Recording it as 0
        puts it at the bottom of every distribution as though it were measured."""
        frame = cell_metadata(
            obs_file, derived={"fraction": ("numerator", "denominator")}
        )
        assert np.isnan(frame["fraction"].iloc[0])
        assert frame["fraction"].iloc[1] == pytest.approx(0.1)

    def test_derived_inputs_are_fetched_even_if_not_requested(self, obs_file):
        frame = cell_metadata(
            obs_file,
            columns=["grouping"],
            derived={"fraction": ("numerator", "denominator")},
        )
        assert list(frame.columns) == ["grouping", "fraction"]

    def test_missing_derived_input_is_named(self, obs_file):
        with pytest.raises(KeyError, match="cannot derive"):
            cell_metadata(obs_file, derived={"x": ("numerator", "absent")})

    def test_no_derived_columns_by_default(self, obs_file):
        frame = cell_metadata(obs_file)
        assert "fraction" not in frame.columns


class TestReadLoadings:
    def _npz(self, tmp_path, n=10, k=3):
        path = tmp_path / "loadings.npz"
        rng = np.random.default_rng(0)
        np.savez(
            path,
            cells=np.array([f"c{i}" for i in range(n)], dtype=object),
            factors=np.array([f"dim_{j}" for j in range(k)]),
            loadings=rng.random((n, k)).astype(np.float32),
        )
        return path

    def _tsv(self, tmp_path, n=10, k=3):
        path = tmp_path / "loadings.tsv"
        rng = np.random.default_rng(0)
        pd.DataFrame(
            rng.random((n, k)).astype(np.float32),
            index=[f"c{i}" for i in range(n)],
            columns=[f"dim_{j}" for j in range(k)],
        ).to_csv(path, sep="\t")
        return path

    def test_reads_an_npz(self, tmp_path):
        frame = read_loadings(self._npz(tmp_path))
        assert frame.shape == (10, 3)
        assert list(frame.columns) == ["dim_0", "dim_1", "dim_2"]
        assert frame.index[0] == "c0"

    def test_reads_a_tsv(self, tmp_path):
        frame = read_loadings(self._tsv(tmp_path))
        assert frame.shape == (10, 3)

    @pytest.mark.parametrize("chunk", [1, 3, 1000])
    def test_tsv_chunking_does_not_change_the_result(self, tmp_path, chunk):
        path = self._tsv(tmp_path)
        ref = read_loadings(path, chunk_rows=1000)
        got = read_loadings(path, chunk_rows=chunk)
        pd.testing.assert_frame_equal(ref, got)

    def test_npz_is_preferred_when_both_exist(self, tmp_path):
        npz = self._npz(tmp_path, n=4)
        tsv = self._tsv(tmp_path, n=99)
        frame = read_loadings(tsv, npz=npz)
        assert len(frame) == 4  # the npz, not the 99-row tsv

    def test_tsv_is_used_when_the_npz_is_absent(self, tmp_path):
        tsv = self._tsv(tmp_path, n=7)
        frame = read_loadings(tsv, npz=tmp_path / "absent.npz")
        assert len(frame) == 7

    def test_dims_subset_and_order(self, tmp_path):
        frame = read_loadings(self._npz(tmp_path), dims=["dim_2", "dim_0"])
        assert list(frame.columns) == ["dim_2", "dim_0"]

    def test_missing_dims_explain_the_split_contract(self, tmp_path):
        with pytest.raises(KeyError, match="split contract"):
            read_loadings(self._npz(tmp_path), dims=["dim_99"])

    def test_a_malformed_npz_names_the_missing_array(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, loadings=np.zeros((2, 2)))
        with pytest.raises(ValueError, match="missing array"):
            read_loadings(path)

    def test_nothing_to_read_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="none of these exist"):
            read_loadings(tmp_path / "a.npz", tsv=tmp_path / "b.tsv")
        with pytest.raises(ValueError, match="give a path"):
            read_loadings()


class TestReadUmap:
    def test_reads_two_coordinate_columns(self, tmp_path):
        path = tmp_path / "umap.tsv"
        pd.DataFrame(
            {"cell": ["a", "b"], "x": [0.0, 1.0], "y": [2.0, 3.0], "extra": [9, 9]}
        ).set_index("cell").to_csv(path, sep="\t")
        frame = read_umap(path)
        assert frame.shape == (2, 2)
        assert list(frame.columns) == ["x", "y"]

    def test_axes_can_be_renamed(self, tmp_path):
        path = tmp_path / "umap.tsv"
        pd.DataFrame({"cell": ["a"], "x": [0.0], "y": [1.0]}).set_index(
            "cell"
        ).to_csv(path, sep="\t")
        frame = read_umap(path, columns=["UMAP_1", "UMAP_2"])
        assert list(frame.columns) == ["UMAP_1", "UMAP_2"]

    def test_one_coordinate_is_refused(self, tmp_path):
        path = tmp_path / "umap.tsv"
        pd.DataFrame({"cell": ["a"], "x": [0.0]}).set_index("cell").to_csv(
            path, sep="\t"
        )
        with pytest.raises(ValueError, match="need 2"):
            read_umap(path)

    def test_missing_file_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no embedding"):
            read_umap(tmp_path / "absent.tsv")


class TestLoadInterpretation:
    @pytest.fixture
    def project(self, tmp_path):
        from scads_drvi.config import Project

        proj = Project(root=tmp_path, fit="a_fit", traits=("t1",))
        arm = proj.enrich_dir("arm")
        (arm / "results" / "t1").mkdir(parents=True)
        pd.DataFrame(
            {
                "dim": ["dim_0", "dim_1"],
                "vanished": [False, False],
                "kept": [True, True],
                "drop_reason": ["", ""],
                "annot_index": [1, 2],
            }
        ).to_csv(arm / "factor_map.tsv", sep="\t", index=False)
        for annot, z in (("k1", 4.0), ("k2", 1.0)):
            pd.DataFrame(
                {"Category": [f"{annot}L2_0"], "Coefficient_z-score": [z]}
            ).to_csv(arm / "results" / "t1" / f"{annot}.results", sep="\t", index=False)

        fit = proj.fit_dir()
        fit.mkdir(parents=True)
        np.savez(
            fit / "topic_loadings.npz",
            cells=np.array(["lib_0:bc_0", "lib_1:bc_1"], dtype=object),
            factors=np.array(["dim_0", "dim_1"]),
            loadings=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        return proj

    def test_loads_labels_and_results(self, project):
        interp = load_interpretation(project, "arm")
        assert isinstance(interp, Interpretation)
        assert interp.labels.n_kept == 2
        assert len(interp.results) == 2
        assert interp.traits == ("t1",)

    def test_loadings_are_opt_in(self, project):
        assert load_interpretation(project, "arm").loadings is None
        with_matrix = load_interpretation(project, "arm", loadings=True)
        assert with_matrix.loadings.shape == (2, 2)

    def test_cells_are_empty_without_an_obs_path(self, project):
        assert load_interpretation(project, "arm").cells.empty

    def test_obs_and_umap_are_joined(self, project, tmp_path):
        obs = write_h5ad_obs(tmp_path / "m.h5ad", n=2)
        umap = tmp_path / "umap.tsv"
        pd.DataFrame(
            {"cell": ["lib_0:bc_0", "lib_1:bc_1"], "x": [0.0, 1.0], "y": [2.0, 3.0]}
        ).set_index("cell").to_csv(umap, sep="\t")
        interp = load_interpretation(
            project, "arm", obs_path=obs, umap_path=umap, obs_columns=["grouping"]
        )
        assert interp.n_cells == 2
        assert {"x", "y", "grouping"} <= set(interp.cells.columns)

    def test_for_trait_selects_and_explains(self, project):
        interp = load_interpretation(project, "arm")
        assert len(interp.for_trait("t1")) == 2
        with pytest.raises(KeyError, match="this arm has"):
            interp.for_trait("absent")

    def test_meta_records_provenance(self, project):
        meta = load_interpretation(project, "arm").meta
        assert meta["model"] == "arm"
        assert meta["n_kept"] == 2
        assert meta["is_split"] is False
        assert "factor_map" in meta["paths"]

    def test_no_traits_anywhere_is_refused(self, project):
        bare = project.replace(traits=())
        with pytest.raises(ValueError, match="name the traits"):
            load_interpretation(bare, "arm")
