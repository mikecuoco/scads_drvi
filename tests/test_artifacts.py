"""io.artifacts: the h5ad decode and the loadings reader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from scads_drvi.io.artifacts import (  # noqa: E402
    cell_metadata,
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

    def test_a_missing_categorical_decodes_to_nan_not_the_last_category(self, tmp_path):
        """anndata writes a missing categorical as code -1.

        Indexing the category table with the raw codes makes -1 select the LAST
        category, so every unlabelled cell silently acquires the final category's name.
        A second decoder in io.h5ad did exactly that; this is now the only one.
        """
        path = tmp_path / "missing.h5ad"
        with h5py.File(path, "w") as fh:
            obs = fh.create_group("obs")
            obs.attrs["_index"] = "cell_id"
            obs.create_dataset("cell_id", data=np.array([b"c0", b"c1", b"c2"]))
            cat = obs.create_group("grouping")
            cat.create_dataset("categories", data=np.array([b"alpha", b"beta"]))
            cat.create_dataset("codes", data=np.array([0, -1, 1], dtype="i1"))

        grouping = read_obs(path)["grouping"]
        assert list(grouping.astype(object).iloc[[0, 2]]) == ["alpha", "beta"]
        assert pd.isna(grouping.iloc[1])

    def test_io_h5ad_exposes_no_second_obs_decoder(self):
        """One decode, in one place -- the divergence above is what a copy costs."""
        h5ad = pytest.importorskip("scads_drvi.io.h5ad")
        assert not hasattr(h5ad, "read_obs")

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
