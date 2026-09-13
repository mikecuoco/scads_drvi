"""io.result: build_embed / write_result / read_result / attach_enrich_results /
directional_loadings -- the single per-fit object that replaces Project + load_interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from scads_drvi.io.result import (  # noqa: E402
    attach_enrich_results,
    build_embed,
    directional_loadings,
    read_feature_loadings,
    read_result,
    write_result,
)


class FakeModel:
    """Stands in for a trained ``scvi.external.DRVI``: just enough surface for
    build_embed -- ``get_latent_representation`` and DRVI's own
    ``set_latent_dimension_stats``. Real DRVI's version writes more columns; this
    writes only the ones this package's own code reads off ``var``.
    """

    def __init__(self, z):
        self._z = z

    def get_latent_representation(self, adata=None, indices=None, batch_size=128):
        z = self._z
        if indices is not None:
            z = z[np.asarray(indices)]
        return z

    def set_latent_dimension_stats(
        self, embed, adata=None, datamodule=None, vanished_threshold=0.5
    ):
        max_value = np.abs(embed.X).max(axis=0)
        embed.var["max_value"] = max_value
        embed.var["order"] = np.argsort(-max_value)
        embed.var["title"] = [f"DR {i + 1}" for i in range(embed.n_vars)]
        embed.var["vanished"] = max_value < vanished_threshold


@pytest.fixture
def adata():
    obs = pd.DataFrame({"group": ["a", "b", "a"]}, index=["c0", "c1", "c2"])
    return ad.AnnData(X=np.zeros((3, 5), dtype=np.float32), obs=obs)


@pytest.fixture
def z():
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, 4)).astype(np.float32)


class TestBuildEmbed:
    def test_shape_and_var_names(self, adata, z):
        embed = build_embed(FakeModel(z), adata)
        assert embed.shape == (3, 4)
        assert list(embed.var_names) == ["dim_0", "dim_1", "dim_2", "dim_3"]

    def test_obs_columns_are_explicit_not_copy_everything(self, adata, z):
        model = FakeModel(z)
        assert list(build_embed(model, adata).obs.columns) == []
        embed = build_embed(model, adata, obs_columns=["group"])
        assert list(embed.obs.columns) == ["group"]

    def test_var_carries_drvi_stats(self, adata, z):
        embed = build_embed(FakeModel(z), adata, vanished_threshold=100.0)
        assert "vanished" in embed.var.columns
        assert bool(embed.var["vanished"].all())

    def test_umap_is_reindexed_to_obs_order(self, adata, z):
        umap = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}, index=["c2", "c0", "c1"]
        )
        embed = build_embed(FakeModel(z), adata, umap=umap)
        np.testing.assert_allclose(embed.obsm["X_umap"][0], [2.0, 5.0])  # c0's row

    def test_umap_missing_a_cell_is_refused(self, adata, z):
        umap = pd.DataFrame({"x": [1.0, 2.0], "y": [4.0, 5.0]}, index=["c0", "c1"])
        with pytest.raises(ValueError, match="have no UMAP"):
            build_embed(FakeModel(z), adata, umap=umap)


class TestWriteReadResult:
    def test_round_trips(self, tmp_path, adata, z):
        embed = build_embed(FakeModel(z), adata, obs_columns=["group"])
        path = write_result(tmp_path / "result.h5ad", embed, provenance={"n_latent": 4})
        back = read_result(path)
        assert back.shape == embed.shape
        assert back.uns["provenance"]["n_latent"] == 4
        assert "written_at" in back.uns["provenance"]

    def test_feature_loadings_round_trip(self, tmp_path, adata, z):
        embed = build_embed(FakeModel(z), adata)
        peaks = pd.DataFrame(
            np.arange(8, dtype=np.float32).reshape(2, 4),
            index=["peakA", "peakB"],
            columns=embed.var_names,
        )
        path = write_result(
            tmp_path / "result.h5ad", embed, provenance={}, feature_loadings=peaks
        )
        back = read_result(path)
        loadings = read_feature_loadings(path, embed=back)
        assert list(loadings.obs_names) == ["peakA", "peakB"]
        assert list(loadings.var_names) == list(embed.var_names)

    def test_no_feature_loadings_is_named(self, tmp_path, adata, z):
        path = write_result(tmp_path / "result.h5ad", build_embed(FakeModel(z), adata), provenance={})
        with pytest.raises(FileNotFoundError, match="no feature_loadings_path"):
            read_feature_loadings(path)

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path, adata, z):
        write_result(tmp_path / "result.h5ad", build_embed(FakeModel(z), adata), provenance={})
        assert list(tmp_path.glob("*.tmp.*")) == []


class TestAttachEnrichResults:
    def test_attaches_results_and_drops_annot_index(self, adata, z):
        embed = build_embed(FakeModel(z), adata)
        results = pd.DataFrame(
            {
                "dim": ["dim_0"],
                "direction": ["pos"],
                "trait": ["t1"],
                "Coefficient_z-score": [3.0],
            }
        )
        selection = pd.DataFrame(
            {
                "dim": ["dim_0"],
                "vanished": [False],
                "kept": [True],
                "drop_reason": [""],
                "annot_index": [1],
            }
        )
        attach_enrich_results(
            embed, "arm", results, factor_selection=selection, params={"top_frac": 0.05}
        )
        arm = embed.uns["enrich"]["arm"]
        assert arm["results"] is results
        assert "annot_index" not in arm["factor_selection"].columns
        assert arm["params"]["top_frac"] == 0.05


class TestDirectionalLoadings:
    def test_relu_and_negated_relu(self):
        embed = ad.AnnData(X=np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32))
        embed.var_names = ["dim_0", "dim_1"]
        pos = directional_loadings(embed, "pos")
        neg = directional_loadings(embed, "neg")
        np.testing.assert_allclose(pos.to_numpy(), [[1.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(neg.to_numpy(), [[0.0, 2.0], [3.0, 0.0]])

    def test_bad_direction_is_refused(self):
        embed = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
        with pytest.raises(ValueError, match="'pos' or 'neg'"):
            directional_loadings(embed, "sideways")
