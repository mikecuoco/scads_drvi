"""enrich.embed: write_result / attach_enrich_results / directional_loadings -- the
data model for the single per-fit object that replaces Project + load_interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from scads_drvi.enrich.embed import (  # noqa: E402
    attach_enrich_results,
    directional_loadings,
    read_feature_loadings,
    write_result,
)


@pytest.fixture
def embed():
    """A cells x K object in the shape `model.set_latent_dimension_stats` would have
    populated -- built directly rather than through a fake model, since none of this
    module's own logic depends on how `embed` was produced."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 4)).astype(np.float32)
    obs = pd.DataFrame({"group": ["a", "b", "a"]}, index=["c0", "c1", "c2"])
    out = ad.AnnData(X=z, obs=obs)
    out.var_names = [f"dim_{i}" for i in range(out.n_vars)]
    return out


class TestWriteResult:
    def test_round_trips(self, tmp_path, embed):
        path = write_result(tmp_path / "result.h5ad", embed, provenance={"n_latent": 4})
        back = ad.read_h5ad(path)
        assert back.shape == embed.shape
        assert back.uns["provenance"]["n_latent"] == 4
        assert "written_at" in back.uns["provenance"]

    def test_feature_loadings_round_trip(self, tmp_path, embed):
        peaks = pd.DataFrame(
            np.arange(8, dtype=np.float32).reshape(2, 4),
            index=["chr1:100-200", "chr2:300-400"],
            columns=embed.var_names,
        )
        path = write_result(
            tmp_path / "result.h5ad", embed, provenance={}, feature_loadings=peaks
        )
        back = ad.read_h5ad(path)
        loadings = read_feature_loadings(path, embed=back)
        assert list(loadings.obs_names) == ["chr1:100-200", "chr2:300-400"]
        assert list(loadings.var_names) == list(embed.var_names)

    def test_non_canonical_peak_name_is_refused(self, tmp_path, embed):
        peaks = pd.DataFrame(
            np.zeros((2, 4), dtype=np.float32),
            index=["chr1_100_200", "chr2:300-400"],  # first is the old underscore form
            columns=embed.var_names,
        )
        with pytest.raises(ValueError, match="non-canonical peak name"):
            write_result(
                tmp_path / "result.h5ad", embed, provenance={}, feature_loadings=peaks
            )

    def test_no_feature_loadings_is_named(self, tmp_path, embed):
        path = write_result(tmp_path / "result.h5ad", embed, provenance={})
        with pytest.raises(FileNotFoundError, match="no feature_loadings_path"):
            read_feature_loadings(path)

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path, embed):
        write_result(tmp_path / "result.h5ad", embed, provenance={})
        assert list(tmp_path.glob("*.tmp.*")) == []


class TestAttachEnrichResults:
    def test_attaches_results_and_drops_annot_index(self, embed):
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
