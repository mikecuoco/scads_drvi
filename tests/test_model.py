"""factorize.model: the traps that fail silently.

Most of this is testable without a GPU or a real checkpoint: the compile-prefix
detection and repair operate on a saved state dict, and the row-order guarantee is a
property of the permutation, not of the model. Only load_fit itself needs scvi, and it
is marked `gpu`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scads_drvi.factorize.model import (  # noqa: E402
    COMPILE_PREFIX,
    FitMeta,
    checkpoint_keys,
    fit_meta,
    has_compile_prefix,
    latent,
    repair_compile_prefix,
    split_responsibility,
)

META = {
    "n_latent": 96,
    "batch_key": "library",
    "min_fragment": 1000,
    "depth_col": "n_fragment",
    "n_cells": 1263026,
    "n_features": 1268438,
    "gene_likelihood": "poisson",
    "dispersion": "gene",
    "seed": 0,
    "smoke_test": False,
    "an_unknown_future_key": 7,
}


def write_checkpoint(path, *, prefixed: bool):
    keys = ["px_r", "z_encoder.weight", "decoder.bias"]
    if prefixed:
        keys = [COMPILE_PREFIX + k for k in keys]
    state = {k: torch.zeros(2) for k in keys}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": state, "var_names": ["a", "b"], "attr_dict": {}}, path
    )
    return path


class TestFitMeta:
    def test_reads_and_types_the_record(self, tmp_path):
        fit_dir = tmp_path / "f"
        fit_dir.mkdir(parents=True)
        (fit_dir / "fit.meta.json").write_text(json.dumps(META))
        meta = fit_meta(fit_dir)
        assert isinstance(meta, FitMeta)
        assert meta.n_latent == 96
        assert meta.batch_key == "library"
        assert meta.is_gated is True

    def test_reads_the_json_file_directly_too(self, tmp_path):
        fit_dir = tmp_path / "f"
        fit_dir.mkdir(parents=True)
        path = fit_dir / "fit.meta.json"
        path.write_text(json.dumps(META))
        assert fit_meta(path).n_latent == 96

    def test_unknown_keys_are_kept_in_raw_not_dropped(self, tmp_path):
        fit_dir = tmp_path / "f"
        fit_dir.mkdir(parents=True)
        (fit_dir / "fit.meta.json").write_text(json.dumps(META))
        meta = fit_meta(fit_dir)
        assert meta.raw["an_unknown_future_key"] == 7

    def test_ungated_fit(self, tmp_path):
        fit_dir = tmp_path / "f"
        fit_dir.mkdir(parents=True)
        (fit_dir / "fit.meta.json").write_text(
            json.dumps({"n_latent": 32, "min_fragment": 0})
        )
        assert fit_meta(fit_dir).is_gated is False

    def test_missing_record_says_who_writes_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="written by the training run"):
            fit_meta(tmp_path / "f")

    def test_a_record_without_n_latent_is_refused(self, tmp_path):
        fit_dir = tmp_path / "f"
        fit_dir.mkdir(parents=True)
        (fit_dir / "fit.meta.json").write_text(json.dumps({"method": "drvi"}))
        with pytest.raises(ValueError, match="n_latent"):
            fit_meta(fit_dir)

    def test_reads_provenance_from_a_result_h5ad(self, tmp_path):
        ad = pytest.importorskip("anndata")
        np_ = pytest.importorskip("numpy")

        embed = ad.AnnData(X=np_.zeros((2, 2), dtype="float32"))
        embed.uns["provenance"] = dict(META)
        path = tmp_path / "result.h5ad"
        embed.write_h5ad(path)

        meta = fit_meta(path)
        assert isinstance(meta, FitMeta)
        assert meta.n_latent == 96
        assert meta.raw["an_unknown_future_key"] == 7

    def test_h5ad_without_n_latent_is_refused(self, tmp_path):
        ad = pytest.importorskip("anndata")
        np_ = pytest.importorskip("numpy")

        embed = ad.AnnData(X=np_.zeros((2, 2), dtype="float32"))
        embed.uns["provenance"] = {"method": "drvi"}
        path = tmp_path / "result.h5ad"
        embed.write_h5ad(path)

        with pytest.raises(ValueError, match="n_latent"):
            fit_meta(path)


class _FakeDRVI:
    """Stands in for `scvi.external.DRVI`: just enough surface for `train_fit` --
    `setup_anndata` (staticmethod), construction, `train`, `save`,
    `get_latent_representation`, `set_latent_dimension_stats` -- with every call
    recorded so the test can assert on the orchestration order, not just the result.
    """

    calls: list = []

    def __init__(self, adata, n_latent, gene_likelihood="poisson", **kwargs):
        _FakeDRVI.calls.append(("init", n_latent, gene_likelihood, kwargs))
        self.n_latent = n_latent
        self._n_obs = adata.n_obs

    @staticmethod
    def setup_anndata(adata, batch_key=None):
        _FakeDRVI.calls.append(("setup_anndata", batch_key))

    def train(self, max_epochs=None, batch_size=128, **kwargs):
        _FakeDRVI.calls.append(("train", max_epochs, batch_size, kwargs))

    def save(self, path, overwrite=True):
        from pathlib import Path

        _FakeDRVI.calls.append(("save", path, overwrite))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.pt").write_bytes(b"fake")

    def get_latent_representation(self, adata=None, indices=None, batch_size=128):
        return np.zeros((self._n_obs, self.n_latent), dtype=np.float32)

    def set_latent_dimension_stats(
        self, embed, adata=None, datamodule=None, vanished_threshold=0.5
    ):
        embed.var["vanished"] = False


class TestTrainFit:
    """`train_fit` trains, saves the checkpoint, and writes the canonical result h5ad in
    one call -- the counterpart to `load_fit` for the case where training happens here.
    Verified against a fake `scvi`/`scvi.external.DRVI` (injected via `sys.modules`,
    same technique `test_import_surface.py` uses to fake heavy modules), so the
    orchestration is checked without needing a real GPU/scvi-tools install.
    """

    def test_trains_saves_and_writes_the_result(self, tmp_path, monkeypatch):
        import sys
        import types

        ad = pytest.importorskip("anndata")
        import pandas as pd

        _FakeDRVI.calls = []
        fake_external = types.ModuleType("scvi.external")
        fake_external.DRVI = _FakeDRVI
        fake_scvi = types.ModuleType("scvi")
        fake_scvi.external = fake_external
        fake_scvi.settings = types.SimpleNamespace(seed=None)
        monkeypatch.setitem(sys.modules, "scvi", fake_scvi)
        monkeypatch.setitem(sys.modules, "scvi.external", fake_external)

        from scads_drvi.factorize.model import train_fit

        obs = pd.DataFrame({"batch": ["a", "b", "a"]}, index=["c0", "c1", "c2"])
        adata = ad.AnnData(X=np.zeros((3, 5), dtype=np.float32), obs=obs)

        embed = train_fit(
            adata,
            n_latent=4,
            result_path=tmp_path / "result.h5ad",
            model_dir=tmp_path / "model",
            batch_key="batch",
            obs_columns=["batch"],
            max_epochs=2,
            seed=7,
        )

        # trained (with the seed set first) and registered before construction
        assert fake_scvi.settings.seed == 7
        assert _FakeDRVI.calls[0] == ("setup_anndata", "batch")
        assert _FakeDRVI.calls[1][0] == "init"
        assert _FakeDRVI.calls[2][0] == "train"
        assert _FakeDRVI.calls[3][0] == "save"

        # checkpoint saved
        assert (tmp_path / "model" / "model.pt").exists()

        # canonical result written and returned, matching what's on disk
        assert embed.shape == (3, 4)
        assert list(embed.obs.columns) == ["batch"]
        assert embed.uns["provenance"]["n_latent"] == 4
        assert embed.uns["provenance"]["batch_key"] == "batch"
        assert embed.uns["provenance"]["seed"] == 7
        back = ad.read_h5ad(tmp_path / "result.h5ad")
        assert back.uns["provenance"] == embed.uns["provenance"]


class TestCompilePrefix:
    def test_detects_a_clean_checkpoint(self, tmp_path):
        write_checkpoint(tmp_path / "model" / "model.pt", prefixed=False)
        assert has_compile_prefix(tmp_path / "model") is False

    def test_detects_a_wrapped_checkpoint(self, tmp_path):
        write_checkpoint(tmp_path / "model" / "model.pt", prefixed=True)
        assert has_compile_prefix(tmp_path / "model") is True

    def test_keys_can_be_listed_from_a_file_or_a_directory(self, tmp_path):
        write_checkpoint(tmp_path / "model" / "model.pt", prefixed=False)
        by_dir = checkpoint_keys(tmp_path / "model")
        by_file = checkpoint_keys(tmp_path / "model" / "model.pt")
        assert by_dir == by_file == ["px_r", "z_encoder.weight", "decoder.bias"]

    def test_missing_checkpoint_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no checkpoint"):
            checkpoint_keys(tmp_path / "absent")

    def test_a_blob_without_a_state_dict_is_refused(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save({"something_else": 1}, path)
        with pytest.raises(ValueError, match="no model_state_dict"):
            checkpoint_keys(path)

    def test_repair_strips_the_prefix_and_leaves_the_original(self, tmp_path):
        src = tmp_path / "model"
        write_checkpoint(src / "model.pt", prefixed=True)
        (src / "extra.json").write_text("{}")

        out = repair_compile_prefix(src, tmp_path / "unwrapped")
        assert has_compile_prefix(out) is False
        assert checkpoint_keys(out) == ["px_r", "z_encoder.weight", "decoder.bias"]
        # a 3 GB checkpoint is not rewritten in place on the strength of an inference
        assert has_compile_prefix(src) is True
        assert (out / "extra.json").exists()

    def test_repair_preserves_the_other_blob_entries(self, tmp_path):
        src = tmp_path / "model"
        write_checkpoint(src / "model.pt", prefixed=True)
        out = repair_compile_prefix(src, tmp_path / "unwrapped")
        blob = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
        assert blob["var_names"] == ["a", "b"]


class FakeModel:
    """Returns one row per index, in ASCENDING index order regardless of request.

    That is what a backed AnnData does, and it is the whole reason `latent` exists.
    """

    def __init__(self, n_latent=3):
        self.n_latent = n_latent
        self.seen = None

    def get_latent_representation(self, adata=None, indices=None, batch_size=None):
        indices = np.asarray(indices)
        self.seen = indices.copy()
        ascending = np.sort(indices)
        # row content encodes which cell it is, so misalignment is detectable
        return np.stack([np.full(self.n_latent, float(i)) for i in ascending])


class TestLatentRowOrder:
    def test_rows_follow_the_requested_order(self):
        model = FakeModel()
        want = np.array([7, 2, 5])
        out = latent(model, indices=want)
        assert out.shape == (3, 3)
        assert list(out[:, 0]) == [7.0, 2.0, 5.0]

    def test_indices_reach_the_model_sorted(self):
        model = FakeModel()
        latent(model, indices=np.array([9, 1, 4]))
        assert list(model.seen) == [1, 4, 9]

    def test_already_sorted_input_is_unchanged(self):
        model = FakeModel()
        out = latent(model, indices=np.array([1, 2, 3]))
        assert list(out[:, 0]) == [1.0, 2.0, 3.0]

    def test_repeated_indices_are_refused(self):
        with pytest.raises(ValueError, match="unique"):
            latent(FakeModel(), indices=np.array([1, 1, 2]))

    def test_empty_and_2d_are_refused(self):
        with pytest.raises(ValueError, match="no indices"):
            latent(FakeModel(), indices=np.array([], dtype=int))
        with pytest.raises(ValueError, match="1-D"):
            latent(FakeModel(), indices=np.zeros((2, 2), dtype=int))

    def test_a_short_return_is_caught(self):
        class Short(FakeModel):
            def get_latent_representation(self, adata=None, indices=None, batch_size=None):
                return np.zeros((1, self.n_latent))

        with pytest.raises(ValueError, match="one row per index"):
            latent(Short(), indices=np.array([1, 2, 3]))

    def test_no_indices_passes_through(self):
        class All(FakeModel):
            def get_latent_representation(self, adata=None, batch_size=None):
                return np.zeros((5, self.n_latent))

        assert latent(All()).shape == (5, 3)


class TestSplitResponsibility:
    def test_responsibilities_sum_to_one_over_splits(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(4, 3, 6))  # cells x splits x features
        out = split_responsibility(logits)
        assert out.shape == (4, 3, 6)
        assert np.allclose(out.sum(axis=-2), 1.0, atol=1e-12)

    def test_it_is_exactly_mu_ratio(self):
        """softmax over splits == mu_ikj / mu_ij, which is the identity being relied on."""
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(2, 3, 5))
        mu = np.exp(logits)
        expected = mu / mu.sum(axis=-2, keepdims=True)
        assert np.allclose(split_responsibility(logits), expected, atol=1e-12)

    def test_the_wrong_axis_gives_a_different_answer(self):
        """Softmaxing features instead of splits yields a well-formed, meaningless array."""
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(2, 3, 5))
        over_splits = split_responsibility(logits, split_dim=-2)
        over_features = split_responsibility(logits, split_dim=-1)
        assert not np.allclose(over_splits, over_features)
        assert np.allclose(over_features.sum(axis=-1), 1.0, atol=1e-12)

    def test_accepts_a_torch_tensor(self):
        logits = torch.zeros(2, 4, 3)
        out = split_responsibility(logits)
        assert np.allclose(out, 0.25)

    def test_too_few_dimensions_is_refused(self):
        with pytest.raises(ValueError, match="at least"):
            split_responsibility(np.zeros(5))

    def test_extreme_logits_do_not_overflow(self):
        logits = np.array([[[1000.0], [-1000.0]]])
        out = split_responsibility(logits)
        assert np.isfinite(out).all()
        assert out[0, 0, 0] == pytest.approx(1.0)
