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

from scads_drvi.config import Project  # noqa: E402
from scads_drvi.factorize.model import (  # noqa: E402
    COMPILE_PREFIX,
    FitMeta,
    checkpoint_keys,
    fit_meta,
    has_compile_prefix,
    latent,
    load_kwargs,
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
        proj = Project(root=tmp_path, fit="f")
        proj.fit_dir().mkdir(parents=True)
        (proj.fit_dir() / "fit.meta.json").write_text(json.dumps(META))
        meta = fit_meta(proj)
        assert isinstance(meta, FitMeta)
        assert meta.n_latent == 96
        assert meta.batch_key == "library"
        assert meta.is_gated is True

    def test_unknown_keys_are_kept_in_raw_not_dropped(self, tmp_path):
        proj = Project(root=tmp_path, fit="f")
        proj.fit_dir().mkdir(parents=True)
        (proj.fit_dir() / "fit.meta.json").write_text(json.dumps(META))
        meta = fit_meta(proj)
        assert meta.raw["an_unknown_future_key"] == 7

    def test_ungated_fit(self, tmp_path):
        proj = Project(root=tmp_path, fit="f")
        proj.fit_dir().mkdir(parents=True)
        (proj.fit_dir() / "fit.meta.json").write_text(
            json.dumps({"n_latent": 32, "min_fragment": 0})
        )
        assert fit_meta(proj).is_gated is False

    def test_missing_record_says_who_writes_it(self, tmp_path):
        proj = Project(root=tmp_path, fit="f")
        with pytest.raises(FileNotFoundError, match="written by the training run"):
            fit_meta(proj)

    def test_a_record_without_n_latent_is_refused(self, tmp_path):
        proj = Project(root=tmp_path, fit="f")
        proj.fit_dir().mkdir(parents=True)
        (proj.fit_dir() / "fit.meta.json").write_text(json.dumps({"method": "drvi"}))
        with pytest.raises(ValueError, match="n_latent"):
            fit_meta(proj)


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


class TestLoadKwargs:
    def test_no_rank_means_no_device_pinning(self):
        assert load_kwargs(None) == {}


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
