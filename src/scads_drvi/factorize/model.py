"""Loading a trained factorization and asking it questions.

This is the only module that imports ``torch`` and ``scvi``, which is why it is here and
not next to the loaders. Everything in it exists because of a specific way the model can
be used wrongly without appearing to fail:

**A compiled module saves keys the loader cannot read.** ``torch.compile`` returns a
wrapper holding the real module at ``_orig_mod``, and its ``state_dict()`` prefixes every
key. A checkpoint saved without unwrapping first cannot be loaded at all -- and the
failure arrives at the end of a long training run, not at the start.

**Row order is not what you asked for.** For a backed AnnData the torch dataset sorts
indices within a batch, so latent rows come back in sorted order regardless of the order
requested. Zipping them against the original index list misaligns every row.

There is deliberately no multi-GPU code here. Inference (everything past `load_fit`) is
cheap enough on one GPU that it does not need it. Training already gets scvi-tools' own
multi-GPU support for free through `train_fit`'s pass-through `train_kwargs` --
``accelerator="gpu", devices=N, strategy="ddp_find_unused_parameters_true"`` -- which is
Lightning DDP, not something this package would gain anything by reimplementing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "COMPILE_PREFIX",
    "FitMeta",
    "fit_meta",
    "checkpoint_keys",
    "has_compile_prefix",
    "repair_compile_prefix",
    "load_fit",
    "train_fit",
    "setup_anndata_like",
    "latent",
    "split_responsibility",
]

#: Prefix ``torch.compile``'s wrapper adds to every state_dict key.
COMPILE_PREFIX = "_orig_mod."


@dataclass(frozen=True)
class FitMeta:
    """The run record of a fit, with the fields callers actually branch on typed."""

    n_latent: int
    batch_key: str | None = None
    min_fragment: int | None = None
    depth_col: str | None = None
    n_cells: int | None = None
    n_features: int | None = None
    n_split_latent: int | None = None
    gene_likelihood: str | None = None
    dispersion: str | None = None
    seed: int | None = None
    smoke_test: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def is_gated(self) -> bool:
        """True when the fit trained on a depth-filtered subset of the cells."""
        return bool(self.min_fragment)


def fit_meta(path: str | Path) -> FitMeta:
    """Read a fit's run record: either its ``fit.meta.json`` (legacy layout, `path`
    naming that file or the fit directory containing it) or a result h5ad's
    ``uns["provenance"]`` (`path` naming the h5ad -- see
    :mod:`scads_drvi.factorize.result`).
    """
    path = Path(path)
    if path.suffix == ".h5ad":
        import anndata as ad

        raw = dict(ad.read_h5ad(path, backed="r").uns.get("provenance", {}))
        if "n_latent" not in raw:
            raise ValueError(f"{path}'s provenance does not record n_latent")
    else:
        json_path = path / "fit.meta.json" if path.is_dir() else path
        if not json_path.exists():
            raise FileNotFoundError(
                f"no fit record at {json_path}. It is written by the training run; "
                f"without it there is no way to tell what this checkpoint was trained on."
            )
        raw = json.loads(json_path.read_text())
        if "n_latent" not in raw:
            raise ValueError(f"{json_path} does not record n_latent")
    known = {f for f in FitMeta.__dataclass_fields__ if f != "raw"}
    return FitMeta(raw=raw, **{k: v for k, v in raw.items() if k in known})


def checkpoint_keys(model_dir: str | Path) -> list[str]:
    """State-dict keys in a saved checkpoint, without building the model."""
    import torch

    path = Path(model_dir)
    if path.is_dir():
        path = path / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("model_state_dict")
    if state is None:
        raise ValueError(f"{path} has no model_state_dict; keys are {list(blob)}")
    return list(state.keys())


def has_compile_prefix(model_dir: str | Path) -> bool:
    """True when the checkpoint was saved from a ``torch.compile``-wrapped module."""
    return any(k.startswith(COMPILE_PREFIX) for k in checkpoint_keys(model_dir))


def repair_compile_prefix(model_dir: str | Path, out_dir: str | Path) -> Path:
    """Write a copy of the checkpoint with the ``_orig_mod.`` prefix stripped.

    The original is left untouched -- a 3 GB checkpoint is not something to rewrite in
    place on the strength of an inference about how it was saved.
    """
    import shutil

    import torch

    src = Path(model_dir)
    src_file = src / "model.pt" if src.is_dir() else src
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        for extra in src.iterdir():
            if extra.name != "model.pt":
                target = out_dir / extra.name
                if not target.exists():
                    (shutil.copytree if extra.is_dir() else shutil.copy2)(extra, target)

    blob = torch.load(src_file, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"]
    blob["model_state_dict"] = {
        (k[len(COMPILE_PREFIX):] if k.startswith(COMPILE_PREFIX) else k): v
        for k, v in state.items()
    }
    out_file = out_dir / "model.pt"
    torch.save(blob, out_file)
    return out_dir


def load_fit(
    model_dir: str | Path,
    *,
    adata,
    device: Any | None = None,
    repair: bool = True,
    repair_dir: str | Path | None = None,
):
    """Load a trained DRVI, applying the compile-prefix fix.

    `model_dir` is the directory a training run's ``model.save(...)`` wrote (what used
    to be found via ``Project.fit_dir(fit) / "model"``; the caller now names it
    directly). `adata` must already be registered for the model -- see
    :func:`setup_anndata_like`, which reads the batch key out of the fit record rather
    than assuming one.
    """
    from scvi.external import DRVI

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"no saved model at {model_dir}")

    load_dir = model_dir
    if has_compile_prefix(model_dir):
        if not repair:
            raise ValueError(
                f"{model_dir} was saved from a torch.compile-wrapped module: every "
                f"state_dict key carries the {COMPILE_PREFIX!r} prefix and the loader "
                f"cannot read them. Pass repair=True to load from a corrected copy, or "
                f"re-save with `model.module = model.module._orig_mod` before save()."
            )
        target = Path(repair_dir) if repair_dir is not None else (
            model_dir.parent / "model.unwrapped"
        )
        if not (target / "model.pt").exists():
            repair_compile_prefix(model_dir, target)
        load_dir = target

    kwargs = {} if device is None else {"device": device}
    return DRVI.load(str(load_dir), adata=adata, **kwargs)


def train_fit(
    adata,
    *,
    n_latent: int,
    result_path: str | Path,
    model_dir: str | Path,
    batch_key: str | None = None,
    obs_columns: Sequence[str] | None = None,
    gene_likelihood: str = "poisson",
    max_epochs: int | None = None,
    batch_size: int = 128,
    seed: int = 0,
    vanished_threshold: float = 0.5,
    model_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
):
    """Train a DRVI model on `adata`, save its checkpoint, and write the canonical
    result h5ad -- the counterpart to :func:`load_fit`, for the case where training
    happens here rather than having already happened elsewhere.

    Registers `adata` with DRVI, trains, saves the checkpoint to `model_dir`, builds the
    embed (:func:`scads_drvi.factorize.result.build_embed`) and writes it to
    `result_path` with a provenance record -- one call from raw obs to a
    downstream-ready result object. Returns the written embed.

    Multi-GPU training, if wanted, is scvi-tools' own: pass e.g. ``train_kwargs={
    "accelerator": "gpu", "devices": 2, "strategy": "ddp_find_unused_parameters_true"}``
    (Lightning DDP). Under Slurm this needs launching with one task per GPU
    (``#SBATCH --ntasks-per-node=N`` and ``srun python ...``) -- ``--gres=gpu:N`` alone
    silently trains on a single GPU with no error.
    """
    import scvi
    from scvi.external import DRVI

    from scads_drvi.factorize.result import build_embed, write_result

    scvi.settings.seed = seed
    DRVI.setup_anndata(adata, batch_key=batch_key)
    model = DRVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood,
                 **(model_kwargs or {}))
    model.train(max_epochs=max_epochs, batch_size=batch_size, **(train_kwargs or {}))

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir), overwrite=True)

    embed = build_embed(
        model, adata, obs_columns=obs_columns, vanished_threshold=vanished_threshold
    )
    provenance = {
        "n_latent": n_latent,
        "batch_key": batch_key,
        "gene_likelihood": gene_likelihood,
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "seed": seed,
        "model_dir": str(model_dir),
    }
    write_result(result_path, embed, provenance=provenance)

    import anndata as ad

    # Re-read rather than return the pre-write `embed`: write_result merges in
    # `written_at`/`scads_drvi_version` defaults on its own copy, and the returned
    # object should match what a caller reading `result_path` back gets, exactly.
    return ad.read_h5ad(result_path)


def setup_anndata_like(adata, meta: FitMeta) -> None:
    """Register `adata` the way the fit was registered.

    The batch key comes from the fit record. Registering with a different one, or none,
    loads without complaint and then returns latents conditioned on the wrong covariate.
    """
    from scvi.external import DRVI

    if meta.batch_key is not None and meta.batch_key not in adata.obs.columns:
        raise KeyError(
            f"this fit was trained with batch_key={meta.batch_key!r}, which is not a "
            f"column of the supplied obs. The model cannot be loaded against it."
        )
    DRVI.setup_anndata(adata, layer=None, batch_key=meta.batch_key)


def latent(
    model,
    *,
    adata=None,
    indices: Sequence[int] | np.ndarray | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Latent representation, in the row order you asked for.

    For a backed AnnData the torch dataset sorts indices within a batch, so results come
    back in ascending index order however they were requested. Indices are sorted before
    the call and the rows permuted back afterwards, so the i-th row always corresponds to
    ``indices[i]``.
    """
    if indices is None:
        return np.asarray(
            model.get_latent_representation(adata, batch_size=batch_size)
        )

    requested = np.asarray(indices)
    if requested.ndim != 1:
        raise ValueError(f"indices must be 1-D, got shape {requested.shape}")
    if requested.size == 0:
        raise ValueError("no indices requested")
    if np.unique(requested).size != requested.size:
        raise ValueError("indices must be unique -- a repeated row cannot be restored")

    order = np.argsort(requested, kind="stable")
    ascending = requested[order]
    out = np.asarray(
        model.get_latent_representation(
            adata, indices=ascending, batch_size=batch_size
        )
        if adata is not None
        else model.get_latent_representation(
            indices=ascending, batch_size=batch_size
        )
    )
    if out.shape[0] != requested.size:
        raise ValueError(
            f"asked for {requested.size} rows and got {out.shape[0]}; the model did not "
            f"return one row per index"
        )
    restored = np.empty_like(out)
    restored[order] = out
    return restored


def split_responsibility(logits, *, split_dim: int = -2) -> np.ndarray:
    """Per-split responsibility from the decoder's per-split rate logits.

    ``softmax(logits, dim=split_dim)[i, k, j]`` is exactly ``mu_ikj / mu_ij`` -- the
    fraction of cell i's expected count at feature j attributable to split k.

    ``split_dim=-2`` is the splits axis and ``-1`` is features. Softmaxing the wrong one
    gives a well-formed array of the right shape whose entries mean nothing.

    The pseudo-split some decoders append for a background offset must **not** be
    included: it is not a factor, and including it moves mass into a component no
    downstream consumer knows about.
    """
    import torch

    tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    if tensor.ndim < 2:
        raise ValueError(
            f"expected at least (splits, features), got shape {tuple(tensor.shape)}"
        )
    with torch.no_grad():
        out = torch.softmax(tensor.to(torch.float64), dim=split_dim)
    return out.cpu().numpy()
