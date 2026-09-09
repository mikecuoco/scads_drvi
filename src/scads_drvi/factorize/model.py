"""Loading a trained factorization and asking it questions.

This is the only module that imports ``torch`` and ``scvi``, which is why it is here and
not next to the loaders. Everything in it exists because of a specific way the model can
be used wrongly without appearing to fail:

**A compiled module saves keys the loader cannot read.** ``torch.compile`` returns a
wrapper holding the real module at ``_orig_mod``, and its ``state_dict()`` prefixes every
key. A checkpoint saved without unwrapping first cannot be loaded at all -- and the
failure arrives at the end of a long training run, not at the start.

**Every rank loads onto one GPU.** ``load`` defaults to a device that resolves to the
first GPU for every process, so a four-rank job runs correctly, reduces correctly, and
takes single-GPU time. It looks like a working multi-GPU run.

**Row order is not what you asked for.** For a backed AnnData the torch dataset sorts
indices within a batch, so latent rows come back in sorted order regardless of the order
requested. Zipping them against the original index list misaligns every row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from scads_drvi.config import Project

__all__ = [
    "COMPILE_PREFIX",
    "FitMeta",
    "fit_meta",
    "checkpoint_keys",
    "has_compile_prefix",
    "repair_compile_prefix",
    "load_fit",
    "setup_anndata_like",
    "latent",
    "split_responsibility",
    "load_kwargs",
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


def fit_meta(project: "Project", fit: str | None = None) -> FitMeta:
    """Read ``fit.meta.json`` for a fit."""
    path = project.fit_dir(fit) / "fit.meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no fit record at {path}. It is written by the training run; without it "
            f"there is no way to tell what this checkpoint was trained on."
        )
    raw = json.loads(path.read_text())
    if "n_latent" not in raw:
        raise ValueError(f"{path} does not record n_latent")
    known = {f for f in FitMeta.__dataclass_fields__ if f != "raw"}
    return FitMeta(raw=raw, **{k: v for k, v in raw.items() if k in known})


def _model_dir(project: "Project", fit: str | None = None) -> Path:
    return project.fit_dir(fit) / "model"


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


def load_kwargs(rank: Any | None = None) -> dict:
    """``accelerator``/``device`` for a model load, pinned to this rank's GPU.

    A list means device *indices*; an int would mean a device *count*, which is how a
    four-rank job ends up with every rank on the first GPU.
    """
    if rank is None:
        return {}
    from scads_drvi.factorize import multigpu

    return multigpu.load_kwargs(rank)


def load_fit(
    project: "Project",
    fit: str | None = None,
    *,
    adata,
    rank: Any | None = None,
    device: Any | None = None,
    repair: bool = True,
    repair_dir: str | Path | None = None,
):
    """Load a trained DRVI, applying the compile-prefix and device fixes.

    `adata` must already be registered for the model -- see :func:`setup_anndata_like`,
    which reads the batch key out of the fit record rather than assuming one.
    """
    from scvi.external import DRVI

    model_dir = _model_dir(project, fit)
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

    kwargs = dict(load_kwargs(rank))
    if device is not None:
        kwargs["device"] = device
    return DRVI.load(str(load_dir), adata=adata, **kwargs)


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
