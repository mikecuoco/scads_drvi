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

**Single-node multi-GPU plumbing, via `torchrun`, lives here too** (`Ranks` and the
functions built on it). Four post-hoc scripts each held a near-identical copy of this
before it was unified. They are inference loops with no Lightning `Trainer`, so
`torch.distributed` directly is the right fit -- and one copy rather than four is the
point, because the staging barrier below is the kind of thing that gets fixed in one
file and forgotten in the other three. No torch at module scope even for this: every
function imports it locally, so a caller's argument guard sits ABOVE its heavy imports
and a stray argv token cannot silently start a multi-hour job.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scads_drvi._util.progress import log

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
    "load_kwargs",
    "Ranks",
    "init_ranks",
    "barrier",
    "stage_once",
    "shard",
    "reduce_",
    "reduce_mean_np",
    "all_gather_rows",
    "exit_unless_main",
    "finish",
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


@dataclass(frozen=True)
class Ranks:
    """This process's place in a `torchrun` job. `world_size == 1` outside torchrun."""

    rank: int
    world_size: int
    local_rank: int
    device: object          # torch.device; typed loosely to keep torch out of module scope

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def load_kwargs(rank: Ranks | None = None) -> dict:
    """``accelerator``/``device`` for a model load, pinned to this rank's GPU.

    A list means device *indices*; an int would mean a device *count*, which is how a
    four-rank job ends up with every rank on the first GPU.

    scvi's ``load`` defaults to ``accelerator="auto", device="auto"``, which resolves to
    **cuda:0 for every process** -- ``torch.cuda.set_device`` does not override an
    explicit device chosen inside scvi. Under torchrun that puts all four ranks' models,
    and every decode, on one GPU: 4x the memory on device 0 (an OOM at any real batch
    size) and 4x the contention, while devices 1-3 sit idle. The failure is worse than
    slow: it looks like a working multi-GPU run. Every rank logs its own shard, the
    reduce is correct, and the answer is right -- just at single-GPU speed with a
    quarter of the memory each.

    scvi passes ``device`` straight through to Lightning's
    ``_AcceleratorConnector(devices=...)``, which reads an int as a COUNT of devices and
    a list as INDICES -- so ``device=0`` means "zero devices" and raises, and ``[local_rank]``
    is the only form that pins a single chosen GPU.
    """
    if rank is None:
        return {}
    return {"accelerator": "gpu", "device": [rank.local_rank]}


def load_fit(
    model_dir: str | Path,
    *,
    adata,
    rank: Ranks | None = None,
    device: Any | None = None,
    repair: bool = True,
    repair_dir: str | Path | None = None,
):
    """Load a trained DRVI, applying the compile-prefix and device fixes.

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

    kwargs = dict(load_kwargs(rank))
    if device is not None:
        kwargs["device"] = device
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
    happens here rather than having already happened elsewhere (e.g. a multi-GPU
    ``torchrun`` job using :func:`init_ranks` and friends below, which produces a
    checkpoint for `load_fit` to pick back up instead).

    Registers `adata` with DRVI, trains, saves the checkpoint to `model_dir`, builds the
    embed (:func:`scads_drvi.factorize.result.build_embed`) and writes it to
    `result_path` with a provenance record -- one call from raw obs to a
    downstream-ready result object. Returns the written embed.
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


# ---------------------------------------------------------------------------
# Single-node multi-GPU plumbing, via torchrun -- see the module docstring.
# ---------------------------------------------------------------------------

def init_ranks() -> Ranks:
    """Resolve the rank triple, bind this process to its GPU, join the process group.

    torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Outside it they are absent and this
    degrades to a single-GPU run with no branch at the call site.
    """
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. These stages decode (batch x K x n_features) per step "
            "and are unusable on CPU, not merely slower.")
    torch.cuda.set_device(local)
    if world > 1:
        # A LONG TIMEOUT, and it is not defensive. `stage_once` has rank 0 copy the input to
        # node-local NVMe while every other rank waits at `barrier()`. The default c10d store
        # timeout is minutes, so a stage that takes longer kills the job -- measured
        # 2026-08-28 on the K=48 input (256 GB): rank 3 died with
        #   DistBackendError: ... store->get('0') got error: Socket Timeout
        # after 11 min, having done nothing but wait. This never surfaced on the K=96 fit
        # because its input was already staged from an earlier run, so `stage_once` returned
        # immediately -- the bug was latent behind a warm cache.
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
    r = Ranks(rank=rank, world_size=world, local_rank=local,
              device=torch.device("cuda", local))
    log(f"rank {rank}/{world} on cuda:{local}")
    return r


def barrier(r: Ranks) -> None:
    import torch.distributed as dist
    if r.distributed:
        dist.barrier()


def stage_once(src: Path, stage_dir: Path | None, r: Ranks, *, stage_fn=None) -> Path:
    """Copy `src` to node-local disk ONCE, then let every rank open it.

    THE reason this helper exists. torchrun starts every rank at line 1 simultaneously, so
    four ranks would call the copier on the same multi-hundred-GB `.partial` and
    silently corrupt it. Lightning's own launcher spawns after staging and does not have
    this problem, which is why a plain Lightning training entry point needs no barrier
    and these post-hoc inference stages do.

    Rank 0 may legitimately DEGRADE to reading in place (a copier should do that when the
    destination is short on space), so the other ranks check for the staged file rather
    than assuming it appeared.

    `stage_fn(src, stage_dir) -> Path` does the copy and is supplied by the caller: what
    counts as fast local disk, and whether there is any, is a property of the machine
    rather than of this package. With no `stage_fn` this degrades to a barrier and a
    warning, which is the honest behaviour when nobody has said where to stage to.
    """
    src = Path(src)
    if stage_dir is None or stage_fn is None:
        if stage_dir is not None and stage_fn is None:
            from scads_drvi._util.advise import check_storage

            check_storage(src, kind="model input")
        barrier(r)
        return src
    if r.is_main:
        out = stage_fn(src, Path(stage_dir))
    else:
        out = Path(stage_dir) / src.name
    barrier(r)
    if not r.is_main and not out.exists():
        log(f"rank {r.rank}: staged copy absent -- rank 0 read in place, following it")
        out = src
    return out


def shard(arr: np.ndarray, r: Ranks) -> np.ndarray:
    """This rank's CONTIGUOUS slice of `arr`.

    Contiguous, not strided: a backed h5ad is row-major CSR so contiguous reads
    coalesce. Safe here in a way it is not during training -- every element is visited
    exactly once regardless of order, so the "rows are grouped in contiguous library
    runs" hazard that makes contiguous minibatches statistically wrong does not apply.
    """
    return np.array_split(np.asarray(arr), r.world_size)[r.rank]


def reduce_(t, r: Ranks, op: str = "sum"):
    """In-place all-reduce of a torch tensor. Returns it for chaining.

    `sum` for accumulators, `max`/`min` for extrema -- all three are exact and
    order-independent, which is what makes sharding these statistics safe at all. Float
    summation ORDER still differs from a single-GPU run, so expect ~1e-12 relative drift
    rather than bit-identity, and say so in the run record rather than claiming otherwise.

    THE CUDA CHECK IS NOT DEFENSIVE PROGRAMMING. `init_ranks` selects the **nccl**
    backend, which accepts only CUDA tensors and otherwise raises "Tensors must be CUDA
    and dense" -- from inside `all_reduce`, naming neither the caller nor the offending
    accumulator. And `reduce_` is a NO-OP at `world_size == 1`, so a CPU accumulator
    passes every single-GPU run and every self-check, then kills a multi-GPU job at the
    reduce, potentially hours in with the whole preceding pass already paid for.
    """
    import torch.distributed as dist

    if not r.distributed:
        return t
    if t.device.type != "cuda":
        raise RuntimeError(
            f"reduce_ was given a {t.device} tensor. The nccl backend all-reduces CUDA "
            f"tensors only. Build the accumulator with `device=r.device` rather than "
            f"moving it here, so the reduce and the arithmetic that fills it agree on "
            f"one device.")
    ops = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}
    if op not in ops:
        raise ValueError(f"op={op!r}; choose from {sorted(ops)}")
    dist.all_reduce(t, op=ops[op])
    return t


def reduce_mean_np(a: np.ndarray, r: Ranks):
    """Mean across ranks of a numpy array each rank holds its own mean of.

    For estimators that are already a per-rank MEAN over independent draws -- summing and
    dividing by world_size is only correct when every rank averaged the SAME number of
    draws, so the caller must guarantee that (assert divisibility) rather than hope.
    """
    import torch

    if not r.distributed:
        return np.asarray(a)
    t = torch.as_tensor(np.ascontiguousarray(a), device=r.device)
    reduce_(t, r, "sum")
    t /= r.world_size
    out = t.cpu().numpy()
    del t
    return out


def all_gather_rows(a: np.ndarray, r: Ranks) -> np.ndarray:
    """Concatenate every rank's row-block, in rank order, on every rank.

    `shard` splits with `np.array_split`, whose blocks differ in length by at most one, so
    this pads to the maximum, gathers a fixed shape, then trims each block back. Rank order
    IS row order because array_split preserves it.
    """
    import torch
    import torch.distributed as dist

    a = np.ascontiguousarray(a)
    if not r.distributed:
        return a

    n = torch.tensor([a.shape[0]], device=r.device, dtype=torch.int64)
    counts = [torch.zeros_like(n) for _ in range(r.world_size)]
    dist.all_gather(counts, n)
    counts = [int(c.item()) for c in counts]
    pad = max(counts)

    buf = torch.zeros((pad,) + a.shape[1:], device=r.device,
                      dtype=torch.as_tensor(a[:1]).dtype)
    buf[: a.shape[0]] = torch.as_tensor(a, device=r.device)
    parts = [torch.zeros_like(buf) for _ in range(r.world_size)]
    dist.all_gather(parts, buf)
    out = np.concatenate([p[:c].cpu().numpy() for p, c in zip(parts, counts, strict=True)], axis=0)
    del buf, parts
    log(f"rank {r.rank}: gathered {out.shape} from blocks {counts}")
    return out


def exit_unless_main(r: Ranks, what: str = "the outputs") -> None:
    """Non-zero ranks stop here.

    The guard has to cover READS of rank-0 artifacts, not just writes: a recorded 2-GPU
    DDP probe died at epoch 3 with rank 1 raising FileNotFoundError inside a contract
    check, which killed every other rank. Exiting is what makes that impossible -- a
    non-zero rank runs no line past this call.
    """
    if r.distributed and not r.is_main:
        log(f"rank {r.rank}: done; rank 0 writes {what}.")
        finish(r)
        raise SystemExit(0)


def finish(r: Ranks) -> None:
    import torch.distributed as dist
    if r.distributed and dist.is_initialized():
        dist.destroy_process_group()
