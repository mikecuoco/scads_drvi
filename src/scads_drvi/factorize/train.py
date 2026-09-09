#!/usr/bin/env python3
"""Training a DRVI fit, on one GPU or several.

The rest of :mod:`scads_drvi.factorize` reads a fit someone else produced. This module is
the other end: it registers the data, builds the model, runs the fit, and writes the
``fit.meta.json`` that :func:`scads_drvi.factorize.model.fit_meta` refuses to work
without. Every field it records is a field a later stage branches on, so the record is
written by the same code that made the choice rather than by hand afterwards.

**Multi-GPU here is Lightning's, not** :mod:`scads_drvi.factorize.multigpu` **'s.** The
post-hoc stages are inference loops with no ``Trainer``, so they own the process group
and drive it through ``multigpu``. Training does the opposite: scvi hands the run to a
Lightning ``Trainer``, which creates and owns the process group. Calling
:func:`multigpu.init` first would hand Lightning an already-initialised group and is the
one thing not to do. What this module borrows from ``multigpu`` is its *reasoning* --
below is the same catalogue of ways a distributed run looks like it worked.

**The strategy string is what turns the sampler on.** scvi decides whether to wrap the
dataloader in a ``DistributedSampler`` by asking ``use_distributed_sampler(strategy)``,
which is literally ``"ddp" in strategy``, reading ``strategy`` out of the keyword
arguments passed to ``train()``. Configure DDP any other way -- an already-launched
process group, a ``Trainer`` built elsewhere -- and every rank iterates *every* cell.
The run finishes, the loss curve looks ordinary, and each epoch was ``world_size``
passes over the data with gradients averaged across duplicates.

**Unused parameters are a speed choice, not a correctness one.** The strategy this
defaults to makes DDP walk the autograd graph before every reduction looking for
parameters that got no gradient. ``find_unused_parameters=False`` takes the faster point
on that curve and lets DDP raise instead of search. Unlike everything else on this list
that failure is loud and immediate -- the first backward pass, not the last epoch -- so
it is worth trying, with the one caveat in :attr:`TrainConfig.find_unused_parameters`
about parameters whose use varies from step to step, which hang rather than raise.

**Early stopping and DDP do not compose.** scvi's ``Trainer`` disables early stopping
when the strategy is a DDP one, because the validation loop must run on a single device.
It warns; a warning in an hours-long job log is not a decision. Asking for both here is
an error instead, so the caller picks which one they meant.

**``batch_size`` is per device.** Four ranks at 128 is an effective batch of 512, and the
learning-rate and KL-warmup schedules are counted in optimiser steps, so the same config
on one GPU and on four is not the same fit. Both numbers go in the record.

**The train/validation split is drawn independently on every rank.** Each rank builds its
own splitter from ``scvi.settings.seed``. Leave the seed unset and the ranks disagree
about which cells are held out, so every cell is somebody's training cell and the
reported validation loss is measured on cells the model has already seen. A seed is
required for a distributed run.

**Everything after ``train()`` runs on every rank.** Lightning's subprocess launcher
re-executes the script in each child, so the save, the record and any post-processing
happen ``world_size`` times unless guarded -- see :func:`is_global_zero`. That same
re-execution is why staging is safe here and needs a barrier in the post-hoc stages: the
launcher spawns *after* the parent's staging has finished, so the children find the
staged copy already in place.

**No torch at module scope**, for the same reason ``multigpu`` gives: the argument guard
in :func:`main` has to be able to reject a bad preset in an environment carrying neither
torch nor scvi, rather than after a minute of imports.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scads_drvi._util.progress import log

if TYPE_CHECKING:  # pragma: no cover
    from scads_drvi.config import Project

__all__ = [
    "DDP_STRATEGY",
    "DDP_STRICT_STRATEGY",
    "Launch",
    "TrainConfig",
    "resolve_launch",
    "check_config",
    "is_global_zero",
    "unwrap_compiled",
    "prepare_adata",
    "build_model",
    "train_kwargs",
    "fit_record",
    "train",
    "main",
]

#: Default strategy for more than one device. ``find_unused_parameters_true`` is what the
#: scvi-tools multi-GPU guide prescribes for a script: DDP tolerates parameters that
#: receive no gradient, at the cost of an extra traversal of the autograd graph per step.
DDP_STRATEGY = "ddp_find_unused_parameters_true"

#: The faster point on the same trade-off: DDP raises on the first parameter that gets no
#: gradient instead of searching for them. Correct only for a model where every parameter
#: is reached on every step -- see :attr:`TrainConfig.find_unused_parameters`.
DDP_STRICT_STRATEGY = "ddp"


# --------------------------------------------------------------------------------------
# Where this process sits in the job
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    """How this run reaches its devices, resolved once and then only read.

    ``devices`` and ``num_nodes`` are what a ``Trainer`` is given; ``world_size`` is how
    many processes the finished run should have had, and is checked against the trainer
    afterwards rather than assumed.
    """

    devices: int | list[int] | str
    num_nodes: int = 1
    strategy: str | None = None
    world_size: int | None = None
    external: bool = False

    @property
    def distributed(self) -> bool:
        """Whether to treat this as a multi-process run.

        An unknown ``world_size`` -- ``devices=-1`` or ``"auto"`` with nothing to count
        them -- counts as distributed. The checks it gates cost a config change when they
        are wrong and a silent wrong answer when they are missed, so unknown resolves
        towards the check.
        """
        return self.world_size is None or self.world_size > 1


def _external_launcher(env: dict) -> tuple[int, int] | None:
    """``(world_size, processes per node)`` when something already launched us.

    ``torchrun`` sets ``WORLD_SIZE``/``LOCAL_WORLD_SIZE``; ``srun`` sets the ``SLURM_``
    pair. Lightning detects either and does **not** spawn again -- which is the whole
    reason this has to be known before ``devices`` is chosen. Told to spawn on top of an
    external launcher, Lightning would either refuse or start ``world_size`` groups.
    """
    world = int(env.get("WORLD_SIZE", 1) or 1)
    if world > 1:
        return world, int(env.get("LOCAL_WORLD_SIZE", world) or world)
    tasks = int(env.get("SLURM_NTASKS", 1) or 1)
    if tasks > 1:
        return tasks, int(env.get("SLURM_NTASKS_PER_NODE", tasks) or tasks)
    return None


def _device_count(devices: int | list[int] | str, available: int | None) -> int | None:
    """How many devices ``devices`` names, or None when only the accelerator knows."""
    if isinstance(devices, list):
        return len(devices)
    if isinstance(devices, int):
        return available if devices == -1 else devices
    return available  # "auto"


def resolve_launch(
    devices: int | list[int] | str = 1,
    *,
    strategy: str | None = None,
    default_strategy: str = DDP_STRATEGY,
    env: dict | None = None,
    available: int | None = None,
) -> Launch:
    """Turn a device request and the environment into one coherent ``Trainer`` config.

    ``strategy`` is what the caller explicitly asked for and ``default_strategy`` what to
    use when they did not -- two arguments rather than one because an explicit strategy on
    a single device is a mistake worth naming, and a default one is not.

    ``available`` is the accelerator's device count, supplied by the caller so that this
    stays importable and testable without torch. Left None, ``-1`` and ``"auto"`` are
    passed through and ``world_size`` is unknown -- which the post-run check then skips
    rather than guesses at.
    """
    env = dict(os.environ) if env is None else env
    ext = _external_launcher(env)

    if ext is not None:
        world, per_node = ext
        if world % per_node:
            raise SystemExit(
                f"ERROR: {world} processes over {per_node} per node is not a whole "
                f"number of nodes. Check WORLD_SIZE/LOCAL_WORLD_SIZE."
            )
        asked = _device_count(devices, available)
        if asked is not None and asked != per_node and devices != 1:
            # `devices=1` is the field default, so it means "unset" here rather than a
            # contradiction; anything else the caller typed and meant.
            raise SystemExit(
                f"ERROR: this process was launched externally with {per_node} processes "
                f"per node, but the config asks for {asked} devices. Under torchrun or "
                f"srun the launcher decides the device count; drop `devices` from the "
                f"config, or launch the script directly and let Lightning spawn."
            )
        return Launch(
            devices=per_node,
            num_nodes=world // per_node,
            strategy=strategy or default_strategy,
            world_size=world,
            external=True,
        )

    count = _device_count(devices, available)
    if count is not None and count < 1:
        raise SystemExit(f"ERROR: devices={devices!r} resolves to {count} devices")
    if count == 1:
        if strategy:
            raise SystemExit(
                f"ERROR: strategy={strategy!r} with a single device. A DDP strategy on "
                f"one device builds a distributed sampler and a process group for a run "
                f"that has neither; leave strategy unset."
            )
        return Launch(devices=devices, world_size=1)
    return Launch(
        devices=devices,
        strategy=strategy or default_strategy,
        world_size=count,
    )


def is_global_zero(model: Any | None = None, env: dict | None = None) -> bool:
    """True in the one process that may write.

    The trainer knows for certain and is asked first. Before it exists -- and in a child
    that Lightning's subprocess launcher re-executed from line 1 -- the answer is in the
    environment, and which variable holds it depends on who launched: ``torchrun`` sets
    ``RANK``, ``srun`` sets ``SLURM_PROCID``, and Lightning's own launcher sets only
    ``NODE_RANK``/``LOCAL_RANK`` in the children it starts.
    """
    trainer = getattr(model, "trainer", None) if model is not None else None
    rank = getattr(trainer, "global_rank", None)
    if rank is not None:
        return int(rank) == 0

    env = dict(os.environ) if env is None else env
    for key in ("RANK", "GLOBAL_RANK", "SLURM_PROCID"):
        if key in env:
            return int(env[key]) == 0
    return int(env.get("NODE_RANK", 0) or 0) == 0 and int(env.get("LOCAL_RANK", 0) or 0) == 0


# --------------------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Everything one fit needs, in one object that can be recorded verbatim.

    Deliberately not a set of command-line flags: a value that can be passed on the
    command line is a value a run record cannot recover. Presets live in the caller's own
    ``*_config.py`` -- see :mod:`scads_drvi._util.presets` and :func:`main`.

    The three ``*_kwargs`` dicts are the escape hatch. Enumerating every DRVI, plan and
    ``Trainer`` argument here would go stale on the next scvi release, and they are
    written into the record whole, so nothing set through them is invisible afterwards.
    """

    # -- what to fit, and on what ------------------------------------------------------
    adata: Path
    fit: str
    layer: str | None = None
    batch_key: str | None = None
    min_fragment: int = 0
    depth_col: str = "n_fragment"
    backed: bool = False

    # -- the model ---------------------------------------------------------------------
    n_latent: int = 32
    n_split_latent: int | None = None
    gene_likelihood: str = "pnb"
    dispersion: str | None = None
    model_kwargs: dict = field(default_factory=dict)

    # -- the optimisation --------------------------------------------------------------
    max_epochs: int = 400
    batch_size: int = 128
    train_size: float | None = None
    validation_size: float | None = None
    early_stopping: bool = False
    load_sparse_tensor: bool = False
    plan_kwargs: dict = field(default_factory=dict)
    seed: int | None = 0

    # -- the machine -------------------------------------------------------------------
    devices: int | list[int] | str = 1
    strategy: str | None = None
    #: Which point on the DDP speed/compatibility trade-off to take. True keeps the
    #: default ``ddp_find_unused_parameters_true``: before each reduction DDP walks the
    #: autograd graph to find parameters that got no gradient this step and leaves them
    #: out. False takes ``ddp``, which skips that traversal -- the faster point -- and
    #: raises the moment a parameter goes unreduced.
    #:
    #: Worth trying False, because the cost of being wrong is small and *early*: an
    #: always-dead parameter raises in the first backward pass, seconds into the run, not
    #: at hour three. The exception is a parameter used on some steps and not others,
    #: where the ranks can disagree about which reductions are coming and the job hangs
    #: instead of raising -- so if a run stops making progress with False, that is the
    #: first thing to put back.
    #:
    #: Ignored on a single device, where there is no reduction to skip.
    find_unused_parameters: bool = True
    precision: str | None = None
    compile: bool = False
    trainer_kwargs: dict = field(default_factory=dict)

    # -- bookkeeping -------------------------------------------------------------------
    smoke_test: bool = False
    smoke_cells: int = 2000
    smoke_epochs: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "adata", Path(self.adata).expanduser())
        if "/" in self.fit or self.fit in (".", ".."):
            raise ValueError(f"fit must be a directory name, not a path: {self.fit!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size={self.batch_size}")
        if self.min_fragment and not self.depth_col:
            raise ValueError("min_fragment needs a depth_col to read it from")
        if self.strategy and "ddp" not in self.strategy:
            # scvi asks `"ddp" in strategy` and nothing else before deciding whether to
            # build a DistributedSampler. Anything else trains every rank on every cell.
            raise ValueError(
                f"strategy={self.strategy!r} has no 'ddp' in it. scvi decides on the "
                f"distributed sampler by that substring alone, so this would give every "
                f"rank the whole dataset and call it an epoch."
            )
        # An explicit strategy wins over the flag, so a disagreement between the two is
        # a config that does the opposite of what one half of it says.
        if self.strategy and ("find_unused_parameters" in self.strategy) != (
            self.find_unused_parameters
        ):
            raise ValueError(
                f"strategy={self.strategy!r} and find_unused_parameters="
                f"{self.find_unused_parameters} disagree. Set one or the other: the "
                f"flag picks between {DDP_STRATEGY!r} and {DDP_STRICT_STRATEGY!r}, and "
                f"an explicit strategy overrides it."
            )

    def effective_batch_size(self, world_size: int | None) -> int | None:
        """What the optimiser actually sees per step. None when the size is unknown."""
        return None if world_size is None else self.batch_size * world_size

    def default_strategy(self) -> str:
        """The DDP strategy implied by :attr:`find_unused_parameters`."""
        return DDP_STRATEGY if self.find_unused_parameters else DDP_STRICT_STRATEGY


def check_config(config: TrainConfig, launch: Launch) -> None:
    """Refuse the combinations that run to completion and mean something else.

    Both of these are things scvi handles by warning. A warning is the right call for a
    library; for a run that costs GPU-hours and writes a record other stages trust, the
    caller should have to say which of the two things they meant.
    """
    if not launch.distributed:
        return
    if config.early_stopping:
        raise SystemExit(
            "ERROR: early_stopping with a DDP strategy. scvi disables early stopping "
            "under DDP -- the validation loop has to run on one device -- so the run "
            "would silently train the full max_epochs. Set early_stopping=False and "
            "choose max_epochs, or train on one device."
        )
    if config.seed is None:
        raise SystemExit(
            "ERROR: seed=None with a DDP strategy. Every rank draws its own "
            "train/validation split from scvi.settings.seed, so an unset seed gives the "
            "ranks different splits: every held-out cell is some other rank's training "
            "cell, and the validation loss is measured on cells the model has seen."
        )


# --------------------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------------------


def _fit_meta_for(config: TrainConfig, **extra) -> Any:
    from scads_drvi.factorize.model import FitMeta

    return FitMeta(
        n_latent=config.n_latent,
        n_split_latent=config.n_split_latent,
        batch_key=config.batch_key,
        layer=config.layer,
        min_fragment=config.min_fragment or None,
        depth_col=config.depth_col if config.min_fragment else None,
        gene_likelihood=config.gene_likelihood,
        dispersion=config.dispersion,
        seed=config.seed,
        smoke_test=config.smoke_test,
        **extra,
    )


def prepare_adata(config: TrainConfig, adata=None):
    """Read the input, apply the depth gate, register it the way a loader will.

    Registration goes through :func:`~scads_drvi.factorize.model.setup_anndata_like`,
    given a :class:`FitMeta` built from this config -- so the fit is registered by
    exactly the code that will re-register it at load time. A second ``setup_anndata``
    call written out here is how a fit ends up trained on one covariate and read back
    conditioned on another.

    An ``adata`` passed in must be the one at ``config.adata``, already read -- it saves
    the read, it does not choose the data. The depth gate is resolved from the path by
    :func:`~scads_drvi.io.h5ad.keep_rows`, so a different object would be gated by row
    indices belonging to another file, and ``config.adata`` is what the run record names
    as the input either way.

    Returns ``(adata, gate_info)``.
    """
    from scads_drvi.factorize.model import setup_anndata_like
    from scads_drvi.io.h5ad import keep_rows

    if adata is None:
        import anndata

        log(f"reading {config.adata}{' backed' if config.backed else ''}")
        adata = anndata.read_h5ad(config.adata, backed="r" if config.backed else None)

    rows, gate = keep_rows(str(config.adata), config.min_fragment, config.depth_col)
    subset = gate is not None and int(gate["n_cells_excluded"]) > 0

    if config.smoke_test and rows.size > config.smoke_cells:
        # Evenly spaced, not the first N: rows arrive grouped by whatever the writer
        # grouped them by, so a head slice is one group rather than a sample of all.
        import numpy as np

        rows = rows[np.linspace(0, rows.size - 1, config.smoke_cells).astype(np.int64)]
        subset = True
        log(f"smoke test: {rows.size:,} cells")

    if subset:
        if config.backed and not config.smoke_test:
            raise SystemExit(
                f"ERROR: the depth gate keeps {rows.size:,} of {adata.n_obs:,} cells, and "
                f"scvi cannot register a view of a backed AnnData. Write the gated subset "
                f"to its own h5ad once and point `adata` at it, or set backed=False and "
                f"pay the memory instead of paying it every epoch."
            )
        adata = adata[rows].to_memory() if config.backed else adata[rows].copy()

    setup_anndata_like(adata, _fit_meta_for(config))
    return adata, gate


def build_model(adata, config: TrainConfig):
    """Construct the DRVI. Every non-default is recorded, including via ``model_kwargs``."""
    from scvi.external import DRVI

    kwargs = dict(config.model_kwargs)
    if config.dispersion is not None:
        kwargs["dispersion"] = config.dispersion
    return DRVI(
        adata,
        n_latent=config.n_latent,
        n_split_latent=config.n_split_latent,
        gene_likelihood=config.gene_likelihood,
        **kwargs,
    )


def train_kwargs(config: TrainConfig, launch: Launch) -> dict:
    """The keyword arguments handed to ``model.train``.

    ``strategy`` is in here rather than set on a ``Trainer`` elsewhere because that is
    where scvi looks for it when deciding on the distributed sampler.
    """
    kwargs: dict[str, Any] = {
        "max_epochs": config.smoke_epochs if config.smoke_test else config.max_epochs,
        "batch_size": config.batch_size,
        "accelerator": "gpu",
        "devices": launch.devices,
        "early_stopping": config.early_stopping,
        "load_sparse_tensor": config.load_sparse_tensor,
    }
    if launch.strategy:
        kwargs["strategy"] = launch.strategy
    if launch.num_nodes > 1:
        kwargs["num_nodes"] = launch.num_nodes
    if config.train_size is not None:
        kwargs["train_size"] = config.train_size
    if config.validation_size is not None:
        kwargs["validation_size"] = config.validation_size
    if config.precision is not None:
        kwargs["precision"] = config.precision
    if config.plan_kwargs:
        kwargs["plan_kwargs"] = dict(config.plan_kwargs)
    kwargs.update(config.trainer_kwargs)
    return kwargs


def unwrap_compiled(model) -> bool:
    """Put the real module back on ``model`` if ``torch.compile`` wrapped it.

    ``torch.compile`` returns a wrapper holding the module at ``_orig_mod``, and its
    ``state_dict`` prefixes every key. Saving through the wrapper produces a checkpoint
    the loader cannot read at all -- and only at the end of the run, with the fit already
    paid for. :func:`~scads_drvi.factorize.model.repair_compile_prefix` exists to rescue
    those; this is how one stops being made. Called unconditionally, not only when this
    run did the compiling, because a plan or callback can compile too.
    """
    inner = getattr(getattr(model, "module", None), "_orig_mod", None)
    if inner is None:
        return False
    model.module = inner
    return True


def fit_record(
    config: TrainConfig,
    launch: Launch,
    *,
    n_cells: int,
    n_features: int,
    gate: dict | None,
    epochs_run: int | None,
    elapsed_s: float,
    compiled: bool,
    world_size: int | None,
    provenance: dict | None = None,
) -> dict:
    """The ``fit.meta.json`` payload.

    The keys :class:`~scads_drvi.factorize.model.FitMeta` types are at the top level and
    keep their names; everything else is nested, so a future field cannot collide with
    one a downstream stage branches on. ``FitMeta`` keeps the whole dict in ``.raw``, so
    nothing recorded here is lost to a consumer that wants it.
    """
    from scads_drvi import __version__

    record = asdict(_fit_meta_for(config, n_cells=n_cells, n_features=n_features))
    record.pop("raw", None)
    record["adata"] = str(config.adata)
    record["fit"] = config.fit
    record["scads_drvi_version"] = __version__
    record["train"] = {
        "max_epochs": config.smoke_epochs if config.smoke_test else config.max_epochs,
        "epochs_run": epochs_run,
        "batch_size": config.batch_size,
        "effective_batch_size": config.effective_batch_size(world_size),
        "world_size": world_size,
        "devices": launch.devices,
        "num_nodes": launch.num_nodes,
        "strategy": launch.strategy,
        "find_unused_parameters": config.find_unused_parameters,
        "external_launcher": launch.external,
        "precision": config.precision,
        "train_size": config.train_size,
        "validation_size": config.validation_size,
        "early_stopping": config.early_stopping,
        "load_sparse_tensor": config.load_sparse_tensor,
        "compiled": compiled,
        "elapsed_s": round(elapsed_s, 1),
        "model_kwargs": dict(config.model_kwargs),
        "plan_kwargs": dict(config.plan_kwargs),
        "trainer_kwargs": dict(config.trainer_kwargs),
    }
    if gate is not None:
        record["gate"] = gate
    if provenance is not None:
        record["config"] = provenance
    return record


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def train(
    project: Project,
    config: TrainConfig,
    *,
    adata=None,
    stage_fn=None,
    stage_dir: str | Path | None = None,
    provenance: dict | None = None,
) -> Path:
    """Fit a DRVI and write it, with its record, under ``project.fit_dir(config.fit)``.

    Returns the fit directory. On a non-zero rank it returns the same path without having
    written anything -- the caller does not have to know which process it is in.

    ``adata`` is an already-read copy of the file at ``config.adata``, for a caller that
    has one; see :func:`prepare_adata` for why it may not be a different object.

    ``stage_fn(src, stage_dir) -> Path`` copies the input to node-local disk. It is
    applied only when Lightning is going to do the spawning, and it must be idempotent:
    the children re-execute the script and will call it again, finding the copy already
    there. Under an external launcher every rank is already running, so there is no
    single process to stage from and no barrier to hold the others behind; staging then
    belongs in the job script, before the launcher.
    """
    import torch

    from scads_drvi.io.meta import save_meta

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Fitting a DRVI on CPU is not merely slower; use a "
            "GPU allocation, or a smoke_test preset on a machine that has one."
        )

    launch = resolve_launch(
        config.devices,
        strategy=config.strategy,
        default_strategy=config.default_strategy(),
        available=torch.cuda.device_count(),
    )
    check_config(config, launch)
    log(
        f"devices={launch.devices} nodes={launch.num_nodes} "
        f"strategy={launch.strategy} world_size={launch.world_size} "
        f"batch={config.batch_size}/device "
        f"effective={config.effective_batch_size(launch.world_size)}"
    )

    if stage_fn is not None:
        if stage_dir is None:
            raise SystemExit("ERROR: stage_fn was given with no stage_dir to copy into")
        if launch.external:
            raise SystemExit(
                "ERROR: stage_fn under an external launcher. Every rank is already "
                "running, so there is no single process to copy from -- they would all "
                "write the same destination. Stage in the job script before torchrun or "
                "srun, or launch this directly and let Lightning spawn."
            )
        config = replace(config, adata=Path(stage_fn(config.adata, Path(stage_dir))))
        log(f"staged input at {config.adata}")

    if config.seed is not None:
        import scvi

        scvi.settings.seed = config.seed

    adata, gate = prepare_adata(config, adata)
    model = build_model(adata, config)

    compiled = False
    if config.compile:
        model.module = torch.compile(model.module)
        compiled = True

    started = time.time()
    model.train(**train_kwargs(config, launch))
    elapsed = time.time() - started

    trainer = getattr(model, "trainer", None)
    observed = getattr(trainer, "world_size", None)
    epochs_run = getattr(trainer, "current_epoch", None)

    if unwrap_compiled(model):
        compiled = True

    if not is_global_zero(model):
        log(f"rank {getattr(trainer, 'global_rank', '?')}: done; rank 0 writes the fit.")
        return project.fit_dir(config.fit)

    out = project.fit_dir(config.fit)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "fit.meta.json"
    model_dir = out / "model"

    # Drop any previous record BEFORE overwriting the checkpoint it describes. Re-running
    # a fit name replaces the model; if this run then fails -- or is withheld by the world
    # size check below -- the old record would be left describing the new checkpoint,
    # which is the mismatch the check exists to prevent, arriving by another door.
    record_path.unlink(missing_ok=True)
    model.save(str(model_dir), overwrite=True, save_anndata=False)
    log(f"saved {model_dir} after {elapsed / 60:.1f} min")

    from scads_drvi.factorize.model import fit_meta, has_compile_prefix

    if has_compile_prefix(model_dir):  # pragma: no cover - unwrap_compiled prevents it
        raise RuntimeError(
            f"{model_dir} was written with compile-prefixed state_dict keys despite the "
            f"unwrap. The checkpoint is on disk and repair_compile_prefix can fix it, "
            f"but do not trust this path until the cause is known."
        )

    if launch.world_size is not None and observed is not None and observed != launch.world_size:
        # No record is written. `fit_meta` then refuses the directory, which is the
        # designed failure -- far better than a plausible fit nobody knows is wrong.
        raise SystemExit(
            f"ERROR: asked for a {launch.world_size}-process run and the trainer reports "
            f"{observed}. The checkpoint is saved at {model_dir} but NO fit.meta.json was "
            f"written, so nothing downstream will read it. Either the ranks trained "
            f"independently on the full data, or the launcher and the config disagree."
        )

    record = fit_record(
        config,
        launch,
        n_cells=int(adata.n_obs),
        n_features=int(adata.n_vars),
        gate=gate,
        epochs_run=None if epochs_run is None else int(epochs_run),
        elapsed_s=elapsed,
        compiled=compiled,
        world_size=observed if observed is not None else launch.world_size,
        provenance=provenance,
    )
    save_meta(record_path, record)

    # Read it back through the consumer. A key renamed here is otherwise found by the
    # stage that needed it, hours later and on another machine.
    check = fit_meta(project, config.fit)
    if check.n_latent != config.n_latent or check.batch_key != config.batch_key:
        raise RuntimeError(f"{record_path} does not read back as it was written")
    log(f"wrote {record_path}")
    return out


# --------------------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    """``python -m scads_drvi.factorize.train <preset> --config <module>``.

    The only things on the command line are which preset and which module, for the reason
    :mod:`scads_drvi._util.presets` gives: a value passed as a flag is a value the run
    record cannot recover. The module defines ``PRESETS = {"train": {...}}`` mapping a
    name to a :class:`TrainConfig`, and may define ``PROJECT``; without one the
    :class:`~scads_drvi.config.Project` comes from ``--root`` or ``$SCADS_DRVI_ROOT``.

    ``--config`` is required rather than globbed from a directory the way ``presets``
    does it: this module lives inside an installed package, so there is no arm directory
    beside it to look in.
    """
    import argparse

    from scads_drvi._util.presets import describe, load_module, load_preset, presets_for
    from scads_drvi._util.presets import provenance as config_provenance

    parser = argparse.ArgumentParser(
        prog="scads_drvi.factorize.train",
        description="Fit a DRVI. Parameters come from the preset, not from flags.",
    )
    parser.add_argument("preset", nargs="?", default=None, help="which preset to run")
    parser.add_argument("--config", required=True, type=Path, help="module defining PRESETS")
    parser.add_argument("--root", default=None, help="analysis root; else $SCADS_DRVI_ROOT")
    parser.add_argument("--list", action="store_true", help="print every preset and exit")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve()
    if args.list:
        table, path = presets_for(here, args.config)
        print(f"# train presets, from {path}")
        for name in sorted(table):
            print(f"{name}:\n{describe(table[name])}")
        raise SystemExit(0)

    config, name, path = load_preset(here, args.preset, config=args.config)
    if not isinstance(config, TrainConfig):
        raise SystemExit(f"ERROR: preset {name!r} in {path.name} is not a TrainConfig")

    from scads_drvi.config import Project

    project = getattr(load_module(args.config), "PROJECT", None)
    if project is None:
        project = Project.from_env(root=args.root)
    elif args.root:
        project = project.replace(root=args.root)

    log(f"preset {name} from {path}")
    return train(project, config, provenance=config_provenance(config, here, name, path))


if __name__ == "__main__":  # pragma: no cover
    # Re-enter through the canonical module instead of running `main` out of this
    # namespace. `python -m scads_drvi.factorize.train` executes this file under the name
    # `__main__`, so the `TrainConfig` defined here is a DIFFERENT CLASS OBJECT from the
    # one a preset module gets by importing `scads_drvi.factorize.train` -- and the
    # isinstance check in `main` rejects every valid preset with "is not a TrainConfig".
    from scads_drvi.factorize.train import main as _main

    _main()
