#!/usr/bin/env python3
"""Single-node multi-GPU plumbing for arm 04's post-hoc scripts, via `torchrun`.

Four scripts (`ood_drvi.py`, `ind_drvi.py`, `pseudobulk_drvi.py`,
`counterfactual_drvi.py`) each held a near-identical copy of this. They are inference loops
with no Lightning `Trainer`, so `torch.distributed` directly is the right fit -- and one copy
rather than four is the point, because the staging barrier below is the kind of thing that
gets fixed in one file and forgotten in the other three.

**No torch at module scope.** Every function imports it locally, the same convention
`lsi_gpu`, `embed_umap`, `embed_metrics`, `stage` and `progress` follow. That is load-bearing:
the callers put an argument guard ABOVE their heavy imports so a stray argv token cannot
silently start a multi-hour job, and that guard has to fire under `env-scads`, which has
neither torch nor a compiled HDF5. `code/tests/test_lsi_gpu.py::TestExtractionContract` pins
it.

Named `multigpu`, not `dist`: `code/common` is `sys.path[0]` for these scripts and
`torch.distributed` is universally aliased `dist`, so that name invites exactly the shadowing
CLAUDE.md warns about.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scads_drvi._util.progress import log


@dataclass(frozen=True)
class Ranks:
    """This process's place in the job. `world_size == 1` outside torchrun."""

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


def init() -> Ranks:
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


def load_kwargs(r: Ranks) -> dict:
    """`accelerator`/`device` for `Model.load`, pinned to THIS rank's GPU.

        scvi's `load` defaults to `accelerator="auto", device="auto"`, which resolves to
        **cuda:0 for every process** -- `torch.cuda.set_device` does not override an explicit
        device chosen inside scvi. Under torchrun that puts all four ranks' models, and every
        decode, on one GPU: 4x the memory on device 0 (an OOM at any real batch size) and 4x
        the contention, while devices 1-3 sit idle. Measured 2026-08-27 in job 25437351, whose
        OOM message listed four processes on "GPU 0".

        The failure is worse than slow: it looks like a working multi-GPU run. Every rank logs
        its own shard, the reduce is correct, and the answer is right -- just at single-GPU
        speed with a quarter of the memory each.
    """
    # A LIST, not an int. scvi passes `device` straight through to Lightning's
    # `_AcceleratorConnector(devices=...)`, which reads an int as a COUNT of devices and a list
    # as INDICES -- so `device=0` means "zero devices" and raises, and `device=2` means "the
    # first two". scvi then takes `_devices_flag[0]` as the index, so `[local_rank]` is the
    # only form that pins a single chosen GPU. Job 25437375 died on the int form.
    return {"accelerator": "gpu", "device": [r.local_rank]}


def barrier(r: Ranks) -> None:
    import torch.distributed as dist
    if r.distributed:
        dist.barrier()


def stage_once(src: Path, stage_dir: Path | None, r: Ranks, *, stage_fn=None) -> Path:
    """Copy `src` to node-local disk ONCE, then let every rank open it.

        THE reason this helper exists. torchrun starts every rank at line 1 simultaneously, so
        four ranks would call the copier on the same multi-hundred-GB `.partial` and
        silently corrupt it. Lightning's own launcher spawns after staging and does not have
        this problem, which is why `fit_drvi.py` needs no barrier and these four do.

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

        Contiguous, not strided: the h5ad is row-major CSR so contiguous reads coalesce. Safe
        here in a way it is not during training -- every element is visited exactly once
        regardless of order, so the "rows are grouped in 257 contiguous library runs" hazard
        that makes contiguous minibatches statistically wrong does not apply.
    """
    return np.array_split(np.asarray(arr), r.world_size)[r.rank]


def reduce_(t, r: Ranks, op: str = "sum"):
    """In-place all-reduce of a torch tensor. Returns it for chaining.

        `sum` for accumulators, `max`/`min` for extrema -- all three are exact and
        order-independent, which is what makes sharding these statistics safe at all. Float
        summation ORDER still differs from a single-GPU run, so expect ~1e-12 relative drift
        rather than bit-identity, and say so in the run record rather than claiming otherwise.

        THE CUDA CHECK IS NOT DEFENSIVE PROGRAMMING. `init` selects the **nccl** backend,
        which accepts only CUDA tensors and otherwise raises "Tensors must be CUDA and dense"
        -- from inside `all_reduce`, naming neither the caller nor the offending accumulator.
        And `reduce_` is a NO-OP at `world_size == 1`, so a CPU accumulator passes every
        single-GPU run and every `selfcheck`, then kills the 4-GPU job at the reduce: for
        `ind_drvi.py` that is ~1.3 h in, with the whole effect pass already paid for. Caught
        exactly that way in `ind_drvi.py` on 2026-08-27, so the error now names the fix.
    """
    import torch.distributed as dist

    if not r.distributed:
        return t
    if t.device.type != "cuda":
        raise RuntimeError(
            f"reduce_ was given a {t.device} tensor. The nccl backend all-reduces CUDA "
            f"tensors only. Build the accumulator with `device=r.device` (as "
            f"pseudobulk_drvi.py and counterfactual_drvi.py do) rather than moving it here, "
            f"so the reduce and the arithmetic that fills it agree on one device.")
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
    out = np.concatenate([p[:c].cpu().numpy() for p, c in zip(parts, counts)], axis=0)
    del buf, parts
    log(f"rank {r.rank}: gathered {out.shape} from blocks {counts}")
    return out


def exit_unless_main(r: Ranks, what: str = "the outputs") -> None:
    """Non-zero ranks stop here.

        The guard has to cover READS of rank-0 artifacts, not just writes: the recorded 2-GPU
        DDP probe died at epoch 3 with rank 1 raising FileNotFoundError inside a contract check,
        which killed every other rank. Exiting is what makes that impossible -- a non-zero rank
        runs no line past this call.
    """
    if r.distributed and not r.is_main:
        log(f"rank {r.rank}: done; rank 0 writes {what}.")
        finish(r)
        raise SystemExit(0)


def finish(r: Ranks) -> None:
    import torch.distributed as dist
    if r.distributed and dist.is_initialized():
        dist.destroy_process_group()
