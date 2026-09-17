"""Worker for ``docs/tutorials/ddp_benchmark.ipynb`` -- trains one DRVI config.

Each config runs in its own process, launched by the notebook via ``subprocess.run``,
rather than as three calls to a function inside one long-lived kernel process. That
isolation is necessary, not just tidy: PyTorch's CUDA context cannot be re-initialized
in a process that forks *after* the parent has already touched CUDA, and Lightning's
notebook-safe DDP strategies (``ddp_notebook``, ``ddp_notebook_find_unused_parameters_true``)
fork worker processes from whichever process calls ``.train()``. Once any config touches
CUDA in the kernel process, every later config's fork -- DDP or not -- inherits a
CUDA-initialized parent and breaks. One fresh process per config sidesteps that.

Usage: ``python ddp_bench_worker.py <name> <devices> [<strategy>]``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import anndata as ad
import scvi
from scvi.external import DRVI

DATA = Path(__file__).parent / "data"


def _poll_gpu_memory_mb(peak: list[int], stop: threading.Event) -> None:
    """Samples ``nvidia-smi`` on a timer, tracking the single highest per-GPU reading.

    DDP's notebook-safe strategies train each rank in its own forked process with its
    own CUDA context -- ``torch.cuda.max_memory_allocated()`` called back in this
    (parent) process after ``.train()`` returns only ever saw this process's own
    negligible allocations, never the child ranks' real usage. Reading physical
    device memory via ``nvidia-smi`` instead is agnostic to which process did the
    allocating, so it works the same way for the single-GPU and DDP configs alike.
    """
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            peak[0] = max(peak[0], max(int(x) for x in out.split()))
        except Exception:
            pass
        stop.wait(0.5)


def main() -> None:
    name, devices_s, *rest = sys.argv[1:]
    devices = int(devices_s)
    strategy = rest[0] if rest else None

    # This worker runs under `srun` (see the notebook's launch instructions), which sets
    # SLURM_* env vars that make Lightning auto-detect `SLURMEnvironment` as the cluster
    # environment -- one that assumes srun itself already launched one OS process per
    # rank. That assumption is wrong here: the notebook-safe DDP strategies below fork
    # their own rank processes *inside* this one srun task, so SLURMEnvironment's
    # rank/rendezvous-port derivation conflicts with DDPStrategy's actual fork-based
    # world size (observed as `EADDRINUSE` on the rendezvous port). Stripping SLURM_*
    # before `.train()` builds the Trainer forces the plain, fork-appropriate
    # `LightningEnvironment` instead.
    for k in [k for k in os.environ if k.startswith("SLURM_")]:
        del os.environ[k]

    atac = ad.read_h5ad(DATA / "_ddp_bench_atac.h5ad")

    scvi.settings.seed = 0
    DRVI.setup_anndata(atac, batch_key=None)
    model = DRVI(atac, n_latent=32)

    train_kwargs = dict(
        max_epochs=200,
        batch_size=256,
        accelerator="gpu",
        devices=devices,
        check_val_every_n_epoch=1,
    )
    if strategy:
        train_kwargs["strategy"] = strategy

    # subprocess.Popen/check_output (nvidia-smi) never touches this process's own CUDA
    # state, so starting the poller before `.train()` is safe even though the DDP
    # configs below fork from this same process inside `.train()`.
    peak_mem_mb = [0]
    stop_polling = threading.Event()
    poller = threading.Thread(target=_poll_gpu_memory_mb, args=(peak_mem_mb, stop_polling))
    poller.start()

    start = time.perf_counter()
    try:
        model.train(**train_kwargs)
    finally:
        stop_polling.set()
        poller.join()
    elapsed = time.perf_counter() - start

    # Every DDP rank reaches here, but BaseModelClass.save() has no rank guard of its
    # own -- every rank would otherwise race to write the same path. Only rank 0
    # records metrics/history and saves below (see the notebook's note on this).
    if not model.trainer.is_global_zero:
        return

    peak_mem_gb = peak_mem_mb[0] / 1e3

    model_dir = DATA / f"ddp_bench_{name}_model"
    model.save(str(model_dir), overwrite=True)

    embed = ad.AnnData(
        model.get_latent_representation(atac),
        obs=atac.obs[["n_genes_by_counts", "total_counts"]].copy(),
    )
    embed.var_names = [f"dr_{i}" for i in range(embed.n_vars)]
    model.set_latent_dimension_stats(embed, vanished_threshold=0.5)

    for split, df in model.history.items():
        df.to_csv(DATA / f"ddp_bench_{name}_history_{split}.csv")

    metrics = {
        "name": name,
        "devices": devices,
        "strategy": strategy,
        "elapsed_s": elapsed,
        "peak_mem_gb": peak_mem_gb,
        "n_vanished": int(embed.var["vanished"].sum()),
        "n_dims": int(embed.n_vars),
        "model_dir": str(model_dir),
    }
    (DATA / f"ddp_bench_{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
