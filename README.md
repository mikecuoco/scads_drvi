# scads_drvi

DRVI factorization → S-LDSC heritability enrichment → per-cell disease scores, for
single-cell ATAC data.

The package is **method-generic**. Nothing in it names a tissue, brain region, cell
grouping, donor-cohort variable, trait, GWAS study or obs column — those are all
caller-supplied. `tests/test_generic.py` enforces this with an AST scan over every
module, so the constraint is checked rather than merely intended.

## Install

```bash
pip install -e code/scads_drvi                    # normal case
pip install -e code/scads_drvi --no-deps          # inside a mamba-solved prefix
```

`--no-deps` matters in a conda/mamba environment: letting pip re-resolve `numpy` or
`torch` across a solved prefix will break the environment. The dependency floors in
`pyproject.toml` document what the code needs; they are not an install plan.

## Getting started

Every path comes from a `Project` you construct. The package derives nothing on its own.

```python
import scads_drvi as sd

proj = sd.Project(root="/path/to/analysis", fit="my_fit", traits=("trait_a", "trait_b"))
proj.enrich_dir("my_arm")
proj.contract("my_arm")["loadings"]
```

Defaults sit under `root` and each is individually overridable:

| field | default |
|---|---|
| `data` | `root/data` |
| `fits` | `data/factorize` |
| `enrich` | `data/enrich` |
| `annotations` | `data/annotations` |
| `figures` | `root/results/figures` |

`fit` and `traits` have **no** defaults on purpose — guessing either would silently point
the pipeline at the wrong inputs.

## Import surface

The top level imports nothing heavier than the standard library; public names resolve
lazily. Heavy dependencies are confined by directory:

| module | needs |
|---|---|
| `factorize/model.py`, `factorize/multigpu.py` | `torch`, `scvi-tools` (function-local) |
| `viz/` | `matplotlib`, `seaborn` (function-local) |
| `io/h5ad.py` | `h5py` at module scope — it *is* the h5ad reader |
| `io/artifacts.py` | `h5py`, function-local, so it imports without one |
| everything else | numpy / pandas / scipy / pyyaml |

This is not cosmetic. The environment that runs the enrichment stages has no torch, no
matplotlib, no seaborn and no h5py, and must still be able to import and use the
loaders, statistics and scoring. `tests/test_import_surface.py` checks it in a
subprocess with those modules blocked.

## Tests

```bash
pytest code/scads_drvi
```

Run this suite **separately** from `code/tests`. That suite's `conftest.py` inserts seven
arm directories onto `sys.path`; collecting both in one session would let those inserts
satisfy an import this package should satisfy itself — a false green hiding exactly the
bug class the package exists to remove.

## LDSC

S-LDSC is provided by the Rust reimplementation
([`sharifhsn/ldsc`](https://github.com/sharifhsn/ldsc), GPL-3.0), pinned by version and
verified by checksum. It replaces the Python original along with the pinned
python-3.10 / numpy-1.23 / pandas-1.5 environment and the three py2→py3 patches that
environment existed to carry.

Facts established against **v0.5.0** on this platform, which the wrapper must respect:

- **The CLI is subcommand-based** (`ldsc h2 …`, `ldsc l2 …`), not flag-based like
  `ldsc.py --h2`. Commands must be translated, not passed through.
- **`--python-compat` is an `l2` flag only.** It sets `--chunk-size 50` plus
  `--global-pass` and disables `--snp-level-masking`, giving bit-identical LD scores.
  There is no equivalent on `h2`, so heritability agreement has to be *measured* against
  known-good output rather than assumed.
- **`--sketch` and `--gpu` are `l2`-only and must stay off by default.** `--sketch` is an
  approximate CountSketch projection — it is where most of the headline speedup comes
  from and it changes the answer. Its own help warns that `d ≤ 50` is numerically
  unstable.
- **`--mmap` is recommended by upstream for networked filesystems** (GPFS/Lustre). Worth
  measuring here, where inputs live on network storage.
- The published Linux asset is a **dynamically linked glibc** binary, not static musl;
  the static build is something you produce yourself with `--features mimalloc`.
- The `.sha256` sidecar records the hash against a `dist/`-prefixed path, so
  `sha256sum -c` fails on it. Parse the hash field instead.

## Storage

The package never copies, stages or reserves disk. When a large read comes off network
storage it emits a `SlowStorageWarning` suggesting you stage the file on node-local disk
first — a warning, filterable by category, not a policy. Deciding *where* data lives is
the caller's job.
