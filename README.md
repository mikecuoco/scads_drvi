# scads_drvi

DRVI factorization → S-LDSC heritability enrichment → per-cell disease scores, for
single-cell ATAC data.

The package is **method-generic**. Nothing in it names a tissue, brain region, cell
grouping, donor-cohort variable, trait, GWAS study or obs column — those are all
caller-supplied. Two tests enforce that rather than trusting it:

- `tests/test_generic.py` — an AST scan over every module rejecting dataset vocabulary in
  string literals and identifiers. Docstrings are **exempt** by design (see
  `tests/conftest.py`): prose may name a domain term as an example, a value used as data
  may not. So the guarantee is precisely "no dataset-specific *value or name*", and
  keeping docstrings clean of them is a convention rather than something the scan checks.
- `tests/test_portability.py` — builds a complete synthetic *plant* single-cell analysis
  (tissues, cultivars, agronomic traits — no shared vocabulary at all) and runs the whole
  chain on it: obs → `Project` → labels → LDSC results → BH → per-cell scores →
  aggregation → seven figures. A scan proves no forbidden *names*; this proves no hidden
  *assumptions*.

## What's in it

| module | role |
|---|---|
| `config.Project` | every path, caller-supplied with defaults under one `root` |
| `labels` | the three names a factor has, and which one may index a matrix |
| `stats` | one-tailed p, Benjamini–Hochberg, BH-boundary z |
| `io.artifacts` | obs decode, loadings, embeddings, `load_interpretation` |
| `io.peaks` / `io.h5ad` / `io.contract` / `io.meta` | peak names, backed-h5ad reads, contract checks, run records |
| `factorize.train` | fit a DRVI on one GPU or several, and write the run record |
| `factorize.model` | load a fit, latent in requested row order, split responsibility |
| `factorize.kernels` / `.multigpu` | interval kernels, torchrun plumbing |
| `enrich.binary` | the pinned Rust LDSC: resolve, verify, build safe commands |
| `enrich.h2_output` | parse what `ldsc h2` prints |
| `enrich.config` | enrichment config loading and factor selection |
| `scores.cell` / `.aggregate` | the two `CS_i` formulas; group summaries and matrices |
| `viz.*` | style, colour policy, frugal boxes, and the figures |

## Where it came from

The package was extracted from a SEA-AD single-cell ATAC capsule, where it replaced
`code/common/` and four near-identical interpretation notebooks totalling 23 MB — three of
which differed by a single line. The commit history here is that work, replayed onto the
package subtree.

That capsule remains the reference consumer: one `interpretation.ipynb` for every arm,
parameterised by `$SCADS_DRVI_MODEL`, with the dataset's own constants in a site module
(`seaad_site.py`) that is deliberately **not** part of this package. Keeping those
constants outside is what the genericity rule above is protecting.

## Install

```bash
pip install -e .                    # normal case
pip install -e . --no-deps          # inside a mamba-solved prefix

pip install -e 'git+https://github.com/mikecuoco/scads_drvi@main#egg=scads-drvi'
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

## Training a fit

```python
from scads_drvi.config import Project
from scads_drvi.factorize.train import TrainConfig, train

proj = Project(root="/path/to/analysis")
cfg  = TrainConfig(
    adata="/path/to/input.h5ad",
    fit="k48",                  # -> data/factorize/k48/{model,fit.meta.json}
    n_latent=48,
    batch_key="a_covariate",    # recorded, and re-used verbatim when the fit is loaded
    min_fragment=1000,          # depth gate; recorded so a later stage can check it
    devices=4,                  # -> strategy="ddp_find_unused_parameters_true"
    batch_size=128,             # PER DEVICE: the effective batch here is 512
    max_epochs=400,
    seed=0,                     # required for a distributed run, see below
)
train(proj, cfg)
```

or as a command, with the parameters in a preset module rather than in flags:

```bash
python -m scads_drvi.factorize.train production --config my_train_config.py
python -m scads_drvi.factorize.train --config my_train_config.py --list
```

```python
# my_train_config.py
from scads_drvi.factorize.train import TrainConfig
PRESETS = {"train": {"production": TrainConfig(...), "smoke": TrainConfig(smoke_test=True, ...)}}
```

Only the preset name and the module reach the command line, for the reason
`_util/presets.py` gives: a value that can be passed as a flag is a value the run record
cannot recover. The record gets the module's SHA-256 as well as the preset name, because
the preset name alone does not pin the values.

**Multi-GPU here is Lightning's, not `factorize.multigpu`'s.** The post-hoc stages are
inference loops with no `Trainer`, so they own the process group and drive it through
`multigpu`; training hands it to scvi, and Lightning owns it. Calling `multigpu.init()`
first is the one thing not to do. Launch it directly and Lightning spawns; launch it
under `torchrun` or `srun` and the trainer detects that, derives `devices`/`num_nodes`
from the launcher, and refuses a config that names a different device count.

Five ways a distributed fit finishes and means something else, each of which this module
turns into an error or a recorded number:

| | what happens | what it does |
|---|---|---|
| strategy not passed to `train()` | scvi decides on the `DistributedSampler` by `"ddp" in strategy` and nothing else | every rank iterates every cell |
| `early_stopping` with DDP | scvi disables it and warns | the run silently trains the full `max_epochs` |
| unset `seed` | each rank draws its own train/validation split | every held-out cell is another rank's training cell |
| `batch_size` read as global | it is per device | 4×128 is a 512 batch, and the LR/KL schedules are per step |
| code after `train()` | the subprocess launcher re-executes the script | the save and the record run `world_size` times |

`find_unused_parameters=False` is the one knob here that is a speed choice rather than a
correctness one: it drops `ddp` in place of `ddp_find_unused_parameters_true`, so DDP
stops walking the autograd graph before each reduction looking for parameters that got no
gradient. Worth trying, because being wrong costs a raise in the *first* backward pass
rather than a wrong answer at the last epoch — with the one caveat that a parameter used
on some steps and not others hangs instead of raising.

Nothing downstream reads a fit without `fit.meta.json`, so the trainer writes it, from the
same config that made the choices. If the trainer's own world size does not match what was
asked for, the checkpoint is saved and the record deliberately is **not** — `fit_meta`
then refuses the directory rather than handing on a plausible fit nobody knows is wrong.

## Reading a finished run

```python
from scads_drvi.labels import load_labels
from scads_drvi.enrich.ldsc import read_results
from scads_drvi.scores.cell import cs_from_z

labels  = load_labels(arm / "factor_map.tsv", model="my_arm")   # 124 kept of 186
results = read_results(arm / "results", traits=["t1", "t2"], labels=labels)
scores  = cs_from_z(loadings, results.query("trait == 't1'"), model="my_arm", trait="t1")

scores.null       # 0.0 -- read this, never hardcode it beside an axis
scores.label      # "$CS_i$ (z-weighted loading sum)"
```

Three things this buys over the code it replaces:

- `read_results` **raises** on a missing `.results` file rather than skipping it. A skip
  drops the factor from the multiplicity denominator and makes every surviving q-value
  optimistic without saying so.
- `labels.assert_index_dims(...)` rejects a display label (`dim_47/neg`) or an
  annotation name (`k7`) where a loadings column is required. Under a split contract
  display and index names are both `dim_`-shaped, so the wrong one silently selects a
  different column.
- `CellScores` carries its own null. Two different formulas were both called `cs` — a
  z-weighted loading sum (null 0) and a mean enrichment ratio (null 1) — and nothing on
  disk recorded which produced a given column.

At production scale (1,263,026 cells × 96 factors) `cs_from_z` takes ~2 s at ~1.4 GB
peak RSS, computed in row chunks.

## Import surface

The top level imports nothing heavier than the standard library; public names resolve
lazily. Heavy dependencies are confined by directory:

| module | needs |
|---|---|
| `factorize/model.py`, `factorize/multigpu.py` | `torch`, `scvi-tools` (function-local) |
| `factorize/train.py` | `torch`, `scvi-tools`, `anndata` — all function-local, and checked |
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
pytest
```

CI runs it on three dependency stacks (`.github/workflows/tests.yml`), mirroring the three
environments the package is deployed into: numpy 2 / pandas 3 with no optional
dependencies at all, the declared floors (numpy 1.26 / pandas 2.1), and a current stack
with h5py and plotting. That is not belt-and-braces — it is how two real bugs were found,
each of which passed in one stack and failed in another.

**If you vendor this package next to another test suite, collect the two separately.** In
the capsule it came from, the sibling `conftest.py` inserts seven directories onto
`sys.path`; collecting both in one session would let those inserts satisfy an import this
package should satisfy itself — a false green hiding exactly the bug class the package
exists to remove.

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

### The swap, measured end to end

`--overlap-annot` now runs against the Rust binary and writes a `.results` carrying
`Coefficient`, **`Coefficient_std_error`** and **`Coefficient_z-score`** — the columns the
whole significance path is built on. Getting there needed two fixes, neither of which was
the flag itself.

**1. The annotation must be full format.** Our per-factor annot is written *thin* — one
bare column. That was a considered choice: the Python LDSC's `annot_parser` drops
`SNP/CHR/BP/CM` with `errors='ignore'`, so the reference layout was unnecessary. The Rust
binary does not share that behaviour and rejects it outright:

```
Error: Annot file '.../ld/k1/annot.1.annot.gz' has 1 columns;
       expected > 4 (full format: CHR SNP BP CM + annotations)
```

`enrich.annotations.write_full_annot` does the widening, and encodes the two invariants
that are silent when lost: every `.bim` row present **in `.bim` order** (the frequency
mask is applied positionally), and *not* restricted to the HapMap3 subset the LD scores
are printed over.

**2. The reference annotations trip the reader's type inference.** With a full annot the
run gets one step further and dies on `baselineLD` itself:

```
Error: reading annot file '.../baselineLD.1.annot.gz'
       Original error: invalid primitive value found during CSV parsing
```

The cause is exact: `CM` is `0` for the **first 166 rows** of chromosome 1 and
`0.000279324` at row 167. The reader infers `Int64` from its leading sample and then
fails on the first decimal. This is an upstream bug and it affects the *standard*
baselineLD v2.2 release, so it blocks `--overlap-annot` for anyone using the canonical
references. `enrich.annotations.force_decimal` writes the affected columns with an
explicit decimal point, applied to a copy — the read-only reference is never touched.

**What the numbers say.** One arm, one trait, 98 categories, against the Python run:

| quantity | agreement |
|---|---|
| coefficient (τ) | max relative difference **4.1e-05** — effectively exact |
| coefficient SE | median **3.3%**, max 43%; 66% within 5%, 99% within 20% |
| z | correlation **0.9983**, max abs difference 0.23 |
| nominal calls at z > 1.645 | **2 of 98 categories flip** |

So this is **not yet a drop-in replacement.** The point estimates are the same tool; the
jackknife standard errors are not, and two categories change significance class on a
single arm. Before adopting it, run both across every arm and trait and decide whether
that movement is acceptable — do not assume the aggregate conclusions survive because the
coefficients match.

**Two further cautions from the same run.** `Prop._SNPs` comes back as `1.57e7` where a
proportion is expected, and the reported per-annotation `M` sums to `9.4e13`, which is
nonsense for ~1.2M variants; the enrichment columns derived from them should not be
trusted without separate checking. The pipeline consumes only the coefficient columns, so
this does not block it. And peak memory is **8.52 GB**, essentially unchanged from
Python's 8.53 GB — the port buys speed, not headroom. SLURM's sampled `MaxRSS` reported
671 MB for the same run and simply missed the peak; size jobs off `/usr/bin/time -v`.

## Storage

The package never copies, stages or reserves disk. When a large read comes off network
storage it emits a `SlowStorageWarning` suggesting you stage the file on node-local disk
first — a warning, filterable by category, not a policy. Deciding *where* data lives is
the caller's job.
