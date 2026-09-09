# scads_drvi

DRVI factorization → S-LDSC heritability enrichment → per-cell disease scores, for
single-cell ATAC data.

The package is **method-generic**. Nothing in it names a tissue, brain region, cell
grouping, donor-cohort variable, trait, GWAS study or obs column — those are all
caller-supplied. Two tests enforce that rather than trusting it:

- `tests/test_generic.py` — an AST scan over every module rejecting dataset vocabulary in
  string literals and identifiers.
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

### What still blocks the swap, measured

The wrapper drives the binary correctly and its **point estimates are exact**: over the
98 categories of one production arm, `max |tau_rust - tau_python| = 0.0`, with no
`--overlap-annot` and no tolerance. What is missing is the *uncertainty*.

**`--overlap-annot` rejects our annotation files.** This, not the absence of a flag, is
the blocker:

```
Error: Annot file '.../ld/k1/annot.1.annot.gz' has 1 columns;
       expected > 4 (full format: CHR SNP BP CM + annotations)
```

The per-factor annot is written **thin** -- a single column holding just the indicator.
That was a deliberate choice, not an oversight: the Python LDSC's `annot_parser` drops
`SNP/CHR/BP/CM` with `errors='ignore'`, so a bare annotation column is accepted as-is and
the reference layout is unnecessary. The Rust binary does not share that behaviour and
requires the full `CHR BP SNP CM + K` form the references use (baselineLD ships 101
columns).

So the fix belongs in the step that splits one joint annot into per-factor files --
`split_annot`, which today writes `joint[[name]]` and would need to prepend the four
identifier columns from that chromosome's `.bim`. Two constraints carry over unchanged
and are easy to lose in a rewrite: every bim row must be present, **in bim order**,
because the MAF mask is applied positionally against the `.frq` file; and the rows must
*not* be restricted to the HapMap3 subset the LD scores are printed over.

**The two "print the uncertainty" flags do not substitute for it.**

- `--print-cov` emits a `99x99` jackknife covariance (98 annotations plus the
  intercept). It is *not* the quantity LDSC divides into a coefficient SE:
  `sqrt(diag(cov))/SE_python` should be one constant across categories and instead
  scatters from `4.2e5` to `7.0e5` -- a 57% spread, so no single `Nbar` reconciles them.
- `--print-delete-vals` writes `200 blocks x 99 params` **to stdout, not to a file**, at
  six decimal places. That is lossless for ordinary values and useless for the case at
  hand: a tau of `-2.4e-17` prints as `-0.000000`.

**The SE convention itself is settled**, checked against the Python tool's own output.
A block jackknife over the partitioned delete values,

```
se = sqrt((n - 1) / n * sum((delete_i - mean(delete))^2))
```

reproduces the logged `Coefficient SE` for all 98 categories to `8e-05` relative -- which
is the rounding in the log's own 5-significant-figure printout, not a disagreement. The
partitioned delete values are **already in tau units**: the ratio is exactly 1.0, so
there is no `Nbar` division, contrary to what the covariance route suggests.

**Memory is unchanged by the port.** `/usr/bin/time -v` records a **8.52 GB** peak,
against the Python LDSC's 8.53 GB. Note that SLURM's sampled `MaxRSS` reported 671 MB
for the same run and is simply wrong here -- it missed the peak. Size jobs off the
former.

## Storage

The package never copies, stages or reserves disk. When a large read comes off network
storage it emits a `SlowStorageWarning` suggesting you stage the file on node-local disk
first — a warning, filterable by category, not a policy. Deciding *where* data lives is
the caller's job.
