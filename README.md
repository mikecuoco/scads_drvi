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
  chain on it: obs → `build_embed`/`write_result` → factor selection → LDSC results →
  `attach_enrich_results` → BH → per-cell scores → aggregation → seven figures. A scan
  proves no forbidden *names*; this proves no hidden *assumptions*.

## What's in it

A fit's results live in **one `AnnData`** (`io.result`), shaped exactly the way DRVI's
own interpretability functions expect it: `obs` = cells, `var` = one row per latent
dimension. There is no path-configuration object — every function takes an explicit
path.

| module | role |
|---|---|
| `io.result` | `build_embed`, `write_result`, `read_result`, `attach_enrich_results`, `directional_loadings` |
| `stats` | one-tailed p, Benjamini–Hochberg, BH-boundary z |
| `io.artifacts` | obs decode, the loadings/embedding TSV readers |
| `io.peaks` / `io.h5ad` | peak names, backed-h5ad reads |
| `factorize.model` | load a fit, latent in requested row order, split responsibility |
| `factorize.kernels` / `.multigpu` | interval kernels, torchrun plumbing |
| `enrich.binary` | the pinned Rust LDSC: resolve, verify, build safe commands |
| `enrich.h2_output` | parse what `ldsc h2` prints |
| `enrich.config` | enrichment config loading and factor selection |
| `enrich.ldsc` | read `.results` files into one tidy `dim`/`direction`/`trait` table |
| `scores.cell` / `.aggregate` | the two `CS_i` formulas; group summaries and matrices |
| `pl.*` | style, colour policy, frugal boxes, and the figures with no DRVI equivalent |

Where DRVI's own package (`drvi-py`) already computes something — per-dimension
vanished/order/title stats, a per-factor UMAP grid, a factor-value-by-category heatmap,
factor↔covariate association scores — this package calls `drvi.utils.pl.*` /
`drvi.utils.metrics.*` / the model's own `set_latent_dimension_stats` directly instead of
reimplementing it.

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

Every function takes an explicit path. A fit's results are one `AnnData`, built once and
read back everywhere else:

```python
from scads_drvi.io.result import build_embed, write_result, read_result

embed = build_embed(model, adata, obs_columns=["cell_type"], umap=umap_coords)
write_result("my_fit.h5ad", embed, provenance={"n_latent": 96, "batch_key": "donor"})

embed = read_result("my_fit.h5ad")
embed.var["vanished"]      # DRVI's own per-dimension flag
```

## Reading a finished run

```python
from scads_drvi.enrich.ldsc import read_results
from scads_drvi.io.result import attach_enrich_results, directional_loadings
from scads_drvi.scores.cell import cs_from_z

results = read_results(arm / "results", traits=["t1", "t2"], annot2dim=annot2dim)
attach_enrich_results(embed, "my_arm", results, factor_selection=fmap)

loadings = directional_loadings(embed, "pos")[keep]   # the ONLY place a pos/neg split
                                                        # is ever materialized
scores = cs_from_z(loadings, results.query("trait == 't1'"), model="my_arm", trait="t1")

scores.null       # 0.0 -- read this, never hardcode it beside an axis
scores.label      # "$CS_i$ (z-weighted loading sum)"
```

Two things this buys over the code it replaces:

- `read_results` **raises** on a missing `.results` file rather than skipping it. A skip
  drops the factor from the multiplicity denominator and makes every surviving q-value
  optimistic without saying so.
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
| `io/result.py` | `anndata` (function-local) |
| `pl/` | `matplotlib`, `seaborn` (function-local) |
| `io/h5ad.py` | `h5py` at module scope — it *is* the h5ad reader |
| `io/artifacts.py` | `h5py`, function-local, so it imports without one |
| everything else | numpy / pandas / scipy / pyyaml |

This is not cosmetic. The environment that runs the enrichment stages has no torch, no
scvi, no drvi, no matplotlib, no seaborn, no h5py and no anndata, and must still be able
to import and use the loaders, statistics and scoring. `tests/test_import_surface.py`
checks it in a subprocess with those modules blocked.

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
