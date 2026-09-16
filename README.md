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
  chain on it: obs → the cells x K embed → written to h5ad → factor selection → LDSC
  results → attached into `uns["enrich"]` → BH → per-cell scores → aggregation → ten
  figures. A scan proves no forbidden *names*; this proves no hidden *assumptions*.

## What's in it

A fit's results live in **one `AnnData`**, shaped exactly the way DRVI's own
interpretability functions expect it: `obs` = cells, `var` = one row per latent
dimension. There is no path-configuration object, and no wrapper around training,
loading a model, or reading/writing the result h5ad either — every function takes an
explicit path, a bare read is just `anndata.read_h5ad(path)`, a write is
`AnnData.write_h5ad(path)`, and training/loading uses `scvi.external.DRVI` directly.

| module | role |
|---|---|
| `stats` | one-tailed p, Benjamini–Hochberg, BH-boundary z |
| `enrich.binary` | the pinned Rust LDSC: resolve, verify, build safe commands |
| `enrich.h2_output` | parse what `ldsc h2` prints |
| `enrich.config` | enrichment config loading and factor selection |
| `enrich.ldsc` | read `.results` files into one tidy `dim`/`direction`/`trait` table |
| `scores.cell` / `.aggregate` | the two `CS_i` formulas; group summaries and matrices |
| `annotate.motif` / `.gc` | weighted factor-motif enrichment against a region x motif score database, calibrated by an exact (never sampled) permutation null |
| `pl.*` | style, colour policy, frugal boxes, and every figure -- including its own in-house per-dimension UMAP grid, stats plot and category heatmap |

Where DRVI's own package (`drvi-py`) already computes something -- per-dimension
vanished/order/title stats, factor↔covariate association scores -- this package calls
`model.set_latent_dimension_stats` / `drvi.utils.metrics.*` directly instead of
reimplementing it. Plotting is the one place this used to extend to `drvi.utils.pl.*`
too; it no longer does -- `pl.umap.latent_umap_grid` and
`pl.factors.latent_dimension_stats`/`latent_heatmap` wrap `scanpy.pl.embedding`/
`seaborn.heatmap` directly, ported to match DRVI's own colours and mechanics exactly, so
no `drvi-py` install is needed just to draw a figure that looks like DRVI's own. A plain categorical or continuous embedding
scatter needs no wrapper at all -- `sc.pl.embedding(embed, basis="umap", color=...)` is
already the whole call. (`scanpy` is accordingly now a hard dependency, and the floor is
Python 3.12 / numpy 2 / pandas 2.3 -- scanpy's own floor.)

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
read back everywhere else with `anndata.read_h5ad` directly. Training and loading a
DRVI model is not wrapped by this package at all — call `scvi.external.DRVI` directly,
same as [DRVI's own tutorial](https://drvi.readthedocs.io/latest/tutorials/external/general_pipeline.html):

```python
from scvi.external import DRVI
import anndata as ad

DRVI.setup_anndata(adata, batch_key="donor")
model = DRVI(adata, n_latent=96)
model.train(max_epochs=200)
model.save("my_fit/model", overwrite=True)

embed = ad.AnnData(model.get_latent_representation(adata), obs=adata.obs[["cell_type"]].copy())
embed.var_names = [f"dim_{i}" for i in range(embed.n_vars)]
model.set_latent_dimension_stats(embed)
embed.obsm["X_umap"] = umap.UMAP().fit_transform(embed.X)   # set once computed, not read
embed.var["vanished"]      # DRVI's own per-dimension flag

embed.uns["provenance"] = {"n_latent": 96, "batch_key": "donor"}
embed.write_h5ad("my_fit.h5ad")
```

Loading a checkpoint back is the same `DRVI.load(model_dir, adata=adata)` call, with
`adata` registered via the same `batch_key` the fit was trained with — nothing checks
that for you, so get it from wherever you recorded it (e.g. your own `provenance` dict).

## Reading a finished run

```python
import numpy as np
import pandas as pd

from scads_drvi.enrich.ldsc import read_results
from scads_drvi.scores.cell import cs_from_z

results = read_results(arm / "results", traits=["t1", "t2"], annot2dim=annot2dim)
embed.uns.setdefault("enrich", {})["my_arm"] = {
    "results": results,
    "factor_selection": fmap.drop(columns="annot_index"),
}

# relu(X) -- the ONLY place a pos/neg split is ever materialized, derived on demand
loadings = pd.DataFrame(
    np.clip(embed.X, 0, None), index=embed.obs_names, columns=embed.var_names
)[keep]
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
| `pl/` | `matplotlib`, `seaborn`, `scanpy` (→ `anndata`), function-local |
| `annotate/` | `pyarrow`, `pyranges`, function-local (the `annotate` extra; only `stream_accumulate`/`build_region_map` need them, not the module import) |
| everything else | numpy / pandas / scipy / pyyaml |

This is not cosmetic. The environment that runs the enrichment stages has no torch, no
scvi, no drvi, no matplotlib, no seaborn, no h5py and no anndata, and must still be able
to import and use the loaders, statistics and scoring -- a result h5ad's own I/O is a
plain `anndata.read_h5ad`/`AnnData.write_h5ad` call at the caller's own site, not
something this package wraps. `tests/test_import_surface.py` checks it in a subprocess
with those modules blocked.

## Tests

```bash
pytest
```

CI runs it on three dependency stacks (`.github/workflows/tests.yml`), mirroring the three
environments the package is deployed into: numpy 2 / pandas 3 with no optional
dependencies at all, the declared floors (Python 3.12 / numpy 2 / pandas 2.3 -- scanpy's
own floor, since `pl.*` now wraps `scanpy.pl.embedding` for every embedding figure), and a
current stack with h5py and plotting. That is not belt-and-braces — it is how two real
bugs were found, each of which passed in one stack and failed in another.

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
