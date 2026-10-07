# Getting Started

## Installation

```bash
# standard editable install
pip install -e .

# inside a mamba-solved environment — do NOT let pip re-resolve numpy/torch
pip install -e . --no-deps

# directly from GitHub
pip install 'git+https://github.com/mikecuoco/scads_drvi@main#egg=scads-drvi'
```

:::{note}
If you are working inside a conda/mamba environment with numpy and torch already solved,
always use `--no-deps`. Letting pip re-resolve those packages across a solved prefix will
break the environment.
:::

## Install documentation dependencies

```bash
pip install -e '.[docs]'
cd docs && make html
# open docs/_build/html/index.html
```

## The result object

A fit's results live in **one `AnnData`**, shaped exactly the way DRVI's own
interpretability functions expect it: `obs` = cells, `var` = one row per latent
dimension, `X` = the signed latent representation. There is no separate
path-configuration object — every function takes an explicit path.

This package does not wrap training or loading a DRVI model at all: call
`scvi.external.DRVI` directly, the same way
[DRVI's own tutorial](https://drvi.readthedocs.io/latest/tutorials/external/general_pipeline.html)
does, and write the result with a plain `AnnData.write_h5ad` call -- there is no h5ad
wrapper here either:

```python
from scvi.external import DRVI
import anndata as ad

DRVI.setup_anndata(adata, batch_key="donor")
model = DRVI(adata, n_latent=96)
model.train(max_epochs=200)
model.save("my_fit/model", overwrite=True)

embed = ad.AnnData(
    model.get_latent_representation(adata),
    obs=adata.obs[["cell_type", "tissue", "donor"]].copy(),
)
embed.var_names = [f"dr_{i}" for i in range(embed.n_vars)]
model.set_latent_dimension_stats(embed)   # vanished, order, title, and friends
embed.obsm["X_umap"] = umap.UMAP().fit_transform(embed.X)   # set once computed, not read

embed.uns["provenance"] = {"n_latent": 96, "batch_key": "donor", "model_dir": "my_fit/model"}
embed.write_h5ad("my_fit_cells.h5ad")
```

`embed.var` is populated by DRVI's own per-dimension statistics
(`model.set_latent_dimension_stats`), so `embed` is immediately usable with
`drvi.utils.metrics.*`, with a plain scanpy embedding scatter
(`sc.pl.embedding(embed, basis="umap", color=...)`), and with this package's own
in-house plotting -- `scads_drvi.pl.umap.latent_umap_grid` and
`scads_drvi.pl.factors.latent_dimension_stats`/`latent_heatmap` -- which wrap
`scanpy.pl.embedding`/`seaborn.heatmap` directly and need no `drvi-py` install.

### The other half: peaks x factors

`embed` is cells x factors. A fit's other half, peaks x factors, is a separate
`AnnData` -- this package keeps the two apart rather than putting both axes on one
object, because cells x factors is the shape every downstream function here (and
DRVI's own interpretability functions) expects `embed` to have.

DRVI is a nonlinear VAE, so there is no linear loadings matrix to read off a weight
tensor. `get_effect_of_splits_within_distribution` is DRVI's own way of asking "how
much does perturbing factor k move peak j's reconstruction", aggregated over cells
weighted by how active each factor is in each cell:

```python
effect = model.get_effect_of_splits_within_distribution(
    directional=False, aggregations="max"
)["max"]   # (n_latent, n_peaks), non-negative

feature_loadings = ad.AnnData(effect.T, obs=adata.var[[]].copy())   # peaks x factors
feature_loadings.obs_names = adata.var_names   # peak ids, e.g. "chr1:1000-1500"
feature_loadings.var_names = embed.var_names   # same dr_0..dr_{K-1} naming as embed
feature_loadings.write_h5ad("my_fit_peaks.h5ad")
```

Loading a checkpoint back is the same `DRVI.load` call DRVI's own tutorial shows —
register `adata` with the **same** `batch_key` the fit was trained with (nothing checks
this for you; get it from wherever you recorded it, e.g. your own `provenance` dict):

```python
from scvi.external import DRVI
import anndata as ad

DRVI.setup_anndata(adata, batch_key="donor")
model = DRVI.load("my_fit/model", adata=adata)

embed = ad.AnnData(
    model.get_latent_representation(adata),
    obs=adata.obs[["cell_type", "tissue", "donor"]].copy(),
)
embed.var_names = [f"dr_{i}" for i in range(embed.n_vars)]
model.set_latent_dimension_stats(embed)
embed.uns["provenance"] = {"n_latent": 96, "batch_key": "donor"}
embed.write_h5ad("my_fit_cells.h5ad")
```

## Reading a finished run

```python
import anndata as ad

embed = ad.read_h5ad("my_fit_cells.h5ad")
embed.var["vanished"]          # DRVI's own per-dimension flag
embed.obsm["X_umap"]           # the embedding, if one was attached
```

## Running S-LDSC enrichment

S-LDSC has no DRVI equivalent, so this package still runs it and reads its output. It
runs through the `ldsc.py` CLI from [CBIIT/ldsc](https://github.com/CBIIT/ldsc), a
Python 3 port of bulik/ldsc (branch `ldsc313`). It is not a dependency of this package:
install it in its own environment, since its `setup.py` pins exact numpy/pandas/scipy
versions -- `pip install git+https://github.com/CBIIT/ldsc.git@ldsc313`.

First, select which dimensions get enriched and under which `k` index, and write each
one's non-negative loadings out as a thin annotation (a bare annotation column, no
`CHR SNP BP CM` — `ldsc.py --l2 --thin-annot` accepts that shape directly):

```python
import subprocess

import pandas as pd

from scads_drvi.enrich.annotations import read_bim
from scads_drvi.enrich.config import kept_dims, select_factors
from scads_drvi.enrich.ldsc import read_results

fmap = select_factors(embed.var, list(embed.var_names))
keep = kept_dims(fmap)                       # dims that survive vanished-filtering
annot2factor = {f"k{i + 1}": factor for i, factor in enumerate(keep)}

bim = read_bim("1000G.EUR.QC.1.bim")

for annot, factor in annot2factor.items():
    thin = ...  # peaks -> per-SNP annotation for `factor`, one 0/1+ column, in bim order
    pd.DataFrame({annot: thin}).to_csv(f"annot/{annot}.1.annot.gz", sep="\t", index=False)
    # --thin-annot accepts that shape directly; --out writes .l2.ldscore.gz/.M/.M_5_50.
    # --annot takes the full file name.
    subprocess.run([
        "ldsc.py", "--l2", "--bfile", "1000G.EUR.QC.1",
        "--annot", f"annot/{annot}.1.annot.gz", "--thin-annot",
        "--ld-wind-cm", "1.0", "--out", f"ld/{annot}.1",
    ], check=True)
    # --print-coefficients writes results/trait_a/{annot}.results.
    subprocess.run([
        "ldsc.py", "--h2", "trait_a.sumstats.gz", "--ref-ld-chr", f"ld/{annot}.",
        "--w-ld-chr", "weights.", "--overlap-annot", "--frqfile-chr", "1000G.EUR.QC.",
        "--print-coefficients", "--out", f"results/trait_a/{annot}",
    ], check=True)
```

Reading the results back attaches one tidy table, keyed by `factor` and `direction`, into
the same h5ad (there is no persisted `dim_j/pos`/`dim_j/neg` split; a directional
loadings view is derived on demand, see below):

```python
results = read_results(results_root, traits=("trait_a", "trait_b"), annot2factor=annot2factor)
embed.uns.setdefault("enrich", {})["my_arm"] = {
    "results": results,
    "factor_selection": fmap.drop(columns="annot_index"),
}
embed.write_h5ad("my_fit_cells.h5ad")

arm = embed.uns["enrich"]["my_arm"]
arm["results"].query("direction == 'pos' and trait == 'trait_a'")
```

### A real run: every factor, both directions, every chromosome

The loop above is a single-chromosome toy example. A real run is the same loop, over
dims × directions × chromosomes, plus a baseline reference layered under each
factor's own annotation in `--ref-ld-chr` (`f"ld/{annot}.,{baseline_stem}"`) and an
`if Path(...).exists(): continue` guard on each `.l2.ldscore.gz`/`.results` output so
re-running only computes what isn't already there — hours, not seconds, so
resumability matters. There is no dedicated sweep class for this; see
`docs/tutorials/pbmc.ipynb`'s "Enrich" section for the full real example.

`scads-drvi` doesn't manage reference panels itself — `baseline_stem` is just a path
you supply. For a GWAS run on UK Biobank itself, in-sample LD is preferred over an
external 1000G panel; the Alkes-group baseline-LF v2.2 UKB reference is a plain
download:

```bash
wget https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/baselineLF_v2.2.UKB.tar.gz
tar xzf baselineLF_v2.2.UKB.tar.gz
```

It extracts to `baselineLF_v2.2.UKB/`, but the per-chromosome file prefix inside it
drops the underscore — `baseline_stem` is
`.../baselineLF_v2.2.UKB/baselineLF2.2.UKB.`, not `baselineLF_v2.2.UKB.`. The
archive is ~11 GB; download it to scratch storage, not a small home directory.

## Annotating peaks (motif enrichment)

Independent of the LDSC path, and works on **peaks**, not cells: it reads back the
`feature_loadings` h5ad written above rather than `embed`, matches it onto a region x
motif score database (aertslab's public SCREEN cisTarget database, resolved and cached
by `annotate.resources`), and scores it against an exact, never-sampled permutation
null, stratified by each region's GC content and width. It reuses the `fmap` the
enrichment step above computed, so a factor `select_factors` dropped there (e.g.
`vanished`) is dropped here too:

```python
import anndata as ad
import numpy as np

from scads_drvi.annotate.motif import build_region_map, gc_width_strata, regionise, weighted_motif_enrichment
from scads_drvi.annotate.resources import ensure_screen_database, read_score_db_regions
from scads_drvi.enrich.config import kept_feature_loadings

feature_loadings = ad.read_h5ad("my_fit_peaks.h5ad")   # peaks x dims
peak_loadings = kept_feature_loadings(feature_loadings, fmap)     # peaks x keep, same dims as above
W = np.clip(peak_loadings.to_numpy(), 0, None).T                  # (K, n_peaks), non-negative

db_path = ensure_screen_database("hg38", "v10_clust", allow_download=True)   # ~14 GB, cached after
used_cols, peak_rows, region_names = build_region_map(
    peak_loadings.index, read_score_db_regions(db_path)
)
keep_mask, strata_ix, n_strata = gc_width_strata(region_names, "genome.fa")

Wr = regionise(W, peak_rows)[:, keep_mask]   # (K, n_regions), averaged onto db regions
rows, meta = weighted_motif_enrichment(
    Wr, labels=peak_loadings.columns, used_cols=used_cols[keep_mask],
    region_names=[r for r, k in zip(region_names, keep_mask) if k],
    strata_ix=strata_ix, n_strata=n_strata, db_path=db_path,
)
# one row per (factor, motif): weighted_score, nes, nes_unstratified
```

## Computing cell scores

```python
import numpy as np
import pandas as pd

from scads_drvi.scores.cell import cs_from_z

primary = arm["results"].loc[arm["results"]["trait"] == "trait_a"]
# relu(X) -- the ONLY place a pos/neg split is ever materialized, derived on demand
loadings = pd.DataFrame(
    np.clip(embed.X, 0, None), index=embed.obs_names, columns=embed.var_names
)[keep]
scores = cs_from_z(loadings, primary, model="my_arm", trait="trait_a")

scores.null    # 0.0  — read this; never hardcode it beside an axis
scores.label   # "$CS_i$ (z-weighted loading sum)"
```

## Import surface

`import scads_drvi` loads nothing heavier than the standard library. Heavy dependencies
are confined by directory and imported inside functions:

| module | needs |
|---|---|
| `pl/` | `matplotlib`, `seaborn`, `scanpy` (→ `anndata`), function-local |
| everything else | `numpy` / `pandas` / `scipy` / `pyyaml` |

This matters because the environment that runs enrichment stages has no torch, scvi,
drvi, matplotlib, seaborn, h5py or anndata, and must still import and use the loaders,
statistics, and scoring. A result h5ad's own I/O is a plain `anndata.read_h5ad`/
`AnnData.write_h5ad` call at the caller's own site -- there is no wrapper for it here.

## Running the tests

```bash
pytest
```

CI runs on three dependency stacks:

1. numpy 2 / pandas 3, no optional dependencies
2. Declared floors (numpy 1.26 / pandas 2.1)
3. Current stack with h5py and plotting

If you vendor this package next to another test suite, collect the two separately.
