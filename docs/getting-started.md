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
embed.var_names = [f"dim_{i}" for i in range(embed.n_vars)]
model.set_latent_dimension_stats(embed)   # vanished, order, title, and friends
embed.obsm["X_umap"] = umap.UMAP().fit_transform(embed.X)   # set once computed, not read

embed.uns["provenance"] = {"n_latent": 96, "batch_key": "donor", "model_dir": "my_fit/model"}
embed.write_h5ad("my_fit.h5ad")
```

`embed.var` is populated by DRVI's own per-dimension statistics
(`model.set_latent_dimension_stats`), so `embed` is immediately usable with
`drvi.utils.metrics.*`, with a plain scanpy embedding scatter
(`sc.pl.embedding(embed, basis="umap", color=...)`), and with this package's own
in-house plotting -- `scads_drvi.pl.umap.latent_umap_grid` and
`scads_drvi.pl.factors.latent_dimension_stats`/`latent_heatmap` -- which wrap
`scanpy.pl.embedding`/`seaborn.heatmap` directly and need no `drvi-py` install.

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
embed.var_names = [f"dim_{i}" for i in range(embed.n_vars)]
model.set_latent_dimension_stats(embed)
embed.uns["provenance"] = {"n_latent": 96, "batch_key": "donor"}
embed.write_h5ad("my_fit.h5ad")
```

## Reading a finished run

```python
import anndata as ad

embed = ad.read_h5ad("my_fit.h5ad")
embed.var["vanished"]          # DRVI's own per-dimension flag
embed.obsm["X_umap"]           # the embedding, if one was attached
```

## Attaching enrichment results

S-LDSC has no DRVI equivalent, so this package still runs it and reads its output — but
the result now lives in the same h5ad, as one tidy table keyed by `dim` and `direction`
(there is no persisted `dim_j/pos`/`dim_j/neg` split; a directional loadings view is
derived on demand, see below):

```python
from scads_drvi.enrich.config import kept_dims, latent_stats_from_embed, select_factors
from scads_drvi.enrich.ldsc import read_results

stats = latent_stats_from_embed(embed)
fmap = select_factors(stats, list(embed.var_names))
keep = kept_dims(fmap)                       # dims that survive vanished-filtering
annot2dim = {f"k{i + 1}": dim for i, dim in enumerate(keep)}

results = read_results(results_root, traits=("trait_a", "trait_b"), annot2dim=annot2dim)
embed.uns.setdefault("enrich", {})["my_arm"] = {
    "results": results,
    "factor_selection": fmap.drop(columns="annot_index"),
}
embed.write_h5ad("my_fit.h5ad")

arm = embed.uns["enrich"]["my_arm"]
arm["results"].query("direction == 'pos' and trait == 'trait_a'")
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
