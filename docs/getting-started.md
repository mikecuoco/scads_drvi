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

```python
from scads_drvi.factorize.model import train_fit

embed = train_fit(
    adata,                      # registered against nothing yet -- train_fit does it
    n_latent=96,
    batch_key="donor",
    obs_columns=["cell_type", "tissue", "donor"],
    result_path="my_fit.h5ad",
    model_dir="my_fit/model",
)
embed.obsm["X_umap"] = umap.UMAP().fit_transform(embed.X)   # set once computed, not read
```

`train_fit` trains, saves the checkpoint to `model_dir`, and writes the canonical result
h5ad to `result_path` in one call, returning the written object. `embed.var` is
populated with DRVI's own per-dimension statistics (`model.set_latent_dimension_stats`)
— `vanished`, `order`, `title`, and friends — so `embed` is immediately usable with
`drvi.utils.pl.*` and `drvi.utils.metrics.*`.

Given a checkpoint trained elsewhere (e.g. a multi-GPU `torchrun` job using
`scads_drvi.factorize.model.init_ranks` and friends), `load_fit` picks it back up and
`build_embed`/`write_result` do the same two steps split apart:

```python
from scads_drvi.factorize.model import load_fit, fit_meta, setup_anndata_like
from scads_drvi.factorize.result import build_embed, write_result

meta = fit_meta("my_fit/model")
setup_anndata_like(adata, meta)
model = load_fit("my_fit/model", adata=adata)
embed = build_embed(model, adata, obs_columns=["cell_type", "tissue", "donor"])
write_result("my_fit.h5ad", embed, provenance=meta.raw)
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
from scads_drvi.factorize.result import attach_enrich_results, write_result

stats = latent_stats_from_embed(embed)
fmap = select_factors(stats, list(embed.var_names))
keep = kept_dims(fmap)                       # dims that survive vanished-filtering
annot2dim = {f"k{i + 1}": dim for i, dim in enumerate(keep)}

results = read_results(results_root, traits=("trait_a", "trait_b"), annot2dim=annot2dim)
attach_enrich_results(embed, "my_arm", results, factor_selection=fmap)
write_result("my_fit.h5ad", embed, provenance=embed.uns["provenance"])

arm = embed.uns["enrich"]["my_arm"]
arm["results"].query("direction == 'pos' and trait == 'trait_a'")
```

## Computing cell scores

```python
from scads_drvi.factorize.result import directional_loadings
from scads_drvi.scores.cell import cs_from_z

primary = arm["results"].loc[arm["results"]["trait"] == "trait_a"]
loadings = directional_loadings(embed, "pos")[keep]   # relu(X), the ONLY place a
                                                        # pos/neg split is materialized
scores = cs_from_z(loadings, primary, model="my_arm", trait="trait_a")

scores.null    # 0.0  — read this; never hardcode it beside an axis
scores.label   # "$CS_i$ (z-weighted loading sum)"
```

## Import surface

`import scads_drvi` loads nothing heavier than the standard library. Heavy dependencies
are confined by directory and imported inside functions:

| module | needs |
|---|---|
| `factorize/model.py` | `torch`, `scvi-tools` (function-local; also carries torchrun/multi-GPU plumbing) |
| `factorize/result.py` | `anndata` (function-local) |
| `pl/` | `matplotlib`, `seaborn` (function-local) |
| `factorize/h5ad.py` | `h5py` at module scope |
| everything else | `numpy` / `pandas` / `scipy` / `pyyaml` |

This matters because the environment that runs enrichment stages has no torch, scvi,
drvi, matplotlib, seaborn, h5py or anndata, and must still import and use the loaders,
statistics, and scoring.

## Running the tests

```bash
pytest
```

CI runs on three dependency stacks:

1. numpy 2 / pandas 3, no optional dependencies
2. Declared floors (numpy 1.26 / pandas 2.1)
3. Current stack with h5py and plotting

If you vendor this package next to another test suite, collect the two separately.
