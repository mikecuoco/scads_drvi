---
hide-toc: true
---

# scads-drvi

**DRVI factorization → S-LDSC heritability enrichment → per-cell disease scores** for
single-cell ATAC data.

The package is **method-generic**. Nothing in it names a tissue, brain region, cell
grouping, donor-cohort variable, trait, GWAS study or obs column — those are all
caller-supplied. Two tests enforce that rather than trusting it:

- `tests/test_generic.py` — an AST scan over every module rejecting dataset vocabulary in
  string literals and identifiers.
- `tests/test_portability.py` — builds a complete synthetic *plant* single-cell analysis
  (tissues, cultivars, agronomic traits — no shared vocabulary at all) and runs the whole
  pipeline end to end.

## What's in it

A fit's results live in **one `AnnData`** (see `factorize.result`), shaped exactly the
way DRVI's own interpretability functions expect it: `obs` = cells, `var` = one row per
latent dimension. There is no separate path-configuration object — every function takes
an explicit path, and a bare read is just `anndata.read_h5ad(path)`.

| module | role |
|---|---|
| `factorize.model` | load or **train** a fit, latent in requested row order, split responsibility, torchrun/multi-GPU plumbing |
| `factorize.result` | `build_embed`, `write_result`, `attach_enrich_results`, `directional_loadings` |
| `factorize.h5ad` | backed-CSR reads, cell gating by depth |
| `stats` | one-tailed p, Benjamini–Hochberg, BH-boundary z |
| `factorize.kernels` | interval kernels |
| `enrich.binary` | the pinned Rust LDSC: resolve, verify, build safe commands |
| `enrich.h2_output` | parse what `ldsc h2` prints |
| `enrich.config` | enrichment config loading and factor selection |
| `enrich.ldsc` | read `.results` files into one tidy `dim`/`direction`/`trait` table |
| `scores.cell` / `.aggregate` | the two `CS_i` formulas; group summaries and matrices |
| `pl.*` | style, colour policy, frugal boxes, and the figures with no DRVI equivalent |

Where DRVI's own package (`drvi-py`) already computes something — per-dimension
vanished/order/title stats, a per-factor UMAP grid, a factor-value-by-category heatmap,
factor↔covariate association scores — this package calls `drvi.utils.pl.*` /
`drvi.utils.metrics.*` / the model's own `set_latent_dimension_stats` directly rather
than reimplementing it. `pl.*` keeps only what has no DRVI equivalent (S-LDSC
visualizations, frugal boxplots, a factor-factor correlation heatmap) plus the shared
style/save infrastructure.

## Getting started

```bash
pip install -e .
```

```python
from scads_drvi.factorize.model import train_fit

embed = train_fit(
    adata, n_latent=96, obs_columns=["cell_type"],
    result_path="my_fit.h5ad", model_dir="my_fit/model",
)
embed.obsm["X_umap"] = umap.UMAP().fit_transform(embed.X)   # set once computed, not read
```

```{toctree}
:maxdepth: 1
:caption: Contents

getting-started
tutorials/index
api/index
changelog
```
