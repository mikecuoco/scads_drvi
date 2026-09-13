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

A fit's results live in **one `AnnData`** (see `io.result`), shaped exactly the way
DRVI's own interpretability functions expect it: `obs` = cells, `var` = one row per
latent dimension. There is no separate path-configuration object — every function takes
an explicit path.

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
`drvi.utils.metrics.*` / the model's own `set_latent_dimension_stats` directly rather
than reimplementing it. `pl.*` keeps only what has no DRVI equivalent (S-LDSC
visualizations, frugal boxplots, a factor-factor correlation heatmap) plus the shared
style/save infrastructure.

## Getting started

```bash
pip install -e .
```

```python
from scads_drvi.io.result import build_embed, write_result, read_result

embed = build_embed(model, adata, obs_columns=["cell_type"], umap=umap_coords)
write_result("my_fit.h5ad", embed, provenance={"n_latent": 96})
embed = read_result("my_fit.h5ad")
```

```{toctree}
:maxdepth: 1
:caption: Contents

getting-started
tutorials/index
api/index
changelog
```
