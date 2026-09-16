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

A fit's results live in **one `AnnData`**, shaped exactly the way DRVI's own
interpretability functions expect it: `obs` = cells, `var` = one row per latent
dimension. There is no separate path-configuration object, and no wrapper around
training, loading a model, or reading/writing the result h5ad either — every function
takes an explicit path, a bare read is just `anndata.read_h5ad(path)`, a write is
`AnnData.write_h5ad(path)`, and training/loading uses `scvi.external.DRVI` directly
(see Getting started).

| module | role |
|---|---|
| `stats` | one-tailed p, Benjamini–Hochberg, BH-boundary z |
| `enrich.binary` | the pinned Rust LDSC: resolve, verify, build safe commands |
| `enrich.h2_output` | parse what `ldsc h2` prints |
| `enrich.config` | enrichment config loading and factor selection |
| `enrich.ldsc` | read `.results` files into one tidy `dim`/`direction`/`trait` table |
| `scores.cell` / `.aggregate` | the two `CS_i` formulas; group summaries and matrices |
| `annotate.motif` / `.gc` | weighted factor-motif enrichment against a region x motif score database, calibrated by an exact (never sampled) permutation null |
| `annotate.resources` | resolves aertslab's public SCREEN cisTarget score database (cache/verify/opt-in download, ~14 GB) |
| `pl.*` | style, colour policy, frugal boxes, and every figure — including its own in-house per-dimension UMAP grid, stats plot and category heatmap |

Where DRVI's own package (`drvi-py`) already computes something — per-dimension
vanished/order/title stats, factor↔covariate association scores — this package calls
`model.set_latent_dimension_stats` / `drvi.utils.metrics.*` directly rather than
reimplementing it. Plotting used to extend that to `drvi.utils.pl.*` too; it no longer
does — `pl.umap.latent_umap_grid` and `pl.factors.latent_dimension_stats`/
`latent_heatmap` wrap `scanpy.pl.embedding`/`seaborn.heatmap` directly, ported to match
DRVI's own colours and mechanics exactly, so no `drvi-py` install is needed just to draw
a figure that looks like DRVI's own. A plain categorical or continuous embedding scatter needs no wrapper at all —
`sc.pl.embedding(embed, basis="umap", color=...)` is already the whole call. (`scanpy`
is accordingly a hard dependency now, and the floor is Python
3.12 / numpy 2 / pandas 2.3 — scanpy's own floor.)

## Getting started

```bash
pip install -e .
```

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
embed.uns["provenance"] = {"n_latent": 96}
embed.write_h5ad("my_fit.h5ad")
```

```{toctree}
:maxdepth: 1
:caption: Contents

getting-started
tutorials/index
api/index
changelog
```
