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

## Getting started

```bash
pip install -e .
```

```python
import scads_drvi as sd

proj = sd.Project(root="/path/to/analysis", fit="my_fit", traits=("trait_a", "trait_b"))
proj.enrich_dir("my_arm")
proj.contract("my_arm")["loadings"]
```

```{toctree}
:maxdepth: 1
:caption: Contents

getting-started
tutorials/index
api/index
changelog
```
