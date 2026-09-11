# Tutorials

These notebooks walk through the `scads-drvi` pipeline from project setup to publication
figures. Each tutorial is self-contained and can be run on your own data; the code cells
show the full API with realistic argument choices.

The tutorials use a **synthetic plant single-cell dataset** (tissues, cultivars, agronomic
traits) that shares no vocabulary with any real analysis. This is the same dataset that
drives `tests/test_portability.py`, so you can verify the pipeline end to end without
providing external data.

## Pipeline overview

```
h5ad obs + DRVI fit
        │
        ▼
  01 · Project setup        ── configure paths, YAML, env vars
        │
        ▼
  02 · Factorize            ── load fit, compute latent repr, plot factor stats
        │
        ▼
  03 · Enrich               ── annotate peaks, run S-LDSC, read results + BH
        │
        ▼
  04 · Cell scores          ── cs_from_z, group summaries, group matrix
        │
        ▼
  05 · Visualise            ── heritability landscape, UMAP grid, grouped heatmap
```

```{toctree}
:maxdepth: 1

01-project-setup
02-factorize
03-enrich
04-scores
05-viz
```
