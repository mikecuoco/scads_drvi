# Tutorials

These notebooks walk through the `scads-drvi` pipeline from a trained DRVI fit to
publication figures. Each tutorial is self-contained and can be run on your own data;
the code cells show the full API with realistic argument choices.

The tutorials use a **synthetic plant single-cell dataset** (tissues, cultivars, agronomic
traits) that shares no vocabulary with any real analysis. This is the same dataset that
drives `tests/test_portability.py`, so you can verify the pipeline end to end without
providing external data.

## Pipeline overview

```
h5ad obs + DRVI fit
        │
        ▼
  01 · Factorize            ── load fit, build_embed, write_result
        │
        ▼
  02 · Enrich                ── annotate peaks, run S-LDSC, attach_enrich_results
        │
        ▼
  03 · Cell scores            ── cs_from_z, group summaries, group matrix
        │
        ▼
  04 · Visualise              ── DRVI's own plots + this package's own S-LDSC figures
```

There is no separate "project setup" step: every function above takes an explicit path,
and a fit's results live in one `AnnData` (see `io.result`) rather than a directory tree
a configuration object derives paths into.

```{toctree}
:maxdepth: 1

01-factorize
02-enrich
03-scores
04-viz
```
