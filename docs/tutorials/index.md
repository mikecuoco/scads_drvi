# Tutorial

One notebook, run end to end on a real public dataset: a 10x Genomics PBMC scATAC-seq
sample. It walks the full pipeline in order, with no synthetic stand-ins except where
the underlying inputs (a GWAS trait's summary statistics, a reference LD panel) are
themselves an external data-acquisition step rather than something this package
computes:

```
peaks × cells (10x PBMC scATAC)
        │
        ▼
  Factorize        ── scvi.external.DRVI directly: fit, save the checkpoint, write the
                       result h5ad
        │
        ▼
  Enrich            ── annotate peaks, run S-LDSC, attach_enrich_results
        │
        ▼
  Cell scores       ── cs_from_z, group summaries, group matrix
        │
        ▼
  Visualise         ── DRVI's own plots + this package's own S-LDSC figures
```

Every function above takes an explicit path; a fit's results live in one `AnnData`
(see `enrich.embed`) rather than a directory tree a configuration object derives
paths into.

```{toctree}
:maxdepth: 1

pbmc
```
