# scads_drvi

DRVI factorization → S-LDSC heritability enrichment → per-cell disease scores, for
single-cell ATAC data.

## Install

```bash
pip install -e .                    # normal case
pip install -e . --no-deps          # inside a mamba-solved prefix
```

`--no-deps` matters in a conda/mamba environment: letting pip re-resolve `numpy`/`torch`
across a solved prefix will break it. Floor: Python 3.12 / numpy 2 / pandas 2.3
(scanpy's own floor).

## Getting started

Four stages, each callable on its own — no orchestrator strings them together, and
every function takes an explicit path or in-memory frame rather than a directory a
config object derives paths into.

### 1. Factorize (DRVI)

Training and loading a DRVI model isn't wrapped by this package — call
`scvi.external.DRVI` directly, same as
[DRVI's own tutorial](https://drvi.readthedocs.io/latest/tutorials/external/general_pipeline.html):

```python
from scvi.external import DRVI
import anndata as ad

DRVI.setup_anndata(adata, batch_key="donor")
model = DRVI(adata, n_latent=96)
model.train(max_epochs=200)
model.save("my_fit/model", overwrite=True)

embed = ad.AnnData(model.get_latent_representation(adata), obs=adata.obs[["cell_type"]].copy())
embed.var_names = [f"dr_{i}" for i in range(embed.n_vars)]
model.set_latent_dimension_stats(embed)
embed.obsm["X_umap"] = umap.UMAP().fit_transform(embed.X)   # set once computed, not read

embed.uns["provenance"] = {"n_latent": 96, "batch_key": "donor"}
embed.write_h5ad("my_fit_cells.h5ad")
```

The cells x factors matrix above (`embed`) is one half of a fit; the other is peaks x
factors. DRVI is a nonlinear VAE, so there is no linear loadings matrix to read off a
weight tensor — `get_effect_of_splits_within_distribution` is DRVI's own way of asking
"how much does perturbing factor k move peak j's reconstruction", aggregated over cells
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

Keeping the two in separate files rather than one `AnnData` with both axes is
deliberate: `embed` stays cells x factors, the shape every downstream function in this
package (and DRVI's own interpretability functions) expects.

Loading a checkpoint back is `DRVI.load(model_dir, adata=adata)`, with `adata`
registered via the same `batch_key` the fit was trained with — get it from wherever you
recorded it (e.g. your own `provenance` dict).

### 2. Enrich (S-LDSC)

Starting from the `embed` written above: `select_factors` reads the `vanished` flags
`set_latent_dimension_stats` already wrote onto `embed.var` and decides which
dimensions get enriched, under which `k` index. Each kept dimension's non-negative
loadings (`np.clip(embed.X, 0, None)`, one column per `keep`) become a per-SNP
annotation, widened to the full format the Rust binary requires, then run through
`l2`/`h2`:

```python
import numpy as np

from scads_drvi.enrich.annotations import read_bim, write_full_annot
from scads_drvi.enrich.config import kept_dims, latent_stats_from_embed, select_factors
from scads_drvi.enrich.run import LdscRun

stats = latent_stats_from_embed(embed)
fmap = select_factors(stats, list(embed.var_names))
keep = kept_dims(fmap)                       # dims that survive vanished-filtering
annot2dim = {f"k{i + 1}": dim for i, dim in enumerate(keep)}

bim = read_bim("1000G.EUR.QC.1.bim")
loadings = np.clip(embed[:, keep].X, 0, None)   # relu -- annotation weights are non-negative

for annot, dim in annot2dim.items():
    thin = ...  # bim-ordered per-SNP annotation derived from loadings[:, keep.index(dim)]
    write_full_annot(f"annot/{annot}.1.annot.gz", thin, bim)

# resolves/downloads/checksum-verifies the pinned v0.5.0 binary; bfile/w_ld_chr/
# overlap_annot are set once and reused by every l2/h2 call below
run = LdscRun.ensure(bfile="1000G.EUR.QC.1", w_ld_chr="weights.", overlap_annot=True)
for annot in annot2dim:
    run.l2(f"annot/{annot}.1.annot.gz", f"ld/{annot}.1")
    run.h2("trait.sumstats.gz", f"ld/{annot}.", f"results/trait/{annot}")

results = run.read_results("results", traits=["trait"], annot2dim=annot2dim)
embed.uns.setdefault("enrich", {})["my_arm"] = {
    "results": results, "factor_selection": fmap.drop(columns="annot_index")
}
# tidy dim/direction/trait table, one row per (factor, trait), with BH q already attached
```

### 3. Annotate (motif enrichment)

Independent of the LDSC path, and works on **peaks**, not cells, so it reads back the
`feature_loadings` h5ad written in step 1 rather than `embed` — restricted to the same
`keep` dims `select_factors` chose above, so a factor dropped from enrichment is
dropped here too:

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

### 4. Score cells

Back on the cell side: the same `keep`-restricted `loadings` from step 2, weighted by
the `results` step 2 read back, give each cell a score for one trait:

```python
import pandas as pd

from scads_drvi.scores.cell import cs_from_z

primary = results.query("direction == 'combined' and trait == 'trait'")
scores = cs_from_z(
    pd.DataFrame(loadings, index=embed.obs_names, columns=keep),
    primary, model="my_arm", trait="trait",
)

scores.null       # 0.0 -- read this, never hardcode it beside an axis
scores.label      # "$CS_i$ (z-weighted loading sum)"
```

`read_results` raises on a missing `.results` file rather than skipping it — a skip
silently drops the factor from the multiplicity denominator. `CellScores` carries its
own null since different formulas (z-weighted loading sum vs. mean enrichment ratio)
have different nulls and nothing on disk records which produced a column.
