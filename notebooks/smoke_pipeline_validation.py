# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: scads_drvi
#     language: python
#     name: scads_drvi
# ---

# %% [markdown]
# # Smoke pipeline validation: a real fit, a real GWAS, a real S-LDSC run
#
# This notebook reports one end-to-end exercise of `scads_drvi` against **real** data on
# **real** hardware, and what fraction of the package that exercise actually touched.
#
# The point was never a biological result. It was to answer: *does the chain run at all,
# with real inputs, when nothing is faked?* Every other fixture in this repo either
# hand-builds its h5ad with `h5py` or fabricates the fit and S-LDSC output outright
# (`tests/smoke_data.py:synthesize_run` generates its `.results` z-scores from the formula
# `z = 4.5 - 0.6 * slot`). Nothing here is fabricated.
#
# ## What is real, and what is a stand-in
#
# **Real, unmodified:**
#
# - The matrix: 10x Genomics' public PBMC scATAC demonstration run, 482 cells x 47,843
#   peaks, checksum-pinned (`tests/smoke_data.py`).
# - The fit: a DRVI actually trained on that matrix on a titanxp GPU, `n_latent=12`,
#   30 epochs, via `scads_drvi.factorize.train`.
# - The GWAS: standing height, Yengo et al. 2018, GWAS Catalog **GCST006901**, munged to
#   1,953,984 SNPs by the pinned `ldsc` binary's own `munge-sumstats`
#   (`tests/smoke_gwas.py`).
# - The reference: Broad Alkesgroup baselineLD v2.2 (GRCh38), 1000G EUR panel.
# - The regression: `ldsc h2 --overlap-annot`, 996,267 SNPs merged, 98 categories.
#
# **Stand-ins, deliberate and load-bearing on the interpretation:**
#
# 1. **One factor, not twelve.** Only `dim_0` was tested. Testing all twelve would
#    multiply the LD-score step twelvefold for a run whose purpose is mechanical.
# 2. **The peak-to-factor mapping is not the production method.** A real arm derives
#    per-peak loadings from DRVI's decoder. That is not exposed as one callable function
#    in this package, so this run substitutes each peak's Pearson correlation with
#    `dim_0`'s latent value across the 482 cells, keeping the top 10% by `|r|`.
# 3. **The peaks were lifted hg19 -> hg38.** The 10x run is hg19 (its own `genome`
#    field says so); the only reference panel available was GRCh38. Lifted with
#    pyliftover and UCSC's `hg19ToHg38` chain rather than sourcing an hg19 reference,
#    because the free Zenodo mirror was returning 504s and the Broad's own bucket is now
#    requester-pays. The production SEA-AD pipeline is GRCh38-native anyway.
#
# **So: no number below is a claim about height biology or about PBMCs.** The enrichment
# figure is a property of a correlation-selected peak set, not of a cell type.

# %% [markdown]
# ## Setup

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scads_drvi.viz.style import WONG, apply_style

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 24)

# The package's own figure style and colourblind-safe palette, so these plots look like
# the ones `viz` produces rather than like matplotlib defaults.
apply_style()

RESULTS = Path("results")
assert RESULTS.is_dir(), f"expected artifacts at {RESULTS.resolve()}"
sorted(p.name for p in RESULTS.iterdir())

# %% [markdown]
# ## 1. Provenance: what was actually trained, fetched, and annotated
#
# `fit.meta.json` is written by `factorize.train` itself, from the same config object
# that made the choices -- not by hand afterwards. It is the record every downstream
# stage keys on, and `factorize.model.fit_meta` refuses a fit directory without it.

# %%
fit_meta = json.loads((RESULTS / "fit.meta.json").read_text())
pd.Series(
    {
        "n_latent": fit_meta["n_latent"],
        "n_cells": fit_meta["n_cells"],
        "n_features": fit_meta["n_features"],
        "gene_likelihood": fit_meta["gene_likelihood"],
        "seed": fit_meta["seed"],
        "batch_key": fit_meta["batch_key"],
        "layer": fit_meta["layer"],
        "scads_drvi_version": fit_meta["scads_drvi_version"],
    },
    name="fit",
).to_frame()

# %% [markdown]
# The `train` block records the hardware and schedule, including the two numbers that
# are easy to conflate: `batch_size` is **per device**, `effective_batch_size` is what
# the optimiser saw. On one device they agree; on four they would not.

# %%
pd.Series(fit_meta["train"], name="train").to_frame()

# %%
gwas = json.loads((RESULTS / "gwas_manifest.json").read_text())
annot = json.loads((RESULTS / "annotation_manifest.json").read_text())
pd.Series(
    {
        "gwas_trait": gwas["trait"],
        "gwas_pubmed_id": gwas["pubmed_id"],
        "gwas_n_snps_munged": gwas["n_snps"],
        "annot_factor": annot["factor"],
        "annot_top_frac": annot["top_frac"],
        "annot_peaks_total": annot["n_peaks_total"],
        "annot_peaks_kept": annot["n_peaks_kept"],
        "annot_peaks_lifted": annot["n_lifted"],
        "annot_peaks_unmapped": annot["n_unmapped"],
    },
    name="inputs",
).to_frame()

# %% [markdown]
# 4,783 of the 4,784 selected peaks lifted successfully; one did not and is simply
# absent from the annotation. Worth stating rather than rounding away.

# %% [markdown]
# ## 2. S-LDSC: the real regression output
#
# `--overlap-annot` is what writes a `.results` carrying `Coefficient_std_error` and
# `Coefficient_z-score`; without it the tool prints a table to stdout and writes nothing,
# and the pipeline's significance testing is built on that z. Our annotation is passed
# **first** in `--ref-ld-chr`, so row 0 is the category under test and the remaining 97
# are baseline.

# %%
results = pd.read_csv(RESULTS / "dim_0_height.results", sep="\t")
print(f"{len(results)} categories")
focal = results.iloc[0]
focal.to_frame().T

# %% [markdown]
# The focal row: **11.8x enrichment** for height heritability, enrichment p = 1.5e-4.
# The coefficient z-score is 1.40 -- positive but not itself significant, which is the
# honest reading: the *proportion* of heritability landing in these peaks is well above
# their SNP share, while the coefficient (which conditions on all 97 baseline categories)
# is not distinguishable from zero at this sample size and with a single annotation.

# %%
pd.Series(
    {
        "Prop._SNPs": focal["Prop._SNPs"],
        "Prop._h2": focal["Prop._h2"],
        "Enrichment": focal["Enrichment"],
        "Enrichment_std_error": focal["Enrichment_std_error"],
        "Enrichment_p": focal["Enrichment_p"],
        "Coefficient_z-score": focal["Coefficient_z-score"],
    },
    name=focal["Category"],
).to_frame()

# %% [markdown]
# For context, the most-enriched baseline categories from the same run. Coding and
# conserved regions dominating is the expected shape for a highly polygenic
# anthropometric trait -- a sanity check that the regression is behaving, not a finding.

# %%
(
    results.loc[1:, ["Category", "Prop._SNPs", "Prop._h2", "Enrichment", "Enrichment_p"]]
    .sort_values("Enrichment", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# %% [markdown]
# ### A trap in that table, worth naming before plotting it
#
# The top two rows -- `MAF_Adj_Predicted_Allele_Age` at 6750x and `MAF_Adj_ASMC` at
# 1908x -- are **not** real enrichments. Both have *negative* `Prop._SNPs` and negative
# `Prop._h2`, and enrichment is their ratio: two negatives divide to a large positive.
# These are continuous, signed baseline-LD annotations, and the proportion-of-SNPs
# framing that enrichment assumes does not apply to them.
#
# Any plot ranking by enrichment has to exclude them or it shows nothing but artefacts.
# Filtering on a positive SNP share is the honest cut, and the count dropped is reported
# rather than quietly applied.

# %%
plottable = results[results["Prop._SNPs"] > 0].copy()
print(
    f"{len(results) - len(plottable)} of {len(results)} categories dropped "
    f"(non-positive Prop._SNPs): "
    f"{results.loc[results['Prop._SNPs'] <= 0, 'Category'].tolist()}"
)

# %% [markdown]
# ### Enrichment, with our annotation in context

# %%
top = plottable.sort_values("Enrichment", ascending=False).head(15).iloc[::-1]
is_focal = top["Category"] == focal["Category"]

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.barh(
    np.arange(len(top)),
    top["Enrichment"],
    xerr=top["Enrichment_std_error"],
    color=[WONG[1] if f else WONG[0] for f in is_focal],
    error_kw={"lw": 0.7, "capsize": 2},
    height=0.78,
)
ax.axvline(1.0, color="#949494", linestyle=":", linewidth=0.8)
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels([c.removesuffix("L2_1").removesuffix("L2_0") for c in top["Category"]], fontsize=7)
ax.set_xlabel("enrichment (Prop. $h^2$ / Prop. SNPs)")
ax.set_title(
    f"Top 15 categories by enrichment — {gwas['trait'].split('_')[0]}\n"
    f"orange = our annotation ({focal['Category']})",
    fontsize=8,
)
ax.text(
    1.05, 0.02, "no enrichment", transform=ax.get_xaxis_transform(),
    fontsize=6, color="#949494", rotation=90, va="bottom",
)
plt.show()

# %% [markdown]
# Our correlation-selected peak set lands mid-pack among the strongest baseline
# categories -- above the conserved/regulatory annotations it sits beside, below coding.
# With one annotation and 482 cells behind it, the error bar is what to read, not the rank.

# %% [markdown]
# ### Where the heritability sits relative to the SNP share
#
# The diagonal is "no enrichment": a category carrying exactly its share of SNPs. Points
# above it carry more heritability than their size would predict.

# %%
fig, ax = plt.subplots(figsize=(4.8, 4.4))
baseline = plottable[plottable["Category"] != focal["Category"]]
ax.scatter(
    baseline["Prop._SNPs"], baseline["Prop._h2"],
    s=14, color=WONG[0], alpha=0.65, linewidths=0, label="baseline-LD categories",
)
ax.scatter(
    focal["Prop._SNPs"], focal["Prop._h2"],
    s=70, color=WONG[1], marker="D", edgecolors="black", linewidths=0.5,
    zorder=5, label=f"ours ({focal['Category']})",
)
lims = [1e-4, 1.3]
ax.plot(lims, lims, color="#949494", linestyle=":", linewidth=0.8, zorder=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims)
ax.set_ylim(*lims)
ax.set_xlabel("proportion of SNPs")
ax.set_ylabel("proportion of $h^2$")
ax.set_title("Heritability share vs SNP share", fontsize=9)
ax.legend(fontsize=6.5, loc="upper left")
plt.show()

# %% [markdown]
# ### Coefficient z-scores across all categories
#
# Enrichment is unconditional; the coefficient conditions on every other category in the
# model, which is why a category can be strongly enriched and still have a coefficient
# indistinguishable from zero. Reference lines are the package's own conventions from
# `viz.color.add_threshold_lines` (one-tailed nominal, and high-confidence).

# %%
from scads_drvi.viz.color import add_threshold_lines

z = results["Coefficient_z-score"].to_numpy(dtype=float)
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.hist(z[np.isfinite(z)], bins=40, color=WONG[0], edgecolor="white", linewidth=0.4)
ax.axvline(
    float(focal["Coefficient_z-score"]), color=WONG[1], linewidth=1.6,
    label=f"ours: z = {float(focal['Coefficient_z-score']):.2f}",
)
add_threshold_lines(ax, axis="x")
ax.set_xlabel("coefficient z-score")
ax.set_ylabel("categories")
ax.set_title(f"Coefficient z across all {len(results)} categories", fontsize=9)
ax.legend(fontsize=6.5)
plt.show()

# %% [markdown]
# Our annotation's z sits just below the nominal one-tailed line, inside the bulk of the
# baseline distribution. Read plainly: this run demonstrates the machinery produces a
# well-formed coefficient and standard error, not that this peak set is significant.

# %% [markdown]
# ## 3. Per-cell scores
#
# `scores.cell.cs_from_z` computes `cs_i = sum_k L_ik * max(0, z_k)`. The clipping is not
# cosmetic: a factor whose annotation carries *less* heritability than baseline is
# evidence against involvement, and letting it subtract would let depletion in one factor
# mask enrichment in another. With one factor and z = 1.40, every cell's score is its
# `dim_0` latent value scaled by that z.
#
# `CellScores` carries its own null (0.0 for this formula) rather than leaving a reader
# to assume one.

# %%
strata = pd.read_csv(RESULTS / "cs_by_frip_stratum.tsv", sep="\t")
strata

# %% [markdown]
# The summary above is what the pipeline wrote. To plot the *distributions* rather than
# five numbers per group, the per-cell scores are recomputed here from the saved latent
# and the z above, through the same `scores.cell.cs_from_z` the run used -- so this is a
# re-derivation, not a second implementation.

# %%
from scads_drvi.scores.cell import cs_from_z

loadings = pd.read_csv(RESULTS / "loadings.tsv", sep="\t", index_col=0)
cells = pd.read_csv(RESULTS / "cell_metadata.tsv", sep="\t", index_col=0)
weights = pd.DataFrame(
    {"dim": [annot["factor"]], "Coefficient_z-score": [float(focal["Coefficient_z-score"])]}
)
scores = cs_from_z(loadings, weights, model="e2e_smoke", trait=gwas["trait"])

print(f"{scores.label}   null = {scores.null}")
print(f"n = {len(scores.values)}, mean = {scores.values.mean():.4f}, "
      f"std = {scores.values.std():.4f}")

# Matches the pipeline's own aggregate to the printed digits, which is the check that
# this re-derivation is the same computation and not a lookalike.
recomputed = scores.values.groupby(cells["frip_stratum"]).mean()
pd.DataFrame(
    {
        "pipeline_mean": strata.set_index("frip_stratum")["mean"],
        "recomputed_mean": recomputed,
    }
).round(6)

# %% [markdown]
# ### Distribution per stratum -- `viz.enrichment.score_by_group`
#
# The module's own figure for this. It takes a `BoxStats` rather than the per-cell
# table, deliberately: "the expensive mistake is unavailable rather than discouraged."
# `box_stats` computes the five summaries in one pass; the null is read off `CellScores`
# rather than typed in.

# %%
from scads_drvi.viz.enrichment import score_by_group
from scads_drvi.viz.frugal import box_stats

stats = box_stats(scores.values.to_numpy(), cells["frip_stratum"].to_numpy(), whis="1.5iqr")
fig, ax = score_by_group(
    stats,
    group_label="frip_stratum",
    value_label=scores.label,
    null=scores.null,
)
fig.set_size_inches(4.4, 3.4)
plt.show()

# %% [markdown]
# Note the ordering: `score_by_group` sorts by median rather than honouring the stratum's
# ordinal order, which is the right default for an unordered grouping (cell types) and
# is worth knowing about for an ordered one like this.

# %% [markdown]
# ### Score against the technical covariates -- `viz.enrichment.covariate_audit`
#
# The module's audit figure: one hexbin panel per covariate with Spearman rho on each.
# Hexbin rather than scatter "because a million-point scatter of a covariate against a
# score is a solid block that hides the relationship it exists to show" -- at 482 cells
# that is not yet true, but the figure is the one a real arm would produce.

# %%
from scads_drvi.viz.enrichment import covariate_audit

audited = cells.assign(cs=scores.values)
fig, axes = covariate_audit(
    audited,
    value="cs",
    covariates=["frip", "tss_fraction", "n_fragment"],
    log_x=["n_fragment"],
)
plt.show()

# %% [markdown]
# ### Two groupings at once -- `scores.aggregate.group_matrix` + `grouped_landscape`
#
# `group_matrix` returns the mean **and** the count matrix, because "a reader of the mean
# alone cannot tell an empty bin from a low one", and `grouped_landscape` draws
# under-populated bins in a distinct grey with a colourbar that says so. `min_cells` is
# dropped to 10 here only because 482 cells over a 6x3 grid leaves thin bins; the default
# is 20.

# %%
from scads_drvi.scores.aggregate import group_matrix
from scads_drvi.viz.enrichment import grouped_landscape

means, counts = group_matrix(
    scores.values,
    cells,
    index="signal_stratum",
    columns="duplicate_stratum",
    min_cells=10,
)
fig, ax = grouped_landscape(
    means,
    counts,
    row_label="signal_stratum",
    column_label="duplicate_stratum",
    value_label=scores.label,
)
plt.show()
counts

# %% [markdown]
# Spearman correlation of the score against FRiP, and against depth, for the record --
# the second is the one that matters for interpreting a 482-cell fit.

# %%
from scipy.stats import spearmanr

association = pd.Series(
    {
        "cs vs frip": spearmanr(scores.values, cells["frip"]).statistic,
        "cs vs n_fragment": spearmanr(scores.values, cells["n_fragment"]).statistic,
        "cs vs tss_fraction": spearmanr(scores.values, cells["tss_fraction"]).statistic,
    },
    name="spearman rho",
)
association.round(3).to_frame()

# %% [markdown]
# **This is the most important number in the notebook, and it is not the enrichment.**
#
# `|rho| = 0.76` against TSS fraction. `viz.factors.covariate_association` flags a
# factor as nuisance-confounded at a default threshold of **0.7** -- so by this package's
# own convention, `dim_0` would be flagged in a real arm and its enrichment would not be
# reported as a biological result.
#
# That is the correct outcome for a 482-cell fit, and it is worth stating in the same
# breath as the 11.8x: the annotation was built by correlating peaks against a latent
# dimension that is itself largely tracking signal quality, so the enrichment is
# substantially an enrichment of *well-measured open chromatin*, which is a real and
# well-known heritability-enriched category on its own.

# %%
flagged = association.abs() >= 0.7
pd.DataFrame(
    {
        "abs_rho": association.abs().round(3),
        "flagged_at_0.7": flagged,
    }
)

# %% [markdown]
# A monotonic gradient across the FRiP strata: mean `cs` falls from 0.796 (low) to 0.255
# (high), with non-overlapping 95% confidence intervals.
#
# **This gradient is designed in, not discovered.** `frip_stratum` is a rank split of
# real FRiP, and the annotation was selected by correlating peaks against `dim_0`, which
# itself tracks depth and signal quality in a 482-cell dataset. A latent dimension
# correlating with QC signal at this scale is the expected outcome, and the whole point
# of naming those columns `*_stratum` rather than anything cell-type-like.

# %%
from IPython.display import Image

Image(filename=str(RESULTS / "cs_by_frip_stratum.png"))

# %% [markdown]
# ## 3b. All twelve factors, through the package's own reader
#
# Everything above concerns one factor, read with raw `pandas`. S-LDSC was then run for
# all twelve, one `h2` per factor with that factor's annotation first in `--ref-ld-chr`,
# and the results read back the way the package intends: `labels.load_labels` for the
# factor/annotation mapping, then `enrich.ldsc.read_results`, which calls
# `stats.add_fdr` internally.
#
# **This section corrects the headline number from section 2.** `Enrichment_p` is an
# unconditional, uncorrected p-value. The test this pipeline is actually built on is the
# **coefficient z** -- which conditions on all 97 baseline categories -- put through
# Benjamini-Hochberg across the factors tested.

# %%
from scads_drvi.enrich.ldsc import read_results
from scads_drvi.labels import load_labels

ALL12 = RESULTS / "all12"
labels12 = load_labels(ALL12 / "factor_map.tsv", model="all12")
print(f"is_split={labels12.is_split}, {len(labels12.annot2dim)} annotations")

# strict=True by default: a kept factor whose .results file is missing RAISES rather
# than being skipped, because skipping shrinks the multiplicity denominator and makes
# every surviving q optimistic without saying so.
enrich12 = read_results(ALL12 / "results", traits=[gwas["trait"]], labels=labels12)
enrich12[
    ["dim", "annot", "Enrichment", "Enrichment_p", "Coefficient_z-score", "fdr_q"]
].sort_values("Coefficient_z-score", ascending=False).reset_index(drop=True).round(4)

# %% [markdown]
# Read the last two columns together. Enrichment runs 10-13x for ten of the twelve
# factors with `Enrichment_p` down to 1e-4, and **not one factor survives BH**: the
# smallest q is 0.146. The coefficients sit at z = 1.0-1.8, which is what an
# unremarkable annotation looks like once the baseline model is conditioned on.
#
# `dim_7` is the interesting exception in the other direction: z = -4.34, a real
# *depletion*. Its q is ~1 because the test is one-tailed for enrichment, which is the
# correct treatment -- a depleted annotation is not a hit.

# %%
print(f"factors with fdr_q < 0.05: {(enrich12['fdr_q'] < 0.05).sum()} of {len(enrich12)}")
print(f"smallest q: {enrich12['fdr_q'].min():.4f}")
print(f"smallest Enrichment_p: {enrich12['Enrichment_p'].min():.3g}")

# %% [markdown]
# ### `viz.enrichment.heritability_landscape`
#
# The enrichment module's flagship figure: ranked z-scores and a volcano sharing one
# significance ramp, so a factor's position in one panel and its colour in the other
# cannot disagree.

# %%
from scads_drvi.viz.enrichment import heritability_landscape

fig, axes = heritability_landscape(
    enrich12, trait=gwas["trait"], labels=labels12, top_n=4, alpha=0.05
)
fig.set_size_inches(9.0, 3.6)
plt.show()

# %% [markdown]
# ### Enrichment against confounding, across all twelve
#
# The twelve-factor table makes visible what one factor could not: the ten factors
# carrying the 10-13x enrichment are the ten confounded by TSS fraction, and the two
# that are not confounded (`dim_6`, `dim_7`) are the two without it.

# %%
rho_by_dim = loadings.apply(lambda col: abs(spearmanr(col, cells["tss_fraction"]).statistic))
merged = enrich12.set_index("dim").join(rho_by_dim.rename("abs_rho_tss"))

fig, ax = plt.subplots(figsize=(5.2, 4.0))
ax.scatter(merged["abs_rho_tss"], merged["Enrichment"], s=48, color=WONG[0],
           edgecolors="black", linewidths=0.4, zorder=3)
for name, row in merged.iterrows():
    ax.annotate(name, (row["abs_rho_tss"], row["Enrichment"]), fontsize=6,
                xytext=(3, 3), textcoords="offset points")
ax.axvline(0.7, color="#D55E00", linestyle=":", linewidth=0.9)
ax.text(0.7, ax.get_ylim()[1], " confounding threshold", fontsize=6, color="#D55E00",
        va="top", rotation=90)
ax.set_xlabel("|Spearman rho| vs TSS fraction")
ax.set_ylabel("S-LDSC enrichment")
ax.set_title("Enrichment tracks the confounder, not biology", fontsize=9)
plt.show()

# %%
print(merged[["abs_rho_tss", "Enrichment", "Coefficient_z-score", "fdr_q"]]
      .sort_values("abs_rho_tss", ascending=False).round(3).to_string())

# %% [markdown]
# ## 3c. The split contract: each direction of each dimension, separately
#
# A latent dimension is signed, and its two directions are not one program: cells far
# positive and cells far negative on `dim_k` are different populations. Section 3b built
# one annotation per dimension from `|correlation|`, which conflates them. The **split
# contract** is this package's answer -- one annotation per (dimension, direction), with
# a `half_map.tsv` recording the provenance so results label as `dim_k/pos`.
#
# Split loadings are ReLU output (`dim_k_pos = max(z_k, 0)`, `dim_k_neg = max(-z_k, 0)`),
# which is exactly the shape `viz.umap.umap_continuous`'s `zero_as_background` exists
# for. Each half selected its own peaks at the same 10% as the unsplit run, so the
# enrichments are directly comparable.

# %%
SPLIT = RESULTS / "split24"
selection = pd.read_csv(SPLIT / "split_selection.tsv", sep="\t")
selection[["annot_dim", "source_dim", "half", "n_nonzero_cells", "n_lifted"]].head(8)

# %% [markdown]
# ### Selection actually did something here
#
# Three halves are degenerate: `dim_7_neg` is non-zero in **1** cell of 482, `dim_6_pos`
# in 8, `dim_10_neg` in 12. An annotation built by correlating peaks against a column
# that is zero for 481 cells is made of noise, and running it would spend a job to obtain
# a contentless number *and* add a test to the multiplicity denominator. They are
# dropped, with the reason recorded, at the `min_cells = 20` floor the package uses
# elsewhere (`scores.aggregate.summarize_by`).
#
# The unsplit run had nothing to drop, so this is the first time `factor_map`'s
# `kept`/`drop_reason` columns carry a real decision.

# %%
fmap = pd.read_csv(SPLIT / "factor_map.tsv", sep="\t")
fmap.loc[~fmap["kept"], ["dim", "kept", "drop_reason", "annot_index"]]

# %%
labels_split = load_labels(
    SPLIT / "factor_map.tsv", model="split24", half_map=SPLIT / "half_map.tsv"
)
print(f"is_split={labels_split.is_split}, {len(labels_split.annot2dim)} annotations kept")
print(f"display for dim_0_neg -> {labels_split.display_label('dim_0_neg')!r}")

# `assert_index_dims` rejects a display label where a loadings column is required. Under
# a split contract both are `dim_`-shaped, so this is the guard that stops `dim_0/neg`
# being used to index a matrix whose column is `dim_0_neg`.
try:
    labels_split.assert_index_dims(["dim_0/neg"])
    print("PROBLEM: a display label was accepted as a column name")
except KeyError as exc:
    print(f"assert_index_dims rejected the display label, as it should:\n  {exc}")

# %%
enrich_split = read_results(SPLIT / "results", traits=[gwas["trait"]], labels=labels_split)
enrich_split["source_dim"] = enrich_split["display"].str.split("/").str[0]
enrich_split["half"] = enrich_split["display"].str.split("/").str[1]
(
    enrich_split[["display", "annot", "Enrichment", "Coefficient_z-score", "fdr_q"]]
    .sort_values("Coefficient_z-score", ascending=False)
    .reset_index(drop=True)
    .round(4)
)

# %% [markdown]
# ### Splitting concentrates signal the unsplit run averaged away
#
# `dim_0` is the clearest case: its two halves come apart at z = 2.73 (neg) against
# 0.90 (pos), while unsplit `dim_0` scored 1.40 -- roughly their average. Combining the
# directions was diluting a real asymmetry.

# %%
pairs = enrich_split.pivot_table(
    index="source_dim", columns="half", values="Coefficient_z-score"
)
unsplit_z = enrich12.set_index("dim")["Coefficient_z-score"]
pairs["unsplit"] = unsplit_z
pairs["gap"] = (pairs["pos"] - pairs["neg"]).abs()
pairs.sort_values("gap", ascending=False).round(3)

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ordered = pairs.dropna(subset=["pos", "neg"]).sort_values("gap", ascending=False)
x = np.arange(len(ordered))
ax.vlines(x, ordered["neg"], ordered["pos"], color="#BBBBBB", linewidth=1.2, zorder=1)
ax.scatter(x, ordered["pos"], s=42, color=WONG[1], label="pos half", zorder=3)
ax.scatter(x, ordered["neg"], s=42, color=WONG[0], label="neg half", zorder=3)
ax.scatter(x, ordered["unsplit"], s=34, color="#444444", marker="_", linewidths=1.8,
           label="unsplit", zorder=4)
ax.axhline(0, color="#949494", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(ordered.index, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("coefficient z")
ax.set_title("Each dimension's two directions, vs the unsplit annotation", fontsize=9)
ax.legend(fontsize=6.5)
plt.show()

# %% [markdown]
# Several pairs also **flip** which direction carries the signal (`dim_5`, `dim_8`,
# `dim_2` favour pos; `dim_0`, `dim_11`, `dim_9` favour neg), so this is not one half
# systematically winning.

# %% [markdown]
# ### What it costs and what it buys
#
# Splitting nearly doubles the multiplicity burden -- 21 tests against 12 -- and the
# best q still improves:

# %%
pd.DataFrame(
    {
        "unsplit (12 factors)": {
            "tests": len(enrich12),
            "best z": enrich12["Coefficient_z-score"].max().round(3),
            "smallest fdr_q": enrich12["fdr_q"].min().round(4),
            "q < 0.10": int((enrich12["fdr_q"] < 0.10).sum()),
            "q < 0.05": int((enrich12["fdr_q"] < 0.05).sum()),
        },
        "split (21 halves)": {
            "tests": len(enrich_split),
            "best z": enrich_split["Coefficient_z-score"].max().round(3),
            "smallest fdr_q": enrich_split["fdr_q"].min().round(4),
            "q < 0.10": int((enrich_split["fdr_q"] < 0.10).sum()),
            "q < 0.05": int((enrich_split["fdr_q"] < 0.05).sum()),
        },
    }
)

# %% [markdown]
# `dim_0/neg` reaches q = 0.067 -- the only annotation in either run to come close to
# 0.05, and it does so *despite* paying for nine extra tests. That is the opposite of
# what dilution alone would produce, which is what makes it worth noting rather than
# dismissing. It is still not significant.

# %%
fig, axes = heritability_landscape(
    enrich_split, trait=gwas["trait"], labels=labels_split, top_n=4, alpha=0.05
)
fig.set_size_inches(9.0, 3.8)
plt.show()

# %% [markdown]
# **The caveat does not move.** A sharper z on a confounded annotation is a sharper
# measurement of the confounder. Ten of the twelve source dimensions correlate with TSS
# fraction at |rho| >= 0.53, `dim_0` among them, and the peak-to-factor mapping is still
# the correlation stand-in rather than DRVI's decoder. What the split run demonstrates
# is that the split contract works end to end -- selection, `half_map`, display labels,
# per-half S-LDSC, and BH over the halves actually tested.

# %% [markdown]
# ## 4. The fit itself, through `viz.factors` and `viz.umap`
#
# The enrichment above concerns one factor. These are the module's standard figures for
# the *fit* -- all twelve dimensions -- and they are what would decide which factors an
# arm carries forward at all.

# %% [markdown]
# ### Per-factor distributions -- `viz.factors.factor_distributions`
#
# `box_stats_by_column` turns the (cells x factors) latent into one box per factor with a
# single `percentile` call, and `factor_distributions` stacks one panel per score source
# over a shared factor axis. Only one source here; the shared axis is what makes two
# comparable when there are more.

# %%
from scads_drvi.viz.factors import factor_distributions
from scads_drvi.viz.frugal import box_stats_by_column

per_factor = box_stats_by_column(loadings, whis="1.5iqr")
fig = factor_distributions(
    {"smoke_drvi_real (latent)": per_factor},
    value_label="latent value",
    yscale=None,
)
fig.set_size_inches(7.0, 2.8)
plt.show()

# %% [markdown]
# ### Which factors are confounded -- `viz.factors.covariate_association`
#
# **This is the figure that matters for interpreting anything above.** Per-factor
# absolute Spearman rho against a nuisance covariate, with the concern threshold drawn.
# The default threshold is 0.7, and the function's own docstring is explicit that the
# level is an argument "because what counts as too much depends on the covariate and the
# claim being made about the factor".

# %%
from scads_drvi.viz.factors import covariate_association

rho_tss = loadings.apply(lambda col: spearmanr(col, cells["tss_fraction"]).statistic)
fig, ax = covariate_association(rho_tss, threshold=0.7, ylabel="|Spearman rho| vs TSS fraction")
fig.set_size_inches(5.2, 3.0)
plt.show()

rho_tss.abs().sort_values(ascending=False).round(3).to_frame("abs_rho_vs_tss_fraction")

# %% [markdown]
# **Six of twelve** dimensions sit at or above the 0.7 concern line against TSS fraction
# alone (a seventh, `dim_9`, is at 0.689). `dim_0` -- the dimension this entire
# enrichment was computed for -- is not even the worst of them; `dim_2` is, at 0.774.
#
# Half the latent space of this fit is tracking one technical covariate. That is the
# honest headline of the notebook: not the 11.8x, but that a 482-cell fit produces a
# latent whose dominant axes are QC structure, and that the package has a standard figure
# whose entire job is to make that impossible to miss before anything reaches S-LDSC.

# %% [markdown]
# ### Factor-factor structure -- `viz.factors.factor_correlation`
#
# Clustered correlation across the latent. Vanished dimensions would be greyed rather
# than dropped, "because removing them would make the matrix look cleaner and quietly
# change what fraction of the model the figure describes" -- this fit reports none.

# %%
from scads_drvi.viz.factors import factor_correlation

fig, ax = factor_correlation(loadings.corr(method="spearman"))
fig.set_size_inches(4.6, 4.0)
plt.show()

# %% [markdown]
# ### Mean latent per stratum -- `viz.factors.group_factor_heatmaps`
#
# Several heatmaps forced to share one row and column order, because "letting each panel
# choose its own ordering is how a reader compares two panels and draws a conclusion
# about factors that are not in the same place in both."

# %%
from scads_drvi.viz.factors import group_factor_heatmaps

# Both panels are over the SAME grouping. That is forced by the function, not a choice:
# it takes one row order for every panel, so panels differing in their row space (say
# frip_stratum against duplicate_stratum) would silently reindex the second to the
# first's labels and draw an all-NaN grid. Panels here differ in the statistic, not the
# grouping. The third tuple element is a colormap name -- signed values, so diverging.
grouped = loadings.groupby(cells["signal_stratum"], observed=True)
panels = [
    ("mean latent value", grouped.mean(), "RdBu_r"),
    ("median latent value", grouped.median(), "RdBu_r"),
]
fig, axes = group_factor_heatmaps(panels, column_order=list(loadings.columns))
fig.set_size_inches(8.4, 3.0)
plt.show()

# %% [markdown]
# ### The embedding -- `viz.umap`
#
# A real UMAP of the real 12-dim latent (seeded, `notebooks/results/embedding.tsv`; see
# the script noted in its provenance). `umap_continuous` applies robust colour limits,
# and `zero_as_background` exists because loadings are ReLU output and a large fraction
# are exactly zero -- not the case for a latent, so it is left off here.

# %%
from scads_drvi.viz.umap import umap_categorical, umap_continuous

embedding = pd.read_csv(RESULTS / "embedding.tsv", sep="\t", index_col=0)
frame = embedding.join(cells).assign(cs=scores.values)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
umap_categorical(frame, "frip_stratum", ax=axes[0], legend="right")
axes[0].set_title("frip_stratum", fontsize=9)
umap_continuous(frame, "cs", ax=axes[1], colorbar_label=scores.label)
axes[1].set_title(scores.label, fontsize=9)
plt.show()

# %% [markdown]
# The strata separate cleanly in the latent's own embedding, and the score follows them.
# For a 482-cell fit whose dominant axis correlates with TSS fraction at 0.76, that is
# the expected picture and not a reassuring one.

# %% [markdown]
# ### One factor per panel -- `viz.umap.umap_factor_grid`

# %%
from scads_drvi.viz.umap import umap_factor_grid

# `zero_as_background=False`: the default is True because loadings are ReLU output and a
# large fraction are exactly zero, which deserves grey rather than the bottom of a colour
# ramp. A latent is signed and dense, so exact zeros carry no special meaning here.
fig = umap_factor_grid(
    embedding,
    loadings,
    list(loadings.columns),
    ncols=4,
    zero_as_background=False,
)
fig.set_size_inches(9.0, 6.4)
plt.show()

# %% [markdown]
# ### Saving through `viz.save.save_figure`
#
# Writes every format in the spec and embeds metadata, "so a stray PDF can be traced back
# to the run that made it".

# %%
from scads_drvi.viz.save import figure_metadata, save_figure

written = save_figure(
    fig,
    "umap_factor_grid",
    RESULTS / "figures",
    metadata=figure_metadata(
        fit=fit_meta["fit"],
        trait=gwas["trait"],
        notebook="smoke_pipeline_validation",
    ),
)
[str(p.relative_to(RESULTS)) for p in written]

# %% [markdown]
# ### What the module could not be asked for, and why
#
# Three of `viz`'s standard figures need inputs this run does not have. Naming them is
# more useful than substituting a lookalike:
#
# | function | needs | why it is missing |
# |---|---|---|
# | `enrichment.heritability_landscape` | one results row per factor, with `fdr_q` | S-LDSC was run for `dim_0` only, so there is no per-factor table and no multiplicity to correct |
# | `enrichment.trait_concordance` | two traits' z-scores | one trait (height) was fetched |
# | `factors.latent_dimension_stats` | DRVI's `latent_stats.tsv` | produced by the fit's inspection stage, which this run never had; `factorize.train` writes the model and the record, not the diagnostics |
#
# `heritability_landscape` is the flagship of the enrichment module, and getting it needs
# the twelve-factor run: 12 x 22 LD-score tasks plus 12 `h2` runs, roughly an hour of
# mostly-parallel cluster time. That is the natural next step, not a limitation of the
# plotting code.

# %% [markdown]
# ## 5. Module coverage: how much of the package this actually exercised
#
# Measured, not estimated: every step was re-executed under `coverage run`, scoped to
# `src/scads_drvi`, and the per-process data combined. Two caveats on the numbers:
#
# - The network-fetch functions (`smoke_data._download` and its `smoke_gwas` twin) ran
#   for real earlier in the session but were **not** re-instrumented here -- re-hitting
#   10x's CDN and EBI's FTP purely for coverage bookkeeping would be a needless repeat
#   load on public services. They read as uncovered below.
# - `l2_one_chrom` was instrumented for one chromosome, not all 22. The code path is
#   identical per chromosome, so line coverage is unaffected.

# %%
cov = json.loads((RESULTS / "coverage.json").read_text())


def as_module(path: str) -> str:
    """`.../src/scads_drvi/enrich/binary.py` -> `scads_drvi/enrich/binary.py`.

    The LAST `scads_drvi` in the path, not the first: the repository directory is also
    called `scads_drvi`, so `parts.index(...)` matches `/repos/scads_drvi/` and every
    module ends up filed under whatever follows it.
    """
    parts = Path(path).parts
    last = len(parts) - 1 - parts[::-1].index("scads_drvi")
    return Path(*parts[last:]).as_posix()


files = pd.DataFrame(
    [
        {
            "module": as_module(path),
            "statements": data["summary"]["num_statements"],
            "covered": data["summary"]["covered_lines"],
            "percent": data["summary"]["percent_covered"],
        }
        for path, data in cov["files"].items()
    ]
)
totals = cov["totals"]
print(
    f"TOTAL: {totals['covered_lines']}/{totals['num_statements']} statements "
    f"= {totals['percent_covered']:.1f}%"
)

# %% [markdown]
# Rolled up by subsystem, which is the level the answer actually lives at.

# %%
def subsystem(module: str) -> str:
    """`scads_drvi/enrich/binary.py` -> `enrich`; `scads_drvi/config.py` -> top level."""
    rest = Path(module).parts[1:]
    return rest[0] if len(rest) > 1 else "(top level)"


files["subsystem"] = files["module"].map(subsystem)
rollup = (
    files.groupby("subsystem")[["statements", "covered"]]
    .sum()
    .assign(percent=lambda d: (100 * d["covered"] / d["statements"]).round(1))
    .sort_values("percent", ascending=False)
)
rollup

# %% [markdown]
# Per module, worst-covered last -- the zeros are the interesting part.

# %%
files["file"] = files["module"].map(lambda m: Path(m).name)
(
    files[["subsystem", "file", "statements", "covered", "percent"]]
    .assign(percent=lambda d: d["percent"].round(1))
    .sort_values(["percent", "statements"], ascending=[False, False])
    .reset_index(drop=True)
)

# %% [markdown]
# Plotted two ways, because percentage alone misleads here: `viz` at 6% and `_util` at
# 11% look comparable until you see that `viz` is 697 statements and `_util` is 149. The
# right-hand panel is statements, which is where the untouched *bulk* of the package
# actually is.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

order = rollup.sort_values("percent")
axes[0].barh(np.arange(len(order)), order["percent"], color=WONG[0], height=0.72)
axes[0].set_yticks(np.arange(len(order)))
axes[0].set_yticklabels(order.index, fontsize=8)
axes[0].set_xlabel("statements covered (%)")
axes[0].set_xlim(0, 100)
axes[0].set_title("Coverage rate by subsystem", fontsize=9)
for i, value in enumerate(order["percent"]):
    axes[0].text(value + 1.5, i, f"{value:.0f}%", va="center", fontsize=7)

size = rollup.sort_values("statements")
missed = size["statements"] - size["covered"]
axes[1].barh(np.arange(len(size)), size["covered"], color=WONG[2], height=0.72, label="covered")
axes[1].barh(
    np.arange(len(size)), missed, left=size["covered"],
    color="#DDDDDD", height=0.72, label="not covered",
)
axes[1].set_yticks(np.arange(len(size)))
axes[1].set_yticklabels(size.index, fontsize=8)
axes[1].set_xlabel("statements")
axes[1].set_title("Where the untouched code actually is", fontsize=9)
axes[1].legend(fontsize=6.5, loc="lower right")
plt.show()

# %% [markdown]
# The largest single blocks of never-executed code, which is the actionable version of
# the same information.

# %%
never = (
    files[files["covered"] == 0]
    .sort_values("statements", ascending=False)
    .head(12)
    .iloc[::-1]
)
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.barh(np.arange(len(never)), never["statements"], color="#BBBBBB", height=0.75)
ax.set_yticks(np.arange(len(never)))
ax.set_yticklabels(
    [f"{s}/{f}" for s, f in zip(never["subsystem"], never["file"], strict=True)],
    fontsize=7,
)
ax.set_xlabel("statements, none executed")
ax.set_title("Largest fully-untouched modules", fontsize=9)
plt.show()

# %% [markdown]
# ### Reading the zeros
#
# The untouched modules split cleanly into three kinds, and only one of them is a gap
# worth acting on.
#
# **Legitimately irrelevant to this run:**
#
# - `factorize.multigpu` -- single GPU; there was no process group to manage.
# - `factorize.kernels` -- peak *calling* (building the input peak set). The 10x run
#   ships peaks already called.
# - `_util.advise` -- no large read off network storage triggered a warning.
# - `_util.presets` -- `train()` was called directly with a hand-built `TrainConfig`,
#   bypassing the preset/CLI layer.
#
# **Not applicable at one factor:**
#
# - `stats` -- Benjamini-Hochberg over one test is not a correction.
# - `labels` -- `FactorLabels` exists to keep display names, annotation names and
#   loadings-column names from being confused with each other across many factors.
# - most of `scores.aggregate` -- `group_matrix`, `block_order`, `eta_squared` are
#   multi-factor tools.
#
# **A real gap, and the honest one to name:** `enrich.ldsc`, `enrich.h2_output` and
# `enrich.config` are at 0% because this run *reimplemented* what they do. The `.results`
# file was read with raw `pandas` and a one-row weights frame hand-built, rather than
# going through `read_results` (which raises on a missing result file instead of skipping
# it -- the behaviour that keeps q-values honest) and the `factor_map` selection stage.
# That was expedient for one ad hoc factor, but it means the readers those modules exist
# to be, and the multiplicity accounting they protect, are still unexercised against real
# S-LDSC output. Same for `io.artifacts`: `.obs` was read straight off `anndata` instead
# of through `read_obs`/`cell_metadata`, so the categorical-decode path that had a real
# bug in it stayed untested here.

# %% [markdown]
# ## Conclusions
#
# 1. **The chain runs on real data.** Train -> save record -> load -> annotate -> LD
#    scores -> S-LDSC -> per-cell scores -> aggregate -> figure, with real inputs at
#    every step and no fabricated intermediate.
# 2. **Six real bugs surfaced only because it was run for real**, four of them silent:
#    an `ldsc l2` output-naming collision that let chromosome 2's scores overwrite
#    chromosome 22's; a printed-SNP-set mismatch against the baseline reference; an
#    annotation-discovery path mismatch; fixed-width field padding in the published GWAS
#    file that silently dropped 99.95% of SNPs while exiting 0; a scientific-notation `N`
#    value; and a CSV dtype-inference trap in the reference's own annotation files.
# 3. **22% of the package ran.** The mechanics are validated; the orchestration and
#    interpretation layer (`labels`, `stats`, `enrich.config`, `enrich.ldsc`,
#    `enrich.h2_output`, `io.artifacts`, most of `viz`) is not, and a multi-factor,
#    multi-trait arm is what would exercise it.
# 4. **Nothing here supports a claim about biology, and the run says so itself.**
#    `dim_0` correlates with TSS fraction at `|rho| = 0.76`, and it is one of **six of
#    twelve** dimensions past the 0.7 threshold this package's own
#    `covariate_association` uses to flag a factor as confounded. The 11.8x enrichment is
#    substantially an enrichment of well-measured open chromatin. A real arm would flag
#    this factor -- and most of this fit -- and not report it.
#
# ## If this were to become a permanent check
#
# The repo already declares a `gpu` marker in `pyproject.toml` that nothing uses. The
# training half of this (train -> record -> load -> latent, ~1 min on one small GPU)
# would fit it directly. The S-LDSC half would not: it needs the 3.3 GB reference, the
# 22-chromosome LD-score step, and a filesystem mount that only some partitions have --
# that belongs in a submitted job, not a test.
