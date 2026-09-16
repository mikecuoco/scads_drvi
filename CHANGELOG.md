# Changelog

Notable changes to `scads_drvi`. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`pl.umap.latent_umap_grid`/`pl.factors.latent_dimension_stats`/`latent_heatmap` now
  match DRVI's own `plot_latent_dims_in_umap`/`plot_latent_dimension_stats`/
  `plot_latent_dims_in_heatmap` exactly**, not just in spirit: DRVI's own
  `SaturatedRdBu`/`SaturatedSky` colormaps (copied verbatim into
  `pl.color.SATURATED_RED_BLUE_CMAP`/`SATURATED_SKY_CMAP`/`SATURATED_JUST_SKY_CMAP`),
  per-dimension colour limits from `embed.var["min"]`/`["max"]` (in place of this
  project's own robust percentile limits), the directional +/- split built the same
  way DRVI builds it (`ad.concat`-ing a negated copy of `embed`, not a temporary `obs`
  column), DRVI's own panel-title-inside-the-axes trick, and DRVI's own hardcoded
  black/blue vanished/kept dot colours and `"Rank based on Explanation Share"` x-axis
  label. `latent_heatmap`'s balanced subsample now uses DRVI's own
  `groupby(...).sample(random_state=0, ...)` mechanism (`seed` default changed
  `42` -> `0` to match) rather than a separately-seeded draw. `dim_subset` still names
  raw `embed.var_names`, not DRVI's own title-keyed `dim_subset`; the layout-safe
  figure legend a past fix already put in `latent_dimension_stats` (avoiding a real
  `bbox_to_anchor` overlap bug) is kept rather than reverted to DRVI's own positioning.
- **Breaking: `enrich.embed` is deleted.** `write_result`, `read_feature_loadings`,
  `attach_enrich_results` and `directional_loadings` were the module's whole surface;
  each was a thin wrapper a caller can write inline just as easily -- a write is
  `embed.write_h5ad(path)` after setting `embed.uns["provenance"]`, an enrichment arm is
  attached with a plain `embed.uns.setdefault("enrich", {})[model] = {...}`, and a
  directional loadings view is `np.clip(embed.X, 0, None)`/`np.clip(-embed.X, 0, None)`.
  The companion peaks x K feature-loadings file this module also wrote/read has no
  replacement; a caller who needs it writes/reads that h5ad directly the same way.
- **Breaking: a fit's results now live in one `AnnData` (`io.result`), not a directory
  of TSVs plus a `Project`.** `config.Project`, `io.contract`, `io.meta`, `labels`
  (`FactorLabels`, `load_labels`, the `dim_j`/`k{i}`/`dim_47/neg` three-name system) and
  `io.artifacts.Interpretation`/`load_interpretation` are **deleted**. In their place:
  `io.result.build_embed`/`write_result`/`read_result`/`attach_enrich_results`/
  `directional_loadings` build, save and load one `obs`=cells, `var`=one-row-per-dim
  object shaped exactly the way DRVI's own interpretability API expects, with S-LDSC
  results embedded as a tidy `dim`/`direction`/`trait` table in `uns["enrich"][model]`.
  There is no persisted pos/neg split — DRVI has none either — a directional loadings
  view is derived on demand from the canonical signed matrix. `factorize.model.fit_meta`
  and `.load_fit` now take explicit paths; `enrich.ldsc.read_results` takes an
  `annot2dim` mapping and a `direction` instead of a `FactorLabels`;
  `scores.cell.cs_from_z` drops its `labels=` argument, made unnecessary by the above.
- **`viz` renamed to `pl`.** Two of its functions were deleted in favor of DRVI's own
  package: `latent_dimension_stats` (`drvi.utils.pl.plot_latent_dimension_stats`) and
  `group_factor_heatmaps` (`drvi.utils.pl.plot_latent_dims_in_heatmap`);
  `umap.umap_factor_grid` was likewise replaced by `drvi.utils.pl.plot_latent_dims_in_umap`.
  Superseded below by in-house replacements — see "Reworked `pl.*` embedding and
  latent-dimension figures".
- **Reworked `pl.*` embedding and latent-dimension figures.** `pl.umap.umap_categorical`/
  `umap_continuous` (plain x/y-frame scatters) are removed, and their would-be
  replacements `embedding_categorical`/`embedding_continuous` were built, then removed
  again: a plain categorical or continuous embedding scatter is just
  `sc.pl.embedding(embed, basis="umap", color=..., vmin="p1", vmax="p99.5")` directly —
  scanpy already does point sizing, a stable per-category palette, a colorbar, and
  (via its own percentile-string `vmin`/`vmax`) this project's own robust colour limits,
  so there is nothing left here worth wrapping. `pl.umap.point_style` and `pl.umap.bare`
  are removed too (scanpy already handles point sizing and frame stripping);
  `pl.umap.subsample` is unchanged and still used by `pl.enrichment.covariate_audit`.
  New: `pl.umap.latent_umap_grid` (the one embedding figure that does earn a wrapper —
  a directional +/- split and per-panel titles/limits from one `sc.pl.embedding` call),
  `pl.factors.latent_dimension_stats` and `pl.factors.latent_heatmap` — this project's
  own in-house replacements for `drvi.utils.pl.plot_latent_dims_in_umap`/
  `plot_latent_dimension_stats`/`plot_latent_dims_in_heatmap`, restyled to this
  package's colour policy (`pl.color.percentile_bounds` maps `ROBUST_LIMITS` onto
  scanpy's own `"pN"` percentile syntax). `latent_heatmap` in particular mirrors DRVI's
  own signature closely (`embed`, `categorical_column`, `title_col`, `order_col`,
  `sort_by_categorical`, `make_balanced`, `remove_vanished`) and draws via
  `seaborn.heatmap` rather than hand-rolled `imshow`/ticks/colorbar; `order="cluster"`
  is this project's own addition (hierarchical clustering via `pl.factors._cluster_order`,
  extracted from `factor_correlation` and now shared by both). `drvi-py`'s plotting is
  no longer needed for any figure this package draws.
- **`scanpy` is now a hard dependency**, and the project floor is raised to match its own
  (`requires-python>=3.12`, `numpy>=2`, `pandas>=2.3`, up from `>=3.10`/`>=1.26`/`>=2.1`) —
  every embedding figure in `pl.*` wraps `scanpy.pl.embedding`, not only the
  DRVI-specific ones, so it is no longer optional the way `drvi-py`/`scvi-tools`/`torch`
  are. The `floors` CI job is repinned to `numpy==2.0.0`/`pandas==2.3.0`/`scanpy==1.12.4`
  accordingly.

### Added
- `enrich.annotations` — widens a thin annotation into the full `CHR BP SNP CM + K`
  form `--overlap-annot` requires, and works around the reader's integer type inference.
  This is what unblocks the Rust S-LDSC swap: with it, `--overlap-annot` runs and writes a
  `.results` carrying `Coefficient_std_error` and `Coefficient_z-score`.
- `ruff` as the linter, pinned in CI, with a rule set chosen for signal rather than
  coverage: pyflakes, bugbear, isort and pyupgrade.
- Continuous integration on three dependency stacks mirroring the three environments the
  package is deployed into — no optional dependencies at all, the declared floors
  (Python 3.12 / numpy 2 / pandas 2.3, scanpy's own floor), and a current stack with
  h5py and plotting.

### Fixed
- `io.h5ad.read_obs` indexed the category table with anndata's raw codes, so a **missing**
  categorical (code `-1`) came back carrying the *last* category's name instead of NaN.
  It was a second, divergent copy of the decode in `io.artifacts.read_obs` — which was
  already correct — and is now deleted rather than repaired; `keep_rows` calls the one
  remaining decoder. The two also disagreed on an absent column: one raised, one returned
  `{}`.
- `viz.frugal.box_stats_by_column(whis="1.5iqr")` returned the **fence**
  (`q1 - 1.5·IQR`, `q3 + 1.5·IQR`) where `box_stats` returns the most extreme observation
  *inside* it. On `[0,1,2,3,4,100]` that draws a whisker at 7.5, a value the column does
  not contain. The two entry points now share one definition.
- `viz.frugal.box_stats_by_column` silently treated any unrecognised `whis` as `1.5iqr`;
  `box_stats` raised for the same argument. Both now validate against `WHIS_KINDS`.
- `scores.aggregate.block_order` returned a `names` list and a `spans` list that disagreed
  when a block was NaN — `sort_values` keeps such a row, `groupby` dropped it. Since the
  spans index `names` positionally, every block label after the gap was shifted, silently,
  in the figure. Spans now cover every name, and the function refuses to return if they
  ever stop doing so.
- `viz.color.add_threshold_lines` labelled the BH line `"BH q < 0.05"` whatever level it
  was drawn at. It takes an `alpha` now, and `viz.enrichment.heritability_landscape`
  passes the same value to the boundary and to its label — the drift the frozen
  `SignificanceRamp` exists to prevent, in the one place the ramp did not reach.
- `stats.bh_threshold_z` returned the smallest passing z unconditionally, which is the
  wrong end of the distribution for a lower-tail test. It takes `tail` now, matching
  `add_fdr` and `p_one_tailed`.
- A closure in `enrich.config` read `key` from the enclosing loop rather than binding it.
  Correct today only because `re.sub` happens to call it synchronously within the same
  iteration — a property of the caller, not of the function.
- `viz.factors.factor_distributions` zipped a **padded** subplot grid against its columns.
  The truncation is intentional there and is now explicit; every other `zip` in the
  package pairs sequences that are equal by construction and now says `strict=True`, so a
  length mismatch raises instead of silently dropping the tail.
- Dead function-local `import pandas` in five modules. The annotations they appeared to
  serve are resolved by the module-level `TYPE_CHECKING` import.

### Changed
- `tests/test_import_surface.py` checks 14 modules where it checked 2. Its docstring and
  the README both claimed the whole loaders / statistics / scoring surface imports with no
  torch, matplotlib, seaborn or h5py, but only the top level and `config` were tested — a
  module-level `import matplotlib` added to `scores.cell` would have passed CI. Every
  newly listed module already passed; the modules left out are now listed with a reason
  each, so an omission is a decision rather than a gap.
- The README's genericity claim is scoped to what the scan actually enforces: docstrings
  are exempt by design (`tests/conftest.py`), so the guarantee is "no dataset-specific
  value or name", and clean docstrings are a convention. Two docstrings naming a GWAS
  study and a specific arm were rewritten to keep their measurements and drop the names.
- Cross-references to the capsule this package was extracted from — `code/common/`,
  `code/tests/…`, `environment/env-ldsc.yaml`, `CLAUDE.md`, "arm 0N" — are gone from the
  modules that carried them. Two error messages had been instructing users to run build
  steps that do not exist in this repository; they now state the requirement instead.
  Notes that reference the origin as *history* are kept, because they are accurate.
- `enrich.annotations.force_decimal`'s docstring said `columns` defaults to every column
  except the identifiers; it excludes only `CHR` and `SNP`, so `BP` is written as
  `1000.0`. The docstring now says so. **Behaviour unchanged** — whether a float `BP`
  matters to the Rust reader is untested here and worth checking against a real
  `--overlap-annot` run.
- Adopting the linter rewrote 144 mechanical items across the package: unnecessary quoted
  annotations, deprecated `typing` imports, import order.

### Known limitations
- **The Rust S-LDSC binary is not yet a drop-in replacement.** It now runs end to end and
  its coefficients match Python to `4.1e-05` relative, but the jackknife standard errors
  do not: median 3.3% apart, and **2 of 98 categories cross z > 1.645 in one and not the
  other** on a single arm and trait. Compare across every arm and trait before adopting
  it; matching coefficients do not imply matching conclusions.
- The same run reports `Prop._SNPs` as `1.57e7` where a proportion belongs, and a summed
  per-annotation `M` of `9.4e13` for ~1.2M variants. The pipeline consumes only the
  coefficient columns, so this does not block it, but the enrichment columns should not be
  trusted without separate checking.
- Peak memory for `h2` is ~8.5 GB and is **not** improved by the port.

## [0.1.0] - 2026-09-09

First release, extracted from the SEA-AD single-cell ATAC capsule where the package
replaced `code/common/` and four near-identical interpretation notebooks totalling 23 MB.

### Added
- `config.Project` — every path caller-supplied, defaulting under one `root`. `fit` and
  `traits` have no defaults on purpose.
- `labels` — reconciles the three names a factor carries (index, annotation, display) and
  `assert_index_dims`, which refuses the wrong one. Under a split contract the display and
  index names are both `dim_`-shaped, so the wrong one selects a different column.
- `stats` — one Benjamini–Hochberg implementation, plus `legacy_bh_qvalues` to quantify
  what the previous inline version did.
- `io.artifacts` / `io.peaks` / `io.h5ad` / `io.contract` / `io.meta` — loaders and
  contract checks.
- `factorize.model` / `.kernels` / `.multigpu` — fit access and torchrun plumbing.
- `enrich.binary` / `.h2_output` / `.config` / `.ldsc` — the pinned Rust LDSC, output
  parsing, and enrichment configuration.
- `scores.cell` / `.aggregate` — the two `CS_i` formulas, named apart and each carrying its
  own null, plus group summaries.
- `viz.*` — style, colour policy, quantile-precomputed boxes, and the figures.
- The genericity rule, enforced by two tests rather than by discipline: an AST scan for
  dataset vocabulary, and a full synthetic plant analysis run end to end.

### Fixed
- The inline Benjamini–Hochberg in the notebooks this replaced was a no-op — a
  `minimum.accumulate` over an already-descending sequence — so the reported `fdr_q` was
  raw `p·m/rank`. The values were **conservative, never anti-conservative**: re-running
  with the correct implementation changes nothing, 68 significant either way across
  7 arms × 2 traits, zero factors gained.
- `normalize_peak_name("chr1:-5-9")` produced `chr1::5-9`, which its own validator rejects.
- The peak-name pattern rejected GRCh38 scaffold names containing dots.
- `np.issubdtype` raises on a pandas 3 `StringDtype`; replaced with `is_numeric_dtype`.
- A zero-width histogram range crashed under numpy 2. The fix needed a *relative*
  tolerance, not a comparison against zero: a delta constant in intent still varies by
  ~1e-16 once both operands have been through floating-point arithmetic.
