# Changelog

Notable changes to `scads_drvi`. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `enrich.annotations` — widens a thin annotation into the full `CHR BP SNP CM + K`
  form `--overlap-annot` requires, and works around the reader's integer type inference.
  This is what unblocks the Rust S-LDSC swap: with it, `--overlap-annot` runs and writes a
  `.results` carrying `Coefficient_std_error` and `Coefficient_z-score`.
- `ruff` as the linter, pinned in CI, with a rule set chosen for signal rather than
  coverage: pyflakes, bugbear, isort and pyupgrade.
- Continuous integration on three dependency stacks mirroring the three environments the
  package is deployed into — no optional dependencies at all, the declared floors
  (numpy 1.26 / pandas 2.1), and a current stack with h5py and plotting.

### Fixed
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
