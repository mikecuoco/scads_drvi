#!/usr/bin/env python3
"""Config loader and factor selection for the enrichment stage.

The caller's config file is the single source of truth and this module is the
only thing that reads it on the Python side.

The one algorithm with real content here is factor selection, and it is a
STAGE rather than a loop filter. K_eff sets the annotation column count, the
number of --h2 runs, a_k, the annotation correlation matrix, and the pool the
empirical-Bayes shrinkage is fitted over -- a vanished dimension left in that
pool moves every other factor's shrunk estimate. So it is resolved once, here,
and everything downstream keys on the factor_map.tsv this writes.

**This module imports numpy, pandas and yaml at module scope**, and is the only
one in the package that does. That is deliberate rather than an oversight: it is
reachable only from an environment that already carries all three, so the lazy
convention the rest of the package follows would buy nothing here.
tests/test_import_surface.py records the exemption.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# `log` and `normalize_peak_name` are re-exported: enrichment scripts import them here.
# `run` used to be re-exported alongside them and was never called by anything, so it is
# gone rather than carried forward.
from scads_drvi._util.progress import log_out as log  # noqa: F401  (re-exported)
from scads_drvi.io.peaks import normalize_peak_name  # noqa: F401  (re-exported)

#: Environment variable naming the enrichment config, for callers with no better handle.
CONFIG_ENV_VAR = "SCADS_DRVI_ENRICH_CONFIG"

#: Contract filenames written by every factorize backend. The enrichment stage consumes
#: exactly these two, plus the GWAS summary statistics.
LOADINGS_TSV = "topic_loadings.tsv"
FACTORS_TSV = "topic_factors.tsv"

#: DRVI's set_latent_dimension_stats output, written by the fit's inspection stage.
#: The `vanished` flags live here and nowhere else.
LATENT_STATS_TSV = "latent_stats.tsv"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _expand_paths(paths: dict[str, str]) -> dict[str, str]:
    """Resolve ${key} references within the paths block. Mirrors peak_config.py."""
    out = dict(paths)
    for _ in range(len(out) + 1):
        pending = [k for k, v in out.items() if isinstance(v, str) and "${" in v]
        if not pending:
            return out
        for key in pending:
            def repl(m: re.Match, key: str = key) -> str:
                ref = m.group(1)
                if ref not in out:
                    raise ValueError(f"paths.{key} references unknown key ${{{ref}}}")
                return str(out[ref])
            out[key] = re.sub(r"\$\{([^}]+)\}", repl, out[key])
    raise ValueError(f"cyclic ${{...}} reference in paths: {pending}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read an enrichment config, resolving ``${key}`` self-references in ``paths``.

    `path` may be omitted only if ``$SCADS_DRVI_ENRICH_CONFIG`` is set. There is no
    built-in default: this module used to sit beside one particular ``config.yaml`` and
    default to it, which made the package's behaviour depend on its own location. A
    caller that has a canonical config supplies the path.
    """
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR)
    if path is None:
        raise ValueError(
            f"no config given. Pass a path, or set ${CONFIG_ENV_VAR}. This function has "
            f"no default because the package must not depend on where it is installed."
        )
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"no enrichment config at {resolved}")
    cfg = yaml.safe_load(resolved.read_text())
    cfg["paths"] = _expand_paths(cfg["paths"])
    cfg.setdefault("_config_path", str(resolved))
    return cfg


def ldsc_env(cfg: dict[str, Any]) -> Path:
    """Root of the pinned LDSC env. $SCADS_LDSC_ENV overrides."""
    return Path(os.environ.get("SCADS_LDSC_ENV", cfg["envs"]["ldsc"]))


def ldsc_script(cfg: dict[str, Any], name: str) -> Path:
    """Path to an LDSC entry point (``ldsc.py`` / ``munge_sumstats.py``).

    LDSC is the one component of this arm we do not control. It is installed by
    pip into its own env with a 2022 numpy/pandas pin, and invoked as a
    subprocess -- its shebang points at that env's interpreter, so it is
    executed directly and our code stays on a current stack.

    ``make_annot.py`` is installed too but deliberately unused;
    build_annotations.py replaces it (and it is the only thing that would need
    pybedtools).
    """
    path = ldsc_env(cfg) / "bin" / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at {path}.\n"
            f"Point $SCADS_LDSC_ENV (or cfg['envs']['ldsc']) at a prefix whose bin/ "
            f"carries the Python LDSC entry points, installed with --no-deps from\n"
            f"  git+https://github.com/CBIIT/ldsc"
            f"@b860edf3898318066eeb307b758e180040b611af\n"
            f"and patched for the two py2->py3 bugs -- see assert_ldsc_patched.\n"
            f"The Rust ldsc resolved by scads_drvi.enrich.binary needs neither."
        )
    return path


def assert_ldsc_patched(cfg: dict[str, Any], patcher: str | Path | None = None) -> None:
    """Fail now, not an hour in, if the pip-installed LDSC is unpatched.

    Every pip-installable LDSC ships two py2->py3 port bugs that make --l2
    unusable, and the fatal one (.M / .M_5_50 opened 'wb' then written as str)
    fires only AFTER the full LD-score computation -- about a minute per
    chromosome, with the .ldscore.gz already on disk. Checking up front turns
    22 wasted minutes into an instant, actionable error.

    The patch script owns the fix list; --check is delegated to it so there is exactly
    one copy of what "patched" means. Its location is a property of a deployment, so it
    is an argument (or ``cfg["paths"]["ldsc_patcher"]``) rather than something derived
    from where this module happens to live.

    This whole function is legacy: it exists for the Python LDSC. The Rust binary
    resolved by :mod:`scads_drvi.enrich.binary` needs no patching and no pinned
    environment, and once the pipeline is on it this check has nothing left to verify.
    """
    if patcher is None:
        patcher = (cfg.get("paths") or {}).get("ldsc_patcher")
    if patcher is None:
        log(
            "WARNING: no ldsc_patcher path given, cannot verify the LDSC patch. "
            "The Rust ldsc (scads_drvi.enrich.binary) needs no patch at all."
        )
        return
    patcher = Path(patcher)
    if not patcher.exists():          # not fatal; the tools may still be fine
        log(f"WARNING: {patcher} missing, cannot verify the LDSC patch")
        return
    proc = subprocess.run(
        [sys.executable, str(patcher), str(ldsc_env(cfg)), "--check"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            (proc.stdout + proc.stderr).strip()
            + f"\n\nFix it with:\n  python {patcher} {ldsc_env(cfg)}"
        )


#: BLAS/OpenMP thread caps for every LDSC subprocess.
#:
#: numpy in the ldsc env links a threaded BLAS that defaults to every core, and
#: an --h2 run was measured at 28 min CPU for 4 min wall. Under Snakemake that
#: nests: total load is (threads per job) x -j, which is the hazard the arm
#: 01a/02 rules already guard against with pin_threads(1). Parallelism here
#: comes from -j alone, so each LDSC process gets exactly one thread.
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def pinned_env(threads: int = 1) -> dict[str, str]:
    """os.environ plus single-threaded BLAS, for handing to subprocess.run."""
    env = dict(os.environ)
    env.update({var: str(threads) for var in THREAD_VARS})
    return env


def width_dir(cfg: dict[str, Any], width: str) -> Path:
    return Path(cfg["paths"]["annot_root"]) / width


def out_dir(cfg: dict[str, Any], width: str) -> Path:
    return Path(cfg["paths"]["scratch"]) / width


#: annot keys a per-width override may set. Anything else is a typo, and a typo in an
#: override is invisible: the run completes using the global default.
ANNOT_OVERRIDABLE = ("mode", "top_frac", "top_n", "threshold", "min_frac_snps", "fail_on_small",
                     "fdr", "lfc_delta", "lfc_source", "lfc_scale", "support",
                     "value_source", "on_small", "continuous_scale")


def resolve_annot(cfg: dict[str, Any], width: str) -> dict[str, Any]:
    """`cfg["annot"]` with `cfg["annot_overrides"][width]` merged over it.

    `annot.mode` is a single global value, which is why config_continuous.yaml exists as
    a separate file at all. But the FDR arm needs a per-width setting for a different
    reason: one width may be binarized from a counterfactual that the control widths do
    not have, and forcing every width onto one mode would either break those controls or
    require a third config file per arm.

    Every consumer -- the builder, the provenance guard, the Snakefile -- must resolve
    through here, so none of them can disagree about which mode a width is running.
    """
    annot = dict(cfg["annot"])
    over = (cfg.get("annot_overrides") or {}).get(width) or {}
    unknown = set(over) - set(ANNOT_OVERRIDABLE)
    if unknown:
        raise SystemExit(
            f"ERROR: annot_overrides[{width!r}] sets unknown key(s) {sorted(unknown)}; "
            f"allowed: {sorted(ANNOT_OVERRIDABLE)}. A typo here would silently run the "
            f"global default.")
    annot.update(over)
    return annot


# ---------------------------------------------------------------------------
# Factor selection (stage 1a)
# ---------------------------------------------------------------------------

def _as_bool(col: pd.Series) -> pd.Series:
    """Coerce a possibly-string boolean column to real bools.

    latent_stats.tsv is written by pandas from numpy bools, so it round-trips
    as literal "True"/"False". pandas usually re-infers that as bool dtype, but
    not if the column picked up any other value -- in which case guessing
    silently is worse than failing.
    """
    if col.dtype == bool:
        return col
    mapped = col.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        bad = sorted(set(col.astype(str)[mapped.isna()]))
        raise ValueError(f"non-boolean values in `vanished`: {bad[:5]}")
    return mapped.astype(bool)


def read_latent_stats(path: str | Path) -> pd.DataFrame:
    """Read DRVI's latent_stats.tsv, or raise.

    Raising is deliberate. Silently enriching all 32 dimensions because the
    inspect stage was never run would waste an S-LDSC run per dead dimension
    AND shift every surviving factor's shrunk enrichment, with nothing in the
    output to say so. After a re-fit, `inspect_drvi.py latent` is a
    prerequisite of this arm.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- factor selection needs DRVI's vanished flags.\n"
            f"They come from the fit's inspection stage (DRVI's\n"
            f"set_latent_dimension_stats), whose output directory must resolve to\n"
            f"{p.parent}."
        )
    df = pd.read_csv(p, sep="\t")
    missing = {"dim", "vanished"} - set(df.columns)
    if missing:
        raise ValueError(f"{p} is missing required column(s): {sorted(missing)}")
    df["vanished"] = _as_bool(df["vanished"])
    for extra in ("vanished_positive_direction", "vanished_negative_direction"):
        if extra in df.columns:
            df[extra] = _as_bool(df[extra])
    return df


def select_factors(
    latent_stats: pd.DataFrame,
    dim_names: Sequence[str],
    exclude_vanished: bool = True,
    exclude_dims: Iterable[str] = (),
) -> pd.DataFrame:
    """Build the factor map: which dimensions get enriched, and under which k index.

    ``dim_names`` is authoritative for ordering and membership -- it comes from
    topic_loadings.tsv, i.e. the contract, not from the inspect artifact.

    Returns a frame with one row per dimension and columns
    ``dim, vanished, kept, drop_reason, annot_index``. ``annot_index`` is the
    1-based k used in S-LDSC filenames (k1..kK_eff) and is assigned over KEPT
    dimensions only; dropped rows get <NA>.

    Upstream SCADS names annotations positionally, so without this mapping the
    route from `k7.results` back to `dim_*` after a drop is guesswork. Nothing
    downstream may renumber on its own.
    """
    dim_names = list(dim_names)
    stats_dims = list(latent_stats["dim"].astype(str))
    if set(stats_dims) != set(dim_names):
        only_stats = sorted(set(stats_dims) - set(dim_names))
        only_contract = sorted(set(dim_names) - set(stats_dims))
        raise ValueError(
            "latent_stats.tsv does not describe the same dimensions as the "
            f"loadings. Only in latent_stats: {only_stats}; only in loadings: "
            f"{only_contract}. These must come from the same fit."
        )

    stats = latent_stats.set_index(latent_stats["dim"].astype(str))
    exclude_dims = set(exclude_dims or ())
    unknown = exclude_dims - set(dim_names)
    if unknown:
        raise ValueError(f"factors.exclude_dims names unknown dimensions: {sorted(unknown)}")

    # The loadings are a ReLU of the latent (positive side only) and the factors
    # are the positive OOD log-fold-change, so `vanished_positive_direction` is
    # strictly the flag that matches the contract. On both current fits it
    # agrees with `vanished` on every dimension; say so loudly if that ever
    # stops being true rather than quietly picking one.
    pos_col = "vanished_positive_direction"
    if pos_col in stats.columns:
        disagree = stats.index[stats["vanished"] != stats[pos_col]].tolist()
        if disagree:
            log(
                f"WARNING: `vanished` and `{pos_col}` disagree on {disagree}. "
                "The contract's loadings are a ReLU, so the positive-direction "
                "flag is the relevant one -- review before trusting this run."
            )

    rows = []
    for dim in dim_names:
        vanished = bool(stats.loc[dim, "vanished"])
        reason = ""
        if exclude_vanished and vanished:
            reason = "vanished"
        elif dim in exclude_dims:
            reason = "exclude_dims"
        rows.append({"dim": dim, "vanished": vanished, "kept": reason == "",
                     "drop_reason": reason})

    fmap = pd.DataFrame(rows)
    fmap["annot_index"] = pd.array(
        [None] * len(fmap), dtype="Int64"
    )
    fmap.loc[fmap["kept"], "annot_index"] = np.arange(1, int(fmap["kept"].sum()) + 1)

    if not fmap["kept"].any():
        raise ValueError(
            "every dimension was excluded -- nothing left to enrich. Check "
            "factors.exclude_vanished / factors.exclude_dims in config.yaml."
        )
    return fmap


def kept_dims(fmap: pd.DataFrame) -> list[str]:
    """Kept dimension names, in annot_index order."""
    keep = fmap[fmap["kept"]].sort_values("annot_index")
    return keep["dim"].tolist()


# ---------------------------------------------------------------------------
# The factorize contract
# ---------------------------------------------------------------------------

def read_loadings(path: str | Path) -> pd.DataFrame:
    """cells x K. Index = cell barcodes, columns = dim_0..dim_{K-1}."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    return df


def read_factors(path: str | Path) -> pd.DataFrame:
    """features x K, index canonicalised to chr:start-end.

    topic_factors.tsv has an empty leading header cell; index_col=0 handles it.
    Peak names are normalised through the shared normalize_peak_name() so the
    underscore and dash spellings behave, and anything unparseable is dropped
    loudly rather than silently becoming an annotation with no coordinates.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    raw = df.index.astype(str)
    norm = [normalize_peak_name(v) for v in raw]
    bad = [r for r, n in zip(raw, norm, strict=True) if n is None]
    if bad:
        log(f"WARNING: dropping {len(bad):,} unparseable feature name(s), e.g. {bad[:3]}")
    keep = [n is not None for n in norm]
    df = df.loc[keep]
    df.index = pd.Index([n for n in norm if n is not None], name="peak")
    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"duplicate feature names after normalisation: {dup[:5]}")
    return df


def load_contract(
    drvi_dir: str | Path,
    fmap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read loadings + factors and subset BOTH to the kept dims, from one map.

    Subsetting from the same list is the point: doing it independently is how
    the two matrices drift out of alignment, and nothing downstream would
    reveal it -- the shapes would still agree.
    """
    drvi_dir = Path(drvi_dir)
    loadings = read_loadings(drvi_dir / LOADINGS_TSV)
    factors = read_factors(drvi_dir / FACTORS_TSV)
    if list(loadings.columns) != list(factors.columns):
        raise ValueError(
            "loadings and factors disagree on dimension columns: "
            f"{list(loadings.columns)[:5]}... vs {list(factors.columns)[:5]}..."
        )
    keep = kept_dims(fmap)
    return loadings[keep], factors[keep]


#: annot params that, if changed, invalidate every LD score downstream. Maps the key as
#: it appears in annotations.meta.json to the resolved-annot key it comes from. This
#: tuple used to be dead code with no consumer; assert_annot_matches_config drives off it
#: now, so adding a parameter to the config and forgetting the guard is one edit, not two.
ANNOT_PROVENANCE = (
    ("annot_mode", "mode"),
    ("annot_top_frac", "top_frac"),
    ("annot_top_n", "top_n"),
    ("annot_threshold", "threshold"),
    ("annot_fdr", "fdr"),
    ("annot_lfc_delta", "lfc_delta"),
    ("annot_lfc_source", "lfc_source"),
    ("annot_lfc_scale", "lfc_scale"),
    ("annot_support", "support"),
    ("annot_value_source", "value_source"),
    ("annot_on_small", "on_small"),
)


def assert_annot_matches_config(cfg: dict[str, Any], odir: str | Path,
                                chroms: Sequence[int] | None = None) -> dict:
    """Refuse to build on annotations that a different config produced.

    Timestamps do not catch this. Editing annot.top_frac leaves every existing
    annot file newer than its inputs, so Snakemake happily skips
    build_annotations and the LD scores get computed against the OLD
    annotations while every log and meta file names the new parameters. The
    result is a complete, plausible, wrong run.
    """
    odir = Path(odir)
    meta_path = odir / "annotations.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found -- run build_annotations.py for this width first."
        )
    meta = json.loads(meta_path.read_text())
    width = meta.get("width", odir.name)
    acfg = resolve_annot(cfg, width)
    fcfg = cfg["factors"]
    want = {mkey: acfg.get(akey) for mkey, akey in ANNOT_PROVENANCE}
    want["exclude_vanished"] = bool(fcfg["exclude_vanished"])
    want["exclude_dims"] = list(fcfg.get("exclude_dims") or ())
    # Annotations written before a parameter existed have no key for it. Absent is not a
    # mismatch -- it is "this run predates the parameter" -- but a PRESENT-and-different
    # value always is. Only compare keys the meta actually carries, except for mode, which
    # every meta has ever written and whose absence would mean the file is not ours.
    if "annot_mode" not in meta:
        raise SystemExit(f"ERROR: {meta_path} has no annot_mode; it was not written by "
                         f"build_annotations.py")
    want = {k: v for k, v in want.items() if k in meta}
    if chroms is not None:
        want["chroms"] = [int(c) for c in chroms]
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        detail = "; ".join(f"{k}: on disk {got!r}, config says {exp!r}"
                           for k, (got, exp) in bad.items())
        raise SystemExit(
            f"ERROR: the annotations in {odir} were built with different "
            f"parameters than config.yaml now specifies ({detail}).\n"
            "Rerun build_annotations.py for this width, or restore the config "
            "that produced them. Continuing would compute LD scores against "
            "the OLD annotations while every log named the NEW parameters."
        )
    return meta


def parse_peaks(index: Iterable[str]) -> pd.DataFrame:
    """Split canonical chr:start-end names into a chrom/start/end frame.

    Coordinates are returned as the peak name states them, which for this pipeline's
    output is 0-based half-open (BED), matching what the peak merge wrote.
    """
    chrom, start, end = [], [], []
    for name in index:
        seq, _, rest = str(name).partition(":")
        lo, _, hi = rest.partition("-")
        chrom.append(seq)
        start.append(int(lo))
        end.append(int(hi))
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end})
