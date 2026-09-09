"""A real, public scATAC dataset, fetched and shaped into what the loaders expect.

Every other fixture in this suite writes its h5ad by hand with ``h5py``. That is fast and
hermetic, and it means **nothing here has ever read a file that anndata actually wrote** --
so `io.h5ad`'s backed-CSR reader, which pulls ``X/data``, ``X/indices``, ``X/indptr`` and
``X.attrs["shape"]`` out with raw h5py, is checked against a hand-built encoding rather
than against the writer it has to survive. `read_rows_csr`, `h5ad_shape`, `h5ad_var_names`
and `keep_rows` have no other caller in this repository at all.

This module closes that hole with the smallest real thing that fits: 10x Genomics' public
PBMC scATAC demonstration run (500 cells, CellRanger ATAC 1.2.0), fetched once,
checksum-verified, and written back out **through anndata** so the read path is exercised
against the real writer.

What is real and what is not
----------------------------
Real, and unmodified: the counts, the barcodes, the peak coordinates, and the per-cell QC
metrics. The peak names 10x ships (``chr1:565153-565499``) are already the canonical
spelling `io.peaks` writes.

Real, and derived here deterministically: the three grouping columns. 10x's own clustering
is behind an 84 MB download, so rather than invent cell types this bins real QC signal by
rank -- `frip_stratum`, `signal_stratum`, `duplicate_stratum`. **They are strata, not cell
types**, and they are named so nobody mistakes them for biology.

Fabricated outright, with seed 0: the fit tree and the S-LDSC results built by
`synthesize_run`. No real DRVI fit or heritability result exists for a public dataset. The
loadings are a *constructed* function of real FRiP and depth, so aggregation over the real
strata has something to find -- that association is designed in, not discovered, and no
number coming out of it means anything biological. Everything fabricated says "synthetic"
in its name, and only the real inputs are ever cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scads_drvi._util.paths import cache_dir
from scads_drvi.config import Project
from scads_drvi.io.peaks import PEAK_RE

__all__ = [
    "SmokeUnavailable",
    "REMOTE",
    "SOURCE_BASE",
    "LAYOUT_VERSION",
    "EXPECTED_N_OBS",
    "EXPECTED_N_VARS",
    "EXPECTED_NNZ",
    "SYNTHETIC_TRAITS",
    "smoke_cache_dir",
    "fetch",
    "ensure_inputs",
    "build_matrix",
    "ensure_matrix",
    "synthesize_run",
]

#: The published run. Pinned by URL and by checksum; a bump is a reviewed change.
SOURCE_BASE = (
    "https://cf.10xgenomics.com/samples/cell-atac/1.2.0/atac_pbmc_500_nextgem"
)
_PREFIX = "atac_pbmc_500_nextgem"

#: Bump whenever `build_matrix` changes what it writes, so a stale derived file is not
#: silently reused -- it is part of the derived filename.
LAYOUT_VERSION = 1

SEED = 0

#: Fabricated trait names. They are not real GWAS, and they say so.
SYNTHETIC_TRAITS: tuple[str, ...] = ("synthetic_trait_a", "synthetic_trait_b")

_FRIP_LEVELS = ("low", "mid", "high")
_PROMOTER_LEVELS = ("distal", "promoter")
_DUPLICATE_LEVELS = ("weak", "moderate", "strong")

#: Columns taken verbatim from the per-barcode CSV. `passed_filters` is renamed to
#: `n_fragment` because that is `keep_rows`'s default `depth_col`, so the gate is exercised
#: without being told where to look.
_CSV_COLUMNS = (
    "barcode",
    "total",
    "duplicate",
    "mitochondrial",
    "passed_filters",
    "is__cell_barcode",
    "TSS_fragments",
    "promoter_region_fragments",
    "enhancer_region_fragments",
    "blacklist_region_fragments",
    "peak_region_fragments",
    "peak_region_cutsites",
)

_RENAME = {"passed_filters": "n_fragment", "total": "total_fragment"}


class SmokeUnavailable(RuntimeError):
    """The dataset is not cached and could not (or may not) be fetched."""


@dataclass(frozen=True)
class RemoteFile:
    """One pinned published file."""

    name: str
    sha256: str
    n_bytes: int

    @property
    def url(self) -> str:
        return f"{SOURCE_BASE}/{self.name}"

    @property
    def pinned(self) -> bool:
        return bool(self.sha256)


#: Published SHA-256 and sizes. A bump is a reviewed change, not a silent upgrade.
REMOTE: Mapping[str, RemoteFile] = {
    "matrix": RemoteFile(
        name=f"{_PREFIX}_filtered_peak_bc_matrix.h5",
        sha256="e83ecf6375e63dcca1623672a5ee7b520d730480cd2a9266d763590b02e10896",
        n_bytes=7_729_733,
    ),
    "per_barcode": RemoteFile(
        name=f"{_PREFIX}_singlecell.csv",
        sha256="974fa4ad9d4f959b1fec631b0ad5aaf5f707c26ede26c7410657420774809709",
        n_bytes=14_393_663,
    ),
}

#: What `build_matrix` produces from the pinned inputs. Deterministic, so asserting it
#: catches builder drift. "500" in the run's name is nominal: 482 barcodes were called.
EXPECTED_N_OBS = 482
EXPECTED_N_VARS = 47_843
EXPECTED_NNZ = 3_716_243


def smoke_cache_dir(cache: str | Path | None = None) -> Path:
    """Where the fetched inputs and the derived h5ad live."""
    if cache is not None:
        return Path(cache) / "scads_drvi" / "smoke" / _PREFIX
    return cache_dir("smoke", _PREFIX)


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _mismatch(path: Path, item: RemoteFile) -> str | None:
    """Why `path` is not `item`, or None. Size first -- it is far cheaper than the hash."""
    size = path.stat().st_size
    if size != item.n_bytes:
        return f"expected {item.n_bytes} bytes, got {size}"
    if item.pinned:
        got = _digest(path)
        if got != item.sha256:
            return f"expected sha256 {item.sha256}, got {got}"
    return None


#: The CDN in front of the published files answers ``Python-urllib/*`` with 403, so the
#: fetcher has to name itself. Verified: that default gets 403 where this gets 206.
_USER_AGENT = "scads-drvi-tests/0.1 (+https://github.com/mikecuoco/scads_drvi)"


def _download(item: RemoteFile, dest: Path) -> None:
    """Fetch to a sibling temporary file, then rename -- never a truncated cache entry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".partial.{os.getpid()}")
    request = urllib.request.Request(item.url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with tmp.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise SmokeUnavailable(f"could not fetch {item.url}: {exc}") from exc
    tmp.replace(dest)


def fetch(
    item: RemoteFile,
    *,
    cache: str | Path | None = None,
    allow_download: bool = False,
    verify: bool = True,
) -> Path:
    """The cached path for `item`, fetching it only when allowed.

    A cached file that fails verification is re-downloaded **once** when downloading is
    allowed, then re-checked. A restored CI cache is the common way to get a stale entry,
    and refusing outright would red the build until someone purged it by hand.
    """
    if verify and not item.pinned:
        raise SmokeUnavailable(
            f"{item.name} has no pinned sha256; run `python tests/smoke_data.py` to "
            f"generate the constants, or pass verify=False to do exactly that"
        )

    dest = smoke_cache_dir(cache) / item.name
    if dest.exists():
        reason = _mismatch(dest, item) if verify else None
        if reason is None:
            return dest
        if not allow_download:
            raise SmokeUnavailable(f"cached {dest} is unusable ({reason}) and downloading is off")
        dest.unlink()

    if not allow_download:
        raise SmokeUnavailable(
            f"{item.name} is not cached and downloading is disabled. Set "
            f"SCADS_DRVI_SMOKE_DOWNLOAD=1, or place the file from {item.url} at {dest}"
        )

    _download(item, dest)
    reason = _mismatch(dest, item) if verify else None
    if reason is not None:
        dest.unlink(missing_ok=True)
        raise SmokeUnavailable(f"downloaded {item.name} failed verification: {reason}")
    return dest


def ensure_inputs(
    *,
    cache: str | Path | None = None,
    allow_download: bool = False,
    verify: bool = True,
) -> dict[str, Path]:
    """Every published input, cached."""
    return {
        key: fetch(item, cache=cache, allow_download=allow_download, verify=verify)
        for key, item in REMOTE.items()
    }


def _rank_strata(values: np.ndarray, labels: Sequence[str]) -> pd.Categorical:
    """Equal-sized ordered strata by rank.

    Deliberately not ``pd.qcut``: an argsort split cannot raise on duplicate bin edges
    (FRiP ties are common at this depth), gives exactly equal group sizes, and returns the
    same thing on pandas 2 and 3.
    """
    values = np.asarray(values, dtype=float)
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    slot = (ranks * len(labels)) // max(len(values), 1)
    slot = np.clip(slot, 0, len(labels) - 1)
    return pd.Categorical(
        [labels[i] for i in slot], categories=list(labels), ordered=True
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """A ratio for *ranking*, where an unmeasured cell sorts lowest rather than raising."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, 0.0)


def build_matrix(inputs: Mapping[str, Path], out: Path) -> dict:
    """Write the fetched run out as a cells x peaks h5ad, and return its manifest."""
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    with h5py.File(inputs["matrix"], "r") as handle:
        group = handle["matrix"]
        n_features, n_barcodes = (int(v) for v in group["shape"][:])
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        barcodes = [b.decode() for b in group["barcodes"][:]]
        features = group["features"]
        key = "id" if "id" in features else "name"
        peaks = [p.decode() for p in features[key][:]]

    # 10x ships features x barcodes with indptr over barcodes -- which IS a CSR of shape
    # (barcodes, features). Reinterpreting the shape is the whole transpose: no re-sort, no
    # copy, no memory spike.
    matrix = sp.csr_matrix(
        (data.astype(np.float32), indices, indptr), shape=(n_barcodes, n_features)
    )
    # Measured: 10x writes every one of the 482 columns with unsorted indices, so this is
    # not a defensive no-op. Sorting reorders within each row and changes no value; without
    # it the file is not in canonical CSR form, which scipy and anndata both assume.
    matrix.sort_indices()

    if len(set(peaks)) != len(peaks):
        raise ValueError(f"{len(peaks) - len(set(peaks))} duplicate peak names")
    bad = [p for p in peaks if not PEAK_RE.match(p)]
    if bad:
        raise ValueError(f"{len(bad)} peak names are not chr:start-end, e.g. {bad[:3]}")

    frame = pd.read_csv(
        inputs["per_barcode"], usecols=list(_CSV_COLUMNS), index_col="barcode"
    )
    # The matrix is authoritative for both membership and order. Reindexing onto it (rather
    # than filtering the CSV and hoping the orders agree) is also what catches the classic
    # bytes-vs-str failure: undecoded barcodes would join to nothing and leave every row NaN
    # while every downstream assertion still "passed" on the resulting garbage.
    cells = frame.reindex(barcodes)
    if cells.isna().to_numpy().any():
        raise ValueError("some matrix barcodes are absent from the per-barcode CSV")
    called = set(frame.index[frame["is__cell_barcode"] == 1])
    if called != set(barcodes):
        raise ValueError(
            f"filtered matrix and is__cell_barcode disagree: "
            f"{len(called - set(barcodes))} / {len(set(barcodes) - called)} one-sided"
        )
    cells = cells.drop(columns=["is__cell_barcode"]).rename(columns=_RENAME)

    obs = pd.DataFrame(index=pd.Index(barcodes, dtype=object, name="barcode"))
    for column in cells.columns:
        # Plain int32, never a pandas nullable dtype: anndata encodes those as a group with
        # values/mask, and io.artifacts.read_obs treats every obs group as a categorical.
        obs[column] = cells[column].to_numpy(dtype=np.int32)

    frip = _safe_ratio(obs["peak_region_fragments"], obs["n_fragment"])
    promoter = _safe_ratio(obs["promoter_region_fragments"], obs["n_fragment"])
    duplication = _safe_ratio(obs["duplicate"], obs["total_fragment"])

    obs["frip_stratum"] = _rank_strata(frip, _FRIP_LEVELS)
    obs["duplicate_stratum"] = _rank_strata(duplication, _DUPLICATE_LEVELS)

    # Nested strictly inside frip_stratum, so there is a real fine-within-coarse axis. The
    # declared order follows the coarse levels, which is not alphabetical -- exactly what
    # block_order has to respect instead of sorting.
    fine = np.empty(len(obs), dtype=object)
    for level in _FRIP_LEVELS:
        mask = np.asarray(obs["frip_stratum"] == level)
        half = _rank_strata(promoter[mask], _PROMOTER_LEVELS)
        fine[mask] = [f"{level}.{h}" for h in half]
    obs["signal_stratum"] = pd.Categorical(
        fine,
        categories=[f"{lv}.{h}" for lv in _FRIP_LEVELS for h in _PROMOTER_LEVELS],
        ordered=True,
    )

    # var carries the peak index and nothing else. Its index is left unnamed on purpose:
    # anndata then writes attrs["_index"] == "_index", which is h5ad_var_names' default
    # branch and is otherwise untested.
    var = pd.DataFrame(index=pd.Index(peaks, dtype=object))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".partial.{os.getpid()}")
    ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(tmp, compression="gzip")
    tmp.replace(out)

    return {
        "layout_version": LAYOUT_VERSION,
        "source_base": SOURCE_BASE,
        "n_obs": int(matrix.shape[0]),
        "n_vars": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "files": {
            key: {"url": item.url, "sha256": item.sha256, "n_bytes": item.n_bytes}
            for key, item in REMOTE.items()
        },
    }


def ensure_matrix(
    *,
    cache: str | Path | None = None,
    allow_download: bool = False,
    verify: bool = True,
) -> tuple[Path, dict]:
    """The derived h5ad and its manifest, building them once and caching both."""
    root = smoke_cache_dir(cache)
    out = root / f"matrix.v{LAYOUT_VERSION}.h5ad"
    manifest_path = root / f"manifest.v{LAYOUT_VERSION}.json"
    if out.exists() and manifest_path.exists():
        return out, json.loads(manifest_path.read_text())

    inputs = ensure_inputs(cache=cache, allow_download=allow_download, verify=verify)
    manifest = build_matrix(inputs, out)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out, manifest


def _read_real_cells(adata_path: Path) -> pd.DataFrame:
    """Barcodes, depth and FRiP, read without the decoder that is under test."""
    import h5py

    with h5py.File(adata_path, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        barcodes = [b.decode() if isinstance(b, bytes) else b for b in obs[index_key][:]]
        depth = obs["n_fragment"][:].astype(float)
        in_peaks = obs["peak_region_fragments"][:].astype(float)
    return pd.DataFrame(
        {"n_fragment": depth, "frip": _safe_ratio(in_peaks, depth)}, index=barcodes
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)


def synthesize_run(
    root: Path,
    adata_path: Path,
    *,
    n_latent: int = 12,
    n_kept: int = 9,
    traits: Sequence[str] = SYNTHETIC_TRAITS,
    seed: int = SEED,
) -> dict:
    """Fabricate a complete fit + enrichment tree over the dataset's real cells.

    None of this is measured. The loadings are a constructed function of real FRiP and
    depth so that aggregation over the real strata is not uniform noise; the S-LDSC numbers
    are made up outright. Nothing here supports a claim about PBMCs.
    """
    cells = _read_real_cells(adata_path)
    n_cells = len(cells)
    dims = [f"dim_{j}" for j in range(n_latent)]

    # One independent stream per artifact, so adding another later does not shift the draws
    # of the ones already written.
    entropy = np.random.SeedSequence(entropy=seed)
    rng_loadings, rng_embedding = (
        np.random.default_rng(child) for child in entropy.spawn(2)
    )

    frip = _zscore(cells["frip"].to_numpy())
    depth = _zscore(np.log1p(cells["n_fragment"].to_numpy()))
    weights = np.linspace(-1.0, 1.0, n_latent)
    loadings = np.clip(
        np.outer(frip, weights)
        + np.outer(depth, weights[::-1])
        + rng_loadings.normal(0.0, 0.25, size=(n_cells, n_latent))
        + 1.5,
        0,
        None,
    ).astype(np.float32)

    project = Project(root=root, fit="synthetic_k12", traits=tuple(traits))
    fit_dir = project.fit_dir()
    (fit_dir / "inspect").mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(loadings, index=cells.index, columns=dims)
    frame.index.name = "barcode"
    frame.to_csv(fit_dir / "topic_loadings.tsv", sep="\t")
    np.savez(
        fit_dir / "topic_loadings.npz",
        cells=np.asarray(cells.index, dtype=object),
        factors=np.array(dims),
        loadings=loadings,
    )

    import h5py

    with h5py.File(adata_path, "r") as handle:
        peaks = [p.decode() if isinstance(p, bytes) else p for p in handle["var"]["_index"][:]]
    factors = np.abs(
        np.linspace(0.1, 1.0, len(peaks))[:, None] * np.linspace(0.5, 1.5, n_latent)[None, :]
    ).astype(np.float32)
    pd.DataFrame(factors, index=pd.Index(peaks, name="peak"), columns=dims).to_csv(
        fit_dir / "topic_factors.tsv", sep="\t"
    )

    (fit_dir / "fit.meta.json").write_text(
        json.dumps(
            {
                "n_latent": n_latent,
                "n_cells": n_cells,
                "n_features": len(peaks),
                "batch_key": None,
                "min_fragment": 0,
                "depth_col": "n_fragment",
                "seed": seed,
                "synthetic": True,
                "generator": "tests/smoke_data.py",
                "layout_version": LAYOUT_VERSION,
            },
            indent=2,
        )
    )
    pd.DataFrame({"dim": dims, "vanished": [False] * n_latent}).to_csv(
        fit_dir / "inspect" / "latent_stats.tsv", sep="\t", index=False
    )

    arm = "synthetic_topfrac"
    arm_dir = project.enrich_dir(arm)
    (arm_dir / "results").mkdir(parents=True, exist_ok=True)
    kept = [True] * n_kept + [False] * (n_latent - n_kept)
    pd.DataFrame(
        {
            "dim": dims,
            "vanished": [False] * n_latent,
            "kept": kept,
            "drop_reason": ["" if k else "annot_too_small" for k in kept],
            "annot_index": [i + 1 if k else None for i, k in enumerate(kept)],
        }
    ).to_csv(arm_dir / "factor_map.tsv", sep="\t", index=False)

    for trait_index, trait in enumerate(traits):
        trait_dir = arm_dir / "results" / trait
        trait_dir.mkdir(parents=True, exist_ok=True)
        for slot in range(n_kept):
            z = 4.5 - 0.6 * slot + 0.3 * trait_index
            pd.DataFrame(
                {
                    "Category": [f"k{slot + 1}L2_0"],
                    "Coefficient": [1e-8 * z],
                    "Coefficient_std_error": [1e-8],
                    "Coefficient_z-score": [z],
                    "n_categories": [40],
                    "total_h2": [0.11],
                }
            ).to_csv(trait_dir / f"k{slot + 1}.results", sep="\t", index=False)

    coords = pd.DataFrame(
        rng_embedding.normal(size=(n_cells, 2)),
        index=cells.index,
        columns=["embed_1", "embed_2"],
    )
    coords.index.name = "barcode"
    coords.to_csv(root / "embedding.tsv", sep="\t")

    return {
        "project": project,
        "arm": arm,
        "fit": "synthetic_k12",
        "traits": tuple(traits),
        "coords": root / "embedding.tsv",
        "n_cells": n_cells,
        "n_features": len(peaks),
    }


if __name__ == "__main__":  # pragma: no cover -- the tool that pins the constants above
    import tempfile

    print(f"fetching from {SOURCE_BASE}")
    paths = ensure_inputs(allow_download=True, verify=False)
    for name, path in paths.items():
        print(f"  {name}: {REMOTE[name].name}")
        print(f"    n_bytes={path.stat().st_size}")
        print(f"    sha256={_digest(path)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        built = build_matrix(paths, Path(tmpdir) / "matrix.h5ad")
    print(f"  n_obs={built['n_obs']} n_vars={built['n_vars']} nnz={built['nnz']}")
