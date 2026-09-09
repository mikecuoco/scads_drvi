"""Typed loaders for what a finished run leaves on disk.

The h5ad obs decode this replaces existed in six places, all slightly different, while
a tested implementation sat unused. It is one function here, and which columns matter is
the caller's decision -- this module knows about cells and factors, not about what any
particular column means.

``h5py`` is imported inside the functions that need it, so this module imports in an
environment that has none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from scads_drvi.config import Project
    from scads_drvi.labels import FactorLabels

__all__ = [
    "obs_columns",
    "read_obs",
    "cell_metadata",
    "read_loadings",
    "read_umap",
    "Interpretation",
    "load_interpretation",
]


def _decode(values) -> np.ndarray:
    """Bytes -> str for an h5py array, leaving anything else alone."""
    array = np.asarray(values)
    if array.dtype.kind in "SO":
        return np.array(
            [v.decode() if isinstance(v, bytes) else str(v) for v in array.ravel()],
            dtype=object,
        ).reshape(array.shape)
    return array


def obs_columns(path: str | Path) -> list[str]:
    """Column names available in an h5ad's ``obs``, without reading any of them."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        return sorted(handle["obs"].keys())


def read_obs(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    index: str | None = None,
) -> "pd.DataFrame":
    """Read obs columns out of an h5ad, decoding anndata's categorical encoding.

    `columns` defaults to every column present. `index` defaults to whatever the file
    records in ``obs.attrs["_index"]``, which is where anndata puts it -- guessing a
    column name instead is how a join silently matches nothing.

    Only obs is touched; ``X`` is never opened, so this is cheap on a matrix of any size.
    """
    import h5py
    import pandas as pd

    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "obs" not in handle:
            raise ValueError(f"{path} has no obs group -- is it an h5ad?")
        obs = handle["obs"]
        available = list(obs.keys())

        if index is None:
            index = obs.attrs.get("_index")
            if isinstance(index, bytes):
                index = index.decode()

        wanted = list(columns) if columns is not None else available
        if index is not None and index in available and index not in wanted:
            wanted.append(index)

        missing = [c for c in wanted if c not in available]
        if missing:
            raise KeyError(
                f"{path} has no obs column(s) {missing}. Available: {available}"
            )

        data: dict[str, np.ndarray] = {}
        for name in wanted:
            item = obs[name]
            if isinstance(item, h5py.Group):
                # anndata's categorical encoding: integer codes into a category table.
                categories = _decode(item["categories"][:])
                codes = np.asarray(item["codes"][:])
                data[name] = pd.Categorical.from_codes(codes, categories=categories)
            else:
                data[name] = _decode(item[:])

    frame = pd.DataFrame(data)
    if index is not None and index in frame.columns:
        frame = frame.set_index(index)
    return frame


def cell_metadata(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    index: str | None = None,
    derived: Mapping[str, tuple[str, str]] | None = None,
) -> "pd.DataFrame":
    """Per-cell metadata, with optional caller-named ratio columns.

    `derived` maps a new column name to a ``(numerator, denominator)`` pair of existing
    columns. The ratio is NaN where the denominator is zero, never 0 -- a cell with no
    observations has an undefined fraction, and recording it as zero puts it at the
    bottom of every distribution as though it had been measured.

    Which columns exist, and what a ratio of two of them means, is the caller's
    knowledge. Nothing is assumed here.
    """
    import pandas as pd

    needed = list(columns) if columns is not None else None
    if needed is not None and derived:
        for numerator, denominator in derived.values():
            for column in (numerator, denominator):
                if column not in needed:
                    needed.append(column)

    frame = read_obs(path, needed, index=index)

    for name, (numerator, denominator) in (derived or {}).items():
        missing = [c for c in (numerator, denominator) if c not in frame.columns]
        if missing:
            raise KeyError(f"cannot derive {name!r}: no column(s) {missing}")
        num = frame[numerator].to_numpy(dtype=float)
        den = frame[denominator].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(den > 0, num / den, np.nan)
        frame[name] = ratio

    if columns is not None:
        keep = [c for c in columns if c in frame.columns]
        keep += [c for c in (derived or {}) if c in frame.columns]
        frame = frame[keep]
    return frame


def read_loadings(
    path: str | Path | None = None,
    *,
    npz: str | Path | None = None,
    tsv: str | Path | None = None,
    dims: Iterable[str] | None = None,
    chunk_rows: int = 200_000,
    dtype=np.float32,
) -> "pd.DataFrame":
    """Read a cells x factors loadings matrix.

    Prefers a ``.npz`` (arrays ``cells``, ``factors``, ``loadings``) over the TSV when
    both are given: same numbers, a fraction of the size, and no text parse. On this
    project's fit that is 268 MB against 736 MB.

    The TSV path is read in chunks, because the whole point of the chunking is to not
    need the peak memory that reading it at once requires.
    """
    import pandas as pd

    from scads_drvi._util.advise import check_storage

    candidates = [p for p in (npz, path, tsv) if p is not None]
    if not candidates:
        raise ValueError("give a path, or npz=/tsv= explicitly")

    chosen: Path | None = None
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.suffix == ".npz" and candidate.exists():
            chosen = candidate
            break
    if chosen is None:
        for candidate in candidates:
            candidate = Path(candidate)
            if candidate.exists():
                chosen = candidate
                break
    if chosen is None:
        raise FileNotFoundError(f"none of these exist: {[str(c) for c in candidates]}")

    check_storage(chosen, kind="loadings matrix")

    if chosen.suffix == ".npz":
        with np.load(chosen, allow_pickle=True) as bundle:
            for key in ("cells", "factors", "loadings"):
                if key not in bundle:
                    raise ValueError(
                        f"{chosen} is missing array {key!r}; found "
                        f"{sorted(bundle.files)}"
                    )
            frame = pd.DataFrame(
                np.asarray(bundle["loadings"], dtype=dtype),
                index=_decode(bundle["cells"]).astype(str),
                columns=_decode(bundle["factors"]).astype(str),
            )
    else:
        blocks = []
        for block in pd.read_csv(
            chosen, sep="\t", index_col=0, chunksize=max(int(chunk_rows), 1)
        ):
            blocks.append(block.astype(dtype))
        if not blocks:
            raise ValueError(f"{chosen} is empty")
        frame = pd.concat(blocks)

    if dims is not None:
        dims = list(dims)
        missing = [d for d in dims if d not in frame.columns]
        if missing:
            raise KeyError(
                f"{chosen} has no column(s) {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}. It has {frame.shape[1]} columns; "
                f"a split contract's loadings have one per annotation column, not one "
                f"per latent dimension."
            )
        frame = frame[dims]
    return frame


def read_umap(
    path: str | Path, *, columns: Sequence[str] | None = None
) -> "pd.DataFrame":
    """Read 2-D embedding coordinates, indexed by cell.

    `columns` renames whatever the file calls its axes; by default the first two
    non-index columns are used as they are.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no embedding coordinates at {path}")
    frame = pd.read_csv(path, sep="\t", index_col=0)
    if frame.shape[1] < 2:
        raise ValueError(f"{path} has {frame.shape[1]} coordinate column(s), need 2")
    frame = frame.iloc[:, :2]
    if columns is not None:
        if len(list(columns)) != 2:
            raise ValueError("columns must name exactly two axes")
        frame.columns = list(columns)
    return frame


@dataclass(frozen=True)
class Interpretation:
    """Everything needed to read one enrichment arm, loaded and joined.

    Replaces the ~89-line setup cell that each interpretation notebook carried its own
    copy of.
    """

    model: str
    labels: "FactorLabels"
    results: "pd.DataFrame"
    cells: "pd.DataFrame"
    loadings: "pd.DataFrame | None" = field(default=None, repr=False)
    meta: dict = field(default_factory=dict)

    @property
    def traits(self) -> tuple[str, ...]:
        return tuple(self.results["trait"].unique())

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def for_trait(self, trait: str) -> "pd.DataFrame":
        """The results rows for one trait, which is what scoring and plotting take."""
        block = self.results.loc[self.results["trait"] == trait]
        if block.empty:
            raise KeyError(
                f"no results for trait {trait!r}; this arm has {list(self.traits)}"
            )
        return block


def load_interpretation(
    project: "Project",
    model: str,
    *,
    traits: Sequence[str] | None = None,
    fit: str | None = None,
    obs_path: str | Path | None = None,
    obs_columns: Sequence[str] | None = None,
    obs_index: str | None = None,
    derived: Mapping[str, tuple[str, str]] | None = None,
    umap_path: str | Path | None = None,
    loadings: bool = False,
    half_map: str | Path | None = None,
    style: str = "dim",
    strict: bool = True,
) -> Interpretation:
    """Load one arm: labels, results, per-cell metadata, optionally the loadings.

    `traits` defaults to ``project.traits``. `loadings=False` by default because the
    matrix is the expensive part and many questions do not need it.

    Everything about the cell table -- which obs columns, which index, which derived
    ratios -- is passed in. This function knows the shape of a run, not the meaning of
    a dataset.
    """
    import pandas as pd

    from scads_drvi.enrich.ldsc import read_results
    from scads_drvi.labels import load_labels

    contract = project.contract(model, fit=fit)
    traits = list(traits) if traits is not None else list(project.traits)
    if not traits:
        raise ValueError(
            "no traits given and Project.traits is empty; name the traits to read"
        )

    if half_map is None:
        candidate = contract["half_map"]
        half_map = candidate if candidate.exists() else None

    labels = load_labels(
        contract["factor_map"], model=model, half_map=half_map, style=style
    )
    results = read_results(
        contract["results"], traits=traits, labels=labels, strict=strict
    )

    cells = pd.DataFrame()
    if obs_path is not None:
        cells = cell_metadata(
            obs_path, columns=obs_columns, index=obs_index, derived=derived
        )

    if umap_path is not None:
        coords = read_umap(umap_path)
        cells = coords.join(cells, how="left") if len(cells) else coords

    matrix = None
    if loadings:
        matrix = read_loadings(
            contract["loadings"], npz=contract["loadings_npz"], dims=labels.kept_dims
        )

    meta = {
        "model": model,
        "fit": fit if fit is not None else project.fit,
        "traits": traits,
        "n_kept": labels.n_kept,
        "is_split": labels.is_split,
        "paths": {k: str(v) for k, v in contract.items()},
    }
    return Interpretation(
        model=model,
        labels=labels,
        results=results,
        cells=cells,
        loadings=matrix,
        meta=meta,
    )
