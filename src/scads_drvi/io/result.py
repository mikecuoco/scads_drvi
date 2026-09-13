"""One ``AnnData`` per fit -- the object this package reads and writes.

Everything a fit produces -- the signed latent representation, DRVI's own per-dimension
statistics, a UMAP embedding, per-cell disease scores and S-LDSC enrichment results --
lives in one h5ad, shaped exactly the way DRVI's own interpretability API expects
(``obs`` = cells, ``var`` = one row per latent dimension). That shape is deliberate:
``model.set_latent_dimension_stats``, ``drvi.utils.pl.plot_latent_dimension_stats``,
``plot_latent_dims_in_umap`` and ``plot_latent_dims_in_heatmap`` all take exactly this
object, so nothing here has to adapt our data to theirs.

**There is no persisted pos/neg split.** DRVI itself has none either -- its own
``directional=True`` machinery only computes or plots things twice at call time and
never doubles ``var``. S-LDSC needs one annotation per direction, and that need is real,
but it is met by deriving a view from the canonical signed matrix at the moment it is
needed (:func:`directional_loadings`), not by storing ``dim_j/pos`` and ``dim_j/neg`` as
separate columns. A signed ``X`` is the only thing ever written to disk; a split is
always a computation, never a file.

Peak-level ("feature") loadings are the one thing that does not fit this shape --
``obs`` = peaks would collide with ``obs`` = cells on the same object -- so they live in
a small companion file, linked by path from ``uns["provenance"]``.
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from anndata import AnnData

__all__ = [
    "build_embed",
    "write_result",
    "read_result",
    "read_feature_loadings",
    "attach_enrich_results",
    "directional_loadings",
]


def _version() -> str:
    try:
        from scads_drvi import __version__

        return __version__
    except Exception:  # pragma: no cover -- defensive only
        return "unknown"


def _atomic_write_h5ad(path: str | Path, adata: AnnData) -> None:
    """Write `adata` to `path` atomically: a temp file, renamed over the target only on
    success.

    A fit can take hours to produce and an enrichment arm's results can take longer
    still; a half-written h5ad after either is worse than none. Local rather than
    imported for the same reason ``io/meta.py`` used to keep this itself: this guarantee
    is small enough to own outright and owning it keeps the module free of any
    dependency on where a deployment puts scratch space.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def build_embed(
    model,
    adata,
    *,
    obs_columns: Sequence[str] | None = None,
    umap: pd.DataFrame | None = None,
    vanished_threshold: float = 0.5,
) -> AnnData:
    """The canonical cells x K object for one fit, in DRVI's own ``embed`` shape.

    `obs_columns` is an explicit list of columns to carry over from `adata.obs` --
    deliberately not "every column", since the raw obs h5ad routinely carries far more
    than any one fit's downstream consumers need. `umap`, if given, is a cells x 2 frame
    (see :func:`scads_drvi.io.artifacts.read_umap`) written to ``obsm["X_umap"]``, which
    is what ``drvi.utils.pl.plot_latent_dims_in_umap`` requires.

    `var` is populated by DRVI's own ``model.set_latent_dimension_stats`` -- this
    function does not reimplement vanished-dimension detection, ordering or titling.
    """
    import anndata as ad
    import numpy as np

    from scads_drvi.factorize.model import latent

    z = latent(model, adata=adata)
    obs = adata.obs[list(obs_columns)].copy() if obs_columns is not None else adata.obs.iloc[:, :0].copy()

    embed = ad.AnnData(X=np.asarray(z, dtype=np.float32), obs=obs)
    embed.var_names = [f"dim_{i}" for i in range(embed.n_vars)]

    model.set_latent_dimension_stats(embed, vanished_threshold=vanished_threshold)

    if umap is not None:
        coords = umap.reindex(embed.obs_names)
        if coords.isna().any().any():
            missing = int(coords.isna().any(axis=1).sum())
            raise ValueError(
                f"{missing} of {embed.n_obs} cells have no UMAP coordinates; `umap` must "
                f"cover every cell in `adata`."
            )
        embed.obsm["X_umap"] = coords.to_numpy(dtype=np.float32)

    return embed


def write_result(
    path: str | Path,
    embed: AnnData,
    *,
    provenance: Mapping[str, Any],
    feature_loadings: pd.DataFrame | None = None,
    feature_loadings_path: str | Path | None = None,
) -> Path:
    """Write `embed` to `path`, with `provenance` merged into ``uns["provenance"]``.

    `feature_loadings`, if given, is a peaks x K frame (index = peak names, columns =
    `embed.var_names`) written to its own companion h5ad -- a peaks x K matrix has
    nowhere sensible to live on a cells x K object, and peaks routinely outnumber the
    thing `embed` is optimised for opening quickly. Its path (default:
    ``<path stem>.loadings.h5ad`` beside `path`) is recorded into
    ``provenance["feature_loadings_path"]`` before the primary write, so
    :func:`read_feature_loadings` can find it later.
    """
    path = Path(path)
    prov = dict(provenance)
    prov.setdefault("written_at", datetime.now(timezone.utc).isoformat())
    prov.setdefault("scads_drvi_version", _version())

    if feature_loadings is not None:
        companion = (
            Path(feature_loadings_path)
            if feature_loadings_path is not None
            else path.with_name(f"{path.stem}.loadings.h5ad")
        )
        _write_feature_loadings(companion, feature_loadings)
        try:
            prov["feature_loadings_path"] = str(companion.relative_to(path.parent))
        except ValueError:
            prov["feature_loadings_path"] = str(companion)

    embed = embed.copy()
    embed.uns["provenance"] = prov
    _atomic_write_h5ad(path, embed)
    return path


def _write_feature_loadings(path: Path, frame: pd.DataFrame) -> None:
    import anndata as ad
    import numpy as np
    import pandas as pd

    obs = pd.DataFrame(index=frame.index.astype(str))
    var = pd.DataFrame(index=frame.columns.astype(str))
    adata = ad.AnnData(X=np.asarray(frame.to_numpy(), dtype=np.float32), obs=obs, var=var)
    _atomic_write_h5ad(path, adata)


def read_result(path: str | Path, *, backed: str | bool | None = None) -> AnnData:
    """Read a result h5ad. The one blessed import path for it, so a future storage
    change has one call site to edit rather than a `ad.read_h5ad` scattered through
    every caller.
    """
    import anndata as ad

    return ad.read_h5ad(Path(path), backed=backed)


def read_feature_loadings(
    path: str | Path, *, embed: AnnData | None = None, backed: str | bool | None = None
) -> AnnData:
    """Read the companion peaks x K loadings file for the result at `path`.

    Its location is resolved from ``uns["provenance"]["feature_loadings_path"]``. Pass
    the already-loaded `embed` (from :func:`read_result`) to avoid a second open of the
    primary file just to read that one field -- ``uns`` is always loaded eagerly, even
    under ``backed=True``.
    """
    import anndata as ad

    path = Path(path)
    prov = (
        embed.uns.get("provenance", {})
        if embed is not None
        else ad.read_h5ad(path, backed="r").uns.get("provenance", {})
    )
    rel = prov.get("feature_loadings_path")
    if rel is None:
        raise FileNotFoundError(
            f"{path} records no feature_loadings_path in its provenance -- it was "
            f"written without feature_loadings, or by write_result() before this field "
            f"existed."
        )
    companion = Path(rel)
    if not companion.is_absolute():
        companion = path.parent / companion
    if not companion.exists():
        raise FileNotFoundError(f"{path} names a feature-loadings file at {companion}, "
                                f"but it does not exist")
    return ad.read_h5ad(companion, backed=backed)


def attach_enrich_results(
    embed: AnnData,
    model: str,
    results: pd.DataFrame,
    *,
    factor_selection: pd.DataFrame | None = None,
    params: Mapping[str, Any] | None = None,
) -> None:
    """Attach one enrichment arm's output into ``embed.uns["enrich"][model]``, in place.

    `results` is the tidy per-trait table :func:`scads_drvi.enrich.ldsc.read_results`
    builds: columns ``dim``, ``direction`` (``"pos"``/``"neg"``/``"combined"``),
    ``trait``, the raw LDSC columns, and the derived ``p``/``q``. This is what replaces
    the old split contract's ``half_map.tsv`` and its three-name label system -- a caller
    filters ``results.query("direction == 'pos' and trait == 'X'")`` instead of resolving
    a `k7` -> `dim_47/pos` alias.

    `factor_selection`, if given, is :func:`scads_drvi.enrich.config.select_factors`'s
    output with its ``annot_index`` column dropped -- that numbering is a purely internal
    detail of how annotation files were named for the LDSC subprocess, never a public
    record. `params` is the arm's resolved annotation config, kept for audit.

    Mutates `embed` in memory only; call :func:`write_result` afterwards to persist it.
    """
    arm: dict[str, Any] = {"results": results}
    if factor_selection is not None:
        keep = [c for c in factor_selection.columns if c != "annot_index"]
        arm["factor_selection"] = factor_selection[keep]
    if params is not None:
        arm["params"] = dict(params)
    embed.uns.setdefault("enrich", {})[model] = arm


def directional_loadings(embed: AnnData, direction: str) -> pd.DataFrame:
    """``relu(X)`` or ``relu(-X)``, as a cells x K frame -- the *only* place a pos/neg
    split is ever materialized.

    DRVI has no native notion of this split (its own ``directional=True`` machinery only
    computes or plots things twice at call time; ``var`` always stays K rows). S-LDSC
    genuinely needs one annotation per direction, so this view is derived fresh from the
    canonical signed `embed.X` wherever that need actually arises, rather than being
    persisted as separate ``dim_j/pos``/``dim_j/neg`` columns.
    """
    import numpy as np
    import pandas as pd

    if direction not in ("pos", "neg"):
        raise ValueError(f"direction must be 'pos' or 'neg', got {direction!r}")

    x = np.asarray(embed.X, dtype=np.float64)
    values = np.clip(x, 0, None) if direction == "pos" else np.clip(-x, 0, None)
    return pd.DataFrame(values, index=embed.obs_names, columns=embed.var_names)
