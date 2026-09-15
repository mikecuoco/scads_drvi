"""This project's own data model for a DRVI fit's results -- not a DRVI wrapper.

Everything a fit produces (signed latent representation, DRVI's own per-dimension
stats, a UMAP embedding, per-cell scores, S-LDSC enrichment results) lives in one
h5ad, shaped the way DRVI's interpretability API expects: ``obs`` = cells, ``var`` =
one row per latent dimension. Building `embed` is a few lines at the call site (see
the getting-started guide), not a function here.

There is no persisted pos/neg split -- S-LDSC's need for one is met by deriving a
view from the canonical signed ``X`` on demand (:func:`directional_loadings`), never
by storing separate columns.

Peak-level ("feature") loadings don't fit this shape (``obs`` = peaks would collide
with ``obs`` = cells), so they live in a small companion h5ad linked from
``uns["provenance"]["feature_loadings_path"]``.
"""

from __future__ import annotations

import contextlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import pandas as pd
    from anndata import AnnData

__all__ = [
    "write_result",
    "read_feature_loadings",
    "attach_enrich_results",
    "directional_loadings",
]

#: Canonical peak-name form (chr:start-end), enforced once here at fit-save time.
_PEAK_RE = re.compile(r"^chr[\w.]+:\d+-\d+$")


def _version() -> str:
    try:
        from scads_drvi import __version__

        return __version__
    except Exception:  # pragma: no cover -- defensive only
        return "unknown"


def _atomic_write_h5ad(path: str | Path, adata: AnnData) -> None:
    """Write `adata` to `path` atomically -- a temp file renamed over the target only
    on success, so a long-running fit or enrichment sweep never leaves a half-written
    h5ad behind."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def write_result(
    path: str | Path,
    embed: AnnData,
    *,
    provenance: Mapping[str, Any],
    feature_loadings: pd.DataFrame | None = None,
    feature_loadings_path: str | Path | None = None,
) -> Path:
    """Write `embed` to `path`, merging `provenance` into ``uns["provenance"]``.

    `feature_loadings`, if given, is a peaks x K frame (index = peak names, columns =
    `embed.var_names`), written to a companion h5ad (default
    ``<path stem>.loadings.h5ad``) and linked via
    ``provenance["feature_loadings_path"]``. Every index entry must already be a
    canonical ``chr:start-end`` peak name; a non-matching name raises, naming the
    first few offenders.
    """
    path = Path(path)
    prov = dict(provenance)
    prov.setdefault("written_at", datetime.now(UTC).isoformat())
    prov.setdefault("scads_drvi_version", _version())

    if feature_loadings is not None:
        bad = [n for n in feature_loadings.index.astype(str) if not _PEAK_RE.match(n)]
        if bad:
            raise ValueError(
                f"feature_loadings has {len(bad)} non-canonical peak name(s), e.g. "
                f"{bad[:5]}. Peak names must already be chr:start-end by the time a fit "
                f"is saved -- rename them before calling write_result, not after."
            )
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


def read_feature_loadings(
    path: str | Path, *, embed: AnnData | None = None, backed: str | bool | None = None
) -> AnnData:
    """Read the companion peaks x K loadings file for the result at `path`, resolved
    from ``uns["provenance"]["feature_loadings_path"]``. Pass an already-loaded
    `embed` to avoid re-opening `path` just for that field."""
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
    """Attach one enrichment arm's output into ``embed.uns["enrich"][model]``, in
    place. `results` is the tidy per-trait table
    :func:`scads_drvi.enrich.ldsc.read_results` builds. `factor_selection` is
    :func:`scads_drvi.enrich.config.select_factors`'s output with its internal
    ``annot_index`` column dropped. `params` is the arm's resolved annotation config,
    kept for audit. Call :func:`write_result` afterwards to persist."""
    arm: dict[str, Any] = {"results": results}
    if factor_selection is not None:
        keep = [c for c in factor_selection.columns if c != "annot_index"]
        arm["factor_selection"] = factor_selection[keep]
    if params is not None:
        arm["params"] = dict(params)
    embed.uns.setdefault("enrich", {})[model] = arm


def directional_loadings(embed: AnnData, direction: str) -> pd.DataFrame:
    """``relu(X)`` or ``relu(-X)`` as a cells x K frame -- the only place a pos/neg
    split is ever materialized, derived fresh from `embed.X` rather than persisted."""
    import numpy as np
    import pandas as pd

    if direction not in ("pos", "neg"):
        raise ValueError(f"direction must be 'pos' or 'neg', got {direction!r}")

    x = np.asarray(embed.X, dtype=np.float64)
    values = np.clip(x, 0, None) if direction == "pos" else np.clip(-x, 0, None)
    return pd.DataFrame(values, index=embed.obs_names, columns=embed.var_names)
