"""Factor naming, reconciled into one object.

A factor has up to three names and they are not interchangeable:

``dim_j``
    the **index** name -- a column of the loadings matrix, and the only thing that may
    be used to subset it.
``k{i}``
    the **annotation** name -- what S-LDSC was run on, and what the result files are
    named after. Numbered 1-based over the *kept* factors only, so it renumbers whenever
    a factor is dropped.
``dim_47/neg``
    the **display** name -- what a reader should see. Under a split contract each latent
    dimension contributes two annotation columns, one per direction, so a bare ``dim_j``
    in that setting names a column and *not* the latent dimension it came from.

The hazard is specific and has bitten this pipeline before: display and index names are
both strings of the same shape, so using one where the other belongs either raises a
KeyError (the lucky case) or silently selects the wrong column and produces a figure
that is confidently mislabelled rather than visibly broken.

:func:`assert_index_dims` exists to make that unrepresentable. Every function that
subsets a loadings matrix calls it first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Mapping

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["FactorLabels", "DisplayStyle", "load_labels", "read_factor_map"]

DisplayStyle = Literal["dim", "dr", "lsi"]

_ANNOT_RE = re.compile(r"^k(\d+)$")
_DIM_RE = re.compile(r"^dim_(\d+)$")


def _as_bool(series: "pd.Series") -> "pd.Series":
    """Coerce the string booleans a TSV round-trip produces back to real ones."""
    import pandas as pd

    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )


def read_factor_map(path: str | Path) -> "pd.DataFrame":
    """Read ``factor_map.tsv``: one row per factor, with kept/annot_index.

    Columns are ``dim``, ``vanished``, ``kept``, ``drop_reason``, ``annot_index``.
    ``annot_index`` is 1-based and blank for dropped factors.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no factor map at {path}. It is written by the annotation-building stage; "
            f"without it there is no mapping from result files back to factors."
        )
    frame = pd.read_csv(path, sep="\t", dtype={"dim": str, "drop_reason": str})
    missing = {"dim", "kept", "annot_index"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    frame["kept"] = _as_bool(frame["kept"])
    if "vanished" in frame.columns:
        frame["vanished"] = _as_bool(frame["vanished"])
    frame["annot_index"] = pd.to_numeric(frame["annot_index"], errors="coerce").astype(
        "Int64"
    )
    return frame


def _default_display(dim: str, style: DisplayStyle) -> str:
    """Display name for a dim under a non-split contract."""
    if style == "dim":
        return dim
    match = _DIM_RE.match(dim)
    if match is None:
        return dim
    index = int(match.group(1))
    if style == "dr":
        return f"DR_{index}"
    if style == "lsi":
        # 1-based: component 0 is the first component, and calling it LSI_0 has confused
        # every reader who compared a figure against an SVD summary.
        return f"LSI_{index + 1}"
    raise ValueError(f"unknown display style {style!r}")


@dataclass(frozen=True)
class FactorLabels:
    """Every name a factor has, and the maps between them."""

    model: str
    annot2dim: Mapping[str, str]
    dim2annot: Mapping[str, str]
    kept_dims: tuple[str, ...]
    display: Mapping[str, str]
    half: Mapping[str, tuple[str, str]] | None = None
    style: DisplayStyle = "dim"
    frame: "pd.DataFrame | None" = field(default=None, repr=False, compare=False)

    @property
    def is_split(self) -> bool:
        """True when each annotation column is one direction of a latent dimension."""
        return self.half is not None

    @property
    def n_kept(self) -> int:
        return len(self.kept_dims)

    def __len__(self) -> int:
        return len(self.kept_dims)

    # -- name resolution -----------------------------------------------------

    def display_label(self, dim: str) -> str:
        """The name a reader should see for an index name."""
        return self.display.get(dim, dim)

    def display_labels(self, dims: Iterable[str]) -> list[str]:
        return [self.display_label(d) for d in dims]

    def index_dim(self, name: str) -> str:
        """Resolve any of the three names to the **index** name.

        Accepts an annotation name (``k7``), a display name (``dim_47/neg``) or an index
        name, and returns the one that may be used to subset a loadings matrix.
        """
        if name in self.dim2annot or name in self.display:
            return name
        if name in self.annot2dim:
            return self.annot2dim[name]
        reverse = {v: k for k, v in self.display.items()}
        if name in reverse:
            return reverse[name]
        raise KeyError(
            f"{name!r} is not a factor of {self.model!r}. Known annotation names look "
            f"like {next(iter(self.annot2dim), 'k1')!r} and index names like "
            f"{next(iter(self.kept_dims), 'dim_0')!r}."
        )

    def index_dims(self, names: Iterable[str]) -> list[str]:
        return [self.index_dim(n) for n in names]

    def assert_index_dims(self, dims: Iterable[str]) -> None:
        """Raise unless every name is an **index** name.

        Called by anything that does ``loadings[dims]``. A display name here would
        either KeyError or, under a split contract where display and index names have
        the same shape, silently select a different column.
        """
        dims = list(dims)
        known = set(self.dim2annot) | set(self.display)
        display_names = {v for k, v in self.display.items() if v != k}
        offenders = []
        for name in dims:
            if name in known:
                continue
            if name in display_names:
                offenders.append(f"{name!r} is a display label, not a column name")
            elif _ANNOT_RE.match(name):
                offenders.append(f"{name!r} is an annotation name, not a column name")
            else:
                offenders.append(f"{name!r} is not a factor of {self.model!r}")
        if offenders:
            raise KeyError(
                "these are not loadings columns:\n  " + "\n  ".join(offenders)
            )


def load_labels(
    factor_map: str | Path,
    *,
    model: str,
    half_map: str | Path | None = None,
    style: DisplayStyle = "dim",
) -> FactorLabels:
    """Build a :class:`FactorLabels` from ``factor_map.tsv`` and an optional half map.

    `half_map`, when present, is the split contract's record of which latent dimension
    and direction each annotation column came from. Without it the display name is the
    index name (or the `style` rendering of it) and :attr:`FactorLabels.is_split` is
    False -- a missing half map is a fact about the contract, not an error.
    """
    import pandas as pd

    frame = read_factor_map(factor_map)
    kept = frame.loc[frame["kept"] & frame["annot_index"].notna()]
    if kept.empty:
        raise ValueError(
            f"{factor_map} keeps no factors; every one was dropped, so there is nothing "
            f"to label or to read results for."
        )

    kept = kept.sort_values("annot_index")
    kept_dims = tuple(kept["dim"].astype(str))
    annot_names = [f"k{int(i)}" for i in kept["annot_index"]]

    annot2dim = dict(zip(annot_names, kept_dims))
    dim2annot = {d: a for a, d in annot2dim.items()}
    if len(annot2dim) != len(kept_dims):
        raise ValueError(
            f"{factor_map} maps two factors to one annotation index; the map must be a "
            f"bijection or results cannot be attributed."
        )

    half: dict[str, tuple[str, str]] | None = None
    if half_map is not None and Path(half_map).exists():
        hmap = pd.read_csv(half_map, sep="\t")
        cols = set(hmap.columns)
        needed = {"annot_dim", "source_dim", "half"}
        if not needed <= cols:
            raise ValueError(
                f"{half_map} is missing {sorted(needed - cols)}; expected one row per "
                f"annotation column naming its source dimension and direction."
            )
        half = {
            str(row.annot_dim): (str(row.source_dim), str(row.half))
            for row in hmap.itertuples()
        }

    if half:
        display = {
            d: (f"{half[d][0]}/{half[d][1]}" if d in half else _default_display(d, style))
            for d in kept_dims
        }
    else:
        display = {d: _default_display(d, style) for d in kept_dims}

    return FactorLabels(
        model=model,
        annot2dim=annot2dim,
        dim2annot=dim2annot,
        kept_dims=kept_dims,
        display=display,
        half=half,
        style=style,
        frame=frame,
    )
