"""Annotation files in the shape S-LDSC's ``--overlap-annot`` will actually read.

An annotation file comes in two shapes. The **thin** shape is bare annotation columns
and nothing else; the **full** shape prefixes them with the four identifier columns
``CHR BP SNP CM`` that the reference annotations carry. The Python LDSC accepts either --
its ``annot_parser`` drops the identifiers with ``errors='ignore'`` -- so pipelines built
against it tend to write thin files, which are smaller and cheaper.

The Rust implementation does not accept thin files:

    Error: Annot file '...' has 1 columns;
           expected > 4 (full format: CHR SNP BP CM + annotations)

so a thin annotation is the thing standing between an existing pipeline and
``--overlap-annot`` -- which is where the coefficient standard error, and therefore every
z-score downstream, comes from. :func:`to_full_annot` performs that widening.

**Two invariants that are easy to lose and silent when lost.** ``--overlap-annot`` filters
rows by minor allele frequency *positionally*, against a separate ``.frq`` file:

    df_annot[(0.95 > df_frq.FRQ) & (df_frq.FRQ > 0.05)]

so the annotation must carry **every row of the ``.bim``, in ``.bim`` order**. It must
NOT be restricted to the HapMap3 subset the LD scores are printed over: the row counts
would disagree and the frequency mask would land on the wrong variants, quietly
selecting a different set than intended. :func:`to_full_annot` refuses a length mismatch
rather than aligning on a key, because a merge that reorders is exactly the failure this
is guarding.

**A dtype trap in the reader, not in your data.** The Rust reader infers each column's
type from a leading sample of rows. A genetic-distance column that is integral for its
first few hundred rows and fractional afterwards -- which is what the standard reference
annotations look like, since the leading variants sit before the first recombination
map entry -- is inferred as an integer and then fails on the first decimal:

    Error: reading annot file '...' - Original error:
           invalid primitive value found during CSV parsing

:func:`force_decimal` rewrites such columns with an explicit decimal point so no column
can be inferred as integral. It is a workaround for the reader, applied to a copy; the
reference itself is not modified.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "BIM_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "read_bim",
    "to_full_annot",
    "write_full_annot",
    "force_decimal",
    "is_full_annot",
]

#: Columns of a PLINK ``.bim``, in file order.
BIM_COLUMNS = ("CHR", "SNP", "CM", "BP", "A1", "A2")

#: The four columns a full-format annotation carries ahead of its annotations, in the
#: order the reference annotations use. Note this is NOT the ``.bim`` order.
IDENTIFIER_COLUMNS = ("CHR", "BP", "SNP", "CM")


def read_bim(path: str | Path) -> pd.DataFrame:
    """Read a PLINK ``.bim``, preserving file order.

    No index is set and no sorting is applied: row order *is* the contract, because
    every consumer of the result aligns positionally.
    """
    import pandas as pd

    frame = pd.read_csv(
        path, sep=r"\s+", header=None, names=list(BIM_COLUMNS), dtype={"SNP": str}
    )
    if frame.empty:
        raise ValueError(f"{path} has no variants")
    return frame


def is_full_annot(columns: Iterable[str]) -> bool:
    """True when `columns` begins with the four identifier columns."""
    head = tuple(list(columns)[: len(IDENTIFIER_COLUMNS)])
    return head == IDENTIFIER_COLUMNS


def to_full_annot(thin: pd.DataFrame, bim: pd.DataFrame) -> pd.DataFrame:
    """Prefix a thin annotation with the identifier columns taken from `bim`.

    The two are combined **positionally**. A length mismatch raises: aligning on variant
    id instead would paper over the one condition -- every bim row present, in bim order
    -- that the positional frequency mask depends on.

    Passing an already-full annotation back through is an error rather than a no-op, so
    a double application cannot silently produce ``CHR CHR BP BP ...``.
    """
    import pandas as pd

    if is_full_annot(thin.columns):
        raise ValueError(
            "annotation already carries the identifier columns "
            f"{IDENTIFIER_COLUMNS}; widening it again would duplicate them"
        )
    if len(thin) != len(bim):
        raise ValueError(
            f"annotation has {len(thin)} rows and the bim has {len(bim)}. Every bim row "
            "must be present, in bim order: --overlap-annot masks by frequency "
            "positionally, so a mismatch applies the mask to the wrong variants. Do not "
            "restrict the annotation to the SNPs the LD scores are printed over."
        )

    identifiers = pd.DataFrame(
        {name: bim[name].to_numpy() for name in IDENTIFIER_COLUMNS}
    )
    return pd.concat([identifiers, thin.reset_index(drop=True)], axis=1)


def force_decimal(
    frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return a copy whose numeric `columns` are typed float rather than integer.

    Works around a reader that infers a column's type from a leading sample and then
    fails on a value that does not fit. Writing the column as float makes the inference
    unambiguous no matter which rows are sampled.

    `columns` defaults to every column except ``CHR`` and ``SNP``, which are genuinely
    integral or textual and are read as such. Note that this **includes** ``BP`` and
    ``CM``, two of the four :data:`IDENTIFIER_COLUMNS`: ``CM`` is the column the reader
    actually mis-infers, and ``BP`` follows from the same default, so a widened
    annotation carries a base position written as ``1000.0`` rather than ``1000``.

    That has not been checked against a real ``--overlap-annot`` run. If a downstream
    integer parse of ``BP`` ever objects, pass ``columns=["CM"]`` explicitly rather than
    changing this default -- what gets written into an annotation file should be a
    deliberate choice at the call site.
    """
    import pandas as pd

    if columns is None:
        columns = [c for c in frame.columns if c not in ("CHR", "SNP")]

    out = frame.copy()
    for name in columns:
        if name not in out.columns:
            raise KeyError(f"{name!r} is not a column of the annotation")
        if pd.api.types.is_numeric_dtype(out[name]):
            out[name] = out[name].astype("float64")
    return out


def write_full_annot(
    dest: str | Path,
    thin: pd.DataFrame,
    bim: pd.DataFrame,
    *,
    decimal: bool = True,
) -> Path:
    """Write `thin` widened by `bim` to `dest`, gzipped if the name says so.

    With `decimal` set, non-identifier columns are written as floats -- see
    :func:`force_decimal` for the reader behaviour that makes this the safe default.
    """
    full = to_full_annot(thin, bim)
    if decimal:
        full = force_decimal(full)

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if dest.suffix == ".gz" else None
    full.to_csv(dest, sep="\t", index=False, compression=compression)
    return dest
