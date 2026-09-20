"""Reading the PLINK ``.bim`` an annotation is built against.

`ldsc_rs.estimate_ldscore(..., thin_annot=True)` accepts a thin annotation (bare
annotation columns, no ``CHR BP SNP CM``) directly -- so all this package needs from
a `.bim` is its row order, since a per-SNP annotation must return one value per `.bim`
row, in `.bim` order.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["BIM_COLUMNS", "read_bim"]

#: Columns of a PLINK ``.bim``, in file order.
BIM_COLUMNS = ("CHR", "SNP", "CM", "BP", "A1", "A2")


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
