"""Names that changed, and the shims that keep the old spelling working for now.

The S-LDSC results table, the peak-risk table and the cell scores all key on a factor
name. They used to call that column ``dim`` (a DRVI term); it is ``factor`` now, so a
non-DRVI decomposition (NMF, topic models) reads naturally. Old tables and old keyword
names still work and warn.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["with_factor_column", "renamed_kwarg"]


def with_factor_column(frame: pd.DataFrame, *, stacklevel: int = 3) -> pd.DataFrame:
    """`frame` with a ``factor`` column; a legacy ``dim`` column is renamed, with a warning.

    A frame that has neither is returned unchanged, so each caller raises its own error
    naming the column it needs.
    """
    if "factor" in frame.columns:
        return frame
    if "dim" in frame.columns:
        warnings.warn(
            "the 'dim' column is now called 'factor'; rename it (the old name still "
            "works for now)",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return frame.rename(columns={"dim": "factor"})
    return frame


def renamed_kwarg(
    new_value: Any, old_value: Any, *, new: str, old: str, stacklevel: int = 3
) -> Any:
    """Resolve a keyword argument that was renamed from `old` to `new`.

    Returns `new_value`, or `old_value` (with a ``DeprecationWarning``) when only the old
    name was given. Both given is an error. Neither given returns None.
    """
    if old_value is None:
        return new_value
    if new_value is not None:
        raise TypeError(f"pass `{new}` only: `{old}` is its deprecated old name")
    warnings.warn(
        f"`{old}` is now `{new}`; the old name still works for now",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return old_value
