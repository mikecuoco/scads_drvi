"""Reading S-LDSC output back in.

One loader replaces the glob-and-aggregate block that was copy-pasted into each
interpretation notebook. Two details in that block were load-bearing and are made
explicit here rather than left as conventions:

**Row 0 is the factor's own annotation.** The heritability run puts our annotation first
in ``--ref-ld-chr``, so the first row of a ``.results`` file is the one being tested and
the rest are baseline categories. That is a property of how the run was invoked, so it
is a parameter with a default, not a hardcoded ``.iloc[0]``.

**A missing result file is an error, not a skip.** The notebooks did ``continue``, which
drops the factor from the table -- and therefore from the multiplicity denominator,
making every surviving q-value optimistic without saying so.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["read_results", "results_files"]

#: Column carrying the coefficient z-score in an LDSC `.results` file.
Z_COLUMN = "Coefficient_z-score"


def results_files(results_dir: str | Path) -> dict[str, Path]:
    """``{annotation name: path}`` for every ``.results`` file in a trait directory."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"no results directory at {results_dir}")
    return {path.stem: path for path in sorted(results_dir.glob("*.results"))}


def read_results(
    results_root: str | Path,
    *,
    traits: Iterable[str],
    annot2dim: Mapping[str, str],
    direction: str | Mapping[str, str] = "combined",
    row: int = 0,
    strict: bool = True,
    fdr: bool = True,
    by: str | None = "trait",
) -> pd.DataFrame:
    """Every trait's per-factor results as one tidy frame.

    `results_root` holds one directory per trait. Each is expected to contain a
    ``.results`` file for every annotation name in `annot2dim` -- typically
    :func:`scads_drvi.enrich.config.select_factors`'s ``k{i}`` numbering mapped back to
    ``dim`` names (:func:`scads_drvi.enrich.config.kept_dims`, zipped against the
    annotation names it was assigned).

    `direction` records which sign of the latent dimension each result came from --
    ``"combined"`` when the arm was not run per-direction, or a mapping from annotation
    name to ``"pos"``/``"neg"`` when it was (a pos/neg loadings view is built with
    ``np.clip(embed.X, 0, None)``/``np.clip(-embed.X, 0, None)``; this function only
    records which one a result file came from). This
    ``dim``/``direction`` pair is what replaces the old split contract's
    ``half_map.tsv`` and three-name label system: a caller filters
    ``results.query("direction == 'pos' and trait == 'X'")`` instead of resolving an
    annotation-name alias.

    With ``strict=True`` a kept factor whose result file is absent raises, naming the
    missing ones. Set it False only when you deliberately want a partial table, and note
    that the FDR correction is then over fewer tests than were intended.

    Adds ``trait``, ``annot``, ``dim`` and ``direction`` columns, and -- with
    ``fdr=True`` -- a one-tailed p and BH q corrected within each `by` group.
    """
    import pandas as pd

    from scads_drvi.stats import add_fdr as _add_fdr

    results_root = Path(results_root)
    traits = list(traits)
    if not traits:
        raise ValueError("no traits requested")

    def _direction_for(annot: str) -> str:
        return direction if isinstance(direction, str) else direction[annot]

    records = []
    for trait in traits:
        found = results_files(results_root / trait)
        missing = [a for a in annot2dim if a not in found]
        if missing and strict:
            raise FileNotFoundError(
                f"trait {trait!r} is missing {len(missing)} of {len(annot2dim)} "
                f"result files, e.g. {missing[:5]}. Skipping them would shrink the "
                f"multiplicity denominator and make every surviving q optimistic; pass "
                f"strict=False if a partial table is genuinely what you want."
            )
        for annot, dim in annot2dim.items():
            path = found.get(annot)
            if path is None:
                continue
            table = pd.read_csv(path, sep="\t")
            if len(table) <= row:
                raise ValueError(
                    f"{path} has {len(table)} rows; row {row} was requested. Row {row} "
                    f"should be the factor's own annotation."
                )
            record = table.iloc[row].to_dict()
            record["trait"] = trait
            record["annot"] = annot
            record["dim"] = dim
            record["direction"] = _direction_for(annot)
            records.append(record)

    if not records:
        raise FileNotFoundError(f"no .results files found under {results_root} for traits {traits}")

    frame = pd.DataFrame.from_records(records)
    if fdr:
        if Z_COLUMN not in frame.columns:
            raise KeyError(
                f"{Z_COLUMN!r} not in the result files; columns are {list(frame.columns)}"
            )
        frame = _add_fdr(frame, by=by, z_col=Z_COLUMN)
    return frame.reset_index(drop=True)
