"""Reading the heritability output the Rust LDSC produces.

Without ``--overlap-annot`` the tool writes **nothing** to ``--out`` -- no ``.results``,
no ``.log`` -- and prints the per-category table to stdout instead. The Python original
behaved similarly enough that the pipeline already had to recover coefficients by
parsing a log; this is the same job against a different, better-defined format.

Measured against the Python output for the same inputs (arm ``k96_ind_exp_split``,
factor ``k1``, trait ``bellenguez``): the coefficient agrees to every printed digit
(``-2.3707e-17``), as do the total h2 (``0.0260`` vs ``0.026``) and the annotation count
(98).

One thing this format does **not** carry is the coefficient standard error, and
therefore not the z-score either. The pipeline's significance testing is built on that
z, so a caller needing it must run with ``--overlap-annot``, which writes a proper
``.results`` carrying ``Coefficient_std_error`` and ``Coefficient_z-score``.
:func:`parse_h2_stdout` reports ``None`` rather than inventing one.

That flag needs a frequency file and, beside every LD-score prefix, an annotation in
**full** format -- ``CHR BP SNP CM`` ahead of the annotation columns. A *thin*
annotation, which the Python original accepted, is rejected outright. See
:mod:`scads_drvi.enrich.annotations`, which does the widening and documents the two
row-alignment invariants that make it safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["H2Summary", "parse_h2_stdout", "h2_table"]

_TOTAL_H2 = re.compile(r"Total observed-scale h2:\s*([-\d.eE+]+)")
_INTERCEPT = re.compile(r"^Intercept:\s*([-\d.eE+]+)", re.MULTILINE)
_LAMBDA = re.compile(r"Lambda GC:\s*([-\d.eE+]+)")
_MEAN_CHI2 = re.compile(r"Mean Chi\^?2:\s*([-\d.eE+]+)")
_N_ANNOT = re.compile(r"Detected K=(\d+) annotation columns")
_HEADER = re.compile(r"^Category\s+Prop_h2\s+Prop_SNPs\s+Enrichment\s+Coefficient\s*$")


@dataclass(frozen=True)
class H2Summary:
    """Parsed heritability output for one run.

    ``coefficient_se`` and ``coefficient_z`` are ``None`` when the output does not carry
    them, which is the case without ``--overlap-annot``. They are not derived from
    anything else: a z-score assembled from a guessed SE would be indistinguishable from
    a real one downstream.
    """

    total_h2: float | None = None
    intercept: float | None = None
    lambda_gc: float | None = None
    mean_chi2: float | None = None
    n_categories: int | None = None
    categories: tuple[str, ...] = ()
    prop_h2: tuple[float, ...] = ()
    prop_snps: tuple[float, ...] = ()
    enrichment: tuple[float, ...] = ()
    coefficient: tuple[float, ...] = ()
    coefficient_se: None = None
    coefficient_z: None = None
    raw: str = field(default="", repr=False)

    @property
    def carries_standard_errors(self) -> bool:
        """False for this format. The pipeline's z-scores need another route."""
        return False

    def category(self, name: str) -> dict[str, float]:
        """The row for one annotation."""
        try:
            slot = self.categories.index(name)
        except ValueError:
            raise KeyError(
                f"no category {name!r}; the first few are {self.categories[:3]}"
            ) from None
        return {
            "prop_h2": self.prop_h2[slot],
            "prop_snps": self.prop_snps[slot],
            "enrichment": self.enrichment[slot],
            "coefficient": self.coefficient[slot],
        }

    @property
    def focal(self) -> dict[str, float]:
        """The first category, which is the annotation under test.

        The run puts our annotation first in ``--ref-ld-chr``, so row 0 is it and the
        rest are baseline categories.
        """
        if not self.categories:
            raise ValueError("no categories were parsed")
        return self.category(self.categories[0])


def _number(text: str) -> float:
    if text.lower() in ("nan", "-nan", "+nan"):
        return float("nan")
    return float(text)


def _first(pattern: re.Pattern[str], text: str, cast=_number):
    match = pattern.search(text)
    return cast(match.group(1)) if match else None


def parse_h2_stdout(text: str) -> H2Summary:
    """Parse the stdout of ``ldsc h2 --print-coefficients``.

    Tolerates the long category names that run into their first column without
    whitespace alignment, which the fixed-width look of the table hides -- splitting on
    runs of whitespace would silently merge a name and a number for any category whose
    name exceeds the column.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if _HEADER.match(line.strip()):
            start = index + 1
            break

    categories: list[str] = []
    values: list[list[float]] = []
    if start is not None:
        for line in lines[start:]:
            stripped = line.strip()
            if not stripped or set(stripped) <= set("- "):
                if categories:
                    break
                continue
            parts = stripped.split()
            if len(parts) < 5:
                break
            # The name is everything before the last four numeric fields.
            try:
                numbers = [_number(p) for p in parts[-4:]]
            except ValueError:
                break
            categories.append(" ".join(parts[:-4]))
            values.append(numbers)

    columns = list(zip(*values, strict=True)) if values else [(), (), (), ()]
    return H2Summary(
        total_h2=_first(_TOTAL_H2, text),
        intercept=_first(_INTERCEPT, text),
        lambda_gc=_first(_LAMBDA, text),
        mean_chi2=_first(_MEAN_CHI2, text),
        n_categories=_first(_N_ANNOT, text, cast=int),
        categories=tuple(categories),
        prop_h2=tuple(columns[0]),
        prop_snps=tuple(columns[1]),
        enrichment=tuple(columns[2]),
        coefficient=tuple(columns[3]),
        raw=text,
    )


def h2_table(summary: H2Summary) -> pd.DataFrame:
    """The parsed per-category rows as a frame, focal annotation first."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Category": summary.categories,
            "Prop_h2": summary.prop_h2,
            "Prop_SNPs": summary.prop_snps,
            "Enrichment": summary.enrichment,
            "Coefficient": summary.coefficient,
        }
    )
