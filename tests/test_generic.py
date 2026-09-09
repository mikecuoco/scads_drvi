"""The package must not know about any particular dataset.

This is the test that keeps the code reusable. Every term below names a brain region,
cell grouping, donor-cohort variable, trait or study that belongs to one experiment.
The package's vocabulary is: cells, features, factors, annotations, LD scores, scores.

If a new analysis genuinely needs one of these concepts, it arrives as a caller
argument, and the caller keeps its own constants (see ``code/05_enrich/seaad_site.py``).
"""

from __future__ import annotations

import re

from conftest import iter_source_files, iter_string_constants

# Terms that name a dataset rather than a method. Matched case-insensitively on word
# boundaries so that, e.g., "region" is fine but "MTG" is not.
DATASET_TERMS = (
    # cell groupings
    "subclass", "supertype", "celltype", "cell_type",
    # a specific cell population
    "immune", "astro", "oligo", "microglia", "excitatory", "inhibitory",
    # brain regions of one atlas
    "mtg", "dfc", "v1c", "ang", "fi", "hip", "lec", "mec",
    # cohort / phenotype variables
    "donor", "braak", "adnc", "apoe", "dementia", "alzheimer",
    # studies and datasets
    "bellenguez", "seaad", "sea_ad", "allen", "gcst",
)

_PATTERNS = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in DATASET_TERMS}


def test_no_dataset_vocabulary_in_string_constants():
    """No non-docstring string literal may name a dataset-specific concept."""
    offenders = []
    for path in iter_source_files():
        for lineno, value in iter_string_constants(path):
            for term, pattern in _PATTERNS.items():
                if pattern.search(value):
                    offenders.append(f"{path.name}:{lineno}: {term!r} in {value!r}")
    assert not offenders, "dataset-specific literals found:\n" + "\n".join(offenders)


def test_no_dataset_vocabulary_in_identifiers():
    """No module, function, class, argument or attribute may be named for a dataset."""
    import ast

    offenders = []
    for path in iter_source_files():
        for term, pattern in _PATTERNS.items():
            if pattern.search(path.stem):
                offenders.append(f"{path.name}: module name contains {term!r}")

        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            for name in names:
                for term, pattern in _PATTERNS.items():
                    if pattern.search(name):
                        offenders.append(
                            f"{path.name}:{getattr(node, 'lineno', '?')}: "
                            f"identifier {name!r} contains {term!r}"
                        )
    assert not offenders, "dataset-specific identifiers found:\n" + "\n".join(offenders)
