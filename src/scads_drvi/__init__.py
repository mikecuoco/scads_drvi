"""DRVI factorization -> S-LDSC heritability enrichment -> per-cell disease scores.

Method-generic: nothing here names a tissue, cell type, donor cohort, trait or obs
column. Those are caller-supplied. See ``tests/test_generic.py``, which enforces it.

This package does not train, load, or otherwise wrap DRVI at all -- that is the
caller's job, done directly against ``scvi.external.DRVI`` (see the getting-started
guide, and DRVI's own tutorial at
https://drvi.readthedocs.io/latest/tutorials/external/general_pipeline.html). There is
no path-configuration object either. A fit's results live in one ``AnnData`` (``obs`` =
cells, ``var`` = one row per latent dimension), built and written with plain
``anndata``/``AnnData.write_h5ad`` calls at the call site (see the getting-started
guide) -- this package has no h5ad read/write wrapper of its own.

The top level imports nothing heavier than the standard library. ``matplotlib``/
``seaborn``/``anndata`` are imported only by :mod:`scads_drvi.pl`, so an environment
without them can still use the loaders, statistics and scoring. ``torch``,
``scvi-tools`` and ``drvi-py`` are not imported anywhere in this package at all; they
are only what a caller uses directly to produce the fit this package reads. Public
names below resolve lazily for the same reason.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

# name -> submodule it lives in. Resolved on first attribute access so that
# `import scads_drvi` stays cheap and dependency-light.
_LAZY: dict[str, str] = {}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    """Resolve a public name from its submodule on first access (PEP 562)."""
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
