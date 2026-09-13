"""DRVI factorization -> S-LDSC heritability enrichment -> per-cell disease scores.

Method-generic: nothing here names a tissue, cell type, donor cohort, trait or obs
column. Those are caller-supplied. See ``tests/test_generic.py``, which enforces it.

There is no path-configuration object here. A fit's results live in one ``AnnData``
(see :mod:`scads_drvi.factorize.result`), built and read with explicit paths --
``build_embed``, ``write_result`` -- rather than through a ``Project`` that derives them;
read one back with ``anndata.read_h5ad`` directly.

The top level imports nothing heavier than the standard library. ``torch``,
``scvi-tools`` and ``drvi-py`` are declared dependencies but are imported only by
:mod:`scads_drvi.factorize`, and ``matplotlib``/``seaborn`` only by :mod:`scads_drvi.pl`
-- so an environment without them can still use the loaders, statistics and scoring.
Public names below resolve lazily for the same reason.
"""

from __future__ import annotations

__version__ = "0.1.0"

# name -> submodule it lives in. Resolved on first attribute access so that
# `import scads_drvi` stays cheap and dependency-light.
_LAZY: dict[str, str] = {}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    """Resolve a public name from its submodule on first access (PEP 562)."""
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
