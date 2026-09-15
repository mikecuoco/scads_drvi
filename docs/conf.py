"""Sphinx configuration for scads-drvi."""

import sys
from pathlib import Path

# Make the package importable from an uninstalled checkout (src layout)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from importlib.metadata import version as _version
    release = _version("scads-drvi")
except Exception:
    release = "0.1.0"

# -- Project information -------------------------------------------------------
project = "scads-drvi"
author = "Mike Cuoco"
copyright = "2026, Mike Cuoco, Allen Institute"  # noqa: A001
version = ".".join(release.split(".")[:2])

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# MyST settings ----------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

# Notebook execution -----------------------------------------------------------
# Notebooks are committed with pre-executed outputs; re-execution at build time
# requires a live environment and may fail on HPC nodes without GPUs.
nb_execution_mode = "auto"

# -- autodoc / autosummary settings -------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__len__",
}
autodoc_typehints = "description"
autodoc_typehints_fully_qualified = False
autodoc_member_order = "bysource"
# Delay heavy imports — the lazy-import pattern means some things resolve only
# at call time; we import the package at build time without heavy deps installed.
# `torch`/`scvi`/`drvi` are mocked defensively even though this package never imports
# them itself (training/loading a DRVI model is the caller's own job, done directly
# against `scvi.external.DRVI`) -- CI's docs job does not install them regardless.
# `anndata` is mocked too, even though it's lighter than the others, because
# `enrich.embed`'s `from __future__ import annotations` leaves its `AnnData`-typed
# signatures as strings that sphinx-autodoc-typehints resolves at build time; without a
# real or mocked `anndata` that resolution raises a NameError.
autodoc_mock_imports = [
    "torch", "scvi", "drvi", "h5py", "matplotlib", "seaborn", "anndata",
]

# -- intersphinx: resolve cross-refs to upstream docs -------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/mikecuoco/scads_drvi",
    "use_repository_button": True,
    "use_download_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "logo": {
        "text": "scads-drvi",
    },
}
html_title = "scads-drvi"
html_static_path = ["_static"]
