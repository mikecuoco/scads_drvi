"""Heavy dependencies stay where they are declared to be.

``torch``/``scvi-tools``/``drvi`` are never imported anywhere in this package at all --
training and loading a DRVI model is the caller's own job, done directly against
``scvi.external.DRVI`` (see the getting-started guide). ``matplotlib``/``seaborn``/
``scanpy``/``h5py``/``anndata`` belong to :mod:`scads_drvi.pl` (every embedding figure
there now wraps ``scanpy.pl.embedding``, function-local, the same as
matplotlib/seaborn always were) -- no core module needs a result h5ad's own I/O, since
that is a plain ``anndata.read_h5ad``/``AnnData.write_h5ad`` call at the caller's own
site, not something this package wraps. Everything else must import in an environment
that has none of them -- which is not hypothetical: the environment that runs the
enrichment stages has no torch, no scvi, no drvi, no matplotlib, no seaborn, no scanpy,
no h5py and no anndata.

The check runs in a subprocess with a meta-path finder blocking those modules, because
by the time this test module is imported they may already be in ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Modules that must import with none of the heavy dependencies present.
#
# This list is the enforcement of the README's claim that "the enrichment stages run with
# no torch, no matplotlib, no seaborn and no h5py, and must still import and use the
# loaders, statistics and scoring". It previously named only the top level and `config`,
# so a module-level `import matplotlib` added to, say, `scores.cell` would have passed CI
# while breaking the environment this test exists to protect.
CORE_MODULES = [
    "scads_drvi",
    "scads_drvi.stats",
    "scads_drvi.scores.cell",
    "scads_drvi.scores.aggregate",
    "scads_drvi.enrich.ldsc",
    "scads_drvi.enrich.h2_output",
    "scads_drvi.enrich.annotations",
    "scads_drvi.enrich.binary",
    "scads_drvi.annotate.gc",
    "scads_drvi.annotate.motif",
    "scads_drvi.annotate.resources",
    "scads_drvi._util.advise",
    "scads_drvi._util.progress",
]

# Deliberately absent, each for a reason rather than an oversight. Listed so that adding
# a module here is a decision someone made, not a gap nobody noticed:
#
#   scads_drvi.pl.*               matplotlib / seaborn / scanpy (-> anndata)
#   scads_drvi.enrich.config      module-scope numpy, pandas and yaml
#   scads_drvi._util.presets      argparse CLI plumbing, not part of the library surface

BLOCKED = ("torch", "scvi", "drvi", "matplotlib", "seaborn", "scanpy", "h5py", "anndata")

_BLOCKER = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked!r}

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            root = name.split(".")[0]
            if root in BLOCKED:
                raise ImportError(f"blocked by the import-surface test: {{name}}")
            return None

    for _name in list(sys.modules):
        if _name.split(".")[0] in BLOCKED:
            del sys.modules[_name]
    sys.meta_path.insert(0, Blocker())

    import importlib
    for _mod in {modules!r}:
        importlib.import_module(_mod)

    for _name in sys.modules:
        assert _name.split(".")[0] not in BLOCKED, _name
    print("ok")
    """
)


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )


def test_core_imports_without_heavy_dependencies():
    script = _BLOCKER.format(blocked=BLOCKED, modules=CORE_MODULES)
    proc = _run(script)
    assert proc.returncode == 0, (
        "core modules pulled in a blocked dependency:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "ok" in proc.stdout


def test_top_level_import_is_cheap():
    """`import scads_drvi` must not drag in numpy, pandas or scipy either.

    The top level exists to hand out names; paying a multi-second import to read one
    attribute is what makes people go back to sys.path hacks.
    """
    proc = _run(
        "import sys; import scads_drvi; "
        "heavy = [m for m in ('numpy','pandas','scipy') if m in sys.modules]; "
        "print(','.join(heavy) or 'none')"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "none", (
        f"import scads_drvi eagerly imported: {proc.stdout.strip()}"
    )


@pytest.mark.parametrize("name", CORE_MODULES)
def test_module_is_importable(name):
    __import__(name)
