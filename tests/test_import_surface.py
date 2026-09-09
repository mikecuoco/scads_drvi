"""Heavy dependencies stay where they are declared to be.

``torch``/``scvi-tools`` belong to :mod:`scads_drvi.factorize.model`,
``matplotlib``/``seaborn`` to :mod:`scads_drvi.viz`, and ``h5py`` is function-local in
:mod:`scads_drvi.io`. Everything else must import in an environment that has none of
them -- which is not hypothetical: the environment that runs the enrichment stages has
no torch, no matplotlib, no seaborn and no h5py.

The check runs in a subprocess with a meta-path finder blocking those modules, because
by the time this test module is imported they may already be in ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Modules that must import with none of the heavy dependencies present.
CORE_MODULES = [
    "scads_drvi",
    "scads_drvi.config",
]

BLOCKED = ("torch", "scvi", "matplotlib", "seaborn", "h5py", "anndata")

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
