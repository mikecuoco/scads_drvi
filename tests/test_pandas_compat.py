"""The package must work on both sides of the pandas 2 -> 3 boundary.

Two of the three environments this runs in are pandas 3.0 and one is 2.1, and the
differences are not cosmetic: from 3.0 a text column is a ``StringDtype`` extension
dtype rather than ``object``, copy-on-write is unconditional, and ``applymap`` is gone.

The AST checks below catch the spellings that break silently or only in one environment,
so a regression is a test failure rather than a crash discovered by whoever next runs
the notebook.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest
from conftest import iter_source_files


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_applymap_is_not_used():
    """Removed in pandas 3.0; DataFrame.map replaces it."""
    offenders = []
    for path in iter_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in _calls(tree):
            if _dotted(call.func).endswith("applymap"):
                offenders.append(f"{path.name}:{call.lineno}")
    assert not offenders, "applymap was removed in pandas 3.0: " + ", ".join(offenders)


def test_issubdtype_is_not_used_on_a_pandas_dtype():
    """`np.issubdtype(series.dtype, ...)` raises TypeError on an extension dtype.

    Under pandas 2 a text column is `object` and the numpy spelling returns False; under
    pandas 3 the same column is a StringDtype and the call raises. Use
    `pandas.api.types.is_numeric_dtype`, which answers the question for both.
    """
    offenders = []
    for path in iter_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in _calls(tree):
            if not _dotted(call.func).endswith("issubdtype"):
                continue
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Attribute) and first.attr == "dtype":
                offenders.append(f"{path.name}:{call.lineno}")
    assert not offenders, (
        "np.issubdtype on a pandas dtype raises under pandas 3: " + ", ".join(offenders)
    )


def test_no_inplace_true():
    """`inplace=True` is a no-op-or-worse under copy-on-write."""
    offenders = []
    for path in iter_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in _calls(tree):
            for keyword in call.keywords:
                if keyword.arg == "inplace" and getattr(
                    keyword.value, "value", False
                ) is True:
                    offenders.append(f"{path.name}:{call.lineno}")
    assert not offenders, "inplace=True under copy-on-write: " + ", ".join(offenders)


def test_no_object_dtype_comparison():
    """`dtype == object` is False for a pandas-3 text column."""
    offenders = []
    for path in iter_source_files():
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.split("#", 1)[0]
            if "dtype == object" in stripped or 'dtype == "object"' in stripped:
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, "dtype == object is unreliable: " + ", ".join(offenders)


@pytest.mark.skipif(
    not hasattr(pd, "StringDtype"), reason="pandas too old to have StringDtype"
)
def test_coordinate_detection_survives_a_string_dtype_column():
    """The concrete regression: a text column beside two coordinates.

    Constructs the pandas-3 shape explicitly, so this fails under pandas 2.1 too rather
    than only in the one environment that defaults to it.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from scads_drvi.viz.umap import umap_continuous

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [1.0, 2.0, 3.0],
            "value": [0.5, 1.5, 2.5],
            "text": pd.array(["a", "b", "c"], dtype=pd.StringDtype()),
        }
    )
    fig, ax = umap_continuous(frame, "value", n=None)
    assert ax.get_xlabel() == "x"
    matplotlib.pyplot.close(fig)


@pytest.mark.skipif(
    not hasattr(pd, "StringDtype"), reason="pandas too old to have StringDtype"
)
def test_latent_dimension_stats_survives_a_string_dtype_column():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from scads_drvi.viz.factors import latent_dimension_stats

    frame = pd.DataFrame(
        {
            "dim": pd.array(["dim_0", "dim_1"], dtype=pd.StringDtype()),
            "vanished": [False, True],
            "effect": [1.0, 0.0],
        }
    )
    fig, _ = latent_dimension_stats(frame)
    assert "1 of 2" in fig.get_suptitle()
    matplotlib.pyplot.close(fig)
