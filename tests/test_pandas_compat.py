"""The package must work on both sides of the pandas 2 -> 3 boundary.

Two of the three environments this runs in are pandas 3.0 and one is the declared floor
(now 2.3), and the differences are not cosmetic: from 3.0 a text column is a
``StringDtype`` extension dtype rather than ``object``, copy-on-write is unconditional,
and ``applymap`` is gone.

The AST checks below catch the spellings that break silently or only in one environment,
so a regression is a test failure rather than a crash discovered by whoever next runs
the notebook.

One regression this module used to guard directly no longer applies: `pl.umap`'s own
``_coords``/numeric-dtype-detection code (the concrete StringDtype-vs-``np.issubdtype``
bug this suite exists because of) was deleted when embedding plotting moved onto
``scanpy.pl.embedding`` -- coordinate/column resolution is scanpy's responsibility now,
with its own upstream pandas-version handling, not this package's.
"""

from __future__ import annotations

import ast

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
