"""Shared fixtures for the scads_drvi suite.

This suite is deliberately NOT collected together with ``code/tests``. That suite's
conftest inserts seven arm directories onto ``sys.path``; running both in one session
would let those inserts satisfy an import this package is supposed to satisfy itself,
turning a real packaging bug into a passing test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "scads_drvi"


def iter_source_files() -> list[Path]:
    """Every Python file in the package, sorted."""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def docstring_nodes(tree: ast.AST) -> set[int]:
    """`id()` of every string constant that is a module/class/function docstring.

    Prose may legitimately mention a domain term as an example; a value used as data
    may not. Separating the two is what keeps the genericity scan honest instead of
    merely loud.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    out.add(id(first.value))
    return out


def iter_string_constants(path: Path):
    """Yield (lineno, value) for each non-docstring string constant in `path`."""
    tree = ast.parse(path.read_text(), filename=str(path))
    skip = docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                yield node.lineno, node.value


@pytest.fixture(scope="session")
def source_files() -> list[Path]:
    return iter_source_files()
