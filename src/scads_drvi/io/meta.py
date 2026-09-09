#!/usr/bin/env python3
"""Run-record load/save. Parameters live in the arm's `*_config.py`; see `presets.py`."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path


@contextlib.contextmanager
def _atomic_write(path: Path, *, suffix: str = ".tmp"):
    """Yield a temp path; rename over `path` only if the body succeeds.

    Local rather than imported: a run record is written after every stage of a run that
    can last hours, and a half-written one is worse than none. That guarantee is small
    enough to own outright, and owning it keeps this module free of any dependency on
    where a particular deployment puts scratch space.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}{suffix}.{os.getpid()}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def load_meta(path: Path, fit_dir: Path, **fields) -> dict:
    """Read a run record, merge `fields`, attach the fit's provenance.

        Read-merge, not overwrite: these run stage by stage, and a later stage must not erase what
        an earlier one recorded. `fit_meta` is unconditional so a missing fit.meta.json fails here
        rather than an hour into a traversal.
    """
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.update(fields)
    meta["fit_meta"] = json.loads((fit_dir / "fit.meta.json").read_text())
    return meta


def save_meta(path: Path, meta: dict) -> None:
    """Write the run record atomically, after EVERY stage -- pseudobulk is a 1.3 h pass."""
    with _atomic_write(path) as tmp:
        tmp.write_text(json.dumps(meta, indent=2, default=str))


def assert_gate_matches_fit(meta: dict, min_fragment: int) -> None:
    """The one coupling between a fit and a stage that reads it.

        A post-hoc statistic estimated on cells the model never saw is not wrong-looking --
        it is a plausible number for a different population. Nothing else about a preset is
        coupled this way, so nothing else is asserted here: re-diagnosing an ungated fit ON
        a gated fit's cells is a deliberate use, and it goes through `--config`.
    """
    fit = (meta.get("fit_meta") or {}).get("min_fragment")
    if fit is not None and int(fit) != int(min_fragment):
        raise SystemExit(
            f"ERROR: this fit trained under min_fragment={fit} but the preset asks for "
            f"{min_fragment}. Use the preset that matches, or point --config at a module "
            f"whose min_fragment is {fit}.")
