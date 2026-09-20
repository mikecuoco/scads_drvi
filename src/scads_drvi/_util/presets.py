#!/usr/bin/env python3
"""Named parameter sets for a stage's scripts, loaded from that stage's `*_config.py`.

Replaces argparse for everything except which preset to run: a value that can be passed on
the command line is a value a run record cannot recover.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (tuple, list, set)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    return v


def find_config(script: Path) -> Path:
    """The single `*_config.py` beside `script`.

    Globbed rather than derived from the directory name so adding an arm needs no table
    here. Two of them is an error, not a pick -- there is no rule for which would win.
    """
    hits = sorted(p for p in script.parent.glob("*_config.py") if p.name != "config.py")
    if len(hits) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one *_config.py in {script.parent}, "
            f"found {[p.name for p in hits]}"
        )
    return hits[0]


def load_module(config: Path) -> ModuleType:
    """Import a config module BY PATH.

    Never via `sys.path`: these scripts already insert their shared directory at
    sys.path[0], so a config module resolved by name could be shadowed by, or shadow,
    another module there.
    A path import cannot collide. `sys.modules` is populated so dataclasses defined in the
    module pickle and `replace()` cleanly.
    """
    config = config.resolve()
    name = f"_armcfg_{config.parent.name}_{config.stem}"
    spec = importlib.util.spec_from_file_location(name, config)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: cannot import config module {config}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PRESETS"):
        raise SystemExit(f"ERROR: {config} defines no PRESETS dict")
    return mod


def presets_for(script: Path, config: Path | None = None) -> tuple[dict[str, Any], Path]:
    """`PRESETS[script.stem]` plus the module path it came from."""
    config = Path(config) if config else find_config(script)
    table = load_module(config).PRESETS
    if script.stem not in table:
        raise SystemExit(
            f"ERROR: {config.name} has no presets for {script.stem!r}; it covers {sorted(table)}"
        )
    return table[script.stem], config


def _pick(names: list[str]) -> str:
    if "production" in names:
        return "production"
    if len(names) == 1:
        return names[0]
    raise SystemExit(
        f"ERROR: no preset given and no default to fall back on. Pass one of: {', '.join(names)}"
    )


def load_preset(
    script: Path, name: str | None = None, *, config: Path | None = None
) -> tuple[Any, str, Path]:
    """Return one preset. An unknown name names the ones that exist rather than KeyError."""
    table, path = presets_for(script, config)
    name = name or _pick(sorted(table))
    if name not in table:
        raise SystemExit(
            f"ERROR: {script.name} has no preset {name!r} in {path.name}; "
            f"choose from: {', '.join(sorted(table))}"
        )
    return table[name], name, path


def describe(cfg: Any) -> str:
    """Paths print bare, not as PosixPath(...) -- this output is meant to be diffed."""
    return "\n".join(
        f"    {f.name} = {_to_jsonable(getattr(cfg, f.name))!r}" for f in dataclasses.fields(cfg)
    )


def provenance(cfg: Any, script: Path, name: str, config: Path) -> dict[str, Any]:
    """The run record's config block.

    `config_sha256` is what makes a recorded run reproducible: the preset name alone does
    not pin the values, because the module is editable.
    """
    return {
        "config_module": str(config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "preset": name,
        "resolved": {f.name: _to_jsonable(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)},
    }


def preset_cli(script: Path, argv: Sequence[str] | None = None) -> tuple[Any, dict[str, Any]]:
    """The whole CLI: a preset name, `--config`, `--list`. No parameter is reachable here.

    Deliberate: a value that can be passed on the command line is a value a run record
    cannot recover, which is the failure this module exists to close.
    """
    script = Path(script).resolve()
    try:
        known = sorted(presets_for(script)[0])
    except SystemExit:
        known = []
    ap = argparse.ArgumentParser(
        description=script.name.replace(".py", "") + " -- parameters come from the arm's "
        "*_config.py, not from flags. See --list.",
        epilog="presets: " + (", ".join(known) if known else "(none found)"),
    )
    ap.add_argument(
        "preset",
        nargs="?",
        default=None,
        help="which preset to run; default 'production' when it exists",
    )
    ap.add_argument(
        "--config",
        default=None,
        type=Path,
        help="a config module other than the one beside this script",
    )
    ap.add_argument("--list", action="store_true", help="print every preset resolved, and exit")
    args = ap.parse_args(argv)

    if args.list:
        table, path = presets_for(script, args.config)
        print(f"# {script.name} presets, from {path}")
        for n in sorted(table):
            print(f"{n}:\n{describe(table[n])}")
        raise SystemExit(0)

    cfg, name, path = load_preset(script, args.preset, config=args.config)
    return cfg, provenance(cfg, script, name, path)
