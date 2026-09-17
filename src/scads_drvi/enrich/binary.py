"""Resolving the pinned LDSC binary: finding it, downloading it, verifying it.

S-LDSC is provided by the Rust reimplementation (``sharifhsn/ldsc``, GPL-3.0), pinned by
version and verified by checksum. It replaces the Python original together with the
python-3.10 / numpy-1.23 / pandas-1.5 environment that existed only to keep that
original running, and the three py2-to-py3 patches that environment had to carry.

Because it is a compiled CLI with no Python bindings it cannot be a
``[project.dependencies]`` entry, so it is the one dependency the package installs and
verifies itself rather than declaring.

What this module does NOT do: build or run an ldsc command. That needs a run's own
config (``bfile``, ``w_ld_chr``, whether ``--python-compat``/``--sketch`` are wanted) and
lives on :class:`~scads_drvi.enrich.run.LdscRun` instead -- this module only gets you a
verified path to the binary, independent of any particular call.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LDSC_VERSION",
    "LDSC_ASSETS",
    "LDSC_SHA256",
    "LDSC_ENV_VAR",
    "LdscBinary",
    "cache_root",
    "ldsc_cache_dir",
    "download_file",
    "asset_for_platform",
    "binary_version",
]

#: Pinned release. A bump is a reviewed change, not a silent upgrade.
LDSC_VERSION = "v0.5.0"

#: Environment variable naming a binary to use instead of resolving one.
LDSC_ENV_VAR = "SCADS_DRVI_LDSC"

#: (system, machine) -> release asset name.
LDSC_ASSETS: Mapping[tuple[str, str], str] = {
    ("linux", "x86_64"): "ldsc_linux-x86_64.tar.gz",
    ("linux", "amd64"): "ldsc_linux-x86_64.tar.gz",
    ("darwin", "arm64"): "ldsc_macos-aarch64.tar.gz",
    ("darwin", "aarch64"): "ldsc_macos-aarch64.tar.gz",
    ("windows", "amd64"): "ldsc_windows-x86_64.zip",
    ("windows", "x86_64"): "ldsc_windows-x86_64.zip",
}

#: Published SHA-256 of each asset at :data:`LDSC_VERSION`.
LDSC_SHA256: Mapping[str, str] = {
    "ldsc_linux-x86_64.tar.gz":
        "03d766d7a3844e2b621468416057a2a614a8d74e92bee0841c163c6ddea03722",
    "ldsc_macos-aarch64.tar.gz":
        "b1c25a2885525d438aac54afb96443670bea1eccf347ca5a02b2837d156cb8d8",
    "ldsc_windows-x86_64.zip":
        "41410252b18d7ecdddf98828b106b2aee4786033a8e5f937ac8d6366ee4b94ef",
}


@dataclass(frozen=True)
class LdscBinary:
    """A resolved binary and how it was found."""

    path: Path
    version: str
    source: str  # "explicit" | "environment" | "path" | "cache" | "download"

    def __fspath__(self) -> str:
        return str(self.path)


def cache_root() -> Path:
    """Where this package's downloaded/cached artifacts live, before any per-artifact
    subpath is appended.

    ``$SCADS_DRVI_CACHE`` wins, then ``$XDG_CACHE_HOME``, then ``~/.cache`` -- so a
    cluster with a small home directory can point it at scratch. Shared by
    :func:`ldsc_cache_dir` (this module's own binary cache) and
    :func:`~scads_drvi.enrich.reference.reference_cache_dir` (the much larger
    reference-dataset cache) so the two never drift out of sync on how the env vars
    are resolved.
    """
    root = os.environ.get("SCADS_DRVI_CACHE")
    if root:
        return Path(root)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg) if xdg else Path.home() / ".cache"


def ldsc_cache_dir(version: str = LDSC_VERSION) -> Path:
    """Where a downloaded binary is kept. See :func:`cache_root` for the env-var
    resolution this builds on."""
    return cache_root() / "scads_drvi" / "ldsc" / version


def asset_for_platform(
    system: str | None = None, machine: str | None = None
) -> str:
    """The release asset for this platform."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    if machine in ("x86-64", "x64"):
        machine = "x86_64"
    try:
        return LDSC_ASSETS[(system, machine)]
    except KeyError:
        raise OSError(
            f"no pinned ldsc release for {system}/{machine}. Available: "
            f"{sorted({v for v in LDSC_ASSETS.values()})}. Build from source with "
            f"`cargo install ldsc --version {LDSC_VERSION.lstrip('v')}` and point "
            f"${LDSC_ENV_VAR} at the result."
        ) from None


def download_file(url: str, dest: Path, *, timeout: int = 300) -> None:
    """Stream `url` to `dest`, atomically -- `dest` never exists half-written.

    Downloads to a ``.partial.{pid}`` sibling first, then renames onto `dest` only
    once the transfer completes. Shared by
    :meth:`~scads_drvi.enrich.run.LdscRun._resolve_binary` and
    :func:`~scads_drvi.enrich.reference.ensure_baseline_ukb`, whose only difference is
    `timeout` (seconds for a request that can be gigabytes).
    """
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + f".partial.{os.getpid()}")
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dest)


def binary_version(binary: str | Path) -> str:
    """``ldsc --version`` as a bare version string, e.g. ``0.5.0``."""
    proc = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise OSError(f"{binary} --version failed: {proc.stderr.strip()}")
    match = re.search(r"(\d+\.\d+\.\d+)", proc.stdout)
    if match is None:
        raise OSError(f"cannot parse a version out of {proc.stdout!r}")
    return match.group(1)
