"""Resolving the LDSC binary, and building commands for it.

S-LDSC is provided by the Rust reimplementation (``sharifhsn/ldsc``, GPL-3.0), pinned by
version and verified by checksum. It replaces the Python original together with the
python-3.10 / numpy-1.23 / pandas-1.5 environment that existed only to keep that
original running, and the three py2-to-py3 patches that environment had to carry.

Because it is a compiled CLI with no Python bindings it cannot be a
``[project.dependencies]`` entry, so it is the one dependency the package installs and
verifies itself rather than declaring.

Three behaviours of the tool are load-bearing and are encoded here rather than left to
whoever types the command:

``--python-compat`` **is an ``l2`` flag only.** It sets the chunk size and the
single-pass traversal that make LD scores bit-identical to the original. There is no
equivalent on ``h2``, so heritability agreement has to be measured against known-good
output rather than assumed.

``--sketch`` **is approximate.** It is a random-projection estimator and it is where most
of the headline speedup comes from. Its own help warns that ``d <= 50`` is numerically
unstable. It changes the answer, so it is off unless a caller says otherwise.

``--gpu`` is experimental and behind a build feature. Off.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LDSC_VERSION",
    "LDSC_ASSETS",
    "LDSC_SHA256",
    "LDSC_ENV_VAR",
    "LdscBinary",
    "ldsc_cache_dir",
    "asset_for_platform",
    "ensure_ldsc",
    "binary_version",
    "build_command",
    "run_ldsc",
]

#: Pinned release. A bump is a reviewed change, not a silent upgrade.
LDSC_VERSION = "v0.5.0"

#: Environment variable naming a binary to use instead of resolving one.
LDSC_ENV_VAR = "SCADS_DRVI_LDSC"

_RELEASE_URL = "https://github.com/sharifhsn/ldsc/releases/download/{version}/{asset}"

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

#: Subcommands, and whether each accepts --python-compat.
SUBCOMMANDS: Mapping[str, bool] = {
    "l2": True,
    "h2": False,
    "rg": False,
    "munge-sumstats": False,
    "make-annot": False,
    "cts-annot": False,
}

#: Flags that change the answer and must be opted into explicitly.
APPROXIMATE_FLAGS = ("sketch", "sketch_maf_aware", "gpu", "gpu_flex32", "fast_f32")


@dataclass(frozen=True)
class LdscBinary:
    """A resolved binary and how it was found."""

    path: Path
    version: str
    source: str  # "explicit" | "environment" | "path" | "cache" | "download"

    def __fspath__(self) -> str:
        return str(self.path)


def ldsc_cache_dir(version: str = LDSC_VERSION) -> Path:
    """Where a downloaded binary is kept.

    ``$SCADS_DRVI_CACHE`` wins, then ``$XDG_CACHE_HOME``, then ``~/.cache`` -- so a
    cluster with a small home directory can point it at scratch.
    """
    root = os.environ.get("SCADS_DRVI_CACHE")
    if root:
        base = Path(root)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "scads_drvi" / "ldsc" / version


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


def _sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _extract(archive: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into)
    for candidate in ("ldsc", "ldsc.exe"):
        found = next(into.rglob(candidate), None)
        if found is not None:
            found.chmod(found.stat().st_mode | 0o755)
            return found
    raise OSError(f"{archive} contains no ldsc executable")


def ensure_ldsc(
    *,
    version: str = LDSC_VERSION,
    explicit: str | Path | None = None,
    cache: str | Path | None = None,
    allow_download: bool = True,
    check_version: bool = True,
) -> LdscBinary:
    """Resolve the ldsc binary.

    Order: `explicit` -> ``$SCADS_DRVI_LDSC`` -> a matching binary on ``PATH`` -> the
    cache -> a checksum-verified download of the pinned release.

    A binary found on ``PATH`` whose version differs from `version` is **refused**, not
    used: silently running a different LD-score implementation than the one recorded is
    exactly the kind of difference that shows up months later as an unreproducible
    number. Pass ``check_version=False`` to accept it deliberately.
    """
    cache_dir = Path(cache) if cache is not None else ldsc_cache_dir(version)
    expected = version.lstrip("v")

    def _accept(path: Path, source: str) -> LdscBinary:
        found = binary_version(path)
        if check_version and found != expected:
            raise OSError(
                f"{path} is ldsc {found}, but this package pins {expected}. Point "
                f"${LDSC_ENV_VAR} at {expected}, or pass check_version=False to accept "
                f"the difference deliberately."
            )
        return LdscBinary(path=path, version=found, source=source)

    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"no ldsc binary at {path}")
        return _accept(path, "explicit")

    from_env = os.environ.get(LDSC_ENV_VAR)
    if from_env:
        path = Path(from_env)
        if not path.exists():
            raise FileNotFoundError(
                f"${LDSC_ENV_VAR} points at {path}, which does not exist"
            )
        return _accept(path, "environment")

    on_path = shutil.which("ldsc")
    if on_path:
        try:
            return _accept(Path(on_path), "path")
        except OSError:
            if check_version:
                raise

    cached = cache_dir / "ldsc"
    if cached.exists():
        return _accept(cached, "cache")

    if not allow_download:
        raise FileNotFoundError(
            f"no ldsc {expected} found and downloading is disabled. Fetch "
            f"{asset_for_platform()} from the {version} release into {cache_dir}, or "
            f"run `cargo install ldsc --version {expected}`."
        )

    asset = asset_for_platform()
    url = _RELEASE_URL.format(version=version, asset=asset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / asset

    if not archive.exists():
        import urllib.request

        tmp = archive.with_suffix(archive.suffix + f".partial.{os.getpid()}")
        with urllib.request.urlopen(url, timeout=300) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(archive)

    want = LDSC_SHA256.get(asset)
    if want is None:
        raise OSError(f"no pinned checksum for {asset} at {version}")
    got = _sha256(archive)
    if got != want:
        archive.unlink(missing_ok=True)
        raise OSError(
            f"checksum mismatch for {asset}: expected {want}, got {got}. The download "
            f"has been removed rather than used."
        )

    extracted = _extract(archive, cache_dir / "unpacked")
    if extracted != cached:
        shutil.copy2(extracted, cached)
        cached.chmod(cached.stat().st_mode | 0o755)
    return _accept(cached, "download")


def build_command(
    subcommand: str,
    options: Mapping[str, object] | None = None,
    *,
    binary: str | Path = "ldsc",
    python_compat: bool = True,
    allow_approximate: bool = False,
) -> list[str]:
    """Build an argv for one ldsc subcommand.

    ``--python-compat`` is added by default where the subcommand accepts it, and
    silently not where it does not -- it exists only on ``l2``.

    An approximate or experimental flag (``sketch``, ``gpu``, ...) raises unless
    `allow_approximate` is set. Those change the answer, and the point of a wrapper is
    that a fast wrong number cannot be produced by accident.
    """
    if subcommand not in SUBCOMMANDS:
        raise ValueError(
            f"unknown subcommand {subcommand!r}; expected one of {sorted(SUBCOMMANDS)}"
        )
    options = dict(options or {})

    used = [flag for flag in APPROXIMATE_FLAGS if options.get(flag)]
    if used and not allow_approximate:
        raise ValueError(
            f"{used} change the estimate rather than only its speed "
            f"(--sketch is a random projection; its own help warns d <= 50 is "
            f"numerically unstable). Pass allow_approximate=True to use them."
        )

    argv = [str(binary), subcommand]
    if python_compat and SUBCOMMANDS[subcommand]:
        argv.append("--python-compat")

    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.extend([flag, ",".join(str(v) for v in value)])
        else:
            argv.extend([flag, str(value)])
    return argv


def run_ldsc(
    subcommand: str,
    options: Mapping[str, object] | None = None,
    *,
    binary: str | Path | LdscBinary = "ldsc",
    python_compat: bool = True,
    allow_approximate: bool = False,
    threads: int | None = None,
    check: bool = True,
    dry_run: bool = False,
    log_fn=None,
) -> subprocess.CompletedProcess:
    """Run one ldsc subcommand.

    `threads` caps the tool's own parallelism, which matters on a shared node: the
    default is a library heuristic that assumes it owns the machine.
    """
    from scads_drvi._util.progress import log

    options = dict(options or {})
    if threads is not None:
        options.setdefault("rayon_threads", int(threads))
        options.setdefault("polars_threads", int(threads))

    argv = build_command(
        subcommand,
        options,
        binary=getattr(binary, "path", binary),
        python_compat=python_compat,
        allow_approximate=allow_approximate,
    )
    (log_fn or log)("$ " + " ".join(argv))
    if dry_run:
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")
    return subprocess.run(argv, capture_output=True, text=True, check=check)
