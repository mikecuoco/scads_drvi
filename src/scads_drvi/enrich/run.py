"""One object for a run of S-LDSC ``l2``/``h2`` calls.

Resolving the pinned binary, building each subcommand's argv, and invoking it are all
``LdscRun``'s own methods -- the only thing left in :mod:`scads_drvi.enrich.binary` is
what has no say from a run's config at all: finding, downloading and verifying the
compiled binary file. :func:`scads_drvi.enrich.ldsc.read_results` stays a plain
function; this class does not replace it, it only stops a caller from repeating the
config that is the same on every call of a single run (``bfile``, ``w_ld_chr``,
``overlap_annot``, ``ld_wind_cm``, which binary) while every call still goes through
the same three subcommand behaviours documented below.

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
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scads_drvi.enrich.binary import (
    LDSC_ENV_VAR,
    LDSC_SHA256,
    LDSC_VERSION,
    LdscBinary,
    asset_for_platform,
    binary_version,
    download_file,
    ldsc_cache_dir,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["LdscRun", "LDSC_VERSION"]

#: Sentinel distinguishing "use the instance default" from an explicit per-call value
#: (including one that turns a default-on flag off, e.g. ``sketch=None``). Typed `Any`
#: so it can stand in as the default for whichever concrete parameter type it sentinels.
_UNSET: Any = object()

_RELEASE_URL = "https://github.com/sharifhsn/ldsc/releases/download/{version}/{asset}"

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


def _sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@dataclass
class LdscRun:
    """Config shared across one run's ``l2``/``h2`` calls, plus the calls themselves.

    `binary`, `bfile`, `ld_wind_cm`, `w_ld_chr` and `overlap_annot` are the instance
    defaults; each is overridable per call (`bfile` on :meth:`l2`, `w_ld_chr` and
    `overlap_annot` on :meth:`h2`) for the case where they legitimately vary within one
    run -- e.g. a per-chromosome sweep passes a different `bfile` to each :meth:`l2`
    call while everything else stays fixed.

    `sketch` and `snp_level_masking` default to on (``200`` and ``True``) for speed:
    the Rust binary's own ``l2 --help`` calls ``d <= 50`` numerically unstable and
    names ``d=200`` "the practical sweet spot" directly, and ``snp_level_masking`` is
    described there as exact-and-free rather than approximate. Because
    ``--python-compat`` disables ``snp_level_masking`` on the binary itself (it exists
    to reproduce Python LDSC's chunk-level approximation exactly), `python_compat`
    defaults to `False` here to match -- the opposite of :meth:`_run`'s own default,
    which is bit-identical reproduction, not speed. `l2`'s `allow_approximate` gate is
    derived automatically from whether an :data:`APPROXIMATE_FLAGS` entry ends up set,
    rather than a separate switch to remember: setting `sketch` (here or per call) is
    itself the opt-in.
    """

    binary: LdscBinary | str | Path
    bfile: str | Path | None = None
    ld_wind_cm: float = 1.0
    w_ld_chr: str | Path | None = None
    overlap_annot: bool = True
    threads: int | None = None
    python_compat: bool = False
    sketch: int | None = 200
    snp_level_masking: bool = True
    check: bool = True
    dry_run: bool = False
    log_fn: Callable[[str], None] | None = None

    @classmethod
    def ensure(
        cls,
        *,
        bfile: str | Path | None = None,
        ld_wind_cm: float = 1.0,
        w_ld_chr: str | Path | None = None,
        overlap_annot: bool = True,
        threads: int | None = None,
        version: str = LDSC_VERSION,
        explicit: str | Path | None = None,
        cache: str | Path | None = None,
        allow_download: bool = True,
        check_version: bool = True,
        **kwargs: Any,
    ) -> LdscRun:
        """Resolve the ldsc binary, then wrap it -- the common case."""
        binary = cls._resolve_binary(
            version=version,
            explicit=explicit,
            cache=cache,
            allow_download=allow_download,
            check_version=check_version,
        )
        return cls(
            binary=binary,
            bfile=bfile,
            ld_wind_cm=ld_wind_cm,
            w_ld_chr=w_ld_chr,
            overlap_annot=overlap_annot,
            threads=threads,
            **kwargs,
        )

    @staticmethod
    def _resolve_binary(
        *,
        version: str = LDSC_VERSION,
        explicit: str | Path | None = None,
        cache: str | Path | None = None,
        allow_download: bool = True,
        check_version: bool = True,
    ) -> LdscBinary:
        """Resolve the ldsc binary.

        Order: `explicit` -> ``$SCADS_DRVI_LDSC`` -> a matching binary on ``PATH`` ->
        the cache -> a checksum-verified download of the pinned release.

        A binary found on ``PATH`` whose version differs from `version` is
        **refused**, not used: silently running a different LD-score implementation
        than the one recorded is exactly the kind of difference that shows up months
        later as an unreproducible number. Pass ``check_version=False`` to accept it
        deliberately.
        """
        cache_dir = Path(cache) if cache is not None else ldsc_cache_dir(version)
        expected = version.lstrip("v")

        def _accept(path: Path, source: str) -> LdscBinary:
            found = binary_version(path)
            if check_version and found != expected:
                raise OSError(
                    f"{path} is ldsc {found}, but this package pins {expected}. Point "
                    f"${LDSC_ENV_VAR} at {expected}, or pass check_version=False to "
                    f"accept the difference deliberately."
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
                f"{asset_for_platform()} from the {version} release into {cache_dir}, "
                f"or run `cargo install ldsc --version {expected}`."
            )

        asset = asset_for_platform()
        url = _RELEASE_URL.format(version=version, asset=asset)
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / asset

        if not archive.exists():
            download_file(url, archive, timeout=300)

        want = LDSC_SHA256.get(asset)
        if want is None:
            raise OSError(f"no pinned checksum for {asset} at {version}")
        got = _sha256(archive)
        if got != want:
            archive.unlink(missing_ok=True)
            raise OSError(
                f"checksum mismatch for {asset}: expected {want}, got {got}. The "
                f"download has been removed rather than used."
            )

        extracted = _extract(archive, cache_dir / "unpacked")
        if extracted != cached:
            shutil.copy2(extracted, cached)
            cached.chmod(cached.stat().st_mode | 0o755)
        return _accept(cached, "download")

    def _build_command(
        self,
        subcommand: str,
        options: Mapping[str, object],
        *,
        python_compat: bool = True,
        allow_approximate: bool = False,
    ) -> list[str]:
        """Build an argv for one ldsc subcommand, against `self.binary`.

        ``--python-compat`` is added by default where the subcommand accepts it, and
        silently not where it does not -- it exists only on ``l2``.

        An approximate or experimental flag (``sketch``, ``gpu``, ...) raises unless
        `allow_approximate` is set. Those change the answer, and the point of a
        wrapper is that a fast wrong number cannot be produced by accident.
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

        argv = [str(getattr(self.binary, "path", self.binary)), subcommand]
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

    def _run(
        self,
        subcommand: str,
        options: Mapping[str, object],
        *,
        python_compat: bool,
        allow_approximate: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run one ldsc subcommand.

        `threads` caps the tool's own parallelism, which matters on a shared node: the
        default is a library heuristic that assumes it owns the machine.
        """
        from scads_drvi._util.progress import log

        options = dict(options or {})
        if self.threads is not None:
            options.setdefault("rayon_threads", int(self.threads))
            options.setdefault("polars_threads", int(self.threads))

        argv = self._build_command(
            subcommand,
            options,
            python_compat=python_compat,
            allow_approximate=allow_approximate,
        )
        (self.log_fn or log)("$ " + " ".join(argv))
        if self.dry_run:
            return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")
        return subprocess.run(argv, capture_output=True, text=True, check=self.check)

    def l2(
        self,
        annot: str | Path,
        out: str | Path,
        *,
        bfile: str | Path | None = None,
        python_compat: bool = _UNSET,
        sketch: int | None = _UNSET,
        snp_level_masking: bool = _UNSET,
        **extra: object,
    ) -> subprocess.CompletedProcess[str]:
        """``ldsc l2`` for one annotation, using the instance's `bfile`/`ld_wind_cm`/
        `sketch`/`snp_level_masking`/`python_compat` unless overridden."""
        bfile = bfile if bfile is not None else self.bfile
        if bfile is None:
            raise ValueError("no bfile: pass one to l2(), or set it on the LdscRun")
        python_compat = self.python_compat if python_compat is _UNSET else python_compat
        sketch = self.sketch if sketch is _UNSET else sketch
        snp_level_masking = (
            self.snp_level_masking if snp_level_masking is _UNSET else snp_level_masking
        )
        if python_compat and snp_level_masking:
            raise ValueError(
                "python_compat and snp_level_masking cannot both be on for one l2() "
                "call: the ldsc binary itself refuses --python-compat with "
                "--snp-level-masking (--python-compat disables snp-level masking to "
                "reproduce Python LDSC's chunk-level approximation exactly). Pass "
                "snp_level_masking=False to keep python_compat, or leave python_compat "
                "at the LdscRun default (False) to keep snp_level_masking."
            )
        options = {
            "bfile": bfile,
            "annot": annot,
            "ld_wind_cm": self.ld_wind_cm,
            "sketch": sketch,
            "snp_level_masking": snp_level_masking,
            "out": out,
            **extra,
        }
        allow_approximate = any(options.get(flag) for flag in APPROXIMATE_FLAGS)
        return self._run(
            "l2", options, python_compat=python_compat, allow_approximate=allow_approximate
        )

    def h2(
        self,
        h2: str | Path,
        ref_ld_chr: str | Path | list[str | Path],
        out: str | Path,
        *,
        w_ld_chr: str | Path | None = None,
        overlap_annot: bool | None = None,
        **extra: object,
    ) -> subprocess.CompletedProcess[str]:
        """``ldsc h2`` for one trait/annotation, using `self.w_ld_chr`/`self.overlap_annot`
        unless overridden. ``--python-compat`` is never passed: it is an ``l2``-only
        flag."""
        w_ld_chr = w_ld_chr if w_ld_chr is not None else self.w_ld_chr
        if w_ld_chr is None:
            raise ValueError("no w_ld_chr: pass one to h2(), or set it on the LdscRun")
        overlap_annot = self.overlap_annot if overlap_annot is None else overlap_annot
        options = {
            "h2": h2,
            "ref_ld_chr": ref_ld_chr,
            "w_ld_chr": w_ld_chr,
            "overlap_annot": overlap_annot,
            "out": out,
            **extra,
        }
        return self._run("h2", options, python_compat=False)

    def read_results(
        self,
        results_root: str | Path,
        *,
        traits: Iterable[str],
        annot2dim: Mapping[str, str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """:func:`scads_drvi.enrich.ldsc.read_results` -- one import for the whole run."""
        from scads_drvi.enrich.ldsc import read_results

        return read_results(
            results_root, traits=traits, annot2dim=annot2dim, **kwargs
        )
