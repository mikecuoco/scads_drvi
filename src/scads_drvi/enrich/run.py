"""One object for a run of S-LDSC ``l2``/``h2`` calls.

:func:`~scads_drvi.enrich.binary.run_ldsc` and :func:`~scads_drvi.enrich.ldsc.read_results`
stay plain functions -- this class does not replace them, it only stops a caller from
repeating the config that is the same on every call of a single run (``bfile``,
``w_ld_chr``, ``overlap_annot``, ``ld_wind_cm``, which binary) while every call still
goes through those same functions underneath.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scads_drvi.enrich.binary import (
    APPROXIMATE_FLAGS,
    LDSC_VERSION,
    LdscBinary,
    ensure_ldsc,
    run_ldsc,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["LdscRun"]

#: Sentinel distinguishing "use the instance default" from an explicit per-call value
#: (including one that turns a default-on flag off, e.g. ``sketch=None``).
_UNSET = object()


@dataclass
class LdscRun:
    """Config shared across one run's ``l2``/``h2`` calls, plus the calls themselves.

    `binary`, `bfile`, `ld_wind_cm`, `w_ld_chr` and `overlap_annot` are the instance
    defaults; each is overridable per call (`bfile` on :meth:`l2`, `w_ld_chr` and
    `overlap_annot` on :meth:`h2`) for the case where they legitimately vary within one
    run -- e.g. a per-chromosome sweep passes a different `bfile` to each :meth:`l2`
    call while everything else stays fixed.

    `sketch` and `snp_level_masking` default to on (``5000`` and ``True``) for speed:
    the Rust binary's own ``l2 --help`` calls ``d <= 50`` numerically unstable and
    ``d=200`` "the practical sweet spot", so ``5000`` is comfortably past both, and
    ``snp_level_masking`` is described there as exact-and-free rather than
    approximate. Because ``--python-compat`` disables ``snp_level_masking`` on the
    binary itself (it exists to reproduce Python LDSC's chunk-level approximation
    exactly), `python_compat` defaults to `False` here to match -- the opposite of
    :func:`~scads_drvi.enrich.binary.run_ldsc`'s own default, which is bit-identical
    reproduction, not speed. `l2`'s `allow_approximate` gate on
    :func:`~scads_drvi.enrich.binary.run_ldsc` is derived automatically from whether an
    :data:`~scads_drvi.enrich.binary.APPROXIMATE_FLAGS` entry ends up set, rather than
    a separate switch to remember: setting `sketch` (here or per call) is itself the
    opt-in.
    """

    binary: LdscBinary | str | Path
    bfile: str | Path | None = None
    ld_wind_cm: float = 1.0
    w_ld_chr: str | Path | None = None
    overlap_annot: bool = True
    threads: int | None = None
    python_compat: bool = False
    sketch: int | None = 5000
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
        **kwargs,
    ) -> LdscRun:
        """:func:`~scads_drvi.enrich.binary.ensure_ldsc` then wrap it -- the common case."""
        binary = ensure_ldsc(
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

    def _run(
        self,
        subcommand: str,
        options: Mapping[str, object],
        *,
        python_compat: bool,
        allow_approximate: bool = False,
    ) -> subprocess.CompletedProcess:
        return run_ldsc(
            subcommand,
            options,
            binary=self.binary,
            python_compat=python_compat,
            allow_approximate=allow_approximate,
            threads=self.threads,
            check=self.check,
            dry_run=self.dry_run,
            log_fn=self.log_fn,
        )

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
    ) -> subprocess.CompletedProcess:
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
    ) -> subprocess.CompletedProcess:
        """``ldsc h2`` for one trait/annotation, using `self.w_ld_chr`/`self.overlap_annot`
        unless overridden. ``--python-compat`` is never passed: it is an ``l2``-only
        flag (see :mod:`scads_drvi.enrich.binary`)."""
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
        traits,
        annot2dim: Mapping[str, str],
        **kwargs,
    ) -> pd.DataFrame:
        """:func:`scads_drvi.enrich.ldsc.read_results` -- one import for the whole run."""
        from scads_drvi.enrich.ldsc import read_results

        return read_results(
            results_root, traits=traits, annot2dim=annot2dim, **kwargs
        )
