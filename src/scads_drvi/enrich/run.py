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

from scads_drvi.enrich.binary import LDSC_VERSION, LdscBinary, ensure_ldsc, run_ldsc

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["LdscRun"]


@dataclass
class LdscRun:
    """Config shared across one run's ``l2``/``h2`` calls, plus the calls themselves.

    `binary`, `bfile`, `ld_wind_cm`, `w_ld_chr` and `overlap_annot` are the instance
    defaults; each is overridable per call (`bfile` on :meth:`l2`, `w_ld_chr` and
    `overlap_annot` on :meth:`h2`) for the case where they legitimately vary within one
    run -- e.g. a per-chromosome sweep passes a different `bfile` to each :meth:`l2`
    call while everything else stays fixed.
    """

    binary: LdscBinary | str | Path
    bfile: str | Path | None = None
    ld_wind_cm: float = 1.0
    w_ld_chr: str | Path | None = None
    overlap_annot: bool = True
    threads: int | None = None
    python_compat: bool = True
    allow_approximate: bool = False
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
        self, subcommand: str, options: Mapping[str, object], *, python_compat: bool
    ) -> subprocess.CompletedProcess:
        return run_ldsc(
            subcommand,
            options,
            binary=self.binary,
            python_compat=python_compat,
            allow_approximate=self.allow_approximate,
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
        **extra: object,
    ) -> subprocess.CompletedProcess:
        """``ldsc l2`` for one annotation, using `self.bfile`/`self.ld_wind_cm` unless
        overridden."""
        bfile = bfile if bfile is not None else self.bfile
        if bfile is None:
            raise ValueError("no bfile: pass one to l2(), or set it on the LdscRun")
        options = {
            "bfile": bfile,
            "annot": annot,
            "ld_wind_cm": self.ld_wind_cm,
            "out": out,
            **extra,
        }
        return self._run("l2", options, python_compat=self.python_compat)

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
