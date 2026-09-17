"""The DRVI-to-S-LDSC sweep: embed + reference panel + sumstats in, tidy results out.

:class:`~scads_drvi.enrich.run.LdscRun` wraps one ``l2``/``h2`` call's shared config;
this class wraps the sweep across kept dimensions, directions and chromosomes that
actually produces a real result -- the manual loop in the getting-started guide, at
real scale (22 chromosomes x several factors x 2 directions, hours -- see
``docs/tutorials/pbmc.ipynb``'s "Enrich" section).

**Peak-to-SNP annotation building is not this class's job.** This package has never
owned that overlap (see :mod:`scads_drvi.enrich.config`'s module docstring and the
tutorial's own words: "scads-drvi deliberately does not own peak-to-SNP annotation
building -- that overlap is caller-supplied, same as the peaks themselves"). The
`annotate` callback is where that logic lives; this class owns reading the bim,
widening and writing the annotation file, running ``l2``/``h2``, and reading results
back.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scads_drvi.enrich.annotations import read_bim, write_full_annot
from scads_drvi.enrich.run import LDSC_VERSION, LdscRun

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from anndata import AnnData

__all__ = ["EnrichmentSweep"]

#: Both directions of DRVI's own ``get_effect_of_splits_within_distribution(...,
#: directional=True)`` output -- the only directions this class knows about.
_DIRECTIONS = ("pos", "neg")
_COMBINED = ("combined",)

#: Sentinel distinguishing "use the default" (:meth:`EnrichmentSweep.ensure` resolves
#: the UKB baseline reference) from an explicit per-call value, including `()` --
#: deliberately no baseline categories at all.
_UNSET = object()


def _key(dim: str, direction: str) -> str:
    """The annotation/LD-score/results file stem for one (dim, direction) pair."""
    return dim if direction == "combined" else f"{dim}_{direction}"


@dataclass
class EnrichmentSweep:
    """Factor selection, the ``l2``/``h2`` sweep, and reading results back -- one object.

    `embed` accepts an h5ad path (the typical case: this reads it with
    ``anndata.read_h5ad`` and exposes the result as :attr:`adata`) or an already-loaded
    `AnnData`. `ldscore_dir`/`results_dir` default to the h5ad's own parent directory
    when `embed` is a path -- output lands next to the input, nothing else to
    configure -- and must be given explicitly when `embed` is already an in-memory
    object with no path to default from.

    Annotation files persist at the same prefix as their ``.l2.ldscore.gz`` (not a
    temp directory): confirmed against the real binary that ``h2 --overlap-annot``
    reads the ``.annot`` file back from ``--ref-ld-chr``'s own prefix to compute each
    category's ``Prop._SNPs`` -- it errors ("No .annot files found for prefix ...")
    if that file isn't still there by the time :meth:`run_h2` runs, even though
    :meth:`run_l2` already finished with it and no code here reads it again.

    `bfile_chr` is a ``str.format`` template with a literal ``"{chrom}"``, e.g.
    ``f"{REF}/plink_files/1000G.EUR.hg38.{{chrom}}"``, formatted per chromosome as
    ``bfile_chr.format(chrom=chrom)`` -- the same plain-string convention already used
    for ``ref_ld_chr``/``frqfile_chr``.

    `annotate(dim, direction, chrom, bim) -> ArrayLike` is called once per (kept dim,
    direction, chromosome). `direction` is ``"pos"``/``"neg"`` when `directional`
    (the default), else the literal ``"combined"``. It must return a 1-D array-like of
    ``len(bim)``, in `bim`'s row order -- peak lookup, `top_frac` thresholds, whatever
    turns loadings into that value, is the callback's own business; it typically
    closes over precomputed pos/neg loadings rather than receiving them as arguments,
    so this one signature stays stable regardless of how a caller derived them.
    """

    ldsc_run: LdscRun
    embed: str | Path | AnnData
    bfile_chr: str
    sumstats: Mapping[str, str | Path]
    annotate: Callable[[str, str, int, pd.DataFrame], Any]
    chroms: Sequence[int] = tuple(range(1, 23))
    ref_ld_chr_extra: Sequence[str | Path] = ()
    frqfile_chr: str | Path | None = None
    print_snps: str | Path | None = None
    print_coefficients: bool = True
    directional: bool = True
    exclude_vanished: bool = True
    exclude_dims: Sequence[str] = ()
    ldscore_dir: str | Path | None = None
    results_dir: str | Path | None = None

    def __post_init__(self) -> None:
        # config.py is the one module in this package that imports numpy/pandas/yaml
        # at its own module scope (see its docstring); importing from it here, not at
        # this module's top, keeps `import scads_drvi.enrich.sweep` itself as cheap as
        # every other enrich module until an EnrichmentSweep actually gets built.
        from scads_drvi.enrich.config import kept_dims, select_factors

        if isinstance(self.embed, (str, Path)):
            import anndata as ad

            embed_path = Path(self.embed)
            self._embed_path: Path | None = embed_path
            self.adata = ad.read_h5ad(embed_path)
        else:
            self._embed_path = None
            self.adata = self.embed

        if self.ldscore_dir is None or self.results_dir is None:
            if self._embed_path is None:
                raise ValueError(
                    "ldscore_dir and results_dir must both be given explicitly when "
                    "embed is an in-memory AnnData rather than an h5ad path -- there "
                    "is no file location to default them from."
                )
            default_dir = self._embed_path.parent
            if self.ldscore_dir is None:
                self.ldscore_dir = default_dir
            if self.results_dir is None:
                self.results_dir = default_dir
        self.ldscore_dir = Path(self.ldscore_dir)
        self.results_dir = Path(self.results_dir)

        if self.ldsc_run.overlap_annot and self.frqfile_chr is None:
            raise ValueError(
                "overlap_annot is on but frqfile_chr is None: the ldsc binary itself "
                "requires --frqfile-chr with --overlap-annot (unless --not-m-5-50 is "
                "set). Better to fail now than after chromosomes of l2 computation."
            )

        self.fmap = select_factors(
            self.adata.var,
            list(self.adata.var_names),
            exclude_vanished=self.exclude_vanished,
            exclude_dims=self.exclude_dims,
        )
        self.keep = kept_dims(self.fmap)

    def _directions(self) -> tuple[str, ...]:
        return _DIRECTIONS if self.directional else _COMBINED

    def _keys(
        self, dims: Iterable[str] | None, directions: Iterable[str] | None
    ) -> list[tuple[str, str, str]]:
        dims = list(dims) if dims is not None else self.keep
        directions = list(directions) if directions is not None else self._directions()
        return [(_key(dim, direction), dim, direction) for dim in dims for direction in directions]

    def annot2dim(
        self, *, dims: Iterable[str] | None = None, directions: Iterable[str] | None = None
    ) -> dict[str, str]:
        return {key: dim for key, dim, _ in self._keys(dims, directions)}

    def direction_map(
        self, *, dims: Iterable[str] | None = None, directions: Iterable[str] | None = None
    ) -> dict[str, str] | str:
        if not self.directional:
            return "combined"
        return {key: direction for key, _, direction in self._keys(dims, directions)}

    def _ld_out(self, key: str, chrom: int) -> Path:
        return self.ldscore_dir / f"{key}.{chrom}"

    def _ref_ld_chr(self, key: str) -> list[str]:
        return [str(self.ldscore_dir / key) + "."] + [str(p) for p in self.ref_ld_chr_extra]

    def _h2_out(self, trait: str, key: str) -> Path:
        return self.results_dir / trait / key

    def run_l2(
        self,
        *,
        dims: Iterable[str] | None = None,
        directions: Iterable[str] | None = None,
        chroms: Iterable[int] | None = None,
        force: bool = False,
    ) -> list[subprocess.CompletedProcess]:
        """``l2`` for every (kept dim, direction, chromosome), resumable on the
        ``.l2.ldscore.gz`` a prior call already wrote."""
        import pandas as pd

        completed = []
        for key, dim, direction in self._keys(dims, directions):
            for chrom in chroms if chroms is not None else self.chroms:
                out = self._ld_out(key, chrom)
                if not force and Path(f"{out}.l2.ldscore.gz").exists():
                    continue
                bfile = self.bfile_chr.format(chrom=chrom)
                bim = read_bim(f"{bfile}.bim")
                values = self.annotate(dim, direction, chrom, bim)
                # Same prefix as `out` (l2's own --out below), not a temp dir -- see
                # the class docstring: h2 --overlap-annot reads this file back later.
                annot_path = write_full_annot(
                    Path(f"{out}.annot.gz"), pd.DataFrame({key: values}), bim
                )
                completed.append(
                    self.ldsc_run.l2(
                        annot_path, out, bfile=bfile, print_snps=self.print_snps
                    )
                )
        return completed

    def run_h2(
        self,
        *,
        traits: Iterable[str] | None = None,
        dims: Iterable[str] | None = None,
        directions: Iterable[str] | None = None,
        force: bool = False,
    ) -> list[subprocess.CompletedProcess]:
        """``h2`` for every (trait, kept dim, direction), resumable on the ``.results``
        a prior call already wrote."""
        completed = []
        for trait in traits if traits is not None else self.sumstats:
            sumstats_path = self.sumstats[trait]
            for key, _dim, _direction in self._keys(dims, directions):
                out = self._h2_out(trait, key)
                if not force and Path(f"{out}.results").exists():
                    continue
                completed.append(
                    self.ldsc_run.h2(
                        sumstats_path,
                        self._ref_ld_chr(key),
                        out,
                        frqfile_chr=self.frqfile_chr,
                        print_coefficients=self.print_coefficients,
                    )
                )
        return completed

    def read_results(
        self, *, traits: Iterable[str] | None = None, **kwargs
    ) -> pd.DataFrame:
        """:meth:`~scads_drvi.enrich.run.LdscRun.read_results` for this sweep's
        `annot2dim`/`direction_map`."""
        return self.ldsc_run.read_results(
            self.results_dir,
            traits=list(traits) if traits is not None else list(self.sumstats),
            annot2dim=self.annot2dim(),
            direction=self.direction_map(),
            **kwargs,
        )

    def run(self, *, force: bool = False, **read_kwargs) -> pd.DataFrame:
        """``run_l2()`` then ``run_h2()`` then ``read_results()``."""
        self.run_l2(force=force)
        self.run_h2(force=force)
        return self.read_results(**read_kwargs)

    @classmethod
    def ensure(
        cls,
        *,
        embed,
        bfile_chr: str,
        sumstats: Mapping[str, str | Path],
        annotate: Callable[[str, str, int, pd.DataFrame], object],
        w_ld_chr: str | Path,
        chroms: Sequence[int] = tuple(range(1, 23)),
        ref_ld_chr_extra: Sequence[str | Path] | object = _UNSET,
        sketch: int | None | object = _UNSET,
        frqfile_chr: str | Path | None = None,
        print_snps: str | Path | None = None,
        print_coefficients: bool = True,
        directional: bool = True,
        exclude_vanished: bool = True,
        exclude_dims: Sequence[str] = (),
        ldscore_dir: str | Path | None = None,
        results_dir: str | Path | None = None,
        ld_wind_cm: float = 1.0,
        overlap_annot: bool = True,
        threads: int | None = None,
        version: str = LDSC_VERSION,
        explicit: str | Path | None = None,
        cache: str | Path | None = None,
        allow_download: bool = True,
        check_version: bool = True,
        reference_cache: str | Path | None = None,
        allow_reference_download: bool = True,
        **ldsc_kwargs,
    ) -> EnrichmentSweep:
        """:meth:`~scads_drvi.enrich.run.LdscRun.ensure` then wrap it -- the common case.

        `bfile` is deliberately left unset on the underlying :class:`LdscRun`: the
        per-chromosome `bfile` is always supplied to `l2()` as an override, from
        `bfile_chr`, never a shared default.

        `ref_ld_chr_extra` defaults to the baseline-LF v2.2 UK Biobank reference
        (:func:`~scads_drvi.enrich.reference.ensure_baseline_ukb`), downloaded once and
        cached thereafter (~11 GB -- point ``$SCADS_DRVI_CACHE`` at scratch storage
        first on a cluster with a small home directory, or pass `reference_cache`).
        In-sample UKB LD is preferred here over an external 1000-Genomes reference,
        matching Alkes-group guidance for a UK-Biobank-scale GWAS. Pass
        `ref_ld_chr_extra=()` explicitly to run with no baseline categories at all, or
        your own stem(s) (e.g. a 1000-Genomes-based ``baselineLD_v2.2``) to use a
        different reference instead.

        `sketch` (the `l2` speed/accuracy knob -- see :class:`LdscRun`'s own
        docstring) follows suit: `5000` when `ref_ld_chr_extra` is left at its UKB
        default (matching the individual count `l2` compresses from at UK Biobank
        scale) and `200` -- `LdscRun`'s own default, already well past both the
        binary's `d <= 50` instability floor and its "practical sweet spot" -- when a
        caller supplies their own `ref_ld_chr_extra` (e.g. a 1000-Genomes-based
        baseline, whose reference panel is a couple orders of magnitude smaller).
        Pass `sketch=` explicitly to override either way.
        """
        used_ukb_default = ref_ld_chr_extra is _UNSET
        if ref_ld_chr_extra is _UNSET:
            from scads_drvi.enrich.reference import ensure_baseline_ukb

            ref_ld_chr_extra = (
                ensure_baseline_ukb(
                    cache=reference_cache, allow_download=allow_reference_download
                ),
            )
        if sketch is _UNSET:
            sketch = 5000 if used_ukb_default else 200
        ldsc_run = LdscRun.ensure(
            w_ld_chr=w_ld_chr,
            ld_wind_cm=ld_wind_cm,
            overlap_annot=overlap_annot,
            threads=threads,
            version=version,
            explicit=explicit,
            cache=cache,
            allow_download=allow_download,
            check_version=check_version,
            sketch=sketch,
            **ldsc_kwargs,
        )
        return cls(
            ldsc_run=ldsc_run,
            embed=embed,
            bfile_chr=bfile_chr,
            sumstats=sumstats,
            annotate=annotate,
            chroms=chroms,
            ref_ld_chr_extra=ref_ld_chr_extra,
            frqfile_chr=frqfile_chr,
            print_snps=print_snps,
            print_coefficients=print_coefficients,
            directional=directional,
            exclude_vanished=exclude_vanished,
            exclude_dims=exclude_dims,
            ldscore_dir=ldscore_dir,
            results_dir=results_dir,
        )
