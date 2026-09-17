"""The default S-LDSC baseline reference: baseline-LF v2.2 on UK Biobank in-sample LD.

In-sample LD (estimated from the same cohort a GWAS was run in) is preferred over an
external reference panel like 1000 Genomes when the target sumstats come from a
UK-Biobank-scale cohort -- which is the common case this package's sweep targets, so
this is the reference :func:`~scads_drvi.enrich.sweep.EnrichmentSweep.ensure` reaches
for unless a caller says otherwise.

Published by the Alkes group (Price lab) at
``https://data.broadinstitute.org/alkesgroup/LDSCORE/baselineLF_v2.2.UKB.tar.gz``,
mirrored here from its S3 bucket. Same shape and role as the 1000-Genomes-based
``baselineLD_v2.2`` reference: one ``{stem}{chrom}.annot.gz`` / ``.l2.ldscore.gz`` /
``.l2.M`` / ``.l2.M_5_50`` per chromosome, meant as extra ``--ref-ld-chr`` baseline
categories layered under a caller's own factor annotation -- **not** a substitute for
a raw genotype ``bfile``. UK Biobank individual-level genotypes are controlled-access
and are not, and cannot be, shipped here; a caller's own ``bfile_chr`` (1000 Genomes or
their own approved-access data) is unaffected by this module.

The exact per-chromosome file stem (``baselineLF2.2.UKB.`` -- note it drops the ``_v``
the extracted directory name carries) was confirmed against a real partial download of
the archive, not guessed; the ``.l2.ldscore.gz`` suffix follows the Alkes group's own
standard naming for every baseline-family release. If a future re-upload changes that
layout, :func:`ensure_baseline_ukb` fails loudly (a missing/misnamed file is refused by
`h2` immediately, not a silently wrong number), naming exactly what it expected.

No checksum is pinned here the way :mod:`scads_drvi.enrich.binary` pins the ldsc binary's:
this is public reference data rather than a compiled binary being executed, so a
successful extraction plus the expected per-chromosome files existing is the
integrity check.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

__all__ = [
    "BASELINE_UKB_URL",
    "BASELINE_UKB_STEM_NAME",
    "reference_cache_dir",
    "ensure_baseline_ukb",
]

BASELINE_UKB_URL = (
    "https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/baselineLF_v2.2.UKB.tar.gz"
)

#: The directory name the tarball extracts to.
_ARCHIVE_DIR = "baselineLF_v2.2.UKB"

#: The per-chromosome file prefix inside it -- see the module docstring for why this
#: differs from `_ARCHIVE_DIR`.
BASELINE_UKB_STEM_NAME = "baselineLF2.2.UKB."


def reference_cache_dir() -> Path:
    """Where the default reference dataset is cached.

    Builds on :func:`~scads_drvi.enrich.binary.cache_root`, the same env-var
    resolution (``$SCADS_DRVI_CACHE`` wins, then ``$XDG_CACHE_HOME``, then
    ``~/.cache``) the ldsc binary's own cache uses. Point ``$SCADS_DRVI_CACHE`` at
    scratch storage before first use here -- this reference is gigabytes, not the
    few megabytes the ldsc binary itself needs.
    """
    from scads_drvi.enrich.binary import cache_root

    return cache_root() / "scads_drvi" / "ldsc_reference"


def ensure_baseline_ukb(
    *, cache: str | Path | None = None, allow_download: bool = True
) -> str:
    """Resolve the baseline-LF v2.2 UKB reference, downloading (~11 GB) if needed.

    Returns the ``--ref-ld-chr`` stem, e.g.
    ``".../baselineLF_v2.2.UKB/baselineLF2.2.UKB."`` -- a trailing-dot prefix `h2`
    expands across chromosomes itself, the same convention every other `ref_ld_chr`
    value in this package already uses.
    """
    cache_dir = Path(cache) if cache is not None else reference_cache_dir()
    extracted = cache_dir / _ARCHIVE_DIR
    stem = str(extracted / BASELINE_UKB_STEM_NAME)
    marker = Path(f"{stem}22.l2.ldscore.gz")
    if marker.exists():
        return stem

    if not allow_download:
        raise FileNotFoundError(
            f"no baseline-LF v2.2 UKB reference at {extracted} and downloading is "
            f"disabled. Fetch {BASELINE_UKB_URL} (~11 GB) and extract it there, or "
            f"pass allow_download=True."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "baselineLF_v2.2.UKB.tar.gz"
    if not archive.exists():
        from scads_drvi.enrich.binary import download_file

        download_file(BASELINE_UKB_URL, archive, timeout=3600)

    with tarfile.open(archive) as tf:
        tf.extractall(cache_dir)  # noqa: S202 -- our own pinned URL, not user input

    if not marker.exists():
        raise OSError(
            f"extracted {archive} but {marker} is still missing -- the archive's "
            f"internal layout may have changed from what this function expects "
            f"({BASELINE_UKB_STEM_NAME}{{chrom}}.l2.ldscore.gz per chromosome)."
        )
    return stem
