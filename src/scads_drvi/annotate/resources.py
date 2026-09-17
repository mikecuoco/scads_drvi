"""Downloading the real cisTarget-style SCREEN score database.

aertslab publish a precomputed regions-vs-motifs score database for the ENCODE SCREEN
region catalogue -- a genuinely public, standard resource (the same one pycisTarget's
own tutorials point at), not anything dataset-specific. Having a resolver for it here
lets a test or a notebook validate :mod:`scads_drvi.annotate.motif` against real
production-scale data without depending on any one researcher's private capsule path.

Mirrors :meth:`scads_drvi.enrich.run.LdscRun._resolve_binary`'s resolve/cache/verify
shape.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import TypedDict

__all__ = [
    "SCREEN_DATABASES",
    "screen_database_cache_dir",
    "ensure_screen_database",
]


class _DatabaseSpec(TypedDict):
    url: str
    size: int
    sha256: str


_BASE_URL = "https://resources.aertslab.org/cistarget/databases"

#: (genome, version) -> where to get it and how to verify it. Only entries listed here
#: can be resolved -- an unlisted genome/version is a decision nobody has vetted a
#: checksum for yet, not something to download blind.
SCREEN_DATABASES: dict[tuple[str, str], _DatabaseSpec] = {
    ("hg38", "v10_clust"): {
        "url": f"{_BASE_URL}/homo_sapiens/hg38/screen/mc_v10_clust/region_based/"
               "hg38_screen_v10_clust.regions_vs_motifs.scores.feather",
        "size": 13_882_267_682,
        "sha256": "9dfaf815b6f9c923f9161a8148cb973838416510b37446cac447245f20bfe574",
    },
}


def screen_database_cache_dir() -> Path:
    """Where a downloaded database is kept.

    ``$SCADS_DRVI_CACHE`` wins, then ``$XDG_CACHE_HOME``, then ``~/.cache`` -- so a
    cluster with a small home directory can point it at scratch. Same resolution order
    as :func:`scads_drvi.enrich.binary.ldsc_cache_dir`.
    """
    root = os.environ.get("SCADS_DRVI_CACHE")
    if root:
        base = Path(root)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "scads_drvi" / "cistarget_screen_db"


def _sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_screen_database(
    genome: str = "hg38",
    version: str = "v10_clust",
    *,
    cache: str | Path | None = None,
    allow_download: bool = False,
    verify: bool = True,
) -> Path:
    """Resolve aertslab's public SCREEN cisTarget score database, downloading if asked.

    THE FILE IS ~14 GB. `allow_download` defaults to **False** on purpose: unlike
    :meth:`scads_drvi.enrich.run.LdscRun._resolve_binary`'s small pinned binary, a
    fresh fetch here takes minutes even on a fast connection and should never happen
    as a surprise side effect of running a test. Pass it explicitly (or point `cache`
    at an existing copy, or set ``$SCADS_DRVI_CACHE``) to opt in.

    `verify` checks the download against the pinned size and SHA-256 in
    :data:`SCREEN_DATABASES`; a mismatch removes the file rather than returning a
    possibly-truncated or corrupted multi-gigabyte database silently. An existing
    cached file is size-checked (not re-hashed -- hashing 14 GB on every call would
    defeat the point of caching it).
    """
    key = (genome, version)
    if key not in SCREEN_DATABASES:
        raise ValueError(
            f"no pinned SCREEN database for genome={genome!r} version={version!r}; "
            f"available: {sorted(SCREEN_DATABASES)}"
        )
    spec = SCREEN_DATABASES[key]
    cache_dir = Path(cache) if cache is not None else screen_database_cache_dir()
    dest = cache_dir / f"{genome}_screen_{version}.regions_vs_motifs.scores.feather"

    if dest.exists():
        if verify and dest.stat().st_size != spec["size"]:
            raise OSError(
                f"{dest} is {dest.stat().st_size:,} bytes, expected {spec['size']:,}. "
                f"Remove it and re-run with allow_download=True to re-fetch."
            )
        return dest

    if not allow_download:
        raise FileNotFoundError(
            f"no cached database at {dest}. It is ~{spec['size'] / 1e9:.1f} GB, so it "
            f"is not fetched automatically -- pass allow_download=True, or fetch it "
            f"yourself from {spec['url']} into {cache_dir}."
        )

    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".partial.{os.getpid()}")
    try:
        with urllib.request.urlopen(spec["url"], timeout=300) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)

        if verify:
            got_size = tmp.stat().st_size
            if got_size != spec["size"]:
                raise OSError(f"downloaded {got_size:,} bytes, expected {spec['size']:,}")
            got_hash = _sha256(tmp)
            if got_hash != spec["sha256"]:
                raise OSError(
                    f"checksum mismatch downloading {spec['url']}: expected "
                    f"{spec['sha256']}, got {got_hash}"
                )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.replace(dest)
    return dest
