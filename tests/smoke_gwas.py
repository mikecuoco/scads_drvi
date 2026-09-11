"""A real, public GWAS, fetched and munged into what `ldsc h2` expects.

`smoke_data.py`'s synthesized S-LDSC `.results` are literally a formula (`z = 4.5 - 0.6 *
slot + ...`), not a regression -- there has never been a real GWAS behind any enrichment
number this suite has produced. This closes that hole with one real, well-powered GWAS:
standing height (Yengo et al. 2018, GWAS Catalog GCST006901), munged by the pinned `ldsc`
binary's own ``munge-sumstats`` subcommand rather than a second, separate tool.

Height, not something more `on brand` for a brain-tissue capsule, because "popular" was
the ask: it is one of the most highly powered, most frequently used example traits in the
S-LDSC literature (the method's own worked examples use it), which is exactly what a
smoke-test regression wants -- a trait `ldsc h2` should find a real, non-degenerate
signal for, not a signal that means anything about this package's actual domain. Nothing
here supports a claim about height biology, only about whether the wrapper runs a real
GWAS correctly.

What munging does and does not filter here: MAF >= 0.01 against the file's own
Freq_Tested_Allele_in_HRS (8,552 unrelated participants -- not the GWAS's own sample, the
file's only frequency column), the classic N-floor (SNPs far below the modal sample size
are dropped), and strand-ambiguous alleles. No ``--merge-alleles`` HapMap3 restriction is
applied: this repo has no LD-score reference panel yet (see the module docstring's
caller -- the h2/l2 step still needs one), so restricting to a SNP list tuned to a
reference we do not have would be pinning to nothing.

Two upstream formatting quirks are fixed before munging, not filtered by it -- neither is
a property of the GWAS, both are artifacts of how the file was written. See
`_clean_fields` for what each one silently did to the SNP count before it was found.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from scads_drvi._util.paths import cache_dir

__all__ = [
    "GwasUnavailable",
    "REMOTE",
    "SOURCE_BASE",
    "TRAIT",
    "PUBMED_ID",
    "LAYOUT_VERSION",
    "smoke_gwas_cache_dir",
    "fetch",
    "ensure_inputs",
    "munge",
    "ensure_sumstats",
]

#: GWAS Catalog GCST006901 -- Height, Yengo et al. 2018 (Hum Mol Genet, PMID 30124842),
#: "Meta-analysis of genome-wide association studies for height and body mass index in
#: ~700,000 individuals of European ancestry". 2.4M SNPs, N up to 693,529.
SOURCE_BASE = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
    "GCST006001-GCST007000/GCST006901"
)
PUBMED_ID = "30124842"

#: Slug for this trait everywhere a caller names one -- factor_map, results dirs, plots.
TRAIT = "height_yengo2018_gcst006901"

#: Bump whenever `munge` changes its flags, so a stale cached `.sumstats.gz` is not
#: silently reused -- it is part of the derived filename, exactly as in smoke_data.py.
LAYOUT_VERSION = 1


class GwasUnavailable(RuntimeError):
    """The GWAS is not cached and could not (or may not) be fetched."""


@dataclass(frozen=True)
class RemoteFile:
    """One pinned published file. Mirrors smoke_data.py's RemoteFile exactly."""

    name: str
    sha256: str
    n_bytes: int

    @property
    def url(self) -> str:
        return f"{SOURCE_BASE}/{self.name}"

    @property
    def pinned(self) -> bool:
        return bool(self.sha256)


#: Published SHA-256 and size. A bump is a reviewed change, not a silent upgrade.
REMOTE: Mapping[str, RemoteFile] = {
    "sumstats": RemoteFile(
        name="Meta-analysis_Wood_et_al+UKBiobank_2018.txt.gz",
        sha256="667d63117d842347bad9e90a80d7e62f6ef50a01609fa4b8cc1c4621c62a792b",
        n_bytes=46_876_053,
    ),
}

#: Column names in the published file. Passed to `ldsc munge-sumstats` explicitly rather
#: than relying on its built-in name map: `Tested_Allele`/`Other_Allele` are this study's
#: names, not the map's `A1`/`A2`, and a silent miss there would drop every SNP as
#: "no alleles found" rather than fail loudly.
_COLUMNS = {
    "snp": "SNP",
    "a1": "Tested_Allele",
    "a2": "Other_Allele",
    "p": "P",
    "N_col": "N",  # capital N: build_command's key->flag transform is case-preserving,
    # and the binary's flag is `--N-col`, not `--n-col`.
    "frq": "Freq_Tested_Allele_in_HRS",
    "signed_sumstats": "BETA,0",
}


def smoke_gwas_cache_dir(cache: str | Path | None = None) -> Path:
    """Where the fetched GWAS and its munged `.sumstats.gz` live."""
    if cache is not None:
        return Path(cache) / "scads_drvi" / "smoke_gwas" / TRAIT
    return cache_dir("smoke_gwas", TRAIT)


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _mismatch(path: Path, item: RemoteFile) -> str | None:
    size = path.stat().st_size
    if size != item.n_bytes:
        return f"expected {item.n_bytes} bytes, got {size}"
    if item.pinned:
        got = _digest(path)
        if got != item.sha256:
            return f"expected sha256 {item.sha256}, got {got}"
    return None


_USER_AGENT = "scads-drvi-tests/0.1 (+https://github.com/mikecuoco/scads_drvi)"


def _download(item: RemoteFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".partial.{os.getpid()}")
    request = urllib.request.Request(item.url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with tmp.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise GwasUnavailable(f"could not fetch {item.url}: {exc}") from exc
    tmp.replace(dest)


def fetch(
    item: RemoteFile,
    *,
    cache: str | Path | None = None,
    allow_download: bool = False,
    verify: bool = True,
) -> Path:
    """The cached path for `item`, fetching it only when allowed. Mirrors smoke_data.py."""
    if verify and not item.pinned:
        raise GwasUnavailable(
            f"{item.name} has no pinned sha256; pass verify=False to accept that"
        )

    dest = smoke_gwas_cache_dir(cache) / item.name
    if dest.exists():
        reason = _mismatch(dest, item) if verify else None
        if reason is None:
            return dest
        if not allow_download:
            raise GwasUnavailable(f"cached {dest} is unusable ({reason}) and downloading is off")
        dest.unlink()

    if not allow_download:
        raise GwasUnavailable(
            f"{item.name} is not cached and downloading is disabled. Set "
            f"SCADS_DRVI_SMOKE_DOWNLOAD=1, or place the file from {item.url} at {dest}"
        )

    _download(item, dest)
    reason = _mismatch(dest, item) if verify else None
    if reason is not None:
        dest.unlink(missing_ok=True)
        raise GwasUnavailable(f"downloaded {item.name} failed verification: {reason}")
    return dest


def ensure_inputs(
    *,
    cache: str | Path | None = None,
    allow_download: bool = False,
    verify: bool = True,
) -> dict[str, Path]:
    """Every published input, cached."""
    return {
        key: fetch(item, cache=cache, allow_download=allow_download, verify=verify)
        for key, item in REMOTE.items()
    }


def _clean_fields(src: Path, dest: Path) -> None:
    """Strip padding whitespace from every field, and rewrite `N` as plain integers.

    Two independent upstream quirks, both invisible in a normal ``zcat | head``:

    Every numeric column is **right-aligned in a fixed-width field**, so a value shorter
    than the column's widest entry carries leading spaces -- ``" 4.8e-01"`` next to
    ``"9.7e-142"`` in the same P column. `ldsc munge-sumstats`'s parser does not trim
    that whitespace, and rather than erroring it treats the value as unparseable and
    silently **drops the row** -- caught here as 1,107 SNPs surviving out of 2,334,001,
    every one of them a value that happened to fill its field exactly (the most extreme
    p-values, coincidentally). A hard parse error is loud; a permissive filter that
    quietly keeps only the coincidentally-unpadded rows is the harder failure to notice,
    because the tool exits 0 and reports a plausible-looking SNP count.

    Separately, one `N` value is written in scientific notation (``7e+05``, apparently a
    spreadsheet's choice for a round number), which the parser's stricter `i64` column
    DOES reject outright -- ``could not parse `7e+05` as dtype `i64``. Every value goes
    through `float` -> `int` rather than special-casing that one string, so a different
    round number failing the same way upstream does not resurface this bug under a
    different repr.
    """
    import gzip

    with gzip.open(src, "rt") as inp, gzip.open(dest, "wt") as out:
        header = inp.readline().rstrip("\n").split("\t")
        out.write("\t".join(header) + "\n")
        n_idx = header.index("N")
        for line in inp:
            fields = [f.strip() for f in line.rstrip("\n").split("\t")]
            fields[n_idx] = str(int(float(fields[n_idx])))
            out.write("\t".join(fields) + "\n")


def munge(sumstats_path: Path, out_prefix: Path, *, binary=None) -> Path:
    """Run the pinned `ldsc munge-sumstats` over the published file.

    Returns the produced `<out_prefix>.sumstats.gz`. Raises `OSError` with the tool's own
    stderr on failure -- a malformed or re-formatted upstream file should fail loudly here,
    not three stages later as "0 SNPs after merge" out of `ldsc h2`.
    """
    import tempfile

    from scads_drvi.enrich.binary import ensure_ldsc, run_ldsc

    resolved = binary if binary is not None else ensure_ldsc()
    out = out_prefix.with_name(out_prefix.name + ".sumstats.gz")

    # The cleaned copy is scratch, not a derived artifact worth keeping: it exists only
    # because the binary's parser cannot read the upstream padding, and every byte of it
    # is already in `sumstats_path`. A temp dir keeps it out of the cache permanently
    # rather than leaving a ~file-sized duplicate beside `out` after every run.
    with tempfile.TemporaryDirectory(prefix="scads_drvi_gwas_clean_") as tmp:
        fixed = Path(tmp) / "cleaned.txt.gz"
        _clean_fields(sumstats_path, fixed)

        options = {
            "sumstats": str(fixed),
            "out": str(out_prefix),
            **{k: v for k, v in _COLUMNS.items() if k != "signed_sumstats"},
            "signed-sumstats": _COLUMNS["signed_sumstats"],
        }
        proc = run_ldsc("munge-sumstats", options, binary=resolved, check=False)
        if proc.returncode != 0 or not out.exists():
            raise OSError(
                f"munge-sumstats failed (exit {proc.returncode}):\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
    return out


def ensure_sumstats(
    *,
    cache: str | Path | None = None,
    allow_download: bool = False,
    verify: bool = True,
) -> tuple[Path, dict]:
    """The munged `.sumstats.gz` and a small manifest, building it once and caching both.

    Mirrors `smoke_data.ensure_matrix`: a `LAYOUT_VERSION`-tagged output and a manifest
    checked before reuse, so a `munge` flag change does not silently serve a stale file.
    """
    root = smoke_gwas_cache_dir(cache)
    prefix = root / f"height.v{LAYOUT_VERSION}"
    out = prefix.with_name(prefix.name + ".sumstats.gz")
    manifest_path = root / f"manifest.v{LAYOUT_VERSION}.json"
    if out.exists() and manifest_path.exists():
        return out, json.loads(manifest_path.read_text())

    inputs = ensure_inputs(cache=cache, allow_download=allow_download, verify=verify)
    produced = munge(inputs["sumstats"], prefix)
    assert produced == out, (produced, out)

    n_lines = sum(1 for _ in _open_maybe_gz(out)) - 1  # header
    manifest = {
        "trait": TRAIT,
        "pubmed_id": PUBMED_ID,
        "source_base": SOURCE_BASE,
        "layout_version": LAYOUT_VERSION,
        "n_snps": n_lines,
        "columns": _COLUMNS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out, manifest


def _open_maybe_gz(path: Path):
    import gzip

    with gzip.open(path, "rt") as handle:
        yield from handle


if __name__ == "__main__":  # pragma: no cover -- the tool that pins the constants above
    print(f"fetching from {SOURCE_BASE}")
    paths = ensure_inputs(allow_download=True, verify=False)
    for name, path in paths.items():
        print(f"  {name}: {REMOTE[name].name}")
        print(f"    n_bytes={path.stat().st_size}")
        print(f"    sha256={_digest(path)}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        out = munge(paths["sumstats"], Path(tmpdir) / "height")
        n_lines = sum(1 for _ in _open_maybe_gz(out)) - 1
        print(f"  munged: {out.name}, {n_lines} SNPs")
