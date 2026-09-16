"""annotate.motif, end to end on real PBMC ATAC peak coordinates.

`tests/test_motif.py` proves the statistics on toy arrays; this proves the full
pipeline -- overlap mapping, regionising, stratification, streaming, NES -- runs
correctly on a real genome's real region names, using the same public 10x PBMC
multiome sample `docs/tutorials/pbmc.ipynb` builds the whole package's tutorial on.

No real motif-score database or DRVI fit is required: the score database is a small
synthetic Feather file built over TRANSLATED copies of the real peak coordinates (so
`build_region_map`'s overlap logic is genuinely exercised, not a trivial identity
join), and the "factor" is peak width -- a real, non-random per-peak quantity standing
in for a DRVI factor's loadings.

``@pytest.mark.data``: skips itself (not the whole run) when the dataset isn't
reachable, the same pattern ``test_ldsc_binary.py`` uses for the real LDSC binary.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyranges")

from scads_drvi.annotate.motif import (
    build_region_map,
    covariate_strata,
    regionise,
    weighted_motif_enrichment,
)
from scads_drvi.enrich.config import parse_peaks

PBMC_URL = (
    "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/"
    "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
)
#: Canonical chr:start-end peaks on primary chromosomes only -- same filter
#: docs/tutorials/pbmc.ipynb applies.
PEAK_RE = re.compile(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y):\d+-\d+$")


def _cache_path() -> Path:
    root = Path(
        os.environ.get("SCADS_DRVI_TEST_CACHE", Path.home() / ".cache" / "scads_drvi_test_data")
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / "pbmc_multiome_filtered_feature_bc_matrix.h5"


@pytest.fixture(scope="module")
def real_pbmc_peaks() -> list[str]:
    """~2,000 real ``chr:start-end`` ATAC peak names from the tutorial's own 10x sample.

    Cached under ``$SCADS_DRVI_TEST_CACHE`` (default ``~/.cache/scads_drvi_test_data``)
    so repeat runs don't re-fetch the 192 MB file.
    """
    sc = pytest.importorskip("scanpy")
    path = _cache_path()
    if not path.exists():
        # Cloudflare 403s the default urllib User-Agent; a browser-like one is enough.
        request = urllib.request.Request(PBMC_URL, headers={"User-Agent": "Mozilla/5.0"})
        tmp = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(request, timeout=60) as resp, tmp.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        except (OSError, urllib.error.URLError) as exc:
            tmp.unlink(missing_ok=True)
            pytest.skip(f"PBMC dataset not reachable: {exc}")
        tmp.rename(path)

    raw = sc.read_10x_h5(path, gex_only=False)
    raw.var_names_make_unique()
    is_peak = raw.var["feature_types"] == "Peaks"
    peaks = [p for p in raw.var_names[is_peak] if PEAK_RE.match(p)]
    if len(peaks) < 100:
        pytest.skip(f"only {len(peaks)} real ATAC peaks found; expected thousands")

    rng = np.random.default_rng(0)
    if len(peaks) > 2000:
        peaks = list(rng.choice(peaks, size=2000, replace=False))
    return sorted(peaks)


def _write_synthetic_score_db(path: Path, region_names: list[str], motif_ids: list[str],
                              rng: np.random.Generator) -> None:
    """A small region x motif score Feather file, shaped like the real cisTarget-style
    databases `stream_accumulate` reads: N region columns + a trailing `motifs` column."""
    import pyarrow as pa
    import pyarrow.feather as pf

    R, M = len(region_names), len(motif_ids)
    scores = rng.lognormal(0, 1, size=(M, R)).astype("float32")
    scores[rng.random((M, R)) < 0.5] = 0.0  # zero-inflated, like the real thing
    columns = {name: scores[:, j] for j, name in enumerate(region_names)}
    columns["motifs"] = motif_ids
    pf.write_feather(pa.table(columns), path)


@pytest.mark.data
def test_weighted_motif_enrichment_on_real_pbmc_peak_coordinates(tmp_path, real_pbmc_peaks):
    peaks = real_pbmc_peaks
    rng = np.random.default_rng(1)

    # The score "database" lives on its OWN region set, distinct from `peaks` -- each
    # real peak translated by a random +/-49bp jitter (same width, shifted), so
    # build_region_map must do real overlap arithmetic rather than a trivial identity
    # join, while every region stays comfortably real (real chromosome, real width).
    coords = parse_peaks(peaks)
    jitter = rng.integers(-49, 50, size=len(peaks))
    starts = np.maximum(0, coords["start"].to_numpy() + jitter)
    widths = (coords["end"] - coords["start"]).to_numpy()
    db_regions = [
        f"{c}:{s}-{s + w}"
        for c, s, w in zip(coords["chrom"], starts, widths, strict=True)
    ]

    motif_ids = [f"motif_{i}" for i in range(30)]
    db_path = tmp_path / "scores.feather"
    _write_synthetic_score_db(db_path, db_regions, motif_ids, rng)

    # A real, non-random per-peak weight standing in for a DRVI factor's loadings --
    # peak width -- as one "pseudo-factor". The mechanics under test don't care where
    # the weights came from; only that they are a real, non-negative (K, n_peaks) array.
    W = widths.astype(np.float64)[None, :]
    labels = ["width_pseudo_factor"]

    used_cols, peak_rows, region_names = build_region_map(peaks, db_regions, fraction_overlap=0.1)
    Wr = regionise(W, peak_rows)

    region_coords = parse_peaks(region_names)
    region_width = (region_coords["end"] - region_coords["start"]).to_numpy(dtype=np.float64)
    keep, strata_ix, n_strata = covariate_strata([region_width], n_bins=4)
    used_cols, Wr = used_cols[keep], Wr[:, keep]
    region_names = [r for r, k in zip(region_names, keep, strict=True) if k]

    rows, meta = weighted_motif_enrichment(
        Wr, labels, used_cols, region_names, strata_ix, n_strata, db_path,
        check_invariance=True,
    )

    assert meta["n_factors"] == 1
    assert meta["n_motifs"] == len(motif_ids)
    assert len(rows) == len(motif_ids)
    assert set(rows["motif_id"]) == set(motif_ids)
    assert np.isfinite(rows["nes"]).all()
    assert np.isfinite(rows["weighted_score"]).all()
