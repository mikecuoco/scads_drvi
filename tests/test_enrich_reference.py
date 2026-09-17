"""enrich.reference: resolving (and, offline in these tests, never actually
downloading) the default baseline-LF v2.2 UKB reference."""

from __future__ import annotations

import pytest

from scads_drvi.enrich.reference import (
    BASELINE_UKB_STEM_NAME,
    BASELINE_UKB_URL,
    ensure_baseline_ukb,
    reference_cache_dir,
)


class TestCacheDir:
    def test_respects_scads_drvi_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SCADS_DRVI_CACHE", str(tmp_path))
        assert reference_cache_dir() == tmp_path / "scads_drvi" / "ldsc_reference"

    def test_falls_back_to_xdg_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SCADS_DRVI_CACHE", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert reference_cache_dir() == tmp_path / "scads_drvi" / "ldsc_reference"


class TestEnsureBaselineUkb:
    def test_returns_cached_stem_without_downloading(self, tmp_path):
        extracted = tmp_path / "baselineLF_v2.2.UKB"
        extracted.mkdir()
        (extracted / f"{BASELINE_UKB_STEM_NAME}22.l2.ldscore.gz").touch()

        stem = ensure_baseline_ukb(cache=tmp_path, allow_download=True)
        assert stem == str(extracted / BASELINE_UKB_STEM_NAME)

    def test_missing_and_download_disabled_names_the_url(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="downloading is disabled") as exc:
            ensure_baseline_ukb(cache=tmp_path, allow_download=False)
        assert BASELINE_UKB_URL in str(exc.value)


@pytest.mark.data
class TestAgainstTheRealReference:
    """Runs only where the (~11 GB) reference is already resolvable offline."""

    def _resolve(self):
        try:
            return ensure_baseline_ukb(allow_download=False)
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"baseline-LF v2.2 UKB reference not available offline: {exc}")

    def test_every_autosome_has_ldscores(self):
        stem = self._resolve()
        from pathlib import Path

        missing = [c for c in range(1, 23) if not Path(f"{stem}{c}.l2.ldscore.gz").exists()]
        assert not missing, f"chromosomes missing ld scores: {missing}"
