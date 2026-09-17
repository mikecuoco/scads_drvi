"""enrich.binary: resolving the pinned LDSC binary (finding/downloading/verifying it).

Building and running an ldsc command is LdscRun's job now (see test_ldsc_run.py) --
this file only covers the parts that don't need a run's config at all: platform/asset
pinning and the cache directory.
"""

from __future__ import annotations

import pytest

from scads_drvi.enrich.binary import (
    LDSC_ASSETS,
    LDSC_SHA256,
    LDSC_VERSION,
    asset_for_platform,
    ldsc_cache_dir,
)


class TestPinning:
    def test_every_asset_has_a_checksum(self):
        for asset in set(LDSC_ASSETS.values()):
            assert asset in LDSC_SHA256, f"{asset} is unpinned"

    def test_checksums_look_like_sha256(self):
        for asset, digest in LDSC_SHA256.items():
            assert len(digest) == 64, asset
            assert set(digest) <= set("0123456789abcdef"), asset

    def test_version_is_tagged(self):
        assert LDSC_VERSION.startswith("v")

    def test_this_platform_is_covered(self):
        assert asset_for_platform("Linux", "x86_64") == "ldsc_linux-x86_64.tar.gz"
        assert asset_for_platform("Darwin", "arm64") == "ldsc_macos-aarch64.tar.gz"

    def test_x64_aliases_are_normalised(self):
        assert asset_for_platform("Linux", "x86-64") == "ldsc_linux-x86_64.tar.gz"

    def test_an_unknown_platform_names_the_build_command(self):
        with pytest.raises(OSError, match="cargo install ldsc"):
            asset_for_platform("Plan9", "vax")


class TestCacheDir:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCADS_DRVI_CACHE", str(tmp_path))
        assert ldsc_cache_dir().is_relative_to(tmp_path)

    def test_xdg_is_respected(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SCADS_DRVI_CACHE", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert ldsc_cache_dir().is_relative_to(tmp_path)

    def test_version_is_part_of_the_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCADS_DRVI_CACHE", str(tmp_path))
        assert LDSC_VERSION in str(ldsc_cache_dir())
