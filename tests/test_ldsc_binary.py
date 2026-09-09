"""enrich.binary: resolving the pinned LDSC binary and building safe commands."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scads_drvi.enrich.binary import (
    APPROXIMATE_FLAGS,
    LDSC_ASSETS,
    LDSC_ENV_VAR,
    LDSC_SHA256,
    LDSC_VERSION,
    SUBCOMMANDS,
    asset_for_platform,
    build_command,
    ensure_ldsc,
    ldsc_cache_dir,
    run_ldsc,
)


def fake_binary(path: Path, version: str = "0.5.0") -> Path:
    """A stand-in that answers --version, so resolution can be tested offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "ldsc ' + version + '"; exit 0; fi\n'
        'echo "$@"\n'
    )
    path.chmod(0o755)
    return path


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


class TestResolutionOrder:
    def test_explicit_wins(self, tmp_path):
        binary = fake_binary(tmp_path / "mine" / "ldsc")
        got = ensure_ldsc(explicit=binary, cache=tmp_path / "cache")
        assert got.source == "explicit"
        assert Path(got) == binary

    def test_explicit_missing_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no ldsc binary"):
            ensure_ldsc(explicit=tmp_path / "absent")

    def test_environment_is_next(self, tmp_path, monkeypatch):
        binary = fake_binary(tmp_path / "env" / "ldsc")
        monkeypatch.setenv(LDSC_ENV_VAR, str(binary))
        got = ensure_ldsc(cache=tmp_path / "cache")
        assert got.source == "environment"

    def test_environment_pointing_nowhere_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setenv(LDSC_ENV_VAR, str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match=LDSC_ENV_VAR):
            ensure_ldsc(cache=tmp_path / "cache")

    def test_cache_is_used_before_downloading(self, tmp_path, monkeypatch):
        monkeypatch.delenv(LDSC_ENV_VAR, raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        cache = tmp_path / "cache"
        fake_binary(cache / "ldsc")
        got = ensure_ldsc(cache=cache, allow_download=False)
        assert got.source == "cache"

    def test_no_binary_and_no_download_explains_both_routes(self, tmp_path, monkeypatch):
        monkeypatch.delenv(LDSC_ENV_VAR, raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError, match="cargo install ldsc"):
            ensure_ldsc(cache=tmp_path / "cache", allow_download=False)


class TestVersionChecking:
    def test_a_mismatched_version_is_refused(self, tmp_path):
        """Silently running a different LD-score implementation than the one recorded
        is how a number becomes unreproducible months later."""
        binary = fake_binary(tmp_path / "old" / "ldsc", version="0.1.2")
        with pytest.raises(OSError, match="pins"):
            ensure_ldsc(explicit=binary, cache=tmp_path / "cache")

    def test_the_mismatch_can_be_accepted_deliberately(self, tmp_path):
        binary = fake_binary(tmp_path / "old" / "ldsc", version="0.1.2")
        got = ensure_ldsc(
            explicit=binary, cache=tmp_path / "cache", check_version=False
        )
        assert got.version == "0.1.2"

    def test_an_unparseable_version_is_an_error(self, tmp_path):
        path = tmp_path / "ldsc"
        path.write_text("#!/bin/sh\necho not-a-version\n")
        path.chmod(0o755)
        with pytest.raises(OSError, match="cannot parse a version"):
            ensure_ldsc(explicit=path, cache=tmp_path / "cache")


class TestBuildCommand:
    def test_python_compat_is_added_to_l2(self):
        argv = build_command("l2", {"bfile": "x"})
        assert "--python-compat" in argv

    def test_python_compat_is_not_added_to_h2(self):
        """It exists only on l2; passing it to h2 would be a hard error from the tool."""
        assert SUBCOMMANDS["h2"] is False
        assert "--python-compat" not in build_command("h2", {"h2": "x.sumstats.gz"})

    def test_python_compat_can_be_turned_off(self):
        assert "--python-compat" not in build_command(
            "l2", {"bfile": "x"}, python_compat=False
        )

    def test_underscores_become_dashes(self):
        argv = build_command("h2", {"ref_ld_chr": "p", "w_ld_chr": "q"})
        assert "--ref-ld-chr" in argv and "--w-ld-chr" in argv

    def test_booleans_are_switches_and_none_is_dropped(self):
        argv = build_command("h2", {"overlap_annot": True, "frqfile": None})
        assert "--overlap-annot" in argv
        assert not any(a.startswith("--frqfile") for a in argv)

    def test_false_is_dropped(self):
        assert "--overlap-annot" not in build_command("h2", {"overlap_annot": False})

    def test_lists_are_comma_joined(self):
        argv = build_command("l2", {"chroms": [1, 2, 22]})
        assert argv[argv.index("--chroms") + 1] == "1,2,22"

    @pytest.mark.parametrize("flag", APPROXIMATE_FLAGS)
    def test_approximate_flags_are_refused_by_default(self, flag):
        """--sketch is a random projection; a fast wrong number must not be reachable
        by accident."""
        with pytest.raises(ValueError, match="change the estimate"):
            build_command("l2", {flag: 200})

    def test_approximate_flags_can_be_opted_into(self):
        argv = build_command("l2", {"sketch": 200}, allow_approximate=True)
        assert "--sketch" in argv

    def test_unknown_subcommand_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="expected one of"):
            build_command("nope")


class TestRunLdsc:
    def test_dry_run_echoes_without_executing(self, tmp_path):
        seen: list[str] = []
        proc = run_ldsc(
            "l2", {"bfile": "x"}, binary=tmp_path / "nonexistent",
            dry_run=True, log_fn=seen.append,
        )
        assert proc.returncode == 0
        assert seen and seen[0].startswith("$ ")
        assert "--python-compat" in seen[0]

    def test_threads_cap_both_pools(self, tmp_path):
        seen: list[str] = []
        run_ldsc(
            "h2", {"h2": "x"}, binary=tmp_path / "b", threads=1,
            dry_run=True, log_fn=seen.append,
        )
        assert "--rayon-threads 1" in seen[0]
        assert "--polars-threads 1" in seen[0]

    def test_it_actually_runs_a_binary(self, tmp_path):
        binary = fake_binary(tmp_path / "ldsc")
        proc = run_ldsc("h2", {"h2": "x"}, binary=binary, check=True)
        assert proc.returncode == 0
        assert "h2" in proc.stdout


@pytest.mark.data
class TestAgainstTheRealBinary:
    """Runs only where the pinned binary is already resolvable."""

    def _resolve(self):
        try:
            return ensure_ldsc(allow_download=False)
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"pinned ldsc not available offline: {exc}")

    def test_version_matches_the_pin(self):
        got = self._resolve()
        assert got.version == LDSC_VERSION.lstrip("v")

    def test_l2_advertises_python_compat(self):
        got = self._resolve()
        help_text = subprocess.run(
            [str(got.path), "l2", "--help"], capture_output=True, text=True, timeout=60
        ).stdout
        assert "--python-compat" in help_text

    def test_h2_does_not_advertise_python_compat(self):
        """The asymmetry this module encodes, checked against the tool itself."""
        got = self._resolve()
        help_text = subprocess.run(
            [str(got.path), "h2", "--help"], capture_output=True, text=True, timeout=60
        ).stdout
        assert "--python-compat" not in help_text
        assert "--overlap-annot" in help_text
        assert "--frqfile-chr" in help_text
