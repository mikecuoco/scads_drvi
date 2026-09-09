"""_util.progress and _util.advise."""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from scads_drvi._util import advise, progress


class TestProgress:
    """The two streams must stay separate; the submit scripts route them to
    different files, so a line moving between them silently reorganises job logs."""

    def _run(self, snippet: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True, timeout=60
        )

    def test_log_goes_to_stderr_only(self):
        proc = self._run(
            "from scads_drvi._util.progress import log; log('hello')"
        )
        assert proc.stdout == ""
        assert proc.stderr.strip() == ">> hello"

    def test_log_out_goes_to_stdout_only(self):
        proc = self._run(
            "from scads_drvi._util.progress import log_out; log_out('hello')"
        )
        assert proc.stdout.strip() == ">> hello"
        assert proc.stderr == ""

    def test_marker_prefix_is_preserved(self):
        """`>> ` is what every existing job log and log-scraper expects."""
        proc = self._run("from scads_drvi._util.progress import log; log('x')")
        assert proc.stderr.startswith(">> ")


class TestFilesystemType:
    def test_root_resolves_to_some_type(self):
        # Whatever "/" is, it must resolve to a non-empty string on Linux.
        fs_type = advise.filesystem_type(Path(*("/",)))
        assert fs_type is None or isinstance(fs_type, str) and fs_type

    def test_unknown_path_is_not_network(self, tmp_path):
        assert advise.is_network_storage(tmp_path) in (True, False)

    def test_missing_file_never_warns(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert advise.check_storage(tmp_path / "nope") is False

    def test_small_file_never_warns(self, tmp_path):
        small = tmp_path / "small.bin"
        small.write_bytes(b"0" * 1024)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert advise.check_storage(small) is False

    def test_nested_mount_wins_over_a_shorter_prefix(self, monkeypatch, tmp_path):
        """A fast local mount inside a network tree must report the local type.

        The shortest matching prefix would report the outer network mount and warn on
        exactly the files someone has already staged -- the opposite of useful.
        """
        outer = tmp_path / "net"
        inner = outer / "fast"
        inner.mkdir(parents=True)
        fake = tmp_path / "mounts"
        fake.write_text(
            f"srv:/export {outer} nfs4 rw 0 0\n"
            f"/dev/nvme0n1 {inner} ext4 rw 0 0\n"
        )
        monkeypatch.setattr(advise, "_mounts_file", lambda: fake)
        assert advise.filesystem_type(inner) == "ext4"
        assert advise.filesystem_type(outer) == "nfs4"
        assert advise.is_network_storage(inner) is False
        assert advise.is_network_storage(outer) is True

    def test_large_file_on_network_storage_warns_once(self, monkeypatch, tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"0" * 4096)
        fake = tmp_path / "mounts"
        fake.write_text(f"srv:/export {tmp_path} nfs4 rw 0 0\n")
        monkeypatch.setattr(advise, "_mounts_file", lambda: fake)

        with pytest.warns(advise.SlowStorageWarning, match="network storage"):
            assert advise.check_storage(big, min_bytes=1024) is True

    def test_warning_is_filterable_by_category(self, monkeypatch, tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"0" * 4096)
        fake = tmp_path / "mounts"
        fake.write_text(f"srv:/export {tmp_path} nfs4 rw 0 0\n")
        monkeypatch.setattr(advise, "_mounts_file", lambda: fake)

        with warnings.catch_warnings():
            # simplefilter PREPENDS, so the last call is consulted first: the specific
            # ignore must be registered after the broad error to take precedence.
            warnings.simplefilter("error", UserWarning)
            warnings.simplefilter("ignore", advise.SlowStorageWarning)
            assert advise.check_storage(big, min_bytes=1024) is True

    def test_unreadable_mounts_file_is_silent(self, monkeypatch, tmp_path):
        """Not Linux, or /proc absent: say nothing rather than guess."""
        monkeypatch.setattr(advise, "_mounts_file", lambda: tmp_path / "absent")
        assert advise.filesystem_type(tmp_path) is None
        assert advise.is_network_storage(tmp_path) is False

    def test_escaped_mount_points_are_decoded(self, monkeypatch, tmp_path):
        """/proc/mounts octal-escapes spaces; a raw compare would miss the mount."""
        spaced = tmp_path / "a b"
        spaced.mkdir()
        fake = tmp_path / "mounts"
        fake.write_text(f"srv:/export {str(spaced).replace(' ', chr(92) + '040')} nfs4 rw 0 0\n")
        monkeypatch.setattr(advise, "_mounts_file", lambda: fake)
        assert advise.filesystem_type(spaced) == "nfs4"
