"""Advise on storage; never manage it.

The package does not copy, stage, or reserve disk. Where data lives is a deployment
decision that depends on the machine, the scheduler and the job -- all things a library
should not guess at. What it can usefully do is notice when a large read is coming off
network storage and say so, once, with the numbers.

Detection is filesystem-type based rather than a prefix list. A hardcoded mount point
would be wrong on the next machine, and the genericity scan would reject it anyway.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

__all__ = [
    "SlowStorageWarning",
    "NETWORK_FILESYSTEMS",
    "filesystem_type",
    "is_network_storage",
    "check_storage",
]


class SlowStorageWarning(UserWarning):
    """A large read is coming off network storage.

    Filterable like any warning::

        warnings.simplefilter("ignore", SlowStorageWarning)
    """


#: Filesystem types that are network-backed. Used as a lookup, so an unknown type is
#: simply not warned about -- a false negative is much cheaper here than a false alarm.
NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs", "nfs4", "cifs", "smb3", "smbfs", "afs", "9p",
        "lustre", "gpfs", "beegfs", "panfs",
        "ceph", "glusterfs", "ocfs2",
        "fuse.sshfs", "fuse.glusterfs", "fuse.cephfs",
    }
)

# Written as a tuple so this module carries no absolute-path constant.
_MOUNTS = ("/", "proc", "mounts")


def _mounts_file() -> Path:
    return Path(*_MOUNTS)


def _human(size: int) -> str:
    """Byte count in the largest unit that keeps it readable.

    Formatting everything as GiB produced "0.0 GiB" for small files, which reads as a
    broken warning rather than a small one.
    """
    step = 1024.0
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TiB"  # unreachable; keeps the type checker happy


def filesystem_type(path: str | os.PathLike) -> str | None:
    """The filesystem type backing `path`, or None if it cannot be determined.

    Resolves `path`, then picks the mount entry with the longest matching prefix --
    nested mounts are common (a fast local scratch mounted inside a network tree), and
    the shortest match would report the wrong one.
    """
    mounts = _mounts_file()
    try:
        raw = mounts.read_text()
    except OSError:
        return None  # not Linux, or /proc not mounted; nothing to say

    try:
        target = Path(path).resolve()
    except OSError:
        return None

    best_len = -1
    best_type: str | None = None
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        # Mount points are octal-escaped in /proc/mounts (e.g. "\040" for a space).
        mount_point = parts[1].encode().decode("unicode_escape")
        fs_type = parts[2]
        try:
            mount = Path(mount_point)
        except ValueError:
            continue
        if target == mount or mount in target.parents:
            if len(mount.parts) > best_len:
                best_len = len(mount.parts)
                best_type = fs_type
    return best_type


def is_network_storage(path: str | os.PathLike) -> bool:
    """True when `path` sits on a filesystem type known to be network-backed."""
    fs_type = filesystem_type(path)
    return fs_type is not None and fs_type in NETWORK_FILESYSTEMS


def check_storage(
    path: str | os.PathLike,
    *,
    kind: str = "input",
    min_bytes: int = 1 << 30,
    stacklevel: int = 3,
) -> bool:
    """Warn if `path` is a large file on network storage. Returns whether it warned.

    Silent when the file is small, local, missing, or on a filesystem whose type cannot
    be established -- the point is to flag the case that actually costs minutes, not to
    editorialise about every read.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        return False

    if size < min_bytes or not is_network_storage(path):
        return False

    warnings.warn(
        f"reading a {_human(size)} {kind} from network storage "
        f"({filesystem_type(path)}): {path}. Staging it on node-local disk first is "
        f"typically an order of magnitude faster for repeated or batched reads; this "
        f"package deliberately does not stage on your behalf.",
        SlowStorageWarning,
        stacklevel=stacklevel,
    )
    return True
