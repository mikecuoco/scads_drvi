"""Where cached downloads are kept.

One resolution chain, shared. It was written for the pinned LDSC binary, but nothing
about it is specific to a binary -- anything this package fetches and keeps belongs
under the same root, so a cluster with a small home directory redirects all of it with
one environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["CACHE_ENV_VAR", "cache_dir"]

#: Environment variable naming the cache root, ahead of the XDG default.
CACHE_ENV_VAR = "SCADS_DRVI_CACHE"


def cache_dir(*parts: str) -> Path:
    """The cache directory for `parts`, created by nobody -- callers decide that.

    ``$SCADS_DRVI_CACHE`` wins, then ``$XDG_CACHE_HOME``, then ``~/.cache`` -- so a
    cluster with a small home directory can point it at scratch.
    """
    root = os.environ.get(CACHE_ENV_VAR)
    if root:
        base = Path(root)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
    return base.joinpath("scads_drvi", *parts)
