"""The progress logger the pipeline's scripts share.

**Two streams, and the split is load-bearing.** :func:`log` writes to stderr;
:func:`log_out` writes to stdout. The submit scripts route ``--output`` and ``--error``
to separate files, so moving a progress line between the two streams silently
reorganises every job log. Both are kept, and each call site keeps the stream it had
before this module absorbed them.

(Historically these were two modules: ``code/common/progress.py`` on stderr and
``code/common/config.py::log`` on stdout. Unifying the streams is a behavioural change
and is deliberately not made here.)
"""

from __future__ import annotations

import sys

__all__ = ["log", "log_out"]


def log(msg: str) -> None:
    """Write a progress marker to **stderr**."""
    print(f">> {msg}", file=sys.stderr, flush=True)


def log_out(msg: str) -> None:
    """Write a progress marker to **stdout**."""
    print(f">> {msg}", flush=True)
