"""Peak-name parsing and canonicalisation.

A peak is an interval on a sequence. Three spellings are accepted on read, because
upstream tools disagree, and exactly one is written::

    chr1:100-200      chr1_100_200      chr1-100-200      ->  chr1:100-200

Coordinates are 0-based half-open (BED), matching every writer in the pipeline. Nothing
here knows what the intervals mean.
"""

from __future__ import annotations

import re

__all__ = ["PEAK_RE", "parse_peak", "normalize_peak_name", "format_peak"]

#: The canonical written form, for validating a whole index at once.
PEAK_RE = re.compile(r"^chr[\w.]+:\d+-\d+$")

# Ordered by how often each spelling turns up. The third is anchored on a "chr" prefix
# because "a-1-2" is ambiguous without it -- a sequence name may itself contain a dash.
#
# The dash form excludes ":" from the sequence name. Without that, "chr1:-5-9" matched
# it with seq="chr1:", and normalisation produced "chr1::5-9" -- a name that PEAK_RE
# then rejects, i.e. the canonicaliser emitting something it considers invalid. Carried
# over from code/common/config.py, fixed here.
_PEAK_PATTERNS = (
    re.compile(r"^(?P<seq>[^:]+):(?P<start>\d+)-(?P<end>\d+)$"),      # chr1:100-200
    re.compile(r"^(?P<seq>[^_]+)_(?P<start>\d+)_(?P<end>\d+)$"),      # chr1_100_200
    re.compile(r"^(?P<seq>chr[^-:]+)-(?P<start>\d+)-(?P<end>\d+)$"),  # chr1-100-200
)


def parse_peak(name: object) -> tuple[str, int, int] | None:
    """Return ``(seqname, start, end)`` for a peak-like string, else ``None``.

    ``None`` rather than an exception: callers routinely run this over an index that
    legitimately mixes peaks with other row labels, and want to partition rather than
    abort.
    """
    text = str(name).strip()
    for pattern in _PEAK_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group("seq"), int(match.group("start")), int(match.group("end"))
    return None


def format_peak(seq: str, start: int, end: int) -> str:
    """Build the canonical name, adding the ``chr`` prefix when absent."""
    if not seq.startswith("chr"):
        seq = f"chr{seq}"
    return f"{seq}:{start}-{end}"


def normalize_peak_name(name: object) -> str | None:
    """Canonicalise any accepted spelling to ``chr<seq>:<start>-<end>``.

    Returns ``None`` when `name` is not peak-like, for the same reason
    :func:`parse_peak` does.
    """
    parsed = parse_peak(name)
    if parsed is None:
        return None
    return format_peak(*parsed)
