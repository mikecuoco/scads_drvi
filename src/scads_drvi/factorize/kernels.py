#!/usr/bin/env python3
"""Pure selection/overlap kernels for arm 04. stdlib + numpy only, which is what lets
both notebook kernels and the pytest env import the identical code. 26 tests in
code/tests/test_drvi_features.py.
"""


# The reason this used to be a duplicated four-line function is that the module had to
# import with nothing but numpy available. _util.progress is stdlib-only, so that
# constraint is satisfied by importing it and the copy can go. Stream is unchanged:
# log_out is the stdout logger this module always used.
from scads_drvi._util.progress import log_out as log  # noqa: E402



def parse_peak_names(names):
    """['chr1:100-601', ...] -> (chrom array, start array, end array)."""
    import numpy as np
    chrom = np.empty(len(names), dtype=object)
    start = np.empty(len(names), dtype=np.int64)
    end = np.empty(len(names), dtype=np.int64)
    for i, nm in enumerate(names):
        c, _, rng = nm.partition(":")
        s, _, e = rng.partition("-")
        chrom[i], start[i], end[i] = c, int(s), int(e)
    return chrom, start, end


def build_chrom_index(chrom, start):
    """chrom -> (row indices sorted by start, their starts). Consensus peaks are
    non-overlapping, so containment is a single searchsorted per query."""
    import numpy as np
    idx = {}
    order = np.argsort(chrom.astype(str), kind="stable")
    for c in np.unique(chrom.astype(str)):
        rows = np.flatnonzero(chrom.astype(str) == c)
        rows = rows[np.argsort(start[rows], kind="stable")]
        idx[c] = (rows, start[rows])
    del order
    return idx


def map_summits(chrom_idx, cons_start, cons_end, q_chrom, q_summit):
    """Return the consensus row containing each query summit, or -1.

    Kept for the self-check and as a diagnostic; the selection ranks on
    `scatter_max_overlap`, which is not summit-limited (see there).
    """
    import numpy as np
    out = np.full(len(q_summit), -1, dtype=np.int64)
    q_chrom_s = q_chrom.astype(str)
    for c in np.unique(q_chrom_s):
        sel = np.flatnonzero(q_chrom_s == c)
        if c not in chrom_idx:
            continue
        rows, starts = chrom_idx[c]
        pos = np.searchsorted(starts, q_summit[sel], side="right") - 1
        ok = pos >= 0
        cand = np.where(ok, rows[np.clip(pos, 0, len(rows) - 1)], -1)
        inside = ok & (q_summit[sel] >= cons_start[cand]) & (q_summit[sel] < cons_end[cand])
        out[sel[inside]] = cand[inside]
    return out


def scatter_max_overlap(chrom_idx, cons_start, cons_end, q_chrom, q_start, q_end,
                        q_val, out_col):
    """Map called peaks to consensus peaks by interval OVERLAP, not summit
        containment. merge_peaks recenters on the WINNING subclass's summit, so a non-winning
        summit often falls outside the retained window -- and how often depends on the width
        (14.3% unmapped at 501 bp vs 6.9% at 1001 bp), which would bias the two widths
        differently. Overlap gives 0.000% at both. Strongest call wins, not last write.
    """
    import numpy as np
    q_chrom_s = q_chrom.astype(str)
    for c in np.unique(q_chrom_s):
        sel = np.flatnonzero(q_chrom_s == c)
        if c not in chrom_idx:
            continue
        rows, starts = chrom_idx[c]
        qs, qe, qv = q_start[sel], q_end[sel], q_val[sel]
        lo = np.maximum(np.searchsorted(starts, qs, side="right") - 1, 0)
        hi = np.searchsorted(starts, qe, side="left") - 1
        span = hi - lo
        max_span = int(span.max()) if len(span) else -1
        for k in range(max(max_span, 0) + 1):
            idx = lo + k
            live = (k <= span) & (idx >= 0) & (idx < len(rows))
            if not live.any():
                continue
            r = rows[np.clip(idx, 0, len(rows) - 1)]
            hit = live & (cons_end[r] > qs) & (cons_start[r] < qe)
            if hit.any():
                np.maximum.at(out_col, r[hit], qv[hit])



def select(mask, n_sub, signal, n_features, max_shared, min_anchor, anchor_frac,
           groups, logfn=log):
    """mask: (n_peaks, n_sub) bool. signal: (n_peaks, n_sub) float, 0 where not called.

    Returns (selected row indices, source label per selected row).
    """
    import numpy as np
    n_anchor_target = int(round(n_features * anchor_frac))
    n_spec_target = n_features - n_anchor_target

    specific = n_sub <= max_shared
    anchor_pool = n_sub >= min_anchor
    logfn(f"  specific pool (n_groups <= {max_shared}): {specific.sum():,}")
    logfn(f"  anchor pool  (n_groups >= {min_anchor}): {anchor_pool.sum():,}")

    # -- per-group balanced draw from the specific pool ---------------------
    # Grow M until the union reaches the target; unions overlap because a peak
    # called in 2-3 groups can be picked by more than one.
    per_sub_rank = []
    for j in range(len(groups)):
        cand = np.flatnonzero(mask[:, j] & specific)
        cand = cand[np.argsort(-signal[cand, j], kind="stable")]
        per_sub_rank.append(cand)

    lo, hi = 1, max((len(c) for c in per_sub_rank), default=1)
    chosen = set()
    while lo < hi:
        mid = (lo + hi) // 2
        u = set()
        for cand in per_sub_rank:
            u.update(cand[:mid].tolist())
        if len(u) < n_spec_target:
            lo = mid + 1
        else:
            hi = mid
    m = lo
    for cand in per_sub_rank:
        chosen.update(cand[:m].tolist())
    logfn(f"  top-{m} per group -> {len(chosen):,} specific peaks")

    # Trim deterministically to target: drop the weakest by best-signal rank.
    spec_rows = np.array(sorted(chosen), dtype=np.int64)
    if len(spec_rows) > n_spec_target:
        best = signal[spec_rows].max(axis=1)
        keep = np.argsort(-best, kind="stable")[:n_spec_target]
        spec_rows = np.sort(spec_rows[keep])
        logfn(f"  trimmed to {len(spec_rows):,}")

    # -- anchor draw -----------------------------------------------------------
    anchor_rows = np.flatnonzero(anchor_pool)
    anchor_rows = np.setdiff1d(anchor_rows, spec_rows, assume_unique=False)
    if len(anchor_rows) > n_anchor_target:
        with np.errstate(invalid="ignore"):
            msig = np.where(mask[anchor_rows], signal[anchor_rows], np.nan)
            mean_sig = np.nanmean(msig, axis=1)
        keep = np.argsort(-mean_sig, kind="stable")[:n_anchor_target]
        anchor_rows = np.sort(anchor_rows[keep])
    logfn(f"  anchor -> {len(anchor_rows):,} peaks")

    rows = np.concatenate([spec_rows, anchor_rows])
    src = np.array(["specific"] * len(spec_rows) + ["anchor"] * len(anchor_rows), dtype=object)
    order = np.argsort(rows, kind="stable")
    return rows[order], src[order]



def self_check():
    import numpy as np
    # -- summit mapping --------------------------------------------------------
    names = ["chr1:100-201", "chr1:300-401", "chr2:50-151"]
    ch, st, en = parse_peak_names(names)
    assert list(st) == [100, 300, 50] and list(en) == [201, 401, 151]
    idx = build_chrom_index(ch, st)
    q_c = np.array(["chr1", "chr1", "chr1", "chr2", "chr3"], dtype=object)
    q_s = np.array([150, 250, 400, 60, 10], dtype=np.int64)
    got = map_summits(idx, st, en, q_c, q_s)
    assert list(got) == [0, -1, 1, 2, -1], list(got)

    out = np.zeros(3, dtype=np.float32)
    # a call spanning both chr1 peaks must hit BOTH; a summit map would hit at most one
    scatter_max_overlap(idx, st, en,
                        np.array(["chr1"], dtype=object),
                        np.array([150], dtype=np.int64),
                        np.array([350], dtype=np.int64),
                        np.array([5.0], dtype=np.float32), out)
    assert list(out) == [5.0, 5.0, 0.0], list(out)
    # strongest call wins; non-overlapping call contributes nothing
    out2 = np.zeros(3, dtype=np.float32)
    scatter_max_overlap(idx, st, en,
                        np.array(["chr1", "chr1", "chr1"], dtype=object),
                        np.array([100, 100, 250], dtype=np.int64),
                        np.array([120, 120, 260], dtype=np.int64),
                        np.array([2.0, 7.0, 9.0], dtype=np.float32), out2)
    assert list(out2) == [7.0, 0.0, 0.0], list(out2)
    # touching-but-not-overlapping boundaries are excluded (half-open intervals)
    out3 = np.zeros(3, dtype=np.float32)
    scatter_max_overlap(idx, st, en,
                        np.array(["chr1", "chr1"], dtype=object),
                        np.array([201, 90], dtype=np.int64),
                        np.array([300, 100], dtype=np.int64),
                        np.array([3.0, 4.0], dtype=np.float32), out3)
    assert list(out3) == [0.0, 0.0, 0.0], list(out3)

    # -- selection: every group represented, anchor honoured ---------------
    rng = np.random.default_rng(0)
    n_peaks, n_s = 400, 4
    mask = rng.random((n_peaks, n_s)) < 0.35
    mask[mask.sum(axis=1) == 0, 0] = True
    mask[:20, :] = True                          # 20 universal peaks -> anchor pool
    n_sub = mask.sum(axis=1)
    signal = np.where(mask, rng.random((n_peaks, n_s)) * 10, 0.0)
    rows, src = select(mask, n_sub, signal, n_features=60, max_shared=2,
                       min_anchor=4, anchor_frac=0.2,
                       groups=[f"s{i}" for i in range(n_s)], logfn=lambda *_: None)
    assert len(rows) == len(set(rows.tolist())), "duplicate rows selected"
    assert (src == "anchor").sum() > 0, "no anchor peaks drawn"
    spec = rows[src == "specific"]
    assert (n_sub[spec] <= 2).all(), "specific selection violated max_shared"
    anc = rows[src == "anchor"]
    assert (n_sub[anc] >= 4).all(), "anchor selection violated min_anchor"
    for j in range(n_s):                          # group balance
        assert mask[spec, j].any(), f"group {j} unrepresented"
    assert list(rows) == sorted(rows.tolist()), "output not in BED order"
    print("self-check: all assertions passed")
    return 0



def read_blacklist(path):
    """chrom -> sorted list of (start, end), for overlap testing."""
    import collections
    bl = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            bl[f[0]].append((int(f[1]), int(f[2])))
    return {c: sorted(v) for c, v in bl.items()}


def in_blacklist(bl, chrom, start, end):
    """True if [start, end) overlaps any blacklist interval. Bisect on sorted starts."""
    import bisect
    iv = bl.get(chrom)
    if not iv:
        return False
    i = bisect.bisect_right([s for s, _ in iv], end) - 1
    while i >= 0:
        s, e = iv[i]
        if e <= start:
            break
        if s < end and start < e:
            return True
        i -= 1
    return False

