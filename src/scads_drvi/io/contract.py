#!/usr/bin/env python3
"""The arm-05 factorize contract, asserted. CLAUDE.md "The factorize contract".

Non-negativity is a hard requirement, not a convention: annotations are a top-frac RANK, so a
negative sorts last instead of "most extreme the other way". Two scripts write different
halves of the contract, which is why this is shared and why topic_factors.tsv is optional.
"""
from __future__ import annotations

import json
from pathlib import Path

from scads_drvi._util.progress import log as _log_stderr
from scads_drvi.io.peaks import PEAK_RE  # noqa: F401  (re-exported: callers import it here)

# PEAK_RE is io.peaks' definition, not a second copy. The copy that used to live here was
# `^chr[\w]+:\d+-\d+$`, which rejects a sequence name containing a dot -- so a contract over
# GRCh38 scaffolds (chrUn_GL000220.1) would have failed validation while normalize_peak_name
# considered the same name canonical. One definition, in the module that writes the form.


def check_contract(outdir: Path, latent_stats: Path | None = None, log_fn=None) -> dict:
    """Assert the emitted files are the contract arm 05 expects.

        `latent_stats` is a parameter because arm 04 has TWO layouts -- inspect_drvi.py writes a
        SIBLING dir (inspect_k48 beside drvi_k48) while older fits carry <fit>/inspect/ -- and
        hardcoding the second meant the check could never validate the production fit.
    """
    import pandas as pd

    log = log_fn or _log_stderr
    stats_path = Path(latent_stats) if latent_stats else outdir / "inspect" / "latent_stats.tsv"

    meta = json.loads((outdir / "fit.meta.json").read_text())
    k = int(meta["n_latent"])
    cols = [f"dim_{i}" for i in range(k)]

    lo = pd.read_csv(outdir / "topic_loadings.tsv", sep="\t", index_col=0)
    assert list(lo.columns) == cols, "loadings columns != dim_0..dim_{K-1}"
    assert (lo.values >= 0).all(), "topic_loadings has negatives (arm 05 requires >= 0)"
    assert not lo.isna().any().any(), "NaNs in topic_loadings"
    assert lo.shape == (meta["n_cells"], k), f"loadings {lo.shape} != {(meta['n_cells'], k)}"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found. latent_stats.tsv comes from `inspect_drvi.py latent`, "
            f"which writes it to a SIBLING inspect dir; only older fits carry it INSIDE the "
            f"fit -- pass the path explicitly for the former, e.g. "
            f"{outdir.parent / ('inspect_' + outdir.name.removeprefix('drvi_'))}/latent_stats.tsv")
    stats = pd.read_csv(stats_path, sep="\t")
    assert {"dim", "vanished"} <= set(stats.columns), \
        f"latent_stats.tsv missing dim/vanished: {list(stats.columns)}"
    assert list(stats["dim"]) == cols, "latent_stats dims != the contract's columns"

    out = {"n_cells": int(lo.shape[0]), "n_latent": k,
           "frac_loadings_zero": round(float((lo.values == 0).mean()), 4),
           "topic_factors": False}
    fa_path = outdir / "topic_factors.tsv"
    if fa_path.exists():
        fa = pd.read_csv(fa_path, sep="\t", index_col=0)
        assert list(fa.columns) == cols, "factors columns != dim_0..dim_{K-1}"
        assert (fa.values >= 0).all(), "topic_factors has negatives"
        bad = [f for f in fa.index if not PEAK_RE.match(str(f))]
        assert not bad, f"{len(bad)} feature names are not chr:start-end, e.g. {bad[:3]}"
        assert fa.shape == (meta["n_features"], k)
        out["topic_factors"] = True
        out["n_features"] = int(fa.shape[0])
    log(f"contract OK: {out}")
    if not out["topic_factors"]:
        log("  topic_factors.tsv absent -- fine for the prototype notebook, which reads the "
            "decoder directly; run `posthoc_drvi.py ood` before using the "
            "top_frac path")
    return out
