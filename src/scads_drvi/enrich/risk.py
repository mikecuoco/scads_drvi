"""How much heritability is in each peak, corrected for confounding.

:mod:`scads_drvi.enrich.ldsc` answers a genome-wide question -- does a factor's whole
peak set carry more heritability than baseline. This module pushes that same fitted
regression down to individual peaks: S-LDSC's coefficient ``tau`` (the `.results`
file's ``Coefficient`` column, already read by :func:`scads_drvi.enrich.ldsc.read_results`)
is the per-SNP heritability attributable to a factor's annotation, fit *jointly* with
the baseline-LD model's confounders (MAF, LD score, gene density, recombination rate,
conservation, ...) -- that joint fit is what "corrected for confounding" means here,
and nothing new needs to be estimated for it. Because the annotation is built
peak-by-peak (each SNP's annotation value comes from whichever peak it falls in), the
category's total heritability is exactly additive over peaks:
``h2_category = tau * sum_peaks(loading_peak * n_snps_in_peak)`` -- so one peak's own
share is ``tau * loading_peak * n_snps_in_peak``.

This gives a continuous, genome-wide picture rather than one restricted to whichever
peak happens to contain a lead SNP exactly -- a peak one SNP away from a lead SNP, on
the same real signal via LD, still gets its own credit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping

    import pandas as pd

__all__ = ["peak_snp_counts", "peak_risk", "top_risk_peaks"]


def peak_snp_counts(peaks_index: Iterable[str], bim: pd.DataFrame) -> pd.Series:
    """How many reference-panel SNPs (from `bim`) overlap each peak in `peaks_index`.

    `bim` may be one chromosome or the whole genome concatenated --
    :func:`scads_drvi.enrich.annotations.assign_peaks` (which this calls) only ever
    compares a SNP against peaks on its own chromosome. Peaks with no overlapping SNP
    get `0`, not a missing entry -- every peak in `peaks_index` gets a row.
    """
    from scads_drvi.enrich.annotations import assign_peaks

    peaks_index = list(peaks_index)
    assigned = assign_peaks(bim, peaks_index)
    counts = assigned.value_counts()
    return counts.reindex(peaks_index, fill_value=0).astype(int)


def peak_risk(
    feature_loadings: Mapping[str, pd.DataFrame],
    results: pd.DataFrame,
    snp_counts: pd.Series | None = None,
    *,
    coefficient_col: str = "Coefficient",
    include_zero: bool = False,
) -> pd.DataFrame:
    """One tidy row per `(peak, dim, direction)`: that peak's share of the factor's
    S-LDSC heritability.

    `feature_loadings` maps each direction the sweep actually tested (``"pos"``/
    ``"neg"``, or ``"combined"`` if it wasn't run per-direction) to its own peaks x
    kept-dims loading frame -- exactly the split :func:`scads_drvi.enrich.ldsc.read_results`'s
    own `direction` parameter already expects to exist (a caller-derived
    ``clip(loadings, 0, None)``/``clip(-loadings, 0, None)`` pair from one signed
    matrix, or DRVI's own already-split ``get_effect_of_splits_within_distribution``
    output directly -- the real S-LDSC sweep in `pbmc.ipynb` uses the latter). This
    function never owns that split, same as `read_results` never derives it either.

    `results` is `read_results`'s own tidy output; every `(dim, direction)` pair
    present in `feature_loadings` must have a matching fitted row there, or this
    raises naming the missing pair -- a factor scored with no fitted coefficient is a
    caller data problem, not something to silently skip.

    `snp_counts` (from :func:`peak_snp_counts`) adds `risk_total` -- the peak's
    absolute share of that factor's heritability (`risk_rate * n_snps_in_peak`) --
    alongside `risk_rate` (`loading * tau`, a per-SNP rate comparable across peaks
    regardless of local SNP density). Without it, only `risk_rate` is reported.

    Peaks with exactly zero loading are excluded by default (`include_zero=False`):
    their risk is zero by construction, not a meaningful comparison point.
    """
    import pandas as pd

    from scads_drvi.enrich.config import parse_peaks

    for column in ("dim", "direction", coefficient_col):
        if column not in results.columns:
            raise KeyError(f"{column!r} not in results; run enrich.ldsc.read_results first")

    tau = results.set_index(["dim", "direction"])[coefficient_col]

    blocks = []
    for direction, loadings in feature_loadings.items():
        peaks = parse_peaks(loadings.index)
        peaks.index = loadings.index
        for dim in loadings.columns:
            key = (str(dim), str(direction))
            if key not in tau.index:
                raise KeyError(
                    f"results has no {coefficient_col!r} for dim={dim!r}, "
                    f"direction={direction!r} -- every scored factor/direction needs "
                    "a fitted result."
                )
            t = float(tau.loc[key])
            values = loadings[dim]
            if not include_zero:
                values = values[values != 0.0]
            if values.empty:
                continue

            block = pd.DataFrame(
                {
                    "peak": values.index,
                    "chrom": peaks.loc[values.index, "chrom"].to_numpy(),
                    "start": peaks.loc[values.index, "start"].to_numpy(),
                    "end": peaks.loc[values.index, "end"].to_numpy(),
                    "dim": str(dim),
                    "direction": str(direction),
                    "loading": values.to_numpy(dtype=float),
                    "tau": t,
                }
            )
            block["risk_rate"] = block["loading"] * t
            if snp_counts is not None:
                block["risk_total"] = block["risk_rate"] * snp_counts.reindex(block["peak"]).to_numpy()
            blocks.append(block)

    if not blocks:
        raise ValueError(
            "no nonzero loadings to score across every (dim, direction) in "
            "feature_loadings -- pass include_zero=True if an all-zero result is "
            "genuinely expected"
        )

    return pd.concat(blocks, ignore_index=True)


def top_risk_peaks(risk: pd.DataFrame, n: int = 20, *, by: str = "risk_total") -> pd.DataFrame:
    """The top-`n` rows of `risk` (:func:`peak_risk`'s output) ranked by `by`.

    Ranks the whole frame as one pool -- group `risk` by `dim`/`direction` first (e.g.
    ``risk.groupby(["dim", "direction"]).apply(top_risk_peaks, n)``) for each factor's
    own top peaks instead of one global ranking.
    """
    if by not in risk.columns:
        raise KeyError(f"{by!r} not in risk; pass snp_counts to peak_risk for risk_total")
    return risk.sort_values(by, ascending=False).head(n).reset_index(drop=True)
