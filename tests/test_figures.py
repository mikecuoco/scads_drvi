"""viz.enrichment / viz.umap / viz.factors -- figures come back, nothing is written."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scads_drvi.pl.enrichment import (  # noqa: E402
    covariate_audit,
    grouped_landscape,
    heritability_landscape,
    score_by_group,
    trait_concordance,
)
from scads_drvi.pl.factors import (  # noqa: E402
    covariate_association,
    factor_correlation,
    factor_distributions,
)
from scads_drvi.pl.frugal import box_stats, box_stats_by_column  # noqa: E402
from scads_drvi.pl.umap import (  # noqa: E402
    bare,
    point_style,
    subsample,
    umap_categorical,
    umap_continuous,
)
from scads_drvi.stats import add_fdr  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def no_writes(tmp_path, monkeypatch):
    """A figure function that writes a file is a figure function you cannot compose.

    Watches a dedicated working directory rather than all of tmp_path, so a test's own
    fixture files do not read as output from the code under test.
    """
    workdir = tmp_path / "_cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    yield
    assert list(workdir.iterdir()) == [], "a figure function wrote to the cwd"


@pytest.fixture
def results():
    rng = np.random.default_rng(0)
    rows = []
    for trait in ("t1", "t2"):
        for k in range(24):
            rows.append(
                {
                    "trait": trait,
                    "dim": f"dim_{k}",
                    "display": f"dim_{k}",
                    "Coefficient_z-score": rng.normal(loc=1.0, scale=1.5),
                }
            )
    return add_fdr(pd.DataFrame(rows), by="trait")


@pytest.fixture
def cells():
    rng = np.random.default_rng(1)
    n = 4000
    return pd.DataFrame(
        {
            "UMAP_1": rng.normal(size=n),
            "UMAP_2": rng.normal(size=n),
            "grouping": rng.choice(list("abcde"), size=n),
            "axis": rng.choice(["p", "q"], size=n),
            "depth": rng.lognormal(mean=8.0, sigma=1.0, size=n),
            "cs": rng.gamma(2.0, 1.0, size=n),
        },
        index=[f"c{i}" for i in range(n)],
    )


class TestPointStyle:
    @pytest.mark.parametrize("n", [1, 100, 5_000, 60_000, 300_000, 1_300_000])
    def test_always_rasterised_and_sized(self, n):
        style = point_style(n)
        assert style["rasterized"] is True
        assert style["s"] > 0

    def test_size_falls_as_the_count_rises(self):
        """The 50x gap between two call sites was a point-count artefact, not a choice."""
        sizes = [point_style(n)["s"] for n in (1_000, 50_000, 1_300_000)]
        assert sizes[0] > sizes[1] > sizes[2]

    def test_zero_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            point_style(0)


class TestSubsample:
    def test_small_frames_pass_through_untouched(self, cells):
        assert subsample(cells, len(cells) + 1) is cells

    def test_reproducible(self, cells):
        a = subsample(cells, 500, seed=7)
        b = subsample(cells, 500, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_stratify_keeps_every_group(self, cells):
        drawn = subsample(cells, 50, stratify="grouping")
        assert set(drawn["grouping"]) == set(cells["grouping"])

    def test_unknown_stratify_column(self, cells):
        with pytest.raises(KeyError, match="not a column"):
            subsample(cells, 10, stratify="absent")

    def test_zero_is_refused(self, cells):
        with pytest.raises(ValueError, match="positive"):
            subsample(cells, 0)


class TestUmapPanels:
    def test_categorical_returns_fig_and_ax(self, cells):
        fig, ax = umap_categorical(cells, "grouping", n=1000)
        assert isinstance(fig, Figure)
        assert ax.get_legend() is not None

    def test_categorical_colours_are_stable_across_subsamples(self, cells):
        """Building the palette from a sampled frame is how one category gets two."""
        from scads_drvi.pl.color import categorical_palette

        palette = categorical_palette(sorted(cells["grouping"].unique()))
        _, ax_small = umap_categorical(cells, "grouping", n=200, palette=palette)
        _, ax_large = umap_categorical(cells, "grouping", n=3000, palette=palette)
        first = {c.get_label(): c.get_facecolor()[0][:3] for c in ax_small.collections}
        second = {c.get_label(): c.get_facecolor()[0][:3] for c in ax_large.collections}
        for label in first:
            assert np.allclose(first[label], second[label])

    def test_continuous_returns_fig_and_ax(self, cells):
        fig, ax = umap_continuous(cells, "cs", n=1000)
        assert isinstance(fig, Figure)

    def test_zero_as_background_draws_a_grey_underlay(self, cells):
        frame = cells.copy()
        frame.loc[frame.index[:2000], "cs"] = 0.0
        _, ax = umap_continuous(frame, "cs", zero_as_background=True, n=None)
        assert len(ax.collections) == 2  # grey zeros, then the coloured rest

    def test_missing_column_is_named(self, cells):
        with pytest.raises(KeyError, match="absent"):
            umap_continuous(cells, "absent")

    def test_explicit_axes_are_honoured(self, cells):
        with pytest.raises(KeyError, match="nope"):
            umap_categorical(cells, "grouping", x="nope", y="UMAP_2")

    def test_bare_strips_the_frame(self, cells):
        _, ax = umap_continuous(cells, "cs", n=100)
        bare(ax)
        assert list(ax.get_xticks()) == []
        assert not any(s.get_visible() for s in ax.spines.values())

class TestEnrichmentFigures:
    def test_heritability_landscape(self, results):
        fig, (bars, volcano) = heritability_landscape(results, trait="t1")
        assert isinstance(fig, Figure)
        assert bars.get_ylabel() == "coefficient z"
        assert volcano.get_legend() is not None

    def test_heritability_landscape_needs_fdr_columns(self):
        bare_frame = pd.DataFrame(
            {"trait": ["t"], "dim": ["dim_0"], "Coefficient_z-score": [1.0]}
        )
        with pytest.raises(KeyError, match="add_fdr"):
            heritability_landscape(bare_frame, trait="t")

    def test_heritability_landscape_empty_trait(self, results):
        with pytest.raises(ValueError, match="no results to plot"):
            heritability_landscape(results, trait="absent")

    def test_trait_concordance(self, results):
        fig, (scatter, hist) = trait_concordance(results, traits=["t1", "t2"])
        assert isinstance(fig, Figure)
        assert "t1" in scatter.get_xlabel()

    def test_trait_concordance_needs_two(self, results):
        with pytest.raises(ValueError, match="exactly two"):
            trait_concordance(results, traits=["t1"])

    def test_trait_concordance_unknown_trait(self, results):
        with pytest.raises(KeyError, match="no results for trait"):
            trait_concordance(results, traits=["t1", "absent"])

    def test_trait_concordance_survives_a_constant_difference(self):
        """Two traits ranking identically give the delta histogram zero width, which
        numpy 2 refuses to bin and numpy 1 quietly widened."""
        rows = []
        for k in range(8):
            rows.append({"trait": "a", "dim": f"dim_{k}", "Coefficient_z-score": 1.0 * k})
            rows.append(
                {"trait": "b", "dim": f"dim_{k}", "Coefficient_z-score": 1.0 * k + 0.3}
            )
        fig, (_, hist) = trait_concordance(pd.DataFrame(rows), traits=["a", "b"])
        assert isinstance(fig, Figure)
        assert "+0.30" in hist.get_title()

    def test_trait_concordance_with_a_single_shared_factor(self):
        rows = [
            {"trait": "a", "dim": "dim_0", "Coefficient_z-score": 2.0},
            {"trait": "b", "dim": "dim_0", "Coefficient_z-score": 2.5},
        ]
        fig, _ = trait_concordance(pd.DataFrame(rows), traits=["a", "b"])
        assert isinstance(fig, Figure)

    def test_covariate_audit(self, cells):
        fig, axes = covariate_audit(
            cells, value="cs", covariates=["depth"], log_x=["depth"]
        )
        assert isinstance(fig, Figure)
        assert axes[0].get_xlabel() == "log10(depth)"

    def test_covariate_audit_needs_covariates(self, cells):
        with pytest.raises(ValueError, match="no covariates"):
            covariate_audit(cells, value="cs", covariates=[])

    def test_score_by_group_takes_only_precomputed_stats(self, cells):
        """Handing it the per-cell table must be impossible, not merely discouraged."""
        stats = box_stats(cells["cs"], cells["grouping"])
        fig, ax = score_by_group(stats, group_label="grouping", null=0.0)
        assert isinstance(fig, Figure)
        with pytest.raises(AttributeError):
            score_by_group(cells, group_label="grouping")

    def test_grouped_landscape_masks_and_says_so(self, cells):
        from scads_drvi.scores.aggregate import group_matrix

        means, counts = group_matrix(
            cells["cs"], cells, index="grouping", columns="axis", min_cells=10_000
        )
        fig, ax = grouped_landscape(
            means, counts, row_label="grouping", column_label="axis"
        )
        assert isinstance(fig, Figure)
        label = fig.axes[-1].get_ylabel() or fig.axes[-1].get_xlabel()
        assert "count threshold" in label

    def test_grouped_landscape_shape_mismatch(self):
        means = pd.DataFrame(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="same shape"):
            grouped_landscape(
                means, pd.DataFrame(np.zeros((3, 2))), row_label="r", column_label="c"
            )


class TestFactorFigures:
    def test_factor_distributions(self):
        rng = np.random.default_rng(0)
        matrix = rng.random((100, 8))
        panels = {
            "source_a": box_stats_by_column(matrix, [f"dim_{i}" for i in range(8)]),
            "source_b": box_stats_by_column(matrix * 2, [f"dim_{i}" for i in range(8)]),
        }
        assert isinstance(factor_distributions(panels), Figure)

    def test_factor_distributions_needs_panels(self):
        with pytest.raises(ValueError, match="no panels"):
            factor_distributions({})

    def test_factor_correlation_greys_dead_factors(self):
        rng = np.random.default_rng(1)
        matrix = rng.random((6, 6))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        names = [f"dim_{i}" for i in range(6)]
        fig, ax = factor_correlation(
            pd.DataFrame(matrix, index=names, columns=names), vanished=["dim_2"]
        )
        assert isinstance(fig, Figure)

    def test_factor_correlation_needs_square(self):
        with pytest.raises(ValueError, match="square"):
            factor_correlation(pd.DataFrame(np.zeros((2, 3))))

    def test_covariate_association(self):
        series = pd.Series(
            [0.9, 0.2, -0.8, 0.1], index=[f"dim_{i}" for i in range(4)]
        )
        fig, ax = covariate_association(series, threshold=0.7)
        assert "2 factor(s)" in ax.get_title()
