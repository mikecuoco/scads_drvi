"""viz.enrichment / viz.umap / viz.factors -- figures come back, nothing is written."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scads_drvi.labels import load_labels  # noqa: E402
from scads_drvi.stats import add_fdr  # noqa: E402
from scads_drvi.viz.enrichment import (  # noqa: E402
    covariate_audit,
    grouped_landscape,
    heritability_landscape,
    score_by_group,
    trait_concordance,
)
from scads_drvi.viz.factors import (  # noqa: E402
    covariate_association,
    factor_correlation,
    factor_distributions,
    group_factor_heatmaps,
    latent_dimension_stats,
)
from scads_drvi.viz.frugal import box_stats, box_stats_by_column  # noqa: E402
from scads_drvi.viz.umap import (  # noqa: E402
    bare,
    point_style,
    subsample,
    umap_categorical,
    umap_continuous,
    umap_factor_grid,
)


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
        from scads_drvi.viz.color import categorical_palette

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

    def test_factor_grid(self, cells):
        loadings = pd.DataFrame(
            np.random.default_rng(2).random((len(cells), 3)),
            index=cells.index,
            columns=["dim_0", "dim_1", "dim_2"],
        )
        fig = umap_factor_grid(
            cells[["UMAP_1", "UMAP_2"]], loadings, ["dim_0", "dim_1"], n=500
        )
        assert isinstance(fig, Figure)

    def test_factor_grid_checks_names_against_labels(self, cells, tmp_path):
        fm = tmp_path / "fm.tsv"
        pd.DataFrame(
            {
                "dim": ["dim_0"],
                "vanished": [False],
                "kept": [True],
                "drop_reason": [""],
                "annot_index": [1],
            }
        ).to_csv(fm, sep="\t", index=False)
        hm = tmp_path / "hm.tsv"
        pd.DataFrame(
            {"annot_dim": ["dim_0"], "source_dim": ["dim_9"], "half": ["pos"]}
        ).to_csv(hm, sep="\t", index=False)
        labels = load_labels(fm, model="m", half_map=hm)
        loadings = pd.DataFrame({"dim_9/pos": np.zeros(len(cells))}, index=cells.index)
        with pytest.raises(KeyError, match="display label"):
            umap_factor_grid(
                cells[["UMAP_1", "UMAP_2"]], loadings, ["dim_9/pos"], labels=labels
            )

    def test_no_shared_cells_is_refused(self, cells):
        loadings = pd.DataFrame({"dim_0": [0.0]}, index=["zzz"])
        with pytest.raises(ValueError, match="share no cells"):
            umap_factor_grid(cells[["UMAP_1", "UMAP_2"]], loadings, ["dim_0"])


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
    def test_latent_dimension_stats_marks_vanished(self):
        frame = pd.DataFrame(
            {
                "dim": [f"dim_{i}" for i in range(6)],
                "vanished": [False, False, True, False, True, False],
                "reconstruction_effect": np.linspace(1, 0, 6),
                "max_value": np.linspace(2, 0, 6),
            }
        )
        fig, axes = latent_dimension_stats(frame)
        assert isinstance(fig, Figure)
        assert "2 of 6" in fig.get_suptitle()

    def test_latent_dimension_stats_needs_numbers(self):
        with pytest.raises(ValueError, match="no numeric columns"):
            latent_dimension_stats(pd.DataFrame({"dim": ["a"]}), columns=[])

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

    def test_group_factor_heatmaps_share_one_order(self):
        rows = [f"g{i}" for i in range(4)]
        cols = [f"dim_{i}" for i in range(5)]
        rng = np.random.default_rng(2)
        first = pd.DataFrame(rng.random((4, 5)), index=rows, columns=cols)
        second = pd.DataFrame(rng.normal(size=(4, 5)), index=rows, columns=cols)
        fig, axes = group_factor_heatmaps(
            [("mean", first, "magma"), ("z", second, "RdBu_r")],
            grey_columns={"dim_3"},
        )
        assert isinstance(fig, Figure)
        # both panels must show the same row order
        left = [t.get_text() for t in axes[0].get_yticklabels()]
        assert left == rows

    def test_group_factor_heatmaps_needs_panels(self):
        with pytest.raises(ValueError, match="no panels"):
            group_factor_heatmaps([])

    def test_covariate_association(self):
        series = pd.Series(
            [0.9, 0.2, -0.8, 0.1], index=[f"dim_{i}" for i in range(4)]
        )
        fig, ax = covariate_association(series, threshold=0.7)
        assert "2 factor(s)" in ax.get_title()
