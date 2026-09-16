"""pl.enrichment / pl.umap / pl.factors -- figures come back, nothing is written."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

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
    latent_dimension_stats,
    latent_heatmap,
    latent_heatmap_with_heritability,
)
from scads_drvi.pl.frugal import box_stats, box_stats_by_column  # noqa: E402
from scads_drvi.pl.umap import latent_umap_grid, subsample  # noqa: E402
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


@pytest.fixture
def embed():
    """A small ``embed``-shaped ``AnnData``: obs = cells, var = one row per dim,
    obsm["X_umap"] = an embedding -- the object every :mod:`scads_drvi.pl.umap` figure
    now takes directly.
    """
    rng = np.random.default_rng(2)
    n, k = 2000, 6
    obs = pd.DataFrame(
        {
            "grouping": rng.choice(list("abcde"), size=n),
            "cs": rng.gamma(2.0, 1.0, size=n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    x = rng.normal(size=(n, k)).astype(np.float32)
    var = pd.DataFrame(
        {
            "order": np.arange(k),
            "vanished": [False] * (k - 2) + [True] * 2,
            "title": [f"DR {i + 1}" for i in range(k)],
            "min": x.min(axis=0),
            "max": x.max(axis=0),
        },
        index=[f"dim_{i}" for i in range(k)],
    )
    e = ad.AnnData(X=x, obs=obs, var=var)
    e.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)
    return e


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


class TestLatentUmapGrid:
    def test_returns_one_panel_per_kept_dim(self, embed):
        fig = latent_umap_grid(embed, directional=False, n=500)
        n_kept = int((~embed.var["vanished"]).sum())
        # each panel is (scatter axis, colorbar axis)
        assert len(fig.axes) == 2 * n_kept

    def test_directional_doubles_the_panel_count(self, embed):
        # directional=True is now the default -- every factor interpretability plot
        # shows split factors unless told not to -- so this compares against the
        # directional=False count explicitly instead of assuming it.
        fig = latent_umap_grid(embed, directional=True, n=500)
        fig_plain = latent_umap_grid(embed, directional=False, n=500)
        assert len(fig.axes) == 2 * len(fig_plain.axes)

    def test_dim_subset_restricts_dims(self, embed):
        fig = latent_umap_grid(embed, dim_subset=["dim_0", "dim_1"], directional=False, n=500)
        assert len(fig.axes) == 2 * 2

    def test_missing_order_col_raises(self, embed):
        e = embed.copy()
        del e.var["order"]
        with pytest.raises(KeyError, match="order"):
            latent_umap_grid(e)

    def test_missing_vanished_raises_when_filtering(self, embed):
        e = embed.copy()
        del e.var["vanished"]
        with pytest.raises(KeyError, match="vanished"):
            latent_umap_grid(e, remove_vanished=True)

    def test_unknown_dim_subset_is_named(self, embed):
        with pytest.raises(KeyError, match="not_a_dim"):
            latent_umap_grid(embed, dim_subset=["not_a_dim"])


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

    def test_heritability_landscape_titles_map_dim_to_dr_names(self):
        rows = [
            {"trait": "t", "dim": "dim_5", "direction": "pos", "Coefficient_z-score": 3.0},
            {"trait": "t", "dim": "dim_5", "direction": "neg", "Coefficient_z-score": -1.0},
        ]
        frame = add_fdr(pd.DataFrame(rows), by="trait")
        fig, (bars, _) = heritability_landscape(frame, trait="t", titles={"dim_5": "DR 6"})
        assert {t.get_text() for t in bars.texts} == {"DR 6+", "DR 6-"}

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

    def test_trait_concordance_titles_map_dim_to_dr_names(self):
        rows = [
            {"trait": "a", "dim": "dim_5", "direction": "pos", "Coefficient_z-score": 2.0},
            {"trait": "b", "dim": "dim_5", "direction": "pos", "Coefficient_z-score": 2.5},
        ]
        fig, (scatter, _) = trait_concordance(
            pd.DataFrame(rows), traits=["a", "b"], titles={"dim_5": "DR 6"}
        )
        assert {t.get_text() for t in scatter.texts} == {"DR 6+"}

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


@pytest.fixture
def dim_stats():
    rng = np.random.default_rng(3)
    k = 8
    return pd.DataFrame(
        {
            "order": np.arange(k),
            "vanished": [False] * (k - 2) + [True] * 2,
            "title": [f"DR {i + 1}" for i in range(k)],
            "reconstruction_effect": rng.random(k),
            "max_value": rng.random(k) * 3,
            "mean": rng.normal(size=k),
            "std": rng.random(k) + 0.1,
        },
        index=[f"dim_{i}" for i in range(k)],
    )


class TestLatentDimensionStats:
    def test_one_panel_per_column(self, dim_stats):
        fig, axes = latent_dimension_stats(dim_stats, columns=("mean", "std"))
        assert isinstance(fig, Figure)
        assert len(axes) == 2

    def test_vanished_and_kept_get_two_colours(self, dim_stats):
        # DRVI's own hardcoded convention (drvi.utils.pl.plot_latent_dimension_stats):
        # vanished dims black, kept dims blue.
        _, axes = latent_dimension_stats(dim_stats, columns=("mean",))
        by_label = {c.get_label(): c for c in axes[0].collections}
        assert to_hex(by_label["kept"].get_facecolor()[0]) == "#0000ff"
        assert to_hex(by_label["vanished"].get_facecolor()[0]) == "#000000"

    def test_legend_present_only_when_vanished_kept(self, dim_stats):
        fig, _ = latent_dimension_stats(dim_stats, columns=("mean",))
        assert fig.legends
        fig2, _ = latent_dimension_stats(
            dim_stats, columns=("mean",), remove_vanished=True
        )
        assert not fig2.legends

    def test_log_scale_try_only_for_all_positive_columns(self, dim_stats):
        _, axes = latent_dimension_stats(dim_stats, columns=("max_value", "mean"))
        assert axes[0].get_yscale() == "log"  # max_value is all positive
        assert axes[1].get_yscale() != "log"  # mean can be negative

    def test_missing_order_col_raises(self, dim_stats):
        frame = dim_stats.drop(columns=["order"])
        with pytest.raises(KeyError, match="order"):
            latent_dimension_stats(frame)

    def test_missing_named_column_raises(self, dim_stats):
        with pytest.raises(KeyError, match="not_a_column"):
            latent_dimension_stats(dim_stats, columns=("not_a_column",))


class TestLatentHeatmap:
    @pytest.fixture
    def heatmap_embed(self, dim_stats):
        rng = np.random.default_rng(4)
        n = 400
        x = rng.normal(size=(n, len(dim_stats))).astype(np.float32)
        obs = pd.DataFrame(
            {"grouping": rng.choice(["a", "b", "c"], size=n)},
            index=[f"c{i}" for i in range(n)],
        )
        return ad.AnnData(X=x, obs=obs, var=dim_stats.copy())

    def test_returns_fig_and_ax(self, heatmap_embed):
        fig, ax = latent_heatmap(heatmap_embed, "grouping", make_balanced=False)
        assert isinstance(fig, Figure)

    def test_remove_vanished_changes_column_count(self, heatmap_embed):
        _, ax_kept = latent_heatmap(
            heatmap_embed, "grouping", remove_vanished=True, make_balanced=False
        )
        _, ax_all = latent_heatmap(
            heatmap_embed, "grouping", remove_vanished=False, make_balanced=False
        )
        assert len(ax_kept.get_xticklabels()) < len(ax_all.get_xticklabels())

    def test_make_balanced_gives_every_category_the_same_row_count(self, heatmap_embed):
        # scanpy's own sc.pl.heatmap draws with imshow (ax.images), not pcolormesh.
        _, ax = latent_heatmap(heatmap_embed, "grouping", make_balanced=True, seed=1)
        counts = heatmap_embed.obs["grouping"].value_counts()
        expected_n = max(10, int(counts.min()))
        assert ax.images[0].get_array().shape[0] == expected_n * counts.size

    def test_make_balanced_is_reproducible(self, heatmap_embed):
        _, ax1 = latent_heatmap(heatmap_embed, "grouping", make_balanced=True, seed=1)
        _, ax2 = latent_heatmap(heatmap_embed, "grouping", make_balanced=True, seed=1)
        np.testing.assert_array_equal(
            ax1.images[0].get_array(), ax2.images[0].get_array()
        )

    def test_column_labels_match_titles_in_rank_order(self, heatmap_embed):
        _, ax = latent_heatmap(
            heatmap_embed, "grouping", remove_vanished=False, make_balanced=False,
            directional=False,
        )
        labels = [t.get_text() for t in ax.get_xticklabels()]
        kept = heatmap_embed.var.sort_values("order")
        assert labels == kept["title"].tolist()

    def test_cluster_order_can_differ_from_rank_order(self, dim_stats):
        # Two dims deliberately anti-correlated, so a correlation-clustering leaf order
        # must not coincide with the plain rank order for every possible outcome.
        rng = np.random.default_rng(5)
        n = 300
        base = rng.normal(size=n)
        x = np.column_stack(
            [base, -base] + [rng.normal(size=n) for _ in range(len(dim_stats) - 2)]
        ).astype(np.float32)
        obs = pd.DataFrame(
            {"grouping": rng.choice(["a", "b"], size=n)}, index=[f"c{i}" for i in range(n)]
        )
        heatmap_embed = ad.AnnData(X=x, obs=obs, var=dim_stats.copy())

        _, ax_rank = latent_heatmap(heatmap_embed, "grouping", order="rank", make_balanced=False)
        _, ax_cluster = latent_heatmap(
            heatmap_embed, "grouping", order="cluster", make_balanced=False
        )
        rank_labels = [t.get_text() for t in ax_rank.get_xticklabels()]
        cluster_labels = [t.get_text() for t in ax_cluster.get_xticklabels()]
        assert rank_labels != cluster_labels

    def test_sort_by_categorical_reproduces_drvis_own_heuristic(self, heatmap_embed):
        kept = heatmap_embed[:, ~heatmap_embed.var["vanished"].to_numpy(dtype=bool)]
        expected = np.asarray(kept.var["title"])[
            np.argsort(np.abs(np.asarray(kept.X)).argmax(axis=0))
        ]
        _, ax = latent_heatmap(
            heatmap_embed, "grouping", sort_by_categorical=True, make_balanced=False,
            directional=False,
        )
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == list(expected)

    def test_missing_categorical_column_raises(self, heatmap_embed):
        with pytest.raises(KeyError, match="absent"):
            latent_heatmap(heatmap_embed, "absent")

    def test_missing_order_col_raises(self, heatmap_embed):
        del heatmap_embed.var["order"]
        with pytest.raises(KeyError, match="order"):
            latent_heatmap(heatmap_embed, "grouping")

    def test_missing_vanished_raises_when_filtering(self, heatmap_embed):
        del heatmap_embed.var["vanished"]
        with pytest.raises(KeyError, match="vanished"):
            latent_heatmap(heatmap_embed, "grouping", remove_vanished=True)

    def test_heritability_adds_a_bar_above_sharing_column_order(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        fig, (bar, ax) = latent_heatmap(
            heatmap_embed, "grouping", heritability=heritability, make_balanced=False,
            directional=False,
        )
        assert isinstance(fig, Figure)
        assert bar is not ax
        assert len(bar.patches) == len(ax.get_xticklabels())

    def test_missing_heritability_value_raises(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        ).drop(heatmap_embed.var_names[0])
        with pytest.raises(KeyError, match="heritability has no value"):
            latent_heatmap(
                heatmap_embed, "grouping", heritability=heritability, make_balanced=False,
                directional=False,
            )

    def test_heritability_se_draws_error_bars(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        se = pd.Series(
            rng.random(heatmap_embed.n_vars) * 0.5 + 0.1, index=heatmap_embed.var_names
        )
        fig, (bar, ax) = latent_heatmap(
            heatmap_embed, "grouping", heritability=heritability, heritability_se=se,
            make_balanced=False, directional=False,
        )
        assert bar.containers[0].has_yerr

    def test_missing_heritability_se_value_raises(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        se = pd.Series(
            rng.random(heatmap_embed.n_vars), index=heatmap_embed.var_names
        ).drop(heatmap_embed.var_names[0])
        with pytest.raises(KeyError, match="heritability_se has no value"):
            latent_heatmap(
                heatmap_embed, "grouping", heritability=heritability, heritability_se=se,
                make_balanced=False, directional=False,
            )

    def test_cell_scores_draw_group_sem_error_bars(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(
            rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names
        )
        fig, (ax, score) = latent_heatmap(
            heatmap_embed, "grouping", cell_scores=cell_scores, make_balanced=False
        )
        assert score.containers[0].has_xerr

    def test_heritability_q_colors_and_draws_significance_legend(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        q = pd.Series(
            np.linspace(0.001, 0.5, heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        fig, (bar, ax) = latent_heatmap(
            heatmap_embed, "grouping", heritability=heritability, heritability_q=q,
            make_balanced=False, directional=False,
        )
        assert bar.legend_ is not None
        assert bar.lines

    def test_cell_scores_adds_one_bar_per_group_on_the_right(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(
            rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names
        )
        fig, (ax, score) = latent_heatmap(
            heatmap_embed, "grouping", cell_scores=cell_scores, make_balanced=False
        )
        assert isinstance(fig, Figure)
        assert score is not ax
        assert len(score.patches) == heatmap_embed.obs["grouping"].nunique()

    def test_missing_cell_score_value_raises(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(
            rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names
        ).drop(heatmap_embed.obs_names[0])
        with pytest.raises(KeyError, match="cell_scores has no value"):
            latent_heatmap(
                heatmap_embed, "grouping", cell_scores=cell_scores, make_balanced=False
            )

    def test_heritability_and_cell_scores_together_return_three_axes(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        cell_scores = pd.Series(
            rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names
        )
        fig, (bar, ax, score) = latent_heatmap(
            heatmap_embed, "grouping", heritability=heritability, cell_scores=cell_scores,
            make_balanced=False, directional=False,
        )
        assert len({id(bar), id(ax), id(score)}) == 3

    def test_invalid_cell_score_agg_raises(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(
            rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names
        )
        with pytest.raises(ValueError, match="cell_score_agg"):
            latent_heatmap(
                heatmap_embed, "grouping", cell_scores=cell_scores,
                cell_score_agg="bogus", make_balanced=False,
            )

    def test_directional_is_the_default_and_doubles_columns(self, heatmap_embed):
        _, ax_split = latent_heatmap(heatmap_embed, "grouping", make_balanced=False)
        _, ax_plain = latent_heatmap(
            heatmap_embed, "grouping", make_balanced=False, directional=False
        )
        assert len(ax_split.get_xticklabels()) == 2 * len(ax_plain.get_xticklabels())

    def test_directional_titles_get_plus_minus_suffix(self, heatmap_embed):
        _, ax = latent_heatmap(heatmap_embed, "grouping", make_balanced=False)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels  # non-empty
        assert all(label.endswith(("+", "-")) for label in labels)

    def test_directional_uses_a_one_sided_colormap_from_zero(self, heatmap_embed):
        # A ReLU'd split's values are never negative -- a diverging colormap centred
        # at 0 would waste half its range, so directional=True switches to a one-sided
        # scale instead.
        from scads_drvi.pl.color import SATURATED_JUST_SKY_CMAP, SATURATED_RED_BLUE_CMAP

        _, ax_split = latent_heatmap(heatmap_embed, "grouping", make_balanced=False)
        assert ax_split.images[0].get_cmap().name == SATURATED_JUST_SKY_CMAP.name
        assert ax_split.images[0].norm.vmin == 0

        _, ax_plain = latent_heatmap(
            heatmap_embed, "grouping", make_balanced=False, directional=False
        )
        assert ax_plain.images[0].get_cmap().name == SATURATED_RED_BLUE_CMAP.name
        assert ax_plain.images[0].norm.vcenter == 0

    def test_directional_heritability_uses_plus_minus_index(self, heatmap_embed):
        rng = np.random.default_rng(6)
        kept = heatmap_embed.var_names[~heatmap_embed.var["vanished"].to_numpy(dtype=bool)]
        signed = [f"{d}{sign}" for d in kept for sign in ("+", "-")]
        heritability = pd.Series(rng.normal(size=len(signed)), index=signed)
        fig, (bar, ax) = latent_heatmap(
            heatmap_embed, "grouping", heritability=heritability, make_balanced=False
        )
        assert len(bar.patches) == len(ax.get_xticklabels()) == len(signed)

    def test_cell_score_agg_sum_uses_raw_group_totals(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(
            rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names
        )
        fig, (ax, score) = latent_heatmap(
            heatmap_embed, "grouping", cell_scores=cell_scores, cell_score_agg="sum",
            make_balanced=False,
        )
        expected = cell_scores.groupby(heatmap_embed.obs["grouping"]).sum()
        actual = sorted(b.get_width() for b in score.patches)
        np.testing.assert_allclose(actual, sorted(expected.to_numpy()))

    def test_colorbar_moves_to_top_right_when_cell_scores_shown(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(
            rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names
        )
        fig, (ax, score) = latent_heatmap(
            heatmap_embed, "grouping", cell_scores=cell_scores, make_balanced=False
        )
        # the relocated colorbar is a small box flush against the figure's top right
        # corner -- distinct from the heatmap/groupby/score axes, which all sit well
        # inside the figure.
        assert any(
            a.get_position().x1 > 0.9 and a.get_position().y1 > 0.9
            and a.get_position().width < 0.2
            for a in fig.axes
        )


class TestLatentHeatmapWithHeritability:
    @pytest.fixture
    def values_and_categories(self, dim_stats):
        rng = np.random.default_rng(4)
        n = 400
        values = pd.DataFrame(
            rng.normal(size=(n, len(dim_stats))),
            columns=dim_stats.index,
            index=[f"c{i}" for i in range(n)],
        )
        categories = pd.Series(
            rng.choice(["a", "b", "c"], size=n), index=values.index
        )
        return values, categories

    @pytest.fixture
    def heritability(self, dim_stats):
        rng = np.random.default_rng(6)
        return pd.Series(rng.normal(size=len(dim_stats)), index=dim_stats.index)

    def test_returns_fig_and_two_axes(self, values_and_categories, dim_stats, heritability):
        values, categories = values_and_categories
        fig, (bar, heat) = latent_heatmap_with_heritability(
            values, categories, dim_stats, heritability
        )
        assert isinstance(fig, Figure)
        assert bar is not heat

    def test_bar_and_heatmap_share_column_order(
        self, values_and_categories, dim_stats, heritability
    ):
        values, categories = values_and_categories
        _, (bar, heat) = latent_heatmap_with_heritability(
            values, categories, dim_stats, heritability, remove_vanished=False
        )
        kept = dim_stats.sort_values("order").index.tolist()
        np.testing.assert_allclose(
            [b.get_height() for b in bar.patches], heritability.loc[kept].to_numpy()
        )
        assert len(bar.patches) == len(heat.get_xticklabels())

    def test_missing_heritability_value_raises(
        self, values_and_categories, dim_stats, heritability
    ):
        values, categories = values_and_categories
        # dim_stats.index[0] is a kept (non-vanished) dim, so dropping it must be seen.
        with pytest.raises(KeyError, match="heritability has no value"):
            latent_heatmap_with_heritability(
                values, categories, dim_stats, heritability.drop(dim_stats.index[0])
            )

    def test_q_colors_and_draws_significance_legend(
        self, values_and_categories, dim_stats, heritability
    ):
        values, categories = values_and_categories
        q = pd.Series(
            np.linspace(0.001, 0.5, len(dim_stats)), index=dim_stats.index
        )
        fig, (bar, heat) = latent_heatmap_with_heritability(
            values, categories, dim_stats, heritability, heritability_q=q
        )
        assert bar.legend_ is not None
        assert bar.lines  # threshold lines drawn

    def test_missing_q_value_raises(self, values_and_categories, dim_stats, heritability):
        values, categories = values_and_categories
        q = pd.Series(np.full(len(dim_stats), 0.01), index=dim_stats.index)
        with pytest.raises(KeyError, match="heritability_q has no value"):
            latent_heatmap_with_heritability(
                values, categories, dim_stats, heritability,
                heritability_q=q.drop(dim_stats.index[0]),
            )
