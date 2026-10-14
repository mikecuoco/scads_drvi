"""pl.enrichment / pl.umap / pl.factors -- figures come back, nothing is written."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.container import ErrorbarContainer  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from scads_drvi.pl import enrichment as _pl_enrichment  # noqa: E402
from scads_drvi.pl.enrichment import (  # noqa: E402
    chrom_sort_key,
    heritability_landscape,
    risk_manhattan,
    trait_concordance,
)
from scads_drvi.pl.factors import (  # noqa: E402
    BarSeries,
    LatentHeatmapAxes,
    factor_correlation,
    factor_distributions,
    latent_dimension_stats,
    latent_heatmap,
)
from scads_drvi.pl.frugal import box_stats_by_column  # noqa: E402
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
                    "factor": f"dim_{k}",
                    "display": f"dim_{k}",
                    "Coefficient": 1e-8 * rng.normal(loc=1.0, scale=1.5),
                    "Coefficient_std_error": 1e-8,
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

    def test_split_arms_are_non_negative(self, embed):
        fig = latent_umap_grid(embed, directional=True, n=500)
        for i in range(len(fig.axes) // 2):
            cbar = fig.axes[1 + 2 * i]
            assert cbar.get_ylim()[0] == 0 and not cbar.yaxis_inverted()
            scatter = fig.axes[2 * i].collections[0]
            assert scatter.get_array().min() >= 0

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
        fig, (tau_ax, z_ax) = heritability_landscape(results, trait="t1")
        assert isinstance(fig, Figure)
        assert tau_ax.get_ylabel() == "coefficient tau"
        assert z_ax.get_ylabel() == "coefficient z"
        assert z_ax.get_legend() is not None
        n = int((results["trait"] == "t1").sum())
        assert len(tau_ax.patches) == len(z_ax.patches) == n  # one bar per factor, both panels
        assert any(isinstance(c, ErrorbarContainer) for c in tau_ax.containers)

    def test_heritability_landscape_tau_error_bars_are_optional(self, results):
        _, (tau_ax, _) = heritability_landscape(results, trait="t1", tau_se_column=None)
        assert not any(isinstance(c, ErrorbarContainer) for c in tau_ax.containers)

    def test_heritability_landscape_needs_tau(self, results):
        with pytest.raises(KeyError, match="Coefficient"):
            heritability_landscape(results.drop(columns="Coefficient"), trait="t1")

    def test_heritability_landscape_needs_fdr_columns(self):
        bare_frame = pd.DataFrame(
            {"trait": ["t"], "factor": ["dim_0"], "Coefficient_z-score": [1.0]}
        )
        with pytest.raises(KeyError, match="add_fdr"):
            heritability_landscape(bare_frame, trait="t")

    def test_heritability_landscape_empty_trait(self, results):
        with pytest.raises(ValueError, match="no results to plot"):
            heritability_landscape(results, trait="absent")

    def test_heritability_landscape_titles_map_dim_to_dr_names(self):
        rows = [
            {
                "trait": "t",
                "factor": "dim_5",
                "direction": "pos",
                "Coefficient": 3e-8,
                "Coefficient_std_error": 1e-8,
                "Coefficient_z-score": 3.0,
            },
            {
                "trait": "t",
                "factor": "dim_5",
                "direction": "neg",
                "Coefficient": -1e-8,
                "Coefficient_std_error": 1e-8,
                "Coefficient_z-score": -1.0,
            },
        ]
        frame = add_fdr(pd.DataFrame(rows), by="trait")
        fig, (_, z_ax) = heritability_landscape(frame, trait="t", titles={"dim_5": "DR 6"})
        assert {t.get_text() for t in z_ax.texts} == {"DR 6+", "DR 6-"}

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
            rows.append({"trait": "a", "factor": f"dim_{k}", "Coefficient_z-score": 1.0 * k})
            rows.append({"trait": "b", "factor": f"dim_{k}", "Coefficient_z-score": 1.0 * k + 0.3})
        fig, (_, hist) = trait_concordance(pd.DataFrame(rows), traits=["a", "b"])
        assert isinstance(fig, Figure)
        assert "+0.30" in hist.get_title()

    def test_trait_concordance_with_a_single_shared_factor(self):
        rows = [
            {"trait": "a", "factor": "dim_0", "Coefficient_z-score": 2.0},
            {"trait": "b", "factor": "dim_0", "Coefficient_z-score": 2.5},
        ]
        fig, _ = trait_concordance(pd.DataFrame(rows), traits=["a", "b"])
        assert isinstance(fig, Figure)

    def test_trait_concordance_titles_map_dim_to_dr_names(self):
        rows = [
            {"trait": "a", "factor": "dim_5", "direction": "pos", "Coefficient_z-score": 2.0},
            {"trait": "b", "factor": "dim_5", "direction": "pos", "Coefficient_z-score": 2.5},
        ]
        fig, (scatter, _) = trait_concordance(
            pd.DataFrame(rows), traits=["a", "b"], titles={"dim_5": "DR 6"}
        )
        assert {t.get_text() for t in scatter.texts} == {"DR 6+"}

    @pytest.fixture
    def risk(self):
        return pd.DataFrame(
            {
                "peak": ["chr1:100-200", "chr1:300-400", "chr2:100-200", "chr2:300-400"],
                "chrom": ["chr1", "chr1", "chr2", "chr2"],
                "start": [100, 300, 100, 300],
                "end": [200, 400, 200, 400],
                "factor": ["dim_0", "dim_1", "dim_0", "dim_1"],
                "direction": ["pos", "pos", "neg", "neg"],
                "loading": [0.5, 0.2, 0.3, 0.9],
                "tau": [2.0, 1.0, 1.0, 1.0],
                "risk_rate": [1.0, 0.2, 0.3, 0.9],
                "risk_total": [3.0, 0.4, 0.6, 2.7],
            }
        )

    def test_risk_manhattan_one_point_per_peak_by_default(self, risk):
        fig, ax = risk_manhattan(risk)
        assert isinstance(fig, Figure)
        assert len(ax.collections[0].get_offsets()) == risk["peak"].nunique()

    def test_risk_manhattan_factor_subset_keeps_every_row(self, risk):
        _, ax = risk_manhattan(risk, factor_subset=["dim_0"])
        expected = (risk["factor"] == "dim_0").sum()
        assert len(ax.collections[0].get_offsets()) == expected

    def test_risk_manhattan_unknown_factor_subset_is_refused(self, risk):
        with pytest.raises(ValueError, match="factor_subset"):
            risk_manhattan(risk, factor_subset=["not_a_dim"])

    def test_risk_manhattan_old_dim_subset_and_dim_column_still_work(self, risk):
        legacy = risk.rename(columns={"factor": "dim"})
        with pytest.warns(DeprecationWarning) as caught:
            _, ax = risk_manhattan(legacy, dim_subset=["dim_0"])
        messages = " ".join(str(w.message) for w in caught)
        assert "dim_subset" in messages and "'dim' column" in messages
        assert len(ax.collections[0].get_offsets()) == (risk["factor"] == "dim_0").sum()

    def test_risk_manhattan_needs_risk_columns(self):
        with pytest.raises(KeyError, match="chrom"):
            risk_manhattan(pd.DataFrame({"peak": ["a"]}))

    def test_risk_manhattan_chrom_order_is_numeric_then_alphabetic(self):
        frame = pd.DataFrame(
            {
                "peak": ["chr10:1-2", "chr2:1-2", "chrX:1-2"],
                "chrom": ["chr10", "chr2", "chrX"],
                "start": [1, 1, 1],
                "end": [2, 2, 2],
                "factor": ["dim_0"] * 3,
                "direction": ["pos"] * 3,
                "risk_total": [1.0, 1.0, 1.0],
            }
        )
        _, ax = risk_manhattan(frame)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["chr2", "chr10", "chrX"]


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

    def test_factor_correlation_two_panels(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 3))
        signed = pd.DataFrame(x, columns=[f"DR {i}" for i in range(1, 4)])
        arms = pd.concat(
            {
                f: pd.DataFrame(
                    {"+": np.clip(signed[f], 0, None), "-": np.clip(-signed[f], 0, None)}
                )
                for f in signed
            },
            axis=1,
        )
        arms.columns = [f"{f}{d}" for f, d in arms.columns]
        fig, (left, right) = factor_correlation(signed.corr(), split=arms.corr(), vanished=["DR 2"])
        assert len(fig.axes) == 3  # two matrices and one colourbar
        assert [t.get_text() for t in right.get_xticklabels()].count("DR 2+") <= 1

    def test_factor_correlation_needs_square(self):
        with pytest.raises(ValueError, match="square"):
            factor_correlation(pd.DataFrame(np.zeros((2, 3))))


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
        fig2, _ = latent_dimension_stats(dim_stats, columns=("mean",), remove_vanished=True)
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
        np.testing.assert_array_equal(ax1.images[0].get_array(), ax2.images[0].get_array())

    def test_column_labels_match_titles_in_rank_order(self, heatmap_embed):
        _, ax = latent_heatmap(
            heatmap_embed,
            "grouping",
            remove_vanished=False,
            make_balanced=False,
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
            heatmap_embed,
            "grouping",
            sort_by_categorical=True,
            make_balanced=False,
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
            heatmap_embed,
            "grouping",
            heritability=heritability,
            make_balanced=False,
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
                heatmap_embed,
                "grouping",
                heritability=heritability,
                make_balanced=False,
                directional=False,
            )

    def test_heritability_se_draws_error_bars(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        se = pd.Series(rng.random(heatmap_embed.n_vars) * 0.5 + 0.1, index=heatmap_embed.var_names)
        fig, (bar, ax) = latent_heatmap(
            heatmap_embed,
            "grouping",
            heritability=heritability,
            heritability_se=se,
            make_balanced=False,
            directional=False,
        )
        assert bar.containers[0].has_yerr

    def test_missing_heritability_se_value_raises(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        se = pd.Series(rng.random(heatmap_embed.n_vars), index=heatmap_embed.var_names).drop(
            heatmap_embed.var_names[0]
        )
        with pytest.raises(KeyError, match="heritability_se has no value"):
            latent_heatmap(
                heatmap_embed,
                "grouping",
                heritability=heritability,
                heritability_se=se,
                make_balanced=False,
                directional=False,
            )

    def test_cell_scores_draw_group_sem_error_bars(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names)
        fig, (ax, score) = latent_heatmap(
            heatmap_embed, "grouping", cell_scores=cell_scores, make_balanced=False
        )
        assert score.containers[0].has_xerr

    def test_heritability_q_colors_and_draws_significance_legend(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        q = pd.Series(np.linspace(0.001, 0.5, heatmap_embed.n_vars), index=heatmap_embed.var_names)
        fig, (bar, ax) = latent_heatmap(
            heatmap_embed,
            "grouping",
            heritability=heritability,
            heritability_q=q,
            make_balanced=False,
            directional=False,
        )
        assert bar.legend_ is not None
        assert bar.lines

    def test_cell_scores_adds_one_bar_per_group_on_the_right(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names)
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
            latent_heatmap(heatmap_embed, "grouping", cell_scores=cell_scores, make_balanced=False)

    def test_heritability_and_cell_scores_together_return_three_axes(self, heatmap_embed):
        rng = np.random.default_rng(6)
        heritability = pd.Series(
            rng.normal(size=heatmap_embed.n_vars), index=heatmap_embed.var_names
        )
        cell_scores = pd.Series(rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names)
        fig, (bar, ax, score) = latent_heatmap(
            heatmap_embed,
            "grouping",
            heritability=heritability,
            cell_scores=cell_scores,
            make_balanced=False,
            directional=False,
        )
        assert len({id(bar), id(ax), id(score)}) == 3

    def test_invalid_cell_score_agg_raises(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names)
        with pytest.raises(ValueError, match="cell_score_agg"):
            latent_heatmap(
                heatmap_embed,
                "grouping",
                cell_scores=cell_scores,
                cell_score_agg="bogus",
                make_balanced=False,
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
        cell_scores = pd.Series(rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names)
        fig, (ax, score) = latent_heatmap(
            heatmap_embed,
            "grouping",
            cell_scores=cell_scores,
            cell_score_agg="sum",
            make_balanced=False,
        )
        expected = cell_scores.groupby(heatmap_embed.obs["grouping"]).sum()
        actual = sorted(b.get_width() for b in score.patches)
        np.testing.assert_allclose(actual, sorted(expected.to_numpy()))

    def test_colorbar_moves_to_top_right_when_cell_scores_shown(self, heatmap_embed):
        rng = np.random.default_rng(7)
        cell_scores = pd.Series(rng.normal(size=heatmap_embed.n_obs), index=heatmap_embed.obs_names)
        fig, (ax, score) = latent_heatmap(
            heatmap_embed, "grouping", cell_scores=cell_scores, make_balanced=False
        )
        # the relocated colorbar is a small box flush against the figure's top right
        # corner -- distinct from the heatmap/groupby/score axes, which all sit well
        # inside the figure.
        assert any(
            a.get_position().x1 > 0.9 and a.get_position().y1 > 0.9 and a.get_position().width < 0.2
            for a in fig.axes
        )


class TestLatentHeatmapSeries:
    def test_custom_bar_colors_and_one_font_size(self, embed):
        from matplotlib import colormaps
        from matplotlib.colors import to_rgba

        keys = [f"DR {i + 1}{d}" for i in range(6) for d in "+-"]
        z = pd.Series(np.arange(12.0), index=keys)
        cats = sorted(embed.obs["grouping"].unique())
        row = pd.Series(np.arange(float(len(cats))), index=cats)
        fig, axes = latent_heatmap(
            embed,
            "grouping",
            aggregate="mean",
            column_series=[
                BarSeries(z, "z", color={k: "#D55E00" for k in keys}),
                BarSeries(z, "z2", color=colormaps["viridis"]),
            ],
            row_series=[
                BarSeries(row, "r", color=dict(zip(cats, ["red"] * len(cats), strict=True)))
            ],
        )
        assert to_rgba(axes.columns[0].patches[0].get_facecolor()) == to_rgba("#D55E00")
        assert to_rgba(axes.rows_right[0].patches[0].get_facecolor()) == to_rgba("red")
        sizes = {
            t.get_fontsize() for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()
        }
        assert sizes == {7.0}

    def test_color_missing_key_and_q_conflict(self, embed):
        z = pd.Series(1.0, index=[f"DR {i + 1}{d}" for i in range(6) for d in "+-"])
        with pytest.raises(KeyError, match="no colour"):
            latent_heatmap(embed, "grouping", column_series=[BarSeries(z, color={"DR 1+": "red"})])
        with pytest.raises(ValueError, match="significance"):
            latent_heatmap(embed, "grouping", column_series=[BarSeries(z, q=z, color="red")])

    """The list-based bar series, ``order="specificity"`` and ``aggregate="mean"``."""

    @pytest.fixture
    def embed_h(self, dim_stats):
        rng = np.random.default_rng(4)
        n = 400
        x = rng.normal(size=(n, len(dim_stats))).astype(np.float32)
        obs = pd.DataFrame(
            {"grouping": rng.choice(["a", "b", "c"], size=n)},
            index=[f"c{i}" for i in range(n)],
        )
        return ad.AnnData(X=x, obs=obs, var=dim_stats.copy())

    @staticmethod
    def _kept(embed_h):
        return embed_h.var_names[~embed_h.var["vanished"].to_numpy(dtype=bool)]

    def test_no_series_still_returns_the_heatmap_axes_alone(self, embed_h):
        _, ax = latent_heatmap(embed_h, "grouping", make_balanced=False)
        assert not isinstance(ax, LatentHeatmapAxes)

    def test_column_series_stack_upward_in_list_order(self, embed_h):
        kept = self._kept(embed_h)
        first = BarSeries(pd.Series(np.arange(len(kept), dtype=float), index=kept), label="a")
        second = BarSeries(pd.Series(np.ones(len(kept)), index=kept), label="b", size_in=0.6)
        _, axes = latent_heatmap(
            embed_h,
            "grouping",
            column_series=[first, second],
            make_balanced=False,
            directional=False,
        )
        assert isinstance(axes, LatentHeatmapAxes)
        near, far = axes.columns
        heat = axes.heat.get_position()
        assert heat.y1 <= near.get_position().y0 < near.get_position().y1
        assert near.get_position().y1 <= far.get_position().y0  # no overlap
        assert len(near.patches) == len(far.patches) == len(kept)
        assert near.get_ylabel() == "a" and far.get_ylabel() == "b"
        assert near.get_xlim() == axes.heat.get_xlim()  # shares the heatmap's x axis

    def test_column_series_may_be_keyed_by_displayed_title(self, embed_h):
        titles = embed_h.var.loc[self._kept(embed_h), "title"]
        series = BarSeries(pd.Series(np.arange(len(titles), dtype=float), index=titles.to_numpy()))
        _, axes = latent_heatmap(
            embed_h, "grouping", column_series=[series], make_balanced=False, directional=False
        )
        assert len(axes.columns[0].patches) == len(titles)

    def test_column_series_with_a_missing_column_is_named(self, embed_h):
        kept = self._kept(embed_h)
        series = BarSeries(pd.Series(np.ones(len(kept) - 1), index=kept[1:]), name="tau")
        with pytest.raises(KeyError, match="tau has no value"):
            latent_heatmap(
                embed_h,
                "grouping",
                column_series=[series],
                make_balanced=False,
                directional=False,
            )

    def test_row_series_on_both_sides_sit_outside_the_heatmap(self, embed_h):
        cats = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        _, axes = latent_heatmap(
            embed_h,
            "grouping",
            make_balanced=False,
            row_series=[BarSeries(cats, label="r1"), BarSeries(cats * 2, label="r2")],
            row_series_left=[BarSeries(cats, label="l1", color="#CC0000")],
        )
        heat = axes.heat.get_position()
        assert len(axes.rows_right) == 2 and len(axes.rows_left) == 1
        r1, r2 = axes.rows_right
        assert r1.get_position().x0 >= heat.x1
        assert r2.get_position().x0 >= r1.get_position().x1  # second is farther out
        assert axes.rows_left[0].get_position().x1 <= heat.x0
        assert axes.rows_left[0].xaxis_inverted()  # grows away from the heatmap
        assert len(r1.patches) == 3

    def test_row_series_missing_a_category_is_named(self, embed_h):
        series = BarSeries(pd.Series({"a": 1.0, "b": 2.0}), name="depth")
        with pytest.raises(KeyError, match="depth has no value"):
            latent_heatmap(embed_h, "grouping", row_series=[series], make_balanced=False)

    def test_per_cell_row_series_reduce_to_the_group_statistic(self, embed_h):
        rng = np.random.default_rng(7)
        per_cell = pd.Series(rng.normal(size=embed_h.n_obs), index=embed_h.obs_names)
        _, axes = latent_heatmap(
            embed_h,
            "grouping",
            make_balanced=False,
            row_series=[BarSeries(per_cell, agg="median")],
        )
        expected = per_cell.groupby(embed_h.obs["grouping"]).median()
        actual = sorted(b.get_width() for b in axes.rows_right[0].patches)
        np.testing.assert_allclose(actual, sorted(expected.to_numpy()))

    def test_order_specificity_sorts_columns_descending(self, embed_h):
        kept = self._kept(embed_h)
        score = pd.Series(np.arange(len(kept), dtype=float), index=kept)  # last is highest
        _, ax = latent_heatmap(
            embed_h,
            "grouping",
            order="specificity",
            specificity=score,
            make_balanced=False,
            directional=False,
        )
        shown = [t.get_text() for t in ax.get_xticklabels()]
        expected = embed_h.var.loc[kept[::-1], "title"].tolist()
        assert shown == expected

    def test_order_specificity_accepts_titles_and_needs_scores(self, embed_h):
        kept = self._kept(embed_h)
        titles = embed_h.var.loc[kept, "title"].to_numpy()
        score = pd.Series(np.arange(len(kept), dtype=float), index=titles)
        _, ax = latent_heatmap(
            embed_h,
            "grouping",
            order="specificity",
            specificity=score,
            make_balanced=False,
            directional=False,
        )
        assert [t.get_text() for t in ax.get_xticklabels()][0] == titles[-1]
        with pytest.raises(ValueError, match="needs `specificity`"):
            latent_heatmap(embed_h, "grouping", order="specificity", make_balanced=False)
        with pytest.raises(KeyError, match="specificity has no value"):
            latent_heatmap(
                embed_h,
                "grouping",
                order="specificity",
                specificity=score.iloc[1:],
                make_balanced=False,
                directional=False,
            )

    def test_unknown_order_and_aggregate_are_refused(self, embed_h):
        with pytest.raises(ValueError, match="order must be"):
            latent_heatmap(embed_h, "grouping", order="alphabetical")
        with pytest.raises(ValueError, match="aggregate must be"):
            latent_heatmap(embed_h, "grouping", aggregate="median")

    def test_aggregate_mean_draws_one_row_per_category_of_group_means(self, embed_h):
        kept = self._kept(embed_h)
        _, ax = latent_heatmap(embed_h, "grouping", aggregate="mean", directional=False)
        image = np.asarray(ax.images[0].get_array())
        expected = (
            pd.DataFrame(embed_h[:, kept].X, columns=kept)
            .groupby(embed_h.obs["grouping"].to_numpy())
            .mean()
        )
        assert image.shape == (3, len(kept))
        np.testing.assert_allclose(image, expected.to_numpy(), rtol=1e-4, atol=1e-6)

    def test_aggregate_mean_keeps_row_series_per_category(self, embed_h):
        rng = np.random.default_rng(7)
        per_cell = pd.Series(rng.normal(size=embed_h.n_obs), index=embed_h.obs_names)
        _, axes = latent_heatmap(
            embed_h,
            "grouping",
            aggregate="mean",
            row_series=[BarSeries(per_cell, agg="mean")],
        )
        expected = per_cell.groupby(embed_h.obs["grouping"]).mean()  # all cells, not a subsample
        actual = sorted(b.get_width() for b in axes.rows_right[0].patches)
        np.testing.assert_allclose(actual, sorted(expected.to_numpy()))
        assert len(axes.rows_right[0].patches) == 3

    def test_legacy_arguments_warn_and_keep_the_old_return_shape(self, embed_h):
        scores = pd.Series(np.ones(embed_h.n_obs), index=embed_h.obs_names)
        with pytest.warns(DeprecationWarning, match="cell_scores"):
            _, shape = latent_heatmap(embed_h, "grouping", cell_scores=scores, make_balanced=False)
        assert (
            isinstance(shape, tuple)
            and len(shape) == 2
            and not isinstance(shape, LatentHeatmapAxes)
        )


def test_chrom_sort_key_is_public_and_orders_numerically():
    assert "chrom_sort_key" in _pl_enrichment.__all__
    assert _pl_enrichment._chrom_sort_key is chrom_sort_key  # private alias kept
    labels = ["chrX", "chr10", "chr2", "1", "chrM"]
    assert sorted(labels, key=chrom_sort_key) == ["1", "chr2", "chr10", "chrM", "chrX"]
